"""OpenAI-compatible HTTP server around :class:`~forge.serve.engine.LLMEngine` (Plan SV-1, ARCHITECTURE 5.8).

Endpoints: ``GET /v1/models``, ``POST /v1/completions``, ``POST /v1/chat/completions`` (SSE streaming with
``stream: true``), ``GET /health``, ``GET /stats``, ``POST /abort`` and ``GET /v1/engine/config``. Optional bearer
``api_key`` protects everything except ``/health``.

The engine is driven by one background thread (:class:`EngineLoop`); HTTP handlers only enqueue requests and await
per-request asyncio queues, so slow clients never block the GPU loop. A client disconnect aborts the request.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from forge.core.errors import ConfigError, ForgeError
from forge.serve.sampling import SamplingParams

log = logging.getLogger("forge.serve.server")

MAX_N = 64  # cap on choices per request: every choice is a full sequence in the KV pool


# --------------------------------------------------------------------------------------------------
# engine loop
# --------------------------------------------------------------------------------------------------
class EngineLoop:
    """Runs ``engine.step()`` in a daemon thread and routes outputs to subscribers."""

    def __init__(self, engine: Any, idle_sleep_s: float = 0.002) -> None:
        self.engine = engine
        self.idle_sleep = idle_sleep_s
        self._subs: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.error: str | None = None
        self.crashes = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="forge-engine-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def subscribe(self, request_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs[request_id] = (loop, q)
        return q

    def unsubscribe(self, request_id: str) -> None:
        with self._lock:
            self._subs.pop(request_id, None)

    def submit(self, prompt_ids: list[int], params: SamplingParams, request_id: str, priority: int = 0) -> None:
        self.engine.add_request(prompt_ids, params, request_id=request_id, priority=priority)
        self._wake.set()

    def abort(self, request_id: str) -> None:
        self.engine.abort(request_id)
        self._wake.set()

    def _deliver(self, rid: str, item: Any) -> None:
        with self._lock:
            sub = self._subs.get(rid)
        if sub is None:
            return
        loop, q = sub
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:  # loop closed
            self.unsubscribe(rid)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.engine.has_unfinished():
                    mgr = self.engine.manager
                    if mgr.tiering_enabled:  # idle: push cold prefix-cache blocks down the HBM -> host -> NVMe hierarchy
                        mgr.demote_cold(8)
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue
                for out in self.engine.step():
                    self._deliver(out.request_id, out)
                if not self.engine.last_step_info:
                    time.sleep(self.idle_sleep)  # nothing could be scheduled (memory pressure): do not spin
            except Exception as e:  # noqa: BLE001 - never let the loop die silently
                self.crashes += 1
                self.error = f"{type(e).__name__}: {e}"
                log.exception("engine step failed; failing in-flight requests")
                with self._lock:
                    ids = list(self._subs)
                for rid in ids:
                    self._deliver(rid, e)
                    try:
                        self.engine.abort(rid)
                    except Exception:  # pragma: no cover
                        pass
                time.sleep(self.idle_sleep * 5)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    try:
        return float(o)
    except Exception:
        return str(o)


def _err(message: str, etype: str = "invalid_request_error", code: str | None = None, status: int = 400):
    return JSONResponse({"error": {"message": message, "type": etype, "code": code}}, status_code=status)


def _tok_str(tokenizer: Any, tid: int) -> str:
    if tokenizer is None:
        return str(tid)
    try:
        return tokenizer.decode([tid], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([tid])
    except Exception:
        return str(tid)


def _stop_list(v: Any) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else [str(x) for x in v]


def build_params(engine: Any, body: dict[str, Any], *, chat: bool) -> SamplingParams:
    lp = body.get("logprobs")
    if chat:
        lp = (int(body.get("top_logprobs") or 0)) if lp else None
    elif lp is not None:
        lp = int(lp)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    try:
        if body.get("n") is not None and not (1 <= int(body["n"]) <= MAX_N):
            raise ConfigError(f"n must be between 1 and {MAX_N}")
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad sampling parameters: {e}") from e
    try:
        return SamplingParams.from_defaults(
            engine.cfg.sampling,
            temperature=body.get("temperature"), top_p=body.get("top_p"), top_k=body.get("top_k"), min_p=body.get("min_p"),
            repetition_penalty=body.get("repetition_penalty"), presence_penalty=body.get("presence_penalty"),
            frequency_penalty=body.get("frequency_penalty"), max_tokens=max_tokens, min_tokens=body.get("min_tokens"),
            stop=_stop_list(body.get("stop")), stop_token_ids=body.get("stop_token_ids"), ignore_eos=body.get("ignore_eos"),
            seed=body.get("seed"), logprobs=lp, logit_bias=body.get("logit_bias"), n=body.get("n"),
        )
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad sampling parameters: {e}") from e


def chat_prompt_ids(engine: Any, messages: list[dict[str, Any]], template: str | None) -> list[int]:
    tok = engine.tokenizer
    if tok is None:
        raise ConfigError("this deployment has no tokenizer; send token ids to /v1/completions instead")
    if not messages:
        raise ConfigError("messages must be non-empty")
    msgs = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts -> concatenate text parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        msgs.append({"role": m.get("role", "user"), "content": content or ""})
    if hasattr(tok, "apply_chat_template"):
        kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if template:
            kw["chat_template"] = template
        try:
            text = tok.apply_chat_template(msgs, **kw)
        except Exception as e:  # tokenizer without template
            log.warning("apply_chat_template failed (%s); using a plain-text fallback", e)
            text = _fallback_template(msgs)
    else:
        text = _fallback_template(msgs)
    return _encode(tok, text, add_special=False)


def _fallback_template(msgs: list[dict[str, Any]]) -> str:
    return "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant:"


def _encode(tok: Any, text: str, add_special: bool = True) -> list[int]:
    try:
        ids = tok.encode(text, add_special_tokens=add_special)
    except TypeError:
        ids = tok.encode(text)
    return [int(i) for i in ids]


def completion_prompts(engine: Any, prompt: Any) -> list[list[int]]:
    tok = engine.tokenizer
    if isinstance(prompt, str):
        if tok is None:
            raise ConfigError("this deployment has no tokenizer; send token ids")
        return [_encode(tok, prompt)]
    if isinstance(prompt, list) and prompt and all(isinstance(x, int) for x in prompt):
        return [list(prompt)]
    if isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt):
        if tok is None:
            raise ConfigError("this deployment has no tokenizer; send token ids")
        return [_encode(tok, p) for p in prompt]
    if isinstance(prompt, list) and prompt and all(isinstance(x, list) for x in prompt):
        return [[int(i) for i in p] for p in prompt]
    raise ConfigError("prompt must be a string, a list of strings, a list of token ids or a list of token-id lists")


def _lp_completion(engine: Any, lps: list[dict] | None) -> dict | None:
    if lps is None:
        return None
    tok = engine.tokenizer
    return {
        "tokens": [_tok_str(tok, d["token_id"]) for d in lps],
        "token_logprobs": [d["logprob"] for d in lps],
        "top_logprobs": [{_tok_str(tok, t): lp for t, lp in d["top"]} for d in lps],
        "token_ids": [d["token_id"] for d in lps],
    }


def _lp_chat(engine: Any, lps: list[dict] | None) -> dict | None:
    if lps is None:
        return None
    tok = engine.tokenizer
    return {"content": [{"token": _tok_str(tok, d["token_id"]), "logprob": d["logprob"], "token_id": d["token_id"],
                         "top_logprobs": [{"token": _tok_str(tok, t), "logprob": lp, "token_id": t} for t, lp in d["top"]]}
                        for d in lps]}


# --------------------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------------------
def create_app(engine: Any, *, api_key: str | None = None, model_name: str | None = None, loop: EngineLoop | None = None):
    """Build the FastAPI app. The engine loop thread is started immediately and stopped on app shutdown."""
    cfg = engine.cfg
    api_key = api_key if api_key is not None else cfg.api_key
    name = model_name or cfg.display_name()
    eloop = loop or EngineLoop(engine)
    eloop.start()
    created = int(time.time())

    async def lifespan(app: FastAPI):
        eloop.start()
        yield
        eloop.stop()

    app = FastAPI(title="Forge LLM serving", version="1", lifespan=lifespan)
    app.state.engine = engine
    app.state.loop = eloop

    async def auth(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        token = token or x_api_key
        if token is None or not hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.exception_handler(ForgeError)
    async def forge_error_handler(request: Request, exc: ForgeError):  # noqa: ARG001
        return _err(exc.message or str(exc), etype=exc.code, code=exc.code, status=exc.http_status if exc.http_status < 500 else 500)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):  # noqa: ARG001
        return _err(str(exc.detail), etype="auth_error" if exc.status_code == 401 else "http_error", status=exc.status_code)

    # ---- simple endpoints -------------------------------------------------------------------------
    @app.get("/health")
    async def health():
        st = {"status": "ok" if eloop.alive and not eloop.error else "degraded", "model": name,
              "running": len(engine.scheduler.running), "waiting": len(engine.scheduler.waiting)}
        if eloop.error:
            st["last_error"] = eloop.error
        return JSONResponse(st, status_code=200 if eloop.alive else 503)

    @app.get("/stats", dependencies=[Depends(auth)])
    async def stats():
        return JSONResponse(_json_safe(engine.stats()))

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "forge",
                                            "max_model_len": cfg.max_model_len}]}

    @app.get("/v1/engine/config", dependencies=[Depends(auth)])
    async def engine_config():
        d = cfg.model_dump()
        d["api_key"] = None
        d["resolved"] = {"num_blocks": engine.pool.num_blocks, "kv_dtype": engine.kv_name, "device": str(engine.device),
                         "compute_dtype": str(engine.compute_dtype), "speculation": type(engine.drafter).__name__ if engine.drafter else None}
        return JSONResponse(_json_safe(d))

    @app.post("/abort", dependencies=[Depends(auth)])
    async def abort(request: Request):
        body = await request.json()
        rid = body.get("request_id") if isinstance(body, dict) else None
        if not rid:
            return _err("request_id required")
        eloop.abort(str(rid))
        return {"aborted": True, "request_id": rid}

    # ---- generation core --------------------------------------------------------------------------
    async def run_request(request: Request, prompts: list[list[int]], params: SamplingParams, priority: int, prefix: str):
        """Submit all prompts; returns (base_ids, merged async iterator of RequestOutput)."""
        loop = asyncio.get_running_loop()
        base_ids = [f"{prefix}-{uuid.uuid4().hex[:16]}" for _ in prompts]
        queues = {rid: eloop.subscribe(rid, loop) for rid in base_ids}
        submitted: list[str] = []
        try:
            for rid, p in zip(base_ids, prompts):
                eloop.submit(p, params, rid, priority)
                submitted.append(rid)
        except Exception:
            for rid in base_ids:
                eloop.unsubscribe(rid)
            for rid in submitted:
                eloop.abort(rid)
            raise
        merged: asyncio.Queue = asyncio.Queue()

        async def pump(idx: int, rid: str):
            q = queues[rid]
            remaining = params.n
            while remaining > 0:
                item = await q.get()
                await merged.put((idx, item))
                if isinstance(item, Exception):
                    return
                if getattr(item, "finished", False):
                    remaining -= 1
            eloop.unsubscribe(rid)

        tasks = [asyncio.create_task(pump(i, rid)) for i, rid in enumerate(base_ids)]

        async def gen():
            total = len(prompts) * params.n
            finished = 0
            try:
                while finished < total:
                    try:
                        idx, item = await asyncio.wait_for(merged.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            raise asyncio.CancelledError()  # noqa: B904
                        continue
                    if isinstance(item, Exception):
                        raise HTTPException(status_code=500, detail=f"engine error: {item}")
                    yield idx, item
                    if item.finished:
                        finished += 1
            finally:
                for t in tasks:
                    t.cancel()
                for rid in base_ids:
                    eloop.unsubscribe(rid)
                    if finished < total:
                        eloop.abort(rid)

        return base_ids, gen()

    def usage(prompt_tokens: int, completion_tokens: int) -> dict:
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}

    async def handle(request: Request, chat: bool):
        try:
            body = await request.json()
        except Exception:
            return _err("request body must be JSON")
        if not isinstance(body, dict):
            return _err("request body must be a JSON object")
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        params = build_params(engine, body, chat=chat)
        if chat:
            prompts = [chat_prompt_ids(engine, body.get("messages") or [], body.get("chat_template") or cfg.chat_template)]
        else:
            prompts = completion_prompts(engine, body.get("prompt"))
        try:
            priority = int(body.get("priority") or 0)
        except (TypeError, ValueError):
            return _err("priority must be an integer")
        echo_text = ""
        if not chat and body.get("echo") and isinstance(body.get("prompt"), str):
            echo_text = body["prompt"]
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        obj = "chat.completion" if chat else "text_completion"
        now = int(time.time())
        ptoks = sum(len(p) for p in prompts)
        base_ids, it = await run_request(request, prompts, params, priority, "http")
        lp_fn = _lp_chat if chat else _lp_completion

        if not stream:
            texts: dict[tuple[int, int], str] = {}
            reasons: dict[tuple[int, int], str | None] = {}
            lps: dict[tuple[int, int], list[dict]] = {}
            ctoks = 0
            async for idx, out in it:
                key = (idx, out.index)
                texts[key] = texts.get(key, "") + out.new_text
                if out.logprobs:
                    lps.setdefault(key, []).extend(out.logprobs)
                if out.finished:
                    reasons[key] = out.finish_reason
                    ctoks += out.completion_tokens
            choices = []
            for n_i, key in enumerate(sorted(texts)):
                txt = texts[key]
                fr = reasons.get(key)
                lp = lp_fn(engine, lps.get(key)) if params.logprobs is not None else None
                if chat:
                    choices.append({"index": n_i, "message": {"role": "assistant", "content": txt}, "logprobs": lp, "finish_reason": fr})
                else:
                    choices.append({"index": n_i, "text": echo_text + txt, "logprobs": lp, "finish_reason": fr})
            return JSONResponse({"id": rid, "object": obj, "created": now, "model": name, "choices": choices,
                                 "usage": usage(ptoks, ctoks)})

        async def sse():
            def chunk(choice: dict, usage_obj: dict | None = None) -> str:
                d = {"id": rid, "object": obj + ".chunk" if chat else obj, "created": now, "model": name, "choices": [choice]}
                if usage_obj is not None:
                    d["usage"] = usage_obj
                return "data: " + json.dumps(d) + "\n\n"

            ctoks = 0
            started: set[tuple[int, int]] = set()
            try:
                if echo_text:
                    yield chunk({"index": 0, "text": echo_text, "finish_reason": None})
                async for idx, out in it:
                    ci = idx * params.n + out.index
                    if chat and (idx, out.index) not in started:
                        started.add((idx, out.index))
                        yield chunk({"index": ci, "delta": {"role": "assistant", "content": ""}, "finish_reason": None})
                    if out.new_text or out.logprobs:
                        lp = lp_fn(engine, out.logprobs) if params.logprobs is not None else None
                        if chat:
                            yield chunk({"index": ci, "delta": {"content": out.new_text}, "logprobs": lp, "finish_reason": None})
                        else:
                            yield chunk({"index": ci, "text": out.new_text, "logprobs": lp, "finish_reason": None})
                    if out.finished:
                        ctoks += out.completion_tokens
                        if chat:
                            yield chunk({"index": ci, "delta": {}, "finish_reason": out.finish_reason})
                        else:
                            yield chunk({"index": ci, "text": "", "finish_reason": out.finish_reason})
                if include_usage:
                    d = {"id": rid, "object": obj + ".chunk" if chat else obj, "created": now, "model": name, "choices": [],
                         "usage": usage(ptoks, ctoks)}
                    yield "data: " + json.dumps(d) + "\n\n"
                yield "data: [DONE]\n\n"
            except HTTPException as e:
                yield "data: " + json.dumps({"error": {"message": str(e.detail), "type": "engine_error"}}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await it.aclose()

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/completions", dependencies=[Depends(auth)])
    async def completions(request: Request):
        return await handle(request, chat=False)

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat_completions(request: Request):
        return await handle(request, chat=True)

    return app
