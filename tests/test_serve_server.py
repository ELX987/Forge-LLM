"""OpenAI-compatible server, job entry (run_server) and client helper - all on CPU with a tiny model."""
from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from forge.serve._testing import ByteTokenizer, make_engine, tiny_model
from forge.serve.client import ForgeServeClient, ServeClientError
from forge.serve.server import create_app


@pytest.fixture()
def served():
    m = tiny_model(0)
    eng = make_engine(m)
    app = create_app(eng, model_name="tiny-model")
    with TestClient(app) as c:
        yield c, eng, m
    app.state.loop.stop()


def sse_events(resp) -> list:
    out = []
    for line in resp.iter_lines():
        if line.startswith("data:"):
            d = line[5:].strip()
            out.append(d if d == "[DONE]" else json.loads(d))
    return out


def test_health_models_stats(served):
    c, eng, _ = served
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "tiny-model"
    models = c.get("/v1/models").json()
    assert models["data"][0]["id"] == "tiny-model" and models["data"][0]["max_model_len"] == 160
    st = c.get("/stats").json()
    for k in ("running", "waiting", "kv_usage", "prefix_hit_rate", "tokens_per_s", "ttft_ms_p50", "ttft_ms_p99", "itl_ms_p50",
              "itl_ms_p99", "spec_acceptance_by_position", "kv"):
        assert k in st
    assert "fragmentation" in st["kv"] and "bytes_by_tier" in st["kv"]
    cfg = c.get("/v1/engine/config").json()
    assert cfg["api_key"] is None and cfg["resolved"]["num_blocks"] == eng.pool.num_blocks


def test_completion_matches_engine_greedy(served):
    c, _, m = served
    prompt = "hello world, hello"
    tok = ByteTokenizer()
    ids = tok.encode(prompt, add_special_tokens=True)  # completions use the tokenizer's default special tokens (BOS)
    ref = tok.decode(m.greedy_generate(ids, 12, eos=None))
    r = c.post("/v1/completions", json={"prompt": prompt, "max_tokens": 12, "temperature": 0, "ignore_eos": True}).json()
    assert r["object"] == "text_completion" and r["choices"][0]["text"] == ref
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"] == {"prompt_tokens": len(ids), "completion_tokens": 12, "total_tokens": len(ids) + 12}
    # token-id prompt and batch of prompts
    r2 = c.post("/v1/completions", json={"prompt": ids, "max_tokens": 12, "temperature": 0, "ignore_eos": True}).json()
    assert r2["choices"][0]["text"] == ref
    r3 = c.post("/v1/completions", json={"prompt": [prompt, "abc"], "max_tokens": 4, "temperature": 0, "ignore_eos": True}).json()
    assert len(r3["choices"]) == 2 and r3["choices"][0]["text"] == ref[:4]


def test_chat_completion_and_template(served):
    c, _, m = served
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    r = c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 8, "temperature": 0, "ignore_eos": True}).json()
    ids = ByteTokenizer().apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ref = ByteTokenizer().decode(m.greedy_generate(ids, 8, eos=None))
    ch = r["choices"][0]
    assert r["object"] == "chat.completion" and ch["message"] == {"role": "assistant", "content": ref}
    assert r["usage"]["prompt_tokens"] == len(ids)


def test_streaming_equals_non_streaming(served):
    c, _, _ = served
    body = {"prompt": "the quick brown fox", "max_tokens": 15, "temperature": 0, "ignore_eos": True}
    full = c.post("/v1/completions", json=body).json()["choices"][0]["text"]
    with c.stream("POST", "/v1/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        ev = sse_events(resp)
    assert ev[-1] == "[DONE]"
    text = "".join(e["choices"][0]["text"] for e in ev[:-1] if e["choices"])
    assert text == full
    assert any(e["choices"] and e["choices"][0]["finish_reason"] == "length" for e in ev[:-1])
    assert ev[-2]["usage"]["completion_tokens"] == 15

    cb = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10, "temperature": 0, "ignore_eos": True}
    full_chat = c.post("/v1/chat/completions", json=cb).json()["choices"][0]["message"]["content"]
    with c.stream("POST", "/v1/chat/completions", json={**cb, "stream": True}) as resp:
        ev = sse_events(resp)
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant" and ev[0]["object"] == "chat.completion.chunk"
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1]) == full_chat
    assert ev[-2]["choices"][0]["finish_reason"] in ("length", "stop") and ev[-1] == "[DONE]"


def test_stop_n_seed_logprobs_penalties(served):
    c, _, m = served
    abc = ByteTokenizer().encode("abc", add_special_tokens=True)
    ref = ByteTokenizer().decode(m.greedy_generate(abc, 20, eos=None))
    stop = ref[6:8]
    r = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 20, "temperature": 0, "ignore_eos": True, "stop": [stop]}).json()
    assert r["choices"][0]["finish_reason"] == "stop" and r["choices"][0]["text"] == ref[: ref.find(stop)]
    r = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 6, "n": 3, "temperature": 1.0, "seed": 4, "ignore_eos": True}).json()
    assert len(r["choices"]) == 3 and r["usage"]["completion_tokens"] == 18
    a = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 6, "temperature": 1.0, "seed": 9, "ignore_eos": True}).json()
    b = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 6, "temperature": 1.0, "seed": 9, "ignore_eos": True}).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
    lp = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3, "temperature": 0, "logprobs": 2, "ignore_eos": True}).json()
    lpd = lp["choices"][0]["logprobs"]
    assert len(lpd["tokens"]) == 3 and len(lpd["token_logprobs"]) == 3 and len(lpd["top_logprobs"][0]) <= 2
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 2, "logprobs": True,
                                            "top_logprobs": 1, "temperature": 0, "ignore_eos": True}).json()
    assert len(r["choices"][0]["logprobs"]["content"]) == 2
    r = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 5, "temperature": 0, "logit_bias": {"77": 100},
                                        "repetition_penalty": 1.0, "ignore_eos": True}).json()
    assert r["choices"][0]["text"] == "M" * 5


def test_validation_errors(served):
    c, _, _ = served
    r = c.post("/v1/completions", json={"prompt": "x" * 400, "max_tokens": 5})
    assert r.status_code == 400 and "max_model_len" in r.json()["error"]["message"]
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert c.post("/v1/completions", json={"prompt": 5}).status_code == 400
    assert c.post("/v1/completions", json={"prompt": "a", "temperature": -1}).status_code == 400
    assert c.post("/v1/completions", content=b"not json").status_code == 400


def test_abort_endpoint(served):
    c, eng, _ = served
    assert c.post("/abort", json={"request_id": "nope"}).json()["aborted"] is True  # unknown ids are a no-op
    assert c.post("/abort", json={}).status_code == 400


def test_api_key_auth():
    m = tiny_model(0)
    eng = make_engine(m)
    app = create_app(eng, api_key="sekret")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.post("/v1/completions", json={"prompt": "a", "max_tokens": 1}).status_code == 401
        h = {"Authorization": "Bearer sekret"}
        assert c.get("/v1/models", headers=h).status_code == 200
        assert c.post("/v1/completions", headers=h, json={"prompt": "a", "max_tokens": 1}).status_code == 200
        assert c.get("/stats", headers={"x-api-key": "sekret"}).status_code == 200
        assert c.get("/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
    app.state.loop.stop()


def test_engine_error_is_reported_not_hung(served):
    c, eng, _ = served
    orig = eng.runner.forward

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    eng.runner.forward = boom
    r = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3})
    assert r.status_code == 500 and "synthetic failure" in r.json()["error"]["message"]
    eng.runner.forward = orig
    ok = c.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3, "temperature": 0})
    assert ok.status_code == 200
    assert c.get("/health").json()["status"] == "degraded"  # the error is surfaced to operators


# ---------------------------------------------------------------------------------------- real server + job entry + client
def test_run_server_job_entry_client_and_disconnect():
    from forge.core.context import NullContext
    from forge.serve.config import ServeConfig
    from forge.serve.run import run_server

    m = tiny_model(1)
    eng = make_engine(m)
    cfg = ServeConfig(model="tiny", port=0, host="127.0.0.1", stats_interval_s=0.2, device="cpu", dtype="float32")
    ctx = NullContext()
    result: dict = {}

    def target():
        result["r"] = run_server(cfg, ctx, engine=eng)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    t0 = time.time()
    while "url" not in ctx.outputs and time.time() - t0 < 30:
        time.sleep(0.05)
    assert "url" in ctx.outputs, ctx.logs
    cl = ForgeServeClient(ctx.outputs["url"])
    assert cl.wait_ready(10)
    assert cl.models()["data"][0]["id"] == "tiny"
    out = cl.chat([{"role": "user", "content": "hi"}], max_tokens=6, temperature=0, ignore_eos=True)
    assert out["usage"]["completion_tokens"] == 6
    pieces = list(cl.chat_text([{"role": "user", "content": "hi"}], max_tokens=6, temperature=0, ignore_eos=True))
    assert "".join(pieces) == out["choices"][0]["message"]["content"]
    comp = list(cl.stream_completion("abc", max_tokens=5, temperature=0, ignore_eos=True))
    assert "".join(ch["choices"][0]["text"] for ch in comp) == cl.complete("abc", max_tokens=5, temperature=0, ignore_eos=True)["choices"][0]["text"]
    with pytest.raises(ServeClientError):
        cl.chat([], max_tokens=1)
    # client disconnect mid-stream aborts the request (KV freed, engine idle)
    with httpx.Client(timeout=20) as hc:
        with hc.stream("POST", ctx.outputs["url"] + "/v1/completions",
                       json={"prompt": "abc", "max_tokens": 150, "temperature": 0, "ignore_eos": True, "stream": True}) as r:
            for _ in r.iter_lines():
                break
    t1 = time.time()
    while eng.has_unfinished() and time.time() - t1 < 15:
        time.sleep(0.05)
    assert not eng.has_unfinished()
    assert eng.manager.stats()["blocks_in_use"] == 0
    time.sleep(0.7)
    ctx.request_stop()
    th.join(30)
    assert not th.is_alive()
    keys = {k for _, d in ctx.metrics for k in d}
    for k in ("serve/running", "serve/waiting", "serve/kv_usage", "serve/prefix_hit_rate", "serve/tokens_per_s", "serve/gpu_mem_gb",
              "serve/ttft_ms_p50", "serve/itl_ms_p50"):
        assert k in keys, k
    assert result["r"]["stopped_by"] == "cancel" and result["r"]["generated_tokens"] > 6
    assert ctx.outputs["summary"]["url"] == ctx.outputs["url"]


def test_client_async_proxy_stream():
    import asyncio

    from forge.core.context import NullContext
    from forge.serve.config import ServeConfig
    from forge.serve.run import run_server

    eng = make_engine(tiny_model(2))
    cfg = ServeConfig(model="tiny", port=0, stats_interval_s=0.2, device="cpu", dtype="float32")
    ctx = NullContext()
    th = threading.Thread(target=lambda: run_server(cfg, ctx, engine=eng), daemon=True)
    th.start()
    t0 = time.time()
    while "url" not in ctx.outputs and time.time() - t0 < 30:
        time.sleep(0.05)
    cl = ForgeServeClient(ctx.outputs["url"])

    async def go():
        st = await cl.astats()
        raw = b""
        async for chunk in cl.proxy_stream("/v1/completions", {"prompt": "abc", "max_tokens": 4, "temperature": 0, "ignore_eos": True}):
            raw += chunk
        return st, raw

    st, raw = asyncio.run(go())
    assert "kv_usage" in st and raw.rstrip().endswith(b"data: [DONE]")
    ctx.request_stop()
    th.join(30)
