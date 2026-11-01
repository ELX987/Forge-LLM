"""Job entry point ``forge.serve.run.run_server(cfg, ctx)`` (job type ``serve``).

Loads the model, starts the OpenAI-compatible server with uvicorn in a background thread, then reports engine
statistics through ``ctx.metric`` every ``cfg.stats_interval_s`` seconds with these keys:

``serve/running``, ``serve/waiting``, ``serve/kv_usage``, ``serve/prefix_hit_rate``, ``serve/tokens_per_s``,
``serve/ttft_ms_p50``, ``serve/itl_ms_p50``, ``serve/spec_accept_rate``, ``serve/gpu_mem_gb``
(latency / acceptance keys are omitted until a measurement exists - no fake zeros). It returns when
``ctx.should_stop()`` becomes true or the server dies.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from forge.core.errors import ForgeError
from forge.serve.config import ServeConfig

log = logging.getLogger("forge.serve.run")


def metrics_from_stats(st: dict[str, Any]) -> dict[str, float]:
    """Map ``engine.stats()`` to the flat metric dict streamed to the UI."""
    out = {
        "serve/running": float(st["running"]),
        "serve/waiting": float(st["waiting"]),
        "serve/kv_usage": float(st["kv_usage"]),
        "serve/prefix_hit_rate": float(st["prefix_hit_rate"]),
        "serve/tokens_per_s": float(st["tokens_per_s"]),
        "serve/gpu_mem_gb": float(st["gpu_mem_gb"]),
    }
    mem = st.get("memory") or {}
    if mem.get("weights_gb") is not None:
        out["serve/weights_gb"] = float(mem["weights_gb"])
        out["serve/kv_pool_gb"] = float(mem["kv_pool_gb"])
    for src, dst in (("ttft_ms_p50", "serve/ttft_ms_p50"), ("itl_ms_p50", "serve/itl_ms_p50"),
                     ("spec_accept_rate", "serve/spec_accept_rate")):
        if st.get(src) is not None:
            out[dst] = float(st[src])
    return out


def run_server(cfg: ServeConfig, ctx: Any, *, engine: Any | None = None) -> dict[str, Any]:
    """Run an inference deployment until cancelled. ``engine`` may be injected (tests)."""
    import uvicorn

    from forge.serve.engine import LLMEngine
    from forge.serve.server import create_app

    if isinstance(cfg, dict):
        cfg = ServeConfig(**cfg)
    ctx.log(f"loading model {cfg.model!r} (dtype={cfg.dtype}, kv={cfg.kv_dtype}, quant={cfg.quant})")
    ctx.progress(0.0, "loading model")
    eng = engine or LLMEngine(cfg)
    ctx.log(f"engine ready: {eng.pool.num_blocks} KV blocks x {eng.pool.block_size} tokens ({eng.kv_name}), device {eng.device}")
    ident = dict(getattr(eng, "identity", None) or {})
    mem = getattr(eng, "memory", None)
    ctx.log(f"loaded model: {ident.get('name')} [{ident.get('label') or ('dense ' + str(ident.get('dtype')))}] "
            f"{ident.get('weight_gb')} GB weights from {ident.get('path') or ident.get('requested')}")
    if mem is not None:
        ctx.log("memory: " + mem.summary())
        for w in mem.warnings:
            ctx.log(w, "warning")
        ctx.set_output("memory", mem.to_dict())
    ctx.set_output("model_identity", ident)
    if not cfg.api_key and cfg.host not in ("127.0.0.1", "localhost", "::1"):
        ctx.log(f"WARNING: serving on {cfg.host} without an api_key: anyone who can reach the port can use (and abort) requests; "
                "set ServeConfig.api_key or tunnel over SSH", "warning")
    app = create_app(eng, api_key=cfg.api_key, model_name=cfg.display_name())
    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]  # we run in a worker thread
    thread = threading.Thread(target=server.run, name="forge-uvicorn", daemon=True)
    thread.start()
    t0 = time.time()
    while not server.started:
        if not thread.is_alive():
            raise ForgeError(f"server failed to start on {cfg.host}:{cfg.port} (port in use?)")
        if ctx.should_stop():
            server.should_exit = True
            thread.join(10)
            return {"stopped_before_start": True}
        if time.time() - t0 > 120:
            server.should_exit = True
            raise ForgeError("server did not start within 120 s")
        time.sleep(0.05)
    port = cfg.port
    try:
        port = server.servers[0].sockets[0].getsockname()[1]
    except Exception:  # pragma: no cover
        pass
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    url = f"http://{host}:{port}"
    ctx.set_output("url", url)
    ctx.set_output("port", port)
    ctx.set_output("model", cfg.display_name())
    ctx.log(f"serving {cfg.display_name()} at {url}  (OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models, /stats)")
    ctx.progress(1.0, f"serving at {url}")

    step = 0
    next_emit = time.time()
    try:
        while thread.is_alive() and not ctx.should_stop():
            now = time.time()
            if now >= next_emit:
                try:
                    ctx.metric(step, **metrics_from_stats(eng.stats()))
                except Exception as e:  # noqa: BLE001
                    log.warning("stats emission failed: %s", e)
                step += 1
                next_emit = now + cfg.stats_interval_s
            time.sleep(min(0.1, cfg.stats_interval_s / 4))
    finally:
        ctx.log("shutting down server")
        server.should_exit = True
        thread.join(15)
        app.state.loop.stop()
        eng.shutdown()
    final = eng.stats()
    result = {"url": url, "requests_finished": final["total_finished"], "generated_tokens": final["total_generated_tokens"],
              "prefix_hit_rate": final["prefix_hit_rate"], "stopped_by": "cancel" if ctx.should_stop() else "server_exit"}
    ctx.set_output("summary", result)
    return result
