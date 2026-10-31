"""Server-Sent-Events helpers (no extra dependency): formatting, heartbeats, disconnect handling."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from forge.jobs.repo import to_jsonable

HEARTBEAT_S = 15.0


def sse_format(event: str, data: Any, id_: str | int | None = None) -> str:
    payload = json.dumps(to_jsonable(data), separators=(",", ":"))
    head = f"id: {id_}\n" if id_ is not None else ""
    return f"{head}event: {event}\ndata: {payload}\n\n"


def sse_comment(text: str = "ping") -> str:
    return f": {text}\n\n"


# a step function returns (events, done): events is a list of (event, data); done ends the stream
Step = Callable[[], tuple[list[tuple[str, Any]], bool]]


async def sse_stream(
    request: Request,
    step: Step,
    *,
    interval: float = 0.5,
    max_events: int | None = None,
    hello: tuple[str, Any] | None = None,
    heartbeat_s: float = HEARTBEAT_S,
    run_blocking: Callable[..., Awaitable[Any]] | None = None,
) -> AsyncIterator[str]:
    """Drive ``step()`` every ``interval`` seconds, yielding SSE text. Stops on client disconnect, when
    ``step`` reports done, or after ``max_events`` events (heartbeats do not count)."""
    from starlette.concurrency import run_in_threadpool

    run = run_blocking or run_in_threadpool
    sent = 0
    last_out = time.monotonic()
    seq = 0
    if hello is not None:
        yield sse_format(hello[0], hello[1], seq)
        seq += 1
        sent += 1
        last_out = time.monotonic()
        if max_events is not None and sent >= max_events:
            return
    while True:
        if await request.is_disconnected():
            return
        events, done = await run(step)
        for ev, data in events:
            yield sse_format(ev, data, seq)
            seq += 1
            sent += 1
            last_out = time.monotonic()
            if max_events is not None and sent >= max_events:
                return
        if done:
            return
        if time.monotonic() - last_out >= heartbeat_s:
            yield sse_comment("ping")
            last_out = time.monotonic()
        await asyncio.sleep(interval)


def sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
