"""Small client for the Forge serving API (used by the web API's chat proxy, the CLI and tests).

Works against any OpenAI-compatible endpoint, plus Forge's ``/stats`` and ``/abort``. Sync methods use ``httpx.Client``;
``a*`` methods use ``httpx.AsyncClient``. ``proxy_stream`` yields raw SSE bytes so the API process can forward a
deployment's stream to the browser unchanged.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from forge.core.errors import ForgeError


class ServeClientError(ForgeError):
    code = "serve_client_error"

    def __init__(self, message: str, status: int | None = None, payload: Any = None) -> None:
        super().__init__(message, details={"status": status, "payload": payload})
        self.status = status


class ForgeServeClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ---- plumbing --------------------------------------------------------------------------------
    @property
    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _raise(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                payload = resp.json()
                msg = payload.get("error", {}).get("message") or str(payload)
            except Exception:
                payload, msg = None, resp.text[:300]
            raise ServeClientError(f"{resp.status_code}: {msg}", resp.status_code, payload)

    def _get(self, path: str) -> Any:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(self.base_url + path, headers=self.headers)
        self._raise(r)
        return r.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self.base_url + path, headers=self.headers, json=payload)
        self._raise(r)
        return r.json()

    # ---- simple endpoints ------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=min(self.timeout, 10.0)) as c:
            r = c.get(self.base_url + "/health")
        return {"http_status": r.status_code, **r.json()}

    def wait_ready(self, timeout: float = 120.0, interval: float = 0.5) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.health()["http_status"] == 200:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

    def stats(self) -> dict[str, Any]:
        return self._get("/stats")

    def models(self) -> dict[str, Any]:
        return self._get("/v1/models")

    def engine_config(self) -> dict[str, Any]:
        return self._get("/v1/engine/config")

    def abort(self, request_id: str) -> dict[str, Any]:
        return self._post("/abort", {"request_id": request_id})

    # ---- generation ------------------------------------------------------------------------------
    def complete(self, prompt: Any, **params: Any) -> dict[str, Any]:
        return self._post("/v1/completions", {"prompt": prompt, "stream": False, **params})

    def chat(self, messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        return self._post("/v1/chat/completions", {"messages": messages, "stream": False, **params})

    def _stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with httpx.Client(timeout=httpx.Timeout(self.timeout, read=None)) as c:
            with c.stream("POST", self.base_url + path, headers=self.headers, json={**payload, "stream": True}) as r:
                if r.status_code >= 400:
                    r.read()
                    self._raise(r)
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    obj = json.loads(data)
                    if "error" in obj:
                        raise ServeClientError(obj["error"].get("message", "stream error"), None, obj)
                    yield obj

    def stream_completion(self, prompt: Any, **params: Any) -> Iterator[dict[str, Any]]:
        yield from self._stream("/v1/completions", {"prompt": prompt, **params})

    def stream_chat(self, messages: list[dict[str, Any]], **params: Any) -> Iterator[dict[str, Any]]:
        yield from self._stream("/v1/chat/completions", {"messages": messages, **params})

    def chat_text(self, messages: list[dict[str, Any]], **params: Any) -> Iterator[str]:
        """Convenience: yield only the text deltas of a streamed chat completion."""
        for ch in self.stream_chat(messages, **params):
            for choice in ch.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece

    # ---- async variants (API server proxy) ----------------------------------------------------------
    async def aget(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.get(self.base_url + path, headers=self.headers)
        self._raise(r)
        return r.json()

    async def astats(self) -> dict[str, Any]:
        return await self.aget("/stats")

    async def apost(self, path: str, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(self.base_url + path, headers=self.headers, json=payload)
        self._raise(r)
        return r.json()

    async def proxy_stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        """Yield the raw SSE byte stream of a streaming request (``stream`` forced to true)."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None)) as c:
            async with c.stream("POST", self.base_url + path, headers=self.headers, json={**payload, "stream": True}) as r:
                if r.status_code >= 400:
                    await r.aread()
                    self._raise(r)
                async for chunk in r.aiter_raw():
                    yield chunk
