"""Static front end: serve ``forge/web/static`` at ``/`` with SPA (hash-routing) fallback to index.html."""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"

# defence in depth for the UI shell: no MIME sniffing, no framing (clickjacking of cancel/delete buttons), no referrer leak
_SEC_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"}

_PLACEHOLDER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Forge LLM</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui,sans-serif;max-width:42rem;margin:4rem auto;padding:0 1rem;line-height:1.5}
code{background:#8883;padding:.1em .3em;border-radius:3px}</style></head><body>
<h1>Forge LLM API is running</h1>
<p>The web UI bundle (<code>forge/web/static/index.html</code>) was not found in this installation.</p>
<p>The REST API is available: <a href="/docs">interactive OpenAPI docs</a>, <a href="/api/v1/system/info">/api/v1/system/info</a>.</p>
</body></html>"""


def mount_static(app: FastAPI, static_dir: Path | None = None) -> None:
    root = (static_dir or STATIC_DIR).resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")) and full_path != "":
            raise HTTPException(status_code=404, detail="not found")
        index = root / "index.html"
        if full_path:
            cand = (root / full_path).resolve()
            if root in cand.parents and cand.is_file():
                mt, _ = mimetypes.guess_type(str(cand))
                return FileResponse(cand, media_type=mt or "application/octet-stream", headers={"Cache-Control": "no-cache", **_SEC_HEADERS})
            if "." in Path(full_path).name:  # missing asset: a real 404, not the SPA shell
                raise HTTPException(status_code=404, detail="not found")
        if index.is_file():
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache", **_SEC_HEADERS})
        return HTMLResponse(_PLACEHOLDER, headers=_SEC_HEADERS)
