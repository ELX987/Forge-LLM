"""GET /methods: the catalog over HTTP (read-only metadata)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client  # noqa: E402,F401

API = "/api/v1"


def test_list_filter_and_get(client):
    r = client.get(f"{API}/methods")
    assert r.status_code == 200
    ms = r.json()
    assert len(ms) >= 20 and {"id", "title", "family", "status", "summary", "gpu_validated"} <= set(ms[0])
    assert not any(m["gpu_validated"] for m in ms)
    tern = client.get(f"{API}/methods", params={"family": "ternary"}).json()
    assert {m["id"] for m in tern} >= {"ternary-train", "ternary-ptq", "ternary-convert"}
    planned = client.get(f"{API}/methods", params={"status": "planned"}).json()
    assert planned and all(m["job_type"] is None for m in planned)
    one = client.get(f"{API}/methods/ternary-train").json()
    assert one["job_type"] == "ternary_train" and one["status"] == "experimental"
    assert client.get(f"{API}/methods/nope").status_code == 404
