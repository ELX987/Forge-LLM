"""`forge` CLI: argument handling, local (in-process scheduler) and remote (HTTP) backends, schema, doctor."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge import cli  # noqa: E402


@pytest.fixture(autouse=True)
def _no_default_server(monkeypatch):
    # make sure the CLI never talks to a real server on the default port during tests
    monkeypatch.setenv("FORGE_API_PORT", "59871")
    monkeypatch.delenv("FORGE_API_URL", raising=False)


def run(argv, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as e:  # _die() exits with a code
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_load_params_merges_file_and_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("steps: 4\nnested:\n  a: 1\n")
    p = cli.load_params(str(f), ["nested.b=2", "name=hello world", "flag=true", "lst=[1,2]", "deep.x.y=3.5"])
    assert p == {"steps": 4, "nested": {"a": 1, "b": 2}, "name": "hello world", "flag": True, "lst": [1, 2], "deep": {"x": {"y": 3.5}}}
    w = tmp_path / "w.yaml"
    w.write_text("type: train\nname: x\nparams:\n  lr: 0.1\n")
    assert cli.load_params(str(w), None) == {"lr": 0.1}  # {type,name,params} wrapper accepted
    with pytest.raises(SystemExit):
        cli.load_params(str(tmp_path / "missing.yaml"), None)
    with pytest.raises(SystemExit):
        cli.load_params(None, ["novalue"])


def test_parse_gpus():
    assert cli._parse_gpus(None) is None and cli._parse_gpus("auto") is None
    assert cli._parse_gpus("0,2") == [0, 2] and cli._parse_gpus("cpu") == []
    with pytest.raises(SystemExit):
        cli._parse_gpus("a,b")


def test_parser_covers_documented_commands():
    ap = cli.build_parser()
    for argv in (["serve-ui", "--host", "0.0.0.0", "--port", "9000", "--home", "/tmp/x"], ["job", "submit", "selftest", "--set", "steps=2"],
                 ["job", "list"], ["job", "show", "j"], ["job", "logs", "j", "-f"], ["job", "cancel", "j"], ["probe", "--quick"], ["bench", "-c", "x.yaml"],
                 ["train", "--config", "t.yaml"], ["finetune", "--config", "t.yaml"], ["distill", "--config", "t.yaml"], ["compress", "--config", "c.yaml"],
                 ["eval", "--config", "e.yaml"], ["serve", "--config", "s.yaml"], ["store", "stats"], ["store", "gc", "--execute"], ["store", "verify"],
                 ["models", "list"], ["models", "register", "/m"], ["models", "download", "org/n"], ["schema", "train"], ["doctor", "--json"]):
        ns = ap.parse_args(argv)
        assert callable(ns.fn), argv
    assert ap.parse_args(["serve-ui", "--home", "/tmp/x"]).home == "/tmp/x"


def test_schema_command(capsys, home):
    code, out, _ = run(["schema", "selftest"], capsys)
    sch = json.loads(out)
    assert code == 0 and sch["properties"]["steps"]["minimum"] == 1
    code, out, _ = run(["schema", "selftest", "--defaults"], capsys)
    assert json.loads(out)["steps"] == 30
    code, out, _ = run(["schema", "list"], capsys)
    assert code == 0 and "selftest" in out and "train" in out
    code, _, err = run(["schema", "nope"], capsys)
    assert code == 1 and "unknown job type" in err


def test_doctor_reports_gpu_kernels_untested(capsys, home):
    code, out, _ = run(["doctor"], capsys)
    assert code == 0 and "GPU kernels" in out and "untested on this host" in out
    code, out, _ = run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    names = {c["name"] for c in rep["checks"]}
    assert {"forge", "python", "torch", "nvidia-smi", "FORGE_HOME", "GPU kernels", "job types"} <= names
    assert all(c["level"] in ("ok", "warn", "fail", "info") for c in rep["checks"])
    assert "packages" in rep["info"] and "modules" in rep["info"]


def test_local_submit_follow_runs_job_and_lists_it(capsys, home):
    code, out, err = run(["job", "submit", "selftest", "--set", "steps=4", "--set", "step_delay_s=0.02", "--name", "cli-test"], capsys)
    assert code == 0, (out, err)
    assert "submitted job_" in out and "SYNTHETIC" in out and "job finished" in out
    assert "succeeded" in err
    code, out, _ = run(["job", "list", "--json"], capsys)
    jobs = json.loads(out)
    assert len(jobs) == 1 and jobs[0]["status"] == "succeeded" and jobs[0]["name"] == "cli-test"
    jid = jobs[0]["id"]
    code, out, _ = run(["job", "list"], capsys)
    assert jid in out and "succeeded" in out
    code, out, _ = run(["job", "show", jid], capsys)
    assert json.loads(out)["outputs"]["echo"] == "ok"
    code, out, _ = run(["job", "logs", jid, "--tail", "3"], capsys)
    assert code == 0 and "job finished" in out
    code, out, _ = run(["job", "cancel", jid], capsys)  # already finished: no-op
    assert code == 0 and "succeeded" in out
    code, _, err = run(["job", "show", "job_missing"], capsys)
    assert code == 1 and "not found" in err


def test_local_submit_failure_exit_code_and_validation(capsys, home):
    code, out, err = run(["job", "submit", "selftest", "--set", "steps=5", "--set", "fail_at_step=2", "--set", "step_delay_s=0"], capsys)
    assert code == 1 and "injected failure" in err
    code, out, err = run(["job", "submit", "selftest", "--set", "steps=-1"], capsys)
    assert code == 1 and "steps" in err and "invalid parameters" in err


def test_local_detach_queues_without_running(capsys, home):
    code, out, err = run(["job", "submit", "selftest", "--detach"], capsys)
    assert code == 0 and "forge serve-ui" in out
    code, out, _ = run(["job", "list", "--json", "--status", "queued"], capsys)
    assert len(json.loads(out)) == 1


def test_models_register_and_list(capsys, home, tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                                               "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}))
    (d / "w.safetensors").write_bytes(b"\0" * 100)
    code, out, _ = run(["models", "register", str(d), "--name", "cli-model"], capsys)
    assert code == 0 and json.loads(out)["name"] == "cli-model"
    code, out, _ = run(["models", "list"], capsys)
    assert "cli-model" in out
    code, out, _ = run(["models", "list", "--json"], capsys)
    assert json.loads(out)[0]["source"] == "local"
    code, _, err = run(["models", "register", str(tmp_path / "nothing")], capsys)
    assert code == 1


def test_store_stats(capsys, home):
    try:
        code, out, err = run(["store", "stats"], capsys)
    except SystemExit as e:  # store unavailable
        pytest.skip(f"store unavailable: {e}")
    assert code == 0 and isinstance(json.loads(out), dict)
    code, out, err = run(["store", "gc"], capsys)
    assert code == 0 and json.loads(out)["dry_run"] is True and "dry run" in err
    code, out, _ = run(["store", "verify", "--fast"], capsys)
    assert code == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_remote_backend_against_real_server(capsys, home):
    uvicorn = pytest.importorskip("uvicorn")
    from forge.api.app import create_app

    port = _free_port()
    app = create_app(token="tok", stream_poll_s=0.1, grace_s=2.0)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started
        base = ["--api", url, "--token", "tok"]
        code, out, err = run(["job", "submit", "selftest", "--set", "steps=4", "--set", "step_delay_s=0.02", *base], capsys)
        assert code == 0 and "via remote backend" in out and "job finished" in out
        code, out, _ = run(["job", "list", "--json", *base], capsys)
        assert len(json.loads(out)) == 1
        # wrong token: clean error, not a traceback
        with pytest.raises(SystemExit):
            cli.main(["job", "list", "--api", url, "--token", "bad"])
        assert "401" in capsys.readouterr().err
        # cancel a running job through the API
        code, out, _ = run(["job", "submit", "selftest", "--set", "steps=5000", "--detach", *base], capsys)
        jid = out.split()[1]
        time.sleep(1.0)
        code, out, _ = run(["job", "cancel", jid, *base], capsys)
        assert code == 0
        deadline = time.time() + 30
        status = ""
        while time.time() < deadline:
            code, out, _ = run(["job", "show", jid, *base], capsys)
            status = json.loads(out)["status"]
            if status in ("cancelled", "failed", "succeeded"):
                break
            time.sleep(0.2)
        assert status == "cancelled"
    finally:
        app.state.services.jobs.stop(kill_running=True)
        server.should_exit = True
        th.join(timeout=10)
