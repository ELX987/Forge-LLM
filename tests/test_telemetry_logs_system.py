"""Log capture helpers and the system monitor."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from forge.telemetry.logs import LogTailer, format_line, parse_line, read_log, tail_lines  # noqa: E402
from forge.telemetry.system import SystemMonitor  # noqa: E402


def test_format_and_parse_line():
    line = format_line("hello world", "warning")
    rec = parse_line(line)
    assert rec["level"] == "WARNING" and rec["msg"] == "hello world" and rec["ts"].endswith("Z")
    raw = parse_line("Traceback (most recent call last):")
    assert raw["level"] == "RAW" and raw["ts"] is None


def test_read_log_offsets_and_partial_lines(tmp_path):
    p = tmp_path / "log.txt"
    assert read_log(p, 0) == ("", 0)
    p.write_bytes(b"one\ntwo\npart")
    text, off = read_log(p, 0)
    assert text.startswith("one\ntwo\n")
    p.write_bytes(b"one\ntwo\npartial done\n")
    text2, off2 = read_log(p, 8)
    assert text2 == "partial done\n" and off2 == p.stat().st_size
    assert read_log(p, off2) == ("", off2)
    # truncated file resets the offset
    p.write_bytes(b"x\n")
    assert read_log(p, 999)[0] == "x\n"


def test_read_log_max_bytes_cuts_at_line_boundary(tmp_path):
    p = tmp_path / "log.txt"
    p.write_bytes(("abcdefghij\n" * 10).encode())
    text, off = read_log(p, 0, max_bytes=25)
    assert text == "abcdefghij\n" * 2 and off == 22


def test_utf8_never_split(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("é" * 20 + "\n" + "ü" * 20 + "\n", encoding="utf-8")
    text, off = read_log(p, 0, max_bytes=45)
    assert "�" not in text


def test_tail_lines(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("\n".join(f"line {i}" for i in range(1000)) + "\n")
    lines, end = tail_lines(p, 5)
    assert lines == [f"line {i}" for i in range(995, 1000)] and end == p.stat().st_size
    assert tail_lines(tmp_path / "nope", 5) == ([], 0)
    assert len(tail_lines(p, 5000)[0]) == 1000


def test_log_tailer(tmp_path):
    p = tmp_path / "log.txt"
    t = LogTailer(p)
    assert t.poll() == []
    p.write_text("a\nb\n")
    assert t.poll() == ["a", "b"]
    with open(p, "a") as f:
        f.write("c\n")
    assert t.poll() == ["c"] and t.poll() == []


def test_system_monitor_ring_and_job_series():
    written = []

    class FakeMetrics:
        def write(self, run_id, step, values, ts=None):
            written.append((run_id, step, dict(values)))

    gpus = [{"index": 0, "name": "Fake", "util": 50.0, "mem_used_mb": 1000.0, "mem_total_mb": 8000.0, "temp_c": 60.0, "power_w": 100.0},
            {"index": 1, "name": "Fake", "util": 5.0, "mem_used_mb": 10.0, "mem_total_mb": 8000.0, "temp_c": 40.0, "power_w": 20.0}]
    mon = SystemMonitor(0.2, history=5, sampler=lambda: gpus, running_jobs=lambda: [("job_a", [0]), ("job_b", None)], metrics_store=FakeMetrics())
    mon.start()
    time.sleep(0.7)
    mon.stop()
    assert mon.gpu_indices() == [0, 1]
    assert 1 <= len(mon.history(100)) <= 5
    a = [w for w in written if w[0] == "job_a"]
    assert a and "sys/gpu0/util" in a[0][2] and "sys/gpu1/util" not in a[0][2]  # only assigned GPUs
    b = [w for w in written if w[0] == "job_b"]
    assert b and "sys/gpu1/util" in b[0][2]  # gpus=None -> all
    assert a[0][1] == 0  # step = seconds since the job's first sample
    assert mon.latest()["gpus"][0]["name"] == "Fake"


def test_system_monitor_survives_sampler_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return []

    mon = SystemMonitor(0.2, sampler=flaky)
    mon.start()
    time.sleep(0.6)
    mon.stop()
    assert calls["n"] >= 2
