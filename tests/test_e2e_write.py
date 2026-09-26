"""End-to-end write proof against a real fixture server (`python3 -m server`) started by the test.

WHY a real subprocess and not the in-process double: the write gate's promise is about what lands in
the system of record. This test files one escalation for TCK-1103 (rate_limit_increase -> policy
escalates), checks the stored row has confirm=True and a summary with no draft text, then runs the same
ticket again and asserts the duplicate check kept the count at exactly one. TCK-1101 (billing_proration)
runs in the same write mode and must not write at all.

The whole module is skipped when the fixture server cannot start within 3 s (no free port, no server/).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from agent.config import Config
from agent.loop import run

IC4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_TIMEOUT_S = 3.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            _get_json(base_url + "/v1/escalations", timeout=0.5)
            return True
        except Exception:
            time.sleep(0.05)
    return False


@pytest.fixture(scope="module")
def fixture_server():
    """Fresh `python3 -m server` on a free port; skips the module if it is not up within 3 s."""
    port = _free_port()
    base_url = "http://127.0.0.1:%d" % port
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "server", "--port", str(port)],
            cwd=IC4_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:  # pragma: no cover - environment without a python interpreter
        pytest.skip("cannot launch python -m server: %s" % exc)
    try:
        if not _wait_ready(base_url, proc, START_TIMEOUT_S):
            proc.kill()
            proc.wait(timeout=5)
            pytest.skip("fixture server did not start on %s within %.0f s" % (base_url, START_TIMEOUT_S))
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _cfg(base_url: str, runs_dir: str) -> Config:
    return Config(upstream_base_url=base_url, dry_run=False, deadline_ms=20000, runs_dir=runs_dir)


DRAFT_MARKERS = ("Thanks for", "recommended this ticket for escalation", "has now been filed", 'See "',
                 "The linked article", "What happens next")


def test_write_e2e_files_once_and_dedupes(fixture_server, tmp_path):
    base_url = fixture_server
    cfg = _cfg(base_url, str(tmp_path / "runs"))
    assert _get_json(base_url + "/v1/escalations")["escalations"] == []

    # 1. TCK-1103 escalates on category grounds -> exactly one row, filed by the agent with confirm=True
    r1 = run("TCK-1103", cfg)
    assert r1.outcome == "completed", r1.error
    assert r1.category == "rate_limit_increase" and r1.escalate is True
    assert r1.write is not None and r1.write.status == "filed" and r1.write.escalation_id
    rows = _get_json(base_url + "/v1/escalations")["escalations"]
    assert len(rows) == 1
    row = rows[0]
    assert row["ticket_id"] == "TCK-1103"
    assert row["confirm"] is True
    assert row["escalation_id"] == r1.write.escalation_id
    assert row["reason"] == "rate_limit_increase"
    # summary is built from tool facts, never from the draft
    assert r1.reply not in row["summary"] and r1.diagnosis not in row["summary"]
    assert not any(m in row["summary"] for m in DRAFT_MARKERS), row["summary"]
    assert "rate_limit_increase" in row["summary"]
    # the customer-facing reply is finalised after the write with the real reference (B4)
    assert r1.reply.endswith("This escalation has now been filed (reference %s)." % r1.write.escalation_id)
    assert "has been escalated" not in r1.reply

    # 2. TCK-1101 does not escalate -> no write intent, count unchanged
    r2 = run("TCK-1101", cfg)
    assert r2.outcome == "completed" and r2.escalate is False and r2.write is None
    assert len(_get_json(base_url + "/v1/escalations")["escalations"]) == 1

    # 3. same ticket again in write mode -> duplicate check, still exactly one row
    r3 = run("TCK-1103", cfg)
    assert r3.escalate is True and r3.write is not None
    assert r3.write.status == "skipped_duplicate"
    assert r3.write.escalation_id == r1.write.escalation_id
    rows = _get_json(base_url + "/v1/escalations")["escalations"]
    assert len(rows) == 1 and rows[0]["ticket_id"] == "TCK-1103"

    # the traces landed under runs_dir and record the write events
    for res, event in ((r1, "write_filed"), (r3, "write_skipped_duplicate")):
        summary = json.load(open(os.path.join(cfg.runs_dir, res.run_id, "summary.json"), encoding="utf-8"))
        assert summary.get("events", {}).get(event) == 1, (event, summary.get("events"))
