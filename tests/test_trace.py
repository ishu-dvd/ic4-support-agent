"""RunTracer contract: exact record fields, parent nesting, worker-thread fallback, finish artifacts."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from agent.trace import NoopTracer, RunTracer, call_counts, event_counts, load_trace, step_durations

ID_FIELDS = {"request_id", "correlation_id", "activity_id", "operation_id"}
SPAN_FIELDS = {"kind", "trace_id", "span_id", "parent_span_id", "name", "start_ms", "duration_ms", "attributes"} | ID_FIELDS
EVENT_FIELDS = {"kind", "trace_id", "span_id", "parent_span_id", "name", "ts_ms", "attributes"} | ID_FIELDS


@dataclass
class _Usage:
    input_tokens: int = 12
    output_tokens: int = 34
    model_id: str = "rules"
    cost_usd: float = 0.0


@dataclass
class _Write:
    status: str = "skipped_dry_run"
    escalation_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class _Result:
    run_id: str
    ticket_id: str = "TCK-0001"
    category: str = "billing_proration"
    escalate: bool = True
    write: Optional[_Write] = field(default_factory=_Write)
    injection_suspected: bool = False
    entitlements_degraded: bool = False
    draft_source: str = "rules"
    usage: _Usage = field(default_factory=_Usage)
    duration_ms: int = 42
    outcome: str = "completed"
    kb_cited: List[str] = field(default_factory=lambda: ["kb-0001"])


def _by_kind(records, kind):
    return [r for r in records if r["kind"] == kind]


def test_records_have_exact_fields_and_nesting(tmp_path):
    tr = RunTracer(str(tmp_path), ticket_id="TCK-0001")
    assert tr.run_id and len(tr.run_id.split("-")[-1]) == 6
    assert len(tr.trace_id) == 32

    with tr.step("read_ticket") as step:
        step["unknown_fields"] = 2
        with tr.call("http GET /v1/tickets/{id}", retries=0) as span:
            span["http.status"] = 200
        tr.event("injection_suspected", patterns=1)
    tr.finish(_Result(run_id=tr.run_id), prompt_version="v1")

    records = load_trace(tr.run_dir)
    kinds = [r["kind"] for r in records]
    assert kinds == ["call", "event", "step", "run"]  # closed in order; run record last

    call, event, step_rec, run = records
    assert set(call) == SPAN_FIELDS
    assert set(step_rec) == SPAN_FIELDS
    assert set(event) == EVENT_FIELDS
    assert set(run) == SPAN_FIELDS | {"ticket_id"}
    for r in records:
        assert r["trace_id"] == tr.trace_id

    assert run["parent_span_id"] is None and run["ticket_id"] == "TCK-0001"
    assert step_rec["parent_span_id"] == run["span_id"]
    assert call["parent_span_id"] == step_rec["span_id"]
    assert event["parent_span_id"] == step_rec["span_id"]

    assert call["attributes"] == {"retries": 0, "http.status": 200}  # yielded dict mutation lands
    assert step_rec["attributes"] == {"unknown_fields": 2}
    assert isinstance(call["start_ms"], int) and isinstance(call["duration_ms"], int)
    assert run["attributes"]["prompt_version"] == "v1"
    assert run["attributes"]["model_id"] == "rules"
    assert run["attributes"]["outcome"] == "completed"
    assert run["attributes"]["escalate"] is True
    assert run["attributes"]["retries"] == 0
    assert run["duration_ms"] == 42

    # id model: request/correlation on every record; activity = enclosing step; operation = the call
    assert tr.request_id.startswith("req_") and tr.correlation_id == tr.request_id
    for r in records:
        assert r["request_id"] == tr.request_id and r["correlation_id"] == tr.correlation_id
    assert step_rec["activity_id"] == step_rec["span_id"] and step_rec["operation_id"] is None
    assert call["activity_id"] == step_rec["span_id"] and call["operation_id"] == call["span_id"]
    assert event["activity_id"] == step_rec["span_id"] and event["operation_id"] is None
    assert run["activity_id"] is None and run["operation_id"] is None


def test_caller_ids_are_propagated_and_sanitised(tmp_path):
    tr = RunTracer(str(tmp_path), request_id="req-abc 123<script>", correlation_id="batch/eval-1")
    assert tr.request_id == "req-abc123script"  # unsafe characters dropped, never raw
    assert tr.correlation_id == "batch/eval-1"
    with tr.step("draft"):
        with tr.call("llm draft"):
            tr.event("llm_retry", attempt=1)
    recs = load_trace(tr.run_dir)
    ev = next(r for r in recs if r["kind"] == "event")
    call = next(r for r in recs if r["kind"] == "call")
    assert ev["operation_id"] == call["span_id"]  # an event inside a call points at that operation
    assert all(r["correlation_id"] == "batch/eval-1" for r in recs)
    tr.finish(_Result(run_id=tr.run_id))
    summary = json.loads((tmp_path / tr.run_id / "summary.json").read_text())
    assert summary["request_id"] == tr.request_id and summary["correlation_id"] == "batch/eval-1"
    result = json.loads((tmp_path / tr.run_id / "result.json").read_text())
    assert result["request_id"] == tr.request_id


def test_retry_counts_from_attempt_attribute(tmp_path):
    tr = RunTracer(str(tmp_path))
    with tr.step("read_ticket"):
        for attempt in range(3):
            with tr.call("http GET /v1/tickets/{id}", attempt=attempt):
                pass
    with tr.step("draft"):
        with tr.call("llm chat.completions", attempt=0):
            pass
        with tr.call("llm chat.completions", attempt=1):
            pass
    tr.finish(_Result(run_id=tr.run_id))
    summary = json.loads((tmp_path / tr.run_id / "summary.json").read_text())
    assert summary["retries"] == {"http": 2, "llm": 1}
    run = [r for r in load_trace(tr.run_dir) if r["kind"] == "run"][0]
    assert run["attributes"]["retries"] == 3


def test_worker_thread_call_gets_enclosing_step_as_parent(tmp_path):
    tr = RunTracer(str(tmp_path))
    with tr.step("read_parallel"):
        def work(name):
            with tr.call("http GET /v1/%s" % name) as span:
                span["http.status"] = 200
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(work, ["accounts", "entitlements", "kb/search"]))
    records = load_trace(tr.run_dir)
    step = _by_kind(records, "step")[0]
    calls = _by_kind(records, "call")
    assert len(calls) == 3
    assert all(c["parent_span_id"] == step["span_id"] for c in calls)


def test_finish_writes_result_and_summary(tmp_path):
    tr = RunTracer(str(tmp_path), run_id="fixed-run", ticket_id="TCK-0002")
    assert os.path.isdir(tmp_path / "fixed-run")
    with tr.step("read_ticket"):
        with tr.call("http GET /v1/tickets/{id}"):
            pass
    with tr.step("draft"):
        with tr.call("llm draft"):
            pass
        tr.event("deadline_skip")
        tr.event("deadline_skip")
    tr.finish(_Result(run_id="fixed-run", ticket_id="TCK-0002"))

    result = json.loads((tmp_path / "fixed-run" / "result.json").read_text())
    assert result["ticket_id"] == "TCK-0002" and result["write"]["status"] == "skipped_dry_run"

    summary = json.loads((tmp_path / "fixed-run" / "summary.json").read_text())
    assert set(summary) >= {"run_id", "ticket_id", "outcome", "category", "escalate", "write_status", "cost_usd",
                            "input_tokens", "output_tokens", "duration_ms", "step_durations", "call_counts", "events",
                            "trace_id", "request_id", "correlation_id", "model_id", "retries", "ttft_ms",
                            "cost_diagnosis_usd", "cost_reply_usd", "llm_attempts", "start_ms"}
    assert summary["run_id"] == "fixed-run"
    assert summary["retries"] == {"http": 0, "llm": 0} and summary["model_id"] == "rules"
    assert summary["write_status"] == "skipped_dry_run"
    assert summary["input_tokens"] == 12 and summary["output_tokens"] == 34
    assert set(summary["step_durations"]) == {"read_ticket", "draft"}
    assert summary["call_counts"] == {"http": 1, "llm": 1}
    assert summary["events"] == {"deadline_skip": 2}

    records = load_trace(str(tmp_path / "fixed-run"))
    assert step_durations(records) == summary["step_durations"]
    assert call_counts(records) == summary["call_counts"]
    assert event_counts(records) == summary["events"]


def test_result_json_redacts_reply_diagnosis_and_error(tmp_path):
    """B6: result.json is a customer-text artifact too; the same masking as the JSONL applies."""
    @dataclass
    class _Res(_Result):
        reply: str = "Please write to jane.doe@example.com; card ending 4111111111111111."
        diagnosis: str = "Customer email jane.doe@example.com quoted an invoice."
        error: Optional[str] = "upstream said: contact bob@corp.example"

    tr = RunTracer(str(tmp_path), run_id="r", ticket_id="TCK-0002")
    tr.finish(_Res(run_id="r", ticket_id="TCK-0002"))
    raw = (tmp_path / "r" / "result.json").read_text()
    assert "jane.doe@example.com" not in raw and "bob@corp.example" not in raw
    assert "4111111111111111" not in raw
    result = json.loads(raw)
    assert "invoice" in result["diagnosis"]  # only identifiers are masked, the text survives


def test_trace_is_appended_incrementally_and_survives_exception(tmp_path):
    tr = RunTracer(str(tmp_path))
    with tr.step("read_ticket"):
        pass
    assert len(load_trace(tr.run_dir)) == 1  # on disk before finish()
    try:
        with tr.step("classify_intent"):
            raise ValueError("boom with secret@example.com")
    except ValueError:
        pass
    recs = load_trace(tr.run_dir)
    assert len(recs) == 2
    assert recs[-1]["attributes"] == {"error": "ValueError"}  # type only, never the message


def test_email_like_attribute_values_are_masked(tmp_path):
    tr = RunTracer(str(tmp_path))
    tr.event("write_filed", contact="someone@example.com")
    assert load_trace(tr.run_dir)[0]["attributes"]["contact"] == "<email>"


def test_finish_accepts_plain_dict_and_object(tmp_path):
    tr = RunTracer(str(tmp_path))
    tr.finish({"outcome": "failed", "error": "ticket_not_found", "ticket_id": "TCK-9"})
    summary = json.loads(open(os.path.join(tr.run_dir, "summary.json")).read())
    assert summary["outcome"] == "failed" and summary["ticket_id"] == "TCK-9"
    assert summary["cost_usd"] == 0.0 and summary["write_status"] is None


def test_noop_tracer_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = NoopTracer("runs", ticket_id="TCK-0001")
    assert tr.run_id == "noop"
    with tr.step("read_ticket") as s:
        s["x"] = 1
        with tr.call("http GET /v1/tickets/{id}") as c:
            c["http.status"] = 200
        tr.event("anything")
    tr.finish(_Result(run_id="noop"))
    assert list(tmp_path.iterdir()) == []
