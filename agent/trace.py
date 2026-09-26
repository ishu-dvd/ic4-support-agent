"""Per-run tracing as OTel-shaped JSONL, no SDK (see notes/DECISIONS.md D-007).

Why one JSONL record appended per span close / event: a run that crashes mid-step still leaves every
closed span on disk, and a reader can tail the file while the run is live. Buffering until finish()
would lose exactly the records that explain a failure.

PII rule: the tracer never receives ticket body text; there is deliberately no field for it. Anything
derived from a body must go through redact.redact_for_trace before being passed as an attribute. As a
belt-and-braces measure string attributes that look like emails are masked here too.

Identifiers (every record carries all of them, so a single line can be joined to anything):
  trace_id        one per run; the OTel trace.
  request_id      the inbound request that started the run (HTTP X-Request-ID, or generated for CLI/eval).
  correlation_id  caller-supplied (X-Correlation-ID) and propagated unchanged; groups many runs (an eval batch,
                  a customer conversation, an upstream incident). Defaults to request_id.
  activity_id     the enclosing pipeline step ("read_ticket", "draft", ...): the step's own span_id.
  operation_id    the individual operation inside a step (one HTTP attempt, one model attempt): the call's span_id.
  span_id / parent_span_id  the OTel tree itself.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .redact import redact_for_trace

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# RunResult fields that are derived from ticket/model text and go through redact_for_trace before
# result.json is written. The JSONL trace never receives them at all.
_REDACTED_RESULT_FIELDS = ("reply", "diagnosis", "error")


def new_run_id() -> str:
    """Time-sortable id: YYYYmmddTHHMMSS-<6 hex> (UTC so ids sort the same on every machine)."""
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:20]


_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._:/-]")


def sanitize_id(value: Optional[str], default: Optional[str] = None, max_len: int = 128) -> Optional[str]:
    """Caller-supplied ids are untrusted input: keep a safe charset and a bounded length, never raw."""
    if value is None:
        return default
    cleaned = _ID_SAFE_RE.sub("", str(value).strip())[:max_len]
    return cleaned or default


def _new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _scrub(value: Any) -> Any:
    """Mask email-looking strings anywhere inside an attribute value. Not a substitute for redact.py."""
    if isinstance(value, str):
        return _EMAIL_RE.sub("<email>", value)
    if isinstance(value, dict):
        return {str(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def to_jsonable(obj: Any) -> Any:
    """dataclass -> dict, objects -> __dict__, everything else JSON-friendly (fallback str)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))
    return str(obj)


class RunTracer:
    """Writes {runs_dir}/{run_id}/trace.jsonl incrementally, then result.json + summary.json on finish().

    Parent tracking: each thread has its own span stack. A call made from a worker thread (the parallel
    reads run in a ThreadPoolExecutor) has an empty stack, so it falls back to the most recently opened
    step that is still open, which is the step that spawned the pool.
    """

    def __init__(
        self,
        runs_dir: str,
        run_id: Optional[str] = None,
        ticket_id: str = "",
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        stdout: bool = False,
        service: str = "ic4-agent",
    ):
        self.runs_dir = runs_dir
        self.run_id = run_id or new_run_id()
        self.trace_id = uuid.uuid4().hex
        self.ticket_id = ticket_id
        self.request_id = sanitize_id(request_id, None) or new_request_id()
        self.correlation_id = sanitize_id(correlation_id, None) or self.request_id
        self.service = service
        self.stdout = stdout
        self.run_dir = os.path.join(runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.trace_path = os.path.join(self.run_dir, "trace.jsonl")
        self.run_span_id = _new_span_id()
        self._run_start_ms = _now_ms()
        self._run_start_pc = time.perf_counter()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._open_steps: List[str] = []  # shared across threads; last entry = most recent open step
        self._kinds: Dict[str, str] = {}  # open span_id -> "step" | "call"
        self.records: List[dict] = []
        self._finished = False

    # ---- parent bookkeeping -------------------------------------------------------------------
    def _stack(self) -> List[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _current_parent(self) -> str:
        stack = self._stack()
        if stack:
            return stack[-1]
        with self._lock:
            if self._open_steps:
                return self._open_steps[-1]
        return self.run_span_id

    def _current_activity(self) -> Optional[str]:
        """Nearest open *step* on this thread, else the most recent open step anywhere (worker threads)."""
        stack = self._stack()
        with self._lock:
            for span_id in reversed(stack):
                if self._kinds.get(span_id) == "step":
                    return span_id
            return self._open_steps[-1] if self._open_steps else None

    def _current_operation(self) -> Optional[str]:
        stack = self._stack()
        with self._lock:
            return stack[-1] if stack and self._kinds.get(stack[-1]) == "call" else None

    def _ids(self, activity_id: Optional[str], operation_id: Optional[str]) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "activity_id": activity_id,
            "operation_id": operation_id,
        }

    def _write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self.records.append(record)
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if self.stdout:
                # One JSON object per line: what App Platform / Docker log forwarding ships as-is.
                print(line, flush=True)

    @contextmanager
    def _span(self, kind: str, name: str, attrs: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        span_id = _new_span_id()
        parent_id = self._current_parent()
        activity_id = span_id if kind == "step" else self._current_activity()
        operation_id = span_id if kind == "call" else None
        attributes: Dict[str, Any] = dict(attrs)
        start_ms = _now_ms()
        t0 = time.perf_counter()
        stack = self._stack()
        stack.append(span_id)
        with self._lock:
            self._kinds[span_id] = kind
            if kind == "step":
                self._open_steps.append(span_id)
        try:
            yield attributes
        except BaseException as exc:  # record the type only; messages may carry upstream detail
            attributes.setdefault("error", type(exc).__name__)
            raise
        finally:
            if stack and stack[-1] == span_id:
                stack.pop()
            elif span_id in stack:
                stack.remove(span_id)
            with self._lock:
                self._kinds.pop(span_id, None)
                if kind == "step" and span_id in self._open_steps:
                    self._open_steps.remove(span_id)
            record = {
                "kind": kind,
                "trace_id": self.trace_id,
                "span_id": span_id,
                "parent_span_id": parent_id,
                "name": name,
                "start_ms": start_ms,
                "duration_ms": int(round((time.perf_counter() - t0) * 1000)),
                "attributes": _scrub(attributes),
            }
            record.update(self._ids(activity_id, operation_id))
            self._write(record)

    # ---- public interface ---------------------------------------------------------------------
    @contextmanager
    def step(self, name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
        with self._span("step", name, attrs) as span:
            yield span

    @contextmanager
    def call(self, name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
        with self._span("call", name, attrs) as span:
            yield span

    def event(self, name: str, **attrs: Any) -> None:
        record = {
            "kind": "event",
            "trace_id": self.trace_id,
            "span_id": _new_span_id(),
            "parent_span_id": self._current_parent(),
            "name": name,
            "ts_ms": _now_ms(),
            "attributes": _scrub(dict(attrs)),
        }
        record.update(self._ids(self._current_activity(), self._current_operation()))
        self._write(record)

    def finish(self, result: Any, **attrs: Any) -> None:
        """Close the run: run record, result.json, summary.json. Extra kwargs (e.g. prompt_version) are
        merged into the run attributes because RunResult itself does not carry them."""
        if self._finished:
            return
        self._finished = True
        data = to_jsonable(result) if result is not None else {}
        if not isinstance(data, dict):
            data = {"result": data}
        for key in _REDACTED_RESULT_FIELDS:
            if isinstance(data.get(key), str):
                data[key] = redact_for_trace(data[key])
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        write = data.get("write") or {}
        elapsed_ms = int(round((time.perf_counter() - self._run_start_pc) * 1000))
        duration_ms = int(data.get("duration_ms") or elapsed_ms)
        if not self.ticket_id and data.get("ticket_id"):
            self.ticket_id = str(data["ticket_id"])
        retries = retry_counts(self.records)

        run_attrs: Dict[str, Any] = {
            "outcome": data.get("outcome"),
            "error": data.get("error"),
            "cost_usd": usage.get("cost_usd", 0.0),  # None = unknown price
            "category": data.get("category"),
            "escalate": data.get("escalate"),
            "draft_source": data.get("draft_source"),
            "injection_suspected": data.get("injection_suspected"),
            "entitlements_degraded": data.get("entitlements_degraded"),
            "duration_ms": duration_ms,
            "retries": sum(retries.values()),
            "service.name": self.service,
        }
        if usage.get("model_id"):
            run_attrs["model_id"] = usage["model_id"]
        run_attrs.update(attrs)

        run_record = {
            "kind": "run",
            "trace_id": self.trace_id,
            "span_id": self.run_span_id,
            "parent_span_id": None,
            "name": "run",
            "ticket_id": self.ticket_id,
            "start_ms": self._run_start_ms,
            "duration_ms": duration_ms,
            "attributes": _scrub(run_attrs),
        }
        run_record.update(self._ids(None, None))
        self._write(run_record)

        data.setdefault("request_id", self.request_id)
        data.setdefault("correlation_id", self.correlation_id)
        with open(os.path.join(self.run_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        summary = {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "ticket_id": self.ticket_id,
            "start_ms": self._run_start_ms,
            "outcome": data.get("outcome"),
            "error": data.get("error"),
            "category": data.get("category"),
            "escalate": data.get("escalate"),
            "write_status": write.get("status") if isinstance(write, dict) else None,
            "draft_source": data.get("draft_source"),
            "injection_suspected": bool(data.get("injection_suspected")),
            "entitlements_degraded": bool(data.get("entitlements_degraded")),
            "guard_violations": len(data.get("guard_violations") or []),
            "model_id": usage.get("model_id"),
            "prompt_version": attrs.get("prompt_version"),
            "cost_usd": run_attrs["cost_usd"],
            "cost_input_usd": usage.get("cost_input_usd"),
            "cost_output_usd": usage.get("cost_output_usd"),
            "cost_diagnosis_usd": usage.get("cost_diagnosis_usd"),
            "cost_reply_usd": usage.get("cost_reply_usd"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "usage_estimated": bool(usage.get("usage_estimated")),
            "ttft_ms": usage.get("ttft_ms"),
            "tbt_ms_avg": usage.get("tbt_ms_avg"),
            "tbt_ms_p50": usage.get("tbt_ms_p50"),
            "tbt_ms_p95": usage.get("tbt_ms_p95"),
            "llm_latency_ms": usage.get("latency_ms"),
            "llm_attempts": usage.get("attempts", 0),
            "duration_ms": duration_ms,
            "retries": retries,
            "step_durations": step_durations(self.records),
            "call_counts": call_counts(self.records),
            "events": event_counts(self.records),
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)


class NoopTracer:
    """Same interface as RunTracer; never touches disk. For unit tests and tracer=None call sites."""

    run_id = "noop"
    trace_id = ""
    request_id = ""
    correlation_id = ""
    run_dir = None
    records: List[dict] = []

    def __init__(self, *args: Any, **kwargs: Any):
        self.ticket_id = kwargs.get("ticket_id", "")
        self.request_id = kwargs.get("request_id") or ""
        self.correlation_id = kwargs.get("correlation_id") or self.request_id

    @contextmanager
    def step(self, name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
        yield dict(attrs)

    @contextmanager
    def call(self, name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
        yield dict(attrs)

    def event(self, name: str, **attrs: Any) -> None:
        return None

    def finish(self, result: Any, **attrs: Any) -> None:
        return None


# ---- readers (used by eval/report and by the tracer's own summary) ---------------------------------
def load_trace(run_dir: str) -> List[dict]:
    """Read trace.jsonl from a run dir; tolerates a truncated last line (crash mid-write)."""
    path = os.path.join(run_dir, "trace.jsonl")
    records: List[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def step_durations(records: List[dict]) -> Dict[str, int]:
    """{step_name: total ms}; repeated step names are summed."""
    out: Dict[str, int] = {}
    for r in records:
        if r.get("kind") == "step":
            out[r["name"]] = out.get(r["name"], 0) + int(r.get("duration_ms") or 0)
    return out


def call_counts(records: List[dict]) -> Dict[str, int]:
    """Call spans bucketed by the first word of their name ("http GET ..." -> http, "llm draft" -> llm)."""
    out: Dict[str, int] = {"http": 0, "llm": 0}
    for r in records:
        if r.get("kind") == "call":
            bucket = str(r.get("name", "")).split(" ", 1)[0] or "other"
            out[bucket] = out.get(bucket, 0) + 1
    return out


def event_counts(records: List[dict]) -> Dict[str, int]:
    return dict(Counter(r["name"] for r in records if r.get("kind") == "event"))


def retry_counts(records: List[dict]) -> Dict[str, int]:
    """Retries per call bucket: every call span with attempt > 0 is one retry ("http", "llm", ...)."""
    out: Dict[str, int] = {"http": 0, "llm": 0}
    for r in records:
        if r.get("kind") != "call":
            continue
        attempt = (r.get("attributes") or {}).get("attempt")
        if isinstance(attempt, int) and attempt > 0:
            bucket = str(r.get("name", "")).split(" ", 1)[0] or "other"
            out[bucket] = out.get(bucket, 0) + 1
    return out
