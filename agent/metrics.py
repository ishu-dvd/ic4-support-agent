"""Read-side aggregation over runs/ for the dashboard, the JSON API and the Prometheus endpoint. Stdlib only.

Source of truth is what the tracer already writes: runs/<run_id>/summary.json (one row per run) and
runs/eval-*/summary.json + cases.jsonl (one row per eval). Nothing here re-scores; it filters, sorts,
buckets and takes percentiles. Percentiles are nearest-rank so p99 over 31 runs is an actual observed
value, not an interpolation that never happened.

Cost vocabulary used everywhere below:
  cost_usd            what the run's model call cost (None = unknown price, surfaced, never $0)
  cost_success        sum of cost over runs with outcome completed/degraded and no guard violations
  cost_failure        sum of cost over failed runs (outcome=failed) - money spent for nothing
  cost_degraded       sum over degraded runs (answered, but with an upstream missing)
  cost_per_success    cost_total / successful runs - the number to put in a budget
  cost_diagnosis/reply  output cost attributed by text share (see llm.attribute_cost)
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

STEP_ORDER = ("read_ticket", "read_parallel", "read_account", "read_entitlements", "kb_search", "filter_kb",
              "classify_intent", "escalation_policy", "draft", "output_guard", "write", "summarize")
PERCENTILES = (50, 90, 95, 99)


# ---- loading -----------------------------------------------------------------------------------------
def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_run_summaries(runs_dir: str) -> List[dict]:
    """Every runs/<run_id>/summary.json that is a per-ticket run (eval dirs are skipped)."""
    out: List[dict] = []
    if not os.path.isdir(runs_dir):
        return out
    for name in os.listdir(runs_dir):
        if name.startswith("eval-"):
            continue
        s = _read_json(os.path.join(runs_dir, name, "summary.json"))
        if s and s.get("run_id"):
            s.setdefault("start_ms", _start_ms_from_run_id(s["run_id"]))
            out.append(s)
    out.sort(key=lambda s: (s.get("start_ms") or 0, s["run_id"]), reverse=True)
    return out


def _start_ms_from_run_id(run_id: str) -> Optional[int]:
    """Older summaries have no start_ms; the run_id prefix is UTC YYYYmmddTHHMMSS."""
    try:
        import calendar
        return calendar.timegm(time.strptime(run_id.split("-")[0], "%Y%m%dT%H%M%S")) * 1000
    except (ValueError, IndexError):
        return None


def load_run(runs_dir: str, run_id: str) -> Optional[dict]:
    """summary + result + full trace for one run (the drill-down)."""
    if not run_id or "/" in run_id or run_id.startswith("."):
        return None
    run_dir = os.path.join(runs_dir, run_id)
    summary = _read_json(os.path.join(run_dir, "summary.json"))
    if summary is None:
        return None
    from agent.trace import load_trace  # local import: keeps this module importable without the package path tricks

    return {"summary": summary, "result": _read_json(os.path.join(run_dir, "result.json")) or {},
            "trace": load_trace(run_dir)}


def load_eval_summaries(runs_dir: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.isdir(runs_dir):
        return out
    for name in os.listdir(runs_dir):
        if not name.startswith("eval-"):
            continue
        s = _read_json(os.path.join(runs_dir, name, "summary.json"))
        if s and s.get("eval_id"):
            out.append(s)
    out.sort(key=lambda s: (s.get("timestamp") or "", s["eval_id"]), reverse=True)
    return out


def load_eval(runs_dir: str, eval_id: str) -> Optional[dict]:
    if not eval_id.startswith("eval-") or "/" in eval_id:
        return None
    d = os.path.join(runs_dir, eval_id)
    summary = _read_json(os.path.join(d, "summary.json"))
    if summary is None:
        return None
    cases: List[dict] = []
    try:
        with open(os.path.join(d, "cases.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        cases.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return {"summary": summary, "cases": cases}


# ---- filtering ---------------------------------------------------------------------------------------
def filter_runs(runs: Iterable[dict], outcome: Optional[str] = None, category: Optional[str] = None,
                model: Optional[str] = None, ticket: Optional[str] = None, since_ms: Optional[int] = None,
                until_ms: Optional[int] = None, correlation_id: Optional[str] = None, request_id: Optional[str] = None,
                q: Optional[str] = None) -> List[dict]:
    out = []
    ql = (q or "").lower()
    for s in runs:
        if outcome and s.get("outcome") != outcome:
            continue
        if category and s.get("category") != category:
            continue
        if model and s.get("model_id") != model:
            continue
        if ticket and s.get("ticket_id") != ticket:
            continue
        if correlation_id and s.get("correlation_id") != correlation_id:
            continue
        if request_id and s.get("request_id") != request_id:
            continue
        start = s.get("start_ms") or 0
        if since_ms is not None and start < since_ms:
            continue
        if until_ms is not None and start > until_ms:
            continue
        if ql and ql not in json.dumps(s, default=str).lower():
            continue
        out.append(s)
    return out


# ---- maths -------------------------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, min(len(vals), int(math.ceil(p / 100.0 * len(vals)))))
    return vals[rank - 1]


def pct_block(values: Sequence[float], ps: Sequence[int] = PERCENTILES) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    block: Dict[str, Any] = {"n": len(vals)}
    for p in ps:
        block["p%d" % p] = percentile(vals, p)
    block["mean"] = (round(sum(vals) / len(vals), 3) if vals else None)
    block["max"] = (max(vals) if vals else None)
    return block


def _sum_cost(rows: Iterable[dict], key: str = "cost_usd") -> Dict[str, Any]:
    total = 0.0
    unknown = 0
    for r in rows:
        v = r.get(key)
        if v is None:
            if r.get("model_id") not in (None, "rules", "none"):
                unknown += 1
            continue
        total += float(v)
    return {"usd": round(total, 6), "unknown_price_runs": unknown}


# ---- the aggregate -------------------------------------------------------------------------------------
def aggregate(runs: Sequence[dict]) -> dict:
    """Everything the dashboard's top half shows, from summary rows only."""
    n = len(runs)
    by_outcome = Counter(str(r.get("outcome")) for r in runs)
    success = [r for r in runs if r.get("outcome") in ("completed", "degraded") and not r.get("guard_violations")]
    failed = [r for r in runs if r.get("outcome") == "failed"]
    degraded = [r for r in runs if r.get("outcome") == "degraded"]

    steps: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        for name, ms in (r.get("step_durations") or {}).items():
            steps[name].append(float(ms))
    ordered = sorted(steps, key=lambda s: (STEP_ORDER.index(s) if s in STEP_ORDER else len(STEP_ORDER), s))

    by_model: Dict[str, dict] = {}
    for r in runs:
        m = str(r.get("model_id") or "none")
        b = by_model.setdefault(m, {"runs": 0, "cost_usd": 0.0, "unknown_price_runs": 0, "input_tokens": 0,
                                    "output_tokens": 0, "ttft_ms": [], "e2e_ms": [], "failed": 0})
        b["runs"] += 1
        if r.get("cost_usd") is None:
            if m not in ("rules", "none"):
                b["unknown_price_runs"] += 1
        else:
            b["cost_usd"] += float(r["cost_usd"])
        b["input_tokens"] += int(r.get("input_tokens") or 0)
        b["output_tokens"] += int(r.get("output_tokens") or 0)
        if r.get("ttft_ms") is not None:
            b["ttft_ms"].append(float(r["ttft_ms"]))
        b["e2e_ms"].append(float(r.get("duration_ms") or 0))
        if r.get("outcome") == "failed":
            b["failed"] += 1
    for m, b in by_model.items():
        b["cost_usd"] = round(b["cost_usd"], 6)
        b["cost_per_run"] = round(b["cost_usd"] / b["runs"], 6) if b["runs"] else None
        b["ttft_ms"] = pct_block(b["ttft_ms"], (50, 95))
        b["e2e_ms"] = pct_block(b["e2e_ms"], (50, 95))

    retries_http = sum(int((r.get("retries") or {}).get("http", 0) or 0) for r in runs)
    retries_llm = sum(int((r.get("retries") or {}).get("llm", 0) or 0) for r in runs)
    retry_hist = Counter(sum((r.get("retries") or {}).values()) for r in runs)
    events: Counter = Counter()
    for r in runs:
        events.update(r.get("events") or {})
    errors = Counter(str(r.get("error")) for r in runs if r.get("error"))
    write_status = Counter(str(r.get("write_status") or "none") for r in runs)
    draft_source = Counter(str(r.get("draft_source") or "none") for r in runs)
    categories = Counter(str(r.get("category")) for r in runs)

    cost_total = _sum_cost(runs)
    cost_success = _sum_cost(success)
    cost_failure = _sum_cost(failed)
    cost_degraded = _sum_cost(degraded)

    def _per(total: Dict[str, Any], count: int) -> Optional[float]:
        return round(total["usd"] / count, 6) if count else None

    return {
        "n": n,
        "window": {"from_ms": min((r.get("start_ms") or 0) for r in runs) if runs else None,
                   "to_ms": max((r.get("start_ms") or 0) for r in runs) if runs else None},
        "outcomes": dict(by_outcome),
        "success_rate": round(100.0 * len(success) / n, 1) if n else None,
        "failure_rate": round(100.0 * len(failed) / n, 1) if n else None,
        "escalations": sum(1 for r in runs if r.get("escalate")),
        "write_status": dict(write_status),
        "draft_source": dict(draft_source),
        "categories": dict(categories),
        "injection_suspected": sum(1 for r in runs if r.get("injection_suspected")),
        "entitlements_degraded": sum(1 for r in runs if r.get("entitlements_degraded")),
        "guard_violation_runs": sum(1 for r in runs if r.get("guard_violations")),
        "latency": {
            "e2e_ms": pct_block([float(r.get("duration_ms") or 0) for r in runs]),
            "steps_ms": {name: pct_block(steps[name]) for name in ordered},
            "ttft_ms": pct_block([r["ttft_ms"] for r in runs if r.get("ttft_ms") is not None]),
            "tbt_ms": pct_block([r["tbt_ms_avg"] for r in runs if r.get("tbt_ms_avg") is not None]),
            "llm_ms": pct_block([r["llm_latency_ms"] for r in runs if r.get("llm_latency_ms")]),
        },
        "cost": {
            "total": cost_total,
            "success": cost_success,
            "failure": cost_failure,
            "degraded": cost_degraded,
            "per_run": _per(cost_total, n),
            "per_success": _per(cost_total, len(success)),
            "per_failure": _per(cost_failure, len(failed)),
            "diagnosis_usd": _sum_cost(runs, "cost_diagnosis_usd")["usd"],
            "reply_usd": _sum_cost(runs, "cost_reply_usd")["usd"],
            "input_usd": _sum_cost(runs, "cost_input_usd")["usd"],
            "output_usd": _sum_cost(runs, "cost_output_usd")["usd"],
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in runs),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in runs),
            "estimated_usage_runs": sum(1 for r in runs if r.get("usage_estimated")),
            "by_model": by_model,
        },
        "retries": {
            "http": retries_http,
            "llm": retries_llm,
            "total": retries_http + retries_llm,
            "runs_with_retries": sum(1 for r in runs if sum((r.get("retries") or {}).values()) > 0),
            "histogram": {str(k): v for k, v in sorted(retry_hist.items())},
            "llm_attempts": pct_block([r["llm_attempts"] for r in runs if r.get("llm_attempts")], (50, 95, 99)),
        },
        "events": dict(events.most_common()),
        "errors": dict(errors.most_common()),
    }


def facets(runs: Sequence[dict]) -> dict:
    """Distinct values for the filter controls."""
    return {
        "outcomes": sorted({str(r.get("outcome")) for r in runs}),
        "categories": sorted({str(r.get("category")) for r in runs}),
        "models": sorted({str(r.get("model_id") or "none") for r in runs}),
        "tickets": sorted({str(r.get("ticket_id")) for r in runs}),
        "correlation_ids": sorted({str(r.get("correlation_id")) for r in runs if r.get("correlation_id")})[:200],
    }


# ---- Prometheus exposition (pull-based scraping: DO Managed Prometheus / Grafana Cloud / any agent) ----
def _prom_escape(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prometheus_text(agg: dict, service: str = "ic4-agent") -> str:
    lines: List[str] = []
    lab = 'service="%s"' % _prom_escape(service)

    def gauge(name: str, value: Any, extra: str = "", help_: str = "") -> None:
        if value is None:
            return
        if help_:
            lines.append("# HELP %s %s" % (name, help_))
            lines.append("# TYPE %s gauge" % name)
        labels = lab + ("," + extra if extra else "")
        lines.append("%s{%s} %s" % (name, labels, value))

    gauge("ic4_runs_total", agg["n"], help_="runs in the selected window")
    for outcome, k in agg["outcomes"].items():
        gauge("ic4_runs_by_outcome", k, 'outcome="%s"' % _prom_escape(outcome))
    gauge("ic4_success_rate_percent", agg["success_rate"], help_="completed/degraded and guard-clean")
    gauge("ic4_cost_usd_total", agg["cost"]["total"]["usd"], help_="model cost, known prices only")
    gauge("ic4_cost_usd", agg["cost"]["success"]["usd"], 'bucket="success"')
    gauge("ic4_cost_usd", agg["cost"]["failure"]["usd"], 'bucket="failure"')
    gauge("ic4_cost_usd", agg["cost"]["degraded"]["usd"], 'bucket="degraded"')
    gauge("ic4_cost_usd", agg["cost"]["diagnosis_usd"], 'bucket="diagnosis"')
    gauge("ic4_cost_usd", agg["cost"]["reply_usd"], 'bucket="reply"')
    gauge("ic4_cost_per_success_usd", agg["cost"]["per_success"])
    gauge("ic4_cost_per_failure_usd", agg["cost"]["per_failure"])
    gauge("ic4_unknown_price_runs", agg["cost"]["total"]["unknown_price_runs"])
    for model, b in agg["cost"]["by_model"].items():
        m = 'model="%s"' % _prom_escape(model)
        gauge("ic4_model_runs", b["runs"], m)
        gauge("ic4_model_cost_usd", b["cost_usd"], m)
        gauge("ic4_model_input_tokens", b["input_tokens"], m)
        gauge("ic4_model_output_tokens", b["output_tokens"], m)
        gauge("ic4_model_ttft_ms", b["ttft_ms"].get("p50"), m + ',quantile="0.5"')
        gauge("ic4_model_ttft_ms", b["ttft_ms"].get("p95"), m + ',quantile="0.95"')
    lat = agg["latency"]
    for q in ("p50", "p90", "p95", "p99"):
        qv = q[1:]
        gauge("ic4_e2e_latency_ms", lat["e2e_ms"].get(q), 'quantile="0.%s"' % qv)
        gauge("ic4_ttft_ms", lat["ttft_ms"].get(q), 'quantile="0.%s"' % qv)
        gauge("ic4_tbt_ms", lat["tbt_ms"].get(q), 'quantile="0.%s"' % qv)
        for step, block in lat["steps_ms"].items():
            gauge("ic4_step_latency_ms", block.get(q), 'step="%s",quantile="0.%s"' % (_prom_escape(step), qv))
    gauge("ic4_retries_total", agg["retries"]["http"], 'kind="http"', help_="retries in window")
    gauge("ic4_retries_total", agg["retries"]["llm"], 'kind="llm"')
    gauge("ic4_runs_with_retries", agg["retries"]["runs_with_retries"])
    for name, k in agg["events"].items():
        gauge("ic4_events_total", k, 'event="%s"' % _prom_escape(name))
    for name, k in agg["errors"].items():
        gauge("ic4_errors_total", k, 'error="%s"' % _prom_escape(name))
    return "\n".join(lines) + "\n"
