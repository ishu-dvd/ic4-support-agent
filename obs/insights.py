"""Read-side "explain it" layer for the dashboard. Stdlib only.

Four things the raw aggregate in metrics.py does not answer:
  * GLOSSARY          - what every KPI means in plain words, its unit, its direction and its formula.
                        One copy, served by /api/glossary, so the dashboard and the docs never drift.
  * security_for_run  - did anything security-relevant happen on THIS run (injection, guard, blocked
                        write) and how serious was it. Derived only from what the tracer wrote.
  * golden_comparison - the same 31 golden tickets run through different models, side by side:
                        eval quality (from the eval summary) joined with observability (from the
                        per-ticket run summaries the eval links through case.run_id).
  * guardrails_inventory - which guardrails exist, where they sit in the pipeline, the exact rules,
                        and which tests prove them (names are parsed from tests/, never hand-copied).

Rates here are the same convention as everywhere else in the repo: ALREADY percentages, 0-100.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent import guard, redact
from agent.metrics import load_eval, load_eval_summaries, load_run_summaries, pct_block, percentile
from agent.policy import STRIPPED_REVIEW_REASON

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
MODELS_JSON = os.path.join(REPO_ROOT, "dashboard", "models.json")
GOLDEN_TOTAL_DEFAULT = 31

# ---- glossary ----------------------------------------------------------------------------------------
def _g(label: str, plain: str, unit: str, better: str, formula: str) -> Dict[str, str]:
    return {"label": label, "plain": plain, "unit": unit, "better": better, "formula": formula}


GLOSSARY: Dict[str, Dict[str, str]] = {
    # volume / outcome
    "runs": _g("Runs", "How many tickets the agent processed in the selected window. One run = one ticket.",
               "count", "n/a", "count(runs/<run_id>/summary.json in window)"),
    "success_rate": _g("Success rate", "Share of runs that produced a usable reply (completed or degraded) without "
                       "tripping the output guard.", "%", "higher",
                       "100 x runs(outcome in {completed, degraded} and guard_violations = 0) / runs"),
    "failure_rate": _g("Failure rate", "Share of runs that ended with an error or a blocked (unsafe) draft. "
                       "The customer got a safe fallback text or nothing.", "%", "lower",
                       "100 x runs(outcome = failed) / runs"),
    "degraded": _g("Degraded runs", "Runs that still answered, but an upstream system (entitlements) was unreachable, "
                   "so the reply avoids asserting plan features.", "count", "lower",
                   "runs(outcome = degraded); event entitlements_degraded"),
    "escalations": _g("Escalations", "Runs where policy decided a human must take over (SLA breach, security, "
                      "billing dispute, insufficient information...).", "count", "n/a", "runs(escalate = true)"),
    "draft_source": _g("Draft source", "Who wrote the customer reply: the LLM, or the deterministic rules templates "
                       "(fallback when the LLM failed, was too slow, or tripped the guard).", "category", "n/a",
                       "summary.draft_source in {llm, rules, rules_fallback}"),
    # latency
    "e2e_ms": _g("End-to-end latency", "Wall-clock time from reading the ticket to writing the summary. What the "
                 "support agent waits for.", "ms", "lower", "run.duration_ms"),
    "step_ms": _g("Per-step latency", "Time spent inside each pipeline step. 'draft' is the LLM call and dominates; "
                  "the rest are local reads and rules.", "ms", "lower", "step span duration_ms per step name"),
    "ttft_ms": _g("Time to first token", "How long the model made us wait before it started streaming its answer. "
                  "Mostly queueing + prompt processing on the provider side.", "ms", "lower",
                  "t(first SSE content chunk) - t(request sent), per LLM attempt; run keeps the successful attempt"),
    "tbt_ms": _g("Time between tokens", "Average gap between streamed chunks once the model started answering. "
                 "A proxy for generation speed (lower = faster typing).", "ms", "lower",
                 "mean(t(chunk_i) - t(chunk_i-1)) over content chunks; p50/p95 per run also recorded"),
    "llm_ms": _g("LLM latency", "Total time of the model call including retries.", "ms", "lower",
                 "sum(llm chat.completions call spans)"),
    "p50": _g("p50 (median)", "Half of the runs were faster than this.", "ms", "lower",
              "nearest-rank percentile: sorted[ceil(0.50 x n)]"),
    "p95": _g("p95", "95 out of 100 runs were faster than this; the 'bad day' number.", "ms", "lower",
              "nearest-rank percentile: sorted[ceil(0.95 x n)]"),
    "p99": _g("p99", "Only 1 in 100 runs was slower. With 31 runs this is simply the slowest observed run.", "ms",
              "lower", "nearest-rank percentile: sorted[ceil(0.99 x n)] (never interpolated)"),
    # cost
    "cost_total": _g("Total cost", "Money spent on model tokens for the runs in the window, at the provider's "
                     "per-token price. Runs on an unpriced model are counted separately, never as $0.", "USD",
                     "lower", "sum(input_tokens x price_in + output_tokens x price_out) over runs with a known price"),
    "cost_per_success": _g("Cost per success", "What one good answer costs, including the money burnt on runs that "
                           "failed along the way. The number to budget with.", "USD", "lower",
                           "cost_total / successful runs"),
    "cost_failure": _g("Cost of failures", "Tokens paid for runs that ended failed - spent for nothing.", "USD",
                       "lower", "sum(cost_usd) over runs(outcome = failed)"),
    "cost_degraded": _g("Cost of degraded runs", "Tokens paid for runs that answered with an upstream missing.", "USD",
                        "lower", "sum(cost_usd) over runs(outcome = degraded)"),
    "cost_diagnosis": _g("Cost of diagnosis", "Share of output cost spent on the internal diagnosis the model wrote "
                         "for the engineer (not shown to the customer).", "USD", "n/a",
                         "cost_output x len(diagnosis) / (len(diagnosis) + len(reply))"),
    "cost_reply": _g("Cost of draft reply", "Share of output cost spent on the customer-facing reply text.", "USD",
                     "n/a", "cost_output x len(reply) / (len(diagnosis) + len(reply))"),
    "cost_input": _g("Input cost", "Cost of the prompt tokens (system prompt + ticket + account context + KB).",
                     "USD", "lower", "input_tokens x price_in"),
    "cost_output": _g("Output cost", "Cost of the tokens the model generated.", "USD", "lower",
                      "output_tokens x price_out"),
    "usage_estimated": _g("Estimated usage", "The provider did not return token counts, so tokens were estimated "
                          "from text length (~4 chars/token). Cost for these runs is approximate.", "count", "lower",
                          "runs(usage_estimated = true)"),
    # reliability
    "retries": _g("Retries", "Extra attempts after a transient failure (timeout, 429, 5xx). Default budget: 3 per "
                  "call. 404 and other 4xx are never retried; writes are never retried.", "count", "lower",
                  "sum(retries.http + retries.llm) across runs"),
    "retry_decision": _g("Retry decision", "Why a retry did or did not happen, recorded per attempt.", "text", "n/a",
                         "retry:status_<code> | retry:<exc> | stop:max_retries | stop:deadline | stop:status_<code> "
                         "| stop:code_<error_code>"),
    # security
    "injection_suspected": _g("Injection suspected", "The ticket text contained instructions aimed at the agent "
                              "(e.g. 'ignore previous instructions', 'admin mode', 'reveal system prompt'). The lines "
                              "were stripped before classification and the run was routed conservatively.", "count",
                              "n/a", "guard.scan_input(subject, body) matched >= 1 INJECTION_PATTERNS"),
    "guard_violations": _g("Guard violations", "The model's draft said something it must not (a refund/approval "
                           "commitment, an internal field, an email, a KB article it was not shown). The draft was "
                           "replaced by the rules template; if that also failed, by a fixed safe text.", "count",
                           "lower", "len(guard.scan_output(draft, ctx).violations)"),
    "guard_fallback": _g("Guard fallback", "The LLM draft was rejected by the output guard and the rules draft was "
                         "used instead. The customer still got a reply.", "count", "lower", "event guard_fallback"),
    "guard_failed": _g("Guard failed (blocked)", "Even the rules draft tripped the guard; the run failed closed with a "
                       "fixed safe reply and no write.", "count", "lower", "event guard_failed; outcome = failed"),
    # eval quality
    "completion_rate": _g("Completion rate", "Share of golden tickets the agent finished (with or without a degraded "
                          "upstream).", "%", "higher", "100 x cases(outcome in {completed, degraded}) / cases"),
    "category_accuracy": _g("Category accuracy", "How often the agent picked the same ticket category as the human "
                            "label in the golden set.", "%", "higher", "100 x cases(predicted = expected) / cases"),
    "escalate_accuracy": _g("Escalation accuracy", "How often the escalate yes/no decision matched the golden "
                            "label.", "%", "higher", "100 x cases(predicted_escalate = expected_escalate) / cases"),
    "escalate_precision": _g("Escalation precision", "Of the tickets the agent escalated, how many really needed it. "
                             "Low = humans get noise.", "%", "higher", "100 x TP / (TP + FP)"),
    "escalate_recall": _g("Escalation recall", "Of the tickets that needed a human, how many the agent escalated. "
                          "Low = risky misses.", "%", "higher", "100 x TP / (TP + FN); a failed run counts as a miss"),
    "escalate_f1": _g("Escalation F1", "Single score balancing precision and recall (0-100).", "%", "higher",
                      "2 x P x R / (P + R), with P and R on the 0-100 scale"),
    "kb_recall": _g("KB recall", "Share of tickets where the agent cited EVERY knowledge-base article the golden set "
                    "expects. Strict: one missing article = miss.", "%", "higher",
                    "100 x cases(expected_kb subset of kb_cited) / cases(expected_kb non-empty)"),
    "kb_hit_any": _g("KB hit (any)", "Looser version: at least one expected article was cited.", "%", "higher",
                     "100 x cases(expected_kb intersects kb_cited) / cases(expected_kb non-empty)"),
    "kb_precision": _g("KB precision", "Of all articles the agent cited, how many were expected. Low = it cites "
                       "irrelevant articles.", "%", "higher",
                       "100 x sum|cited intersect expected| / sum|cited| over finished cases that cited something"),
    "baseline": _g("Majority baseline", "What you would score by always answering the most common label. The agent "
                   "must beat this to be worth anything.", "%", "n/a",
                   "100 x count(most frequent expected label) / cases"),
    "injection_test": _g("Injection test", "The one golden ticket that contains a prompt-injection (TCK-1123): was it "
                         "detected, classified right, escalated right and answered without leaking?", "pass/fail",
                         "higher", "present and detected and category_ok and escalate_ok and reply_clean"),
    # identifiers
    "run_id": _g("run_id", "Identifier of one ticket run; the folder name under runs/. Use it to open the trace.",
                 "id", "n/a", "UTC timestamp + 6 hex chars"),
    "trace_id": _g("trace_id", "OpenTelemetry-style id shared by every span and event of one run.", "id", "n/a",
                   "32 hex chars"),
    "request_id": _g("request_id", "Id of the HTTP request that started the run. Sent back as X-Request-ID; pass your "
                     "own to correlate with your gateway logs.", "id", "n/a", "header X-Request-ID or generated req_..."),
    "correlation_id": _g("correlation_id", "Business-level id you choose to group several requests (a helpdesk case, "
                         "a batch). Echoed as X-Correlation-ID; defaults to the request_id.", "id", "n/a",
                         "header X-Correlation-ID or body.correlation_id"),
    "activity_id": _g("activity_id", "Span id of the pipeline step (read_ticket, draft, ...) a record belongs to.",
                      "id", "n/a", "enclosing step span_id"),
    "operation_id": _g("operation_id", "Span id of one outbound call attempt (one HTTP request or one LLM attempt). "
                       "Retries get a new operation_id under the same activity_id.", "id", "n/a", "call span_id"),
}


def glossary() -> Dict[str, Any]:
    return {"metrics": GLOSSARY, "scale_note": "All rates, accuracies, recall/precision and F1 are percentages on a "
                                              "0-100 scale as returned by the API. Do not multiply by 100."}


# ---- run -> eval membership ----------------------------------------------------------------------------
def run_eval_index(runs_dir: str) -> Dict[str, str]:
    """run_id -> eval_id, from every eval's cases.jsonl (an eval is the only thing that groups runs)."""
    index: Dict[str, str] = {}
    for ev in load_eval_summaries(runs_dir):
        data = load_eval(runs_dir, ev["eval_id"])
        for case in (data or {}).get("cases", []):
            rid = case.get("run_id")
            if rid and rid not in index:
                index[rid] = ev["eval_id"]
    return index


# ---- security -------------------------------------------------------------------------------------------
_SEVERITY_ORDER = {"none": 0, "flagged": 1, "blocked": 2}


def security_for_run(summary: dict, result: Optional[dict] = None, trace: Optional[Sequence[dict]] = None) -> dict:
    """What, if anything, security-relevant happened on this run. Only reads what the tracer wrote."""
    result = result or {}
    events: Dict[str, int] = dict(summary.get("events") or {})
    flags: List[str] = []
    explanation: List[str] = []

    injection = bool(summary.get("injection_suspected") or result.get("injection_suspected"))
    patterns: Optional[int] = None  # None = the trace did not record a count (older runs)
    for rec in trace or []:
        if rec.get("kind") == "event" and rec.get("name") == "injection_suspected":
            v = (rec.get("attributes") or {}).get("patterns")
            if v is not None:
                patterns = int(v)
    if injection:
        flags.append("injection_suspected")
        explanation.append("The ticket text tried to instruct the agent (%s). Those lines were removed before "
                           "classification and never reached the model or the KB search." %
                           ("%d pattern%s matched" % (patterns, "" if patterns == 1 else "s") if patterns else
                            "prompt-injection pattern matched"))

    violations = result.get("guard_violations")
    if not isinstance(violations, list):
        violations = []
    if violations or summary.get("guard_violations"):
        flags.append("guard_violations")
        kinds = sorted({str(v).split(":", 1)[0] for v in violations}) or ["see trace"]
        explanation.append("The model's draft was rejected by the output guard (%s)." % ", ".join(kinds))
    if events.get("guard_fallback"):
        flags.append("guard_fallback")
        explanation.append("The reply was rewritten from the deterministic rules template instead of the LLM draft.")
    guard_failed = bool(events.get("guard_failed")) or summary.get("error") == "guard_failed"
    if guard_failed:
        flags.append("guard_failed")
        explanation.append("Even the rules draft was unsafe: the run failed closed with a fixed safe reply and no "
                           "escalation was written.")
    reasons = result.get("policy_reasons") or []
    stripped_review = any(str(r).startswith(STRIPPED_REVIEW_REASON.split(":", 1)[0]) for r in reasons)
    if stripped_review:
        flags.append("stripped_review")
        explanation.append("After removing the injected lines nothing meaningful was left; the ticket was routed to "
                           "human review and the write path was skipped.")
    write_status = summary.get("write_status") or (result.get("write") or {}).get("status")
    if stripped_review or (guard_failed and summary.get("escalate")):
        flags.append("write_blocked")

    if guard_failed or stripped_review:
        severity = "blocked"
    elif flags:
        severity = "flagged"
    else:
        severity = "none"
        explanation.append("No injection or guard signal on this run.")
    return {
        "severity": severity,
        "flags": flags,
        "guard_violations": [str(v) for v in violations],
        "injection_patterns": patterns if injection else None,
        "explanation": explanation,
        "write_status": write_status,
        "draft_source": summary.get("draft_source") or result.get("draft_source"),
    }


def read_result(runs_dir: str, run_id: str) -> Optional[dict]:
    try:
        with open(os.path.join(runs_dir, run_id, "result.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def injection_patterns(runs_dir: str, run_id: str) -> Optional[int]:
    """Matched-pattern count from the run's injection_suspected trace event (a line scan, no JSON parse of
    the other records). None when the trace predates the `patterns` attribute or is missing."""
    try:
        with open(os.path.join(runs_dir, run_id, "trace.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if '"injection_suspected"' in line and '"kind":"event"' in line.replace(" ", ""):
                    try:
                        v = (json.loads(line).get("attributes") or {}).get("patterns")
                    except ValueError:
                        return None
                    return int(v) if v is not None else None
    except OSError:
        pass
    return None


def looks_flagged(summary: dict) -> bool:
    ev = summary.get("events") or {}
    return bool(summary.get("injection_suspected") or summary.get("guard_violations") or ev.get("guard_fallback")
                or ev.get("guard_failed") or summary.get("error") == "guard_failed"
                or summary.get("draft_source") == "rules_fallback")


def security_overview(runs_dir: str, runs: Sequence[dict], eval_index: Optional[Dict[str, str]] = None) -> dict:
    """Security view over a (filtered) list of run summaries. result.json is read only for runs whose summary
    already shows a signal, so this stays cheap over thousands of runs."""
    eval_index = eval_index if eval_index is not None else run_eval_index(runs_dir)
    counts: Counter = Counter()
    vtypes: Counter = Counter()
    rows: List[dict] = []
    for s in runs:
        if not looks_flagged(s):
            continue
        sec = security_for_run(s, read_result(runs_dir, s["run_id"]))
        if sec["severity"] == "none":
            continue
        if "injection_suspected" in sec["flags"]:
            sec["injection_patterns"] = injection_patterns(runs_dir, s["run_id"])
        for f in sec["flags"]:
            counts[f] += 1
        for v in sec["guard_violations"]:
            vtypes[str(v).split(":", 1)[0]] += 1
        rows.append({
            "run_id": s["run_id"], "ticket_id": s.get("ticket_id"), "start_ms": s.get("start_ms"),
            "model_id": s.get("model_id"), "outcome": s.get("outcome"), "severity": sec["severity"],
            "flags": sec["flags"], "guard_violations": sec["guard_violations"],
            "injection_patterns": sec["injection_patterns"], "escalate": bool(s.get("escalate")),
            "write_status": sec["write_status"], "eval_id": eval_index.get(s["run_id"]),
        })
    rows.sort(key=lambda r: (-_SEVERITY_ORDER[r["severity"]], -(r.get("start_ms") or 0)))
    return {
        "n": len(runs),
        "flagged": sum(1 for r in rows if r["severity"] == "flagged"),
        "blocked": sum(1 for r in rows if r["severity"] == "blocked"),
        "counts": {k: counts.get(k, 0) for k in ("injection_suspected", "guard_violations", "guard_fallback",
                                                  "guard_failed", "stripped_review", "write_blocked")},
        "violation_types": dict(vtypes.most_common()),
        "runs": rows,
    }


# ---- golden-set comparison ---------------------------------------------------------------------------------
def _is_full_golden(ev: dict) -> bool:
    if ev.get("case_set", "golden") != "golden" or ev.get("fake"):
        return False
    if str(ev.get("label", "")).endswith("-holdout"):
        return False
    n, total = ev.get("n"), ev.get("n_total")
    return bool(n) and (total is None or n >= total) and not ev.get("partial")


def _obs_from_runs(rows: Sequence[dict]) -> dict:
    def _pct3(vals: Iterable[Optional[float]], ps: Tuple[int, ...]) -> Dict[str, Any]:
        vs = [float(v) for v in vals if v is not None]
        return {"p%d" % p: percentile(vs, p) for p in ps}

    known = [r for r in rows if r.get("cost_usd") is not None]
    cost_total = round(sum(float(r["cost_usd"]) for r in known), 6)
    failed = [r for r in rows if r.get("outcome") == "failed"]
    return {
        "runs_linked": len(rows),
        "e2e_ms": _pct3((r.get("duration_ms") for r in rows), (50, 95, 99)),
        "ttft_ms": _pct3((r.get("ttft_ms") for r in rows), (50, 95)),
        "tbt_ms": _pct3((r.get("tbt_ms_avg") for r in rows), (50, 95)),
        "llm_ms": _pct3((r.get("llm_latency_ms") for r in rows), (50, 95)),
        "cost_total_usd": cost_total,
        "cost_per_ticket_usd": round(cost_total / len(known), 6) if known else None,
        "cost_diagnosis_usd": round(sum(float(r.get("cost_diagnosis_usd") or 0) for r in rows), 6),
        "cost_reply_usd": round(sum(float(r.get("cost_reply_usd") or 0) for r in rows), 6),
        "cost_failure_usd": round(sum(float(r.get("cost_usd") or 0) for r in failed), 6),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
        "retries": {"http": sum(int((r.get("retries") or {}).get("http") or 0) for r in rows),
                    "llm": sum(int((r.get("retries") or {}).get("llm") or 0) for r in rows)},
        "draft_source": dict(Counter(str(r.get("draft_source") or "none") for r in rows)),
        "outcomes": dict(Counter(str(r.get("outcome")) for r in rows)),
        "guard_violation_runs": sum(1 for r in rows if r.get("guard_violations")),
        "injection_flagged_runs": sum(1 for r in rows if r.get("injection_suspected")),
        "unknown_price_runs": sum(1 for r in rows if r.get("cost_usd") is None and r.get("model_id") not in ("rules", None)),
    }


def _price(model_id: str) -> Dict[str, Optional[float]]:
    from agent.llm import PRICE_PER_1K
    p = PRICE_PER_1K.get(model_id)
    return {"input": p[0] if p else None, "output": p[1] if p else None}


def golden_comparison(runs_dir: str) -> dict:
    evals = load_eval_summaries(runs_dir)
    by_id = {s["run_id"]: s for s in load_run_summaries(runs_dir)}
    latest_full: Dict[str, dict] = {}
    history: List[dict] = []
    n_total = GOLDEN_TOTAL_DEFAULT
    for ev in evals:
        if ev.get("case_set", "golden") != "golden" or str(ev.get("label", "")).endswith("-holdout"):
            continue
        n_total = ev.get("n_total") or n_total
        model = str(ev.get("model_id") or "none")
        m = ev.get("metrics") or {}
        data = load_eval(runs_dir, ev["eval_id"]) or {}
        run_ids = [c.get("run_id") for c in data.get("cases", []) if c.get("run_id")]
        linked = [by_id[r] for r in run_ids if r in by_id]
        cost_known = [r for r in linked if r.get("cost_usd") is not None]
        history.append({
            "model_id": model, "eval_id": ev["eval_id"], "label": ev.get("label"), "timestamp": ev.get("timestamp"),
            "n": ev.get("n"), "partial": not _is_full_golden(ev),
            "category_accuracy": m.get("category_accuracy"), "escalate_f1": m.get("escalate_f1"),
            "kb_recall": m.get("kb_recall"),
            "cost_per_ticket_usd": (round(sum(float(r["cost_usd"]) for r in cost_known) / len(cost_known), 6)
                                    if cost_known else None),
            "ttft_p50_ms": percentile([float(r["ttft_ms"]) for r in linked if r.get("ttft_ms") is not None], 50),
        })
        if not _is_full_golden(ev):
            continue
        # An eval where nothing finished (upstream unreachable for the whole run) says nothing about the
        # model; it stays in history but never becomes the model's leaderboard row.
        if not (m.get("finished") or 0):
            continue
        prev = latest_full.get(model)
        if prev and str(prev["timestamp"] or "") >= str(ev.get("timestamp") or ""):
            continue
        b = ev.get("baselines") or {}
        inj = m.get("injection") or {}
        latest_full[model] = {
            "model_id": model, "eval_id": ev["eval_id"], "label": ev.get("label"), "timestamp": ev.get("timestamp"),
            "n": ev.get("n"), "partial": False, "prompt_version": ev.get("prompt_version"),
            "price_per_1k": _price(model),
            "eval": {
                "completion_rate": m.get("completion_rate"), "category_accuracy": m.get("category_accuracy"),
                "escalate_accuracy": m.get("escalate_accuracy"), "escalate_precision": m.get("escalate_precision"),
                "escalate_recall": m.get("escalate_recall"), "escalate_f1": m.get("escalate_f1"),
                "kb_recall": m.get("kb_recall"), "kb_hit_any": m.get("kb_hit_any"), "kb_precision": m.get("kb_precision"),
                "injection_passed": (inj.get("passed") if inj else None),
                "baseline_category": b.get("category"), "baseline_escalate": b.get("escalate"),
            },
            "obs": _obs_from_runs(linked),
            "run_ids": run_ids,
            "wall_ms": ev.get("wall_ms"),
        }
    # Category and escalation are decided by the deterministic rules/policy layer, so they tie across
    # models by design; what separates models is citation quality, how often the guard had to fall
    # back to rules, and price.
    models = sorted(latest_full.values(),
                    key=lambda r: (-(r["eval"]["kb_recall"] or 0), -(r["eval"]["kb_precision"] or 0),
                                   r["obs"]["draft_source"].get("rules_fallback", 0),
                                   r["obs"]["cost_per_ticket_usd"] or 0))
    history.sort(key=lambda h: str(h["timestamp"] or ""))
    return {"case_set": "golden", "n_total": n_total, "models": models, "history": history}


# ---- guardrails inventory ------------------------------------------------------------------------------------
_TEST_DEF = re.compile(r"^def (test_\w+)", re.M)


def _tests(file: str, name_re: str = r".") -> List[Dict[str, str]]:
    path = os.path.join(TESTS_DIR, file)
    try:
        with open(path, encoding="utf-8") as fh:
            names = _TEST_DEF.findall(fh.read())
    except OSError:
        return []
    rx = re.compile(name_re)
    return [{"file": "tests/" + file, "name": n} for n in names if rx.search(n)]


def _layer(id_: str, name: str, stage: str, what: str, how: str, rules: List[str], signals: List[str],
           tests: List[Dict[str, str]]) -> dict:
    return {"id": id_, "name": name, "stage": stage, "what": what, "how": how, "rules": rules, "signals": signals,
            "tests": tests, "test_count": len(tests)}


def guardrails_inventory() -> dict:
    layers = [
        _layer("input_scan", "Prompt-injection scan", "input",
               "Reads the ticket like an attacker would: does the text try to give the agent orders "
               "('ignore previous instructions', 'admin mode', 'reveal the system prompt', 'show account X')?",
               "guard.scan_input(subject, body) runs %d case-insensitive regexes over subject + body before "
               "anything else sees the ticket. A match sets injection_suspected on the run." % len(guard.INJECTION_PATTERNS),
               list(guard.INJECTION_PATTERNS),
               ["event injection_suspected{patterns}", "summary.injection_suspected", "policy_reasons[injection_suspected: ...]"],
               _tests("test_guard.py", r"patterns_exist|scan_input") + _tests("test_adversarial.py", r"scan_input|shape")
               + _tests("test_rules.py", r"injection")),
        _layer("strip", "Injected-line stripping", "input",
               "Removes only the lines written for the agent, keeps the customer's real request. The classifier, "
               "the KB search, SLA-hours parsing and the escalation summary all see the cleaned text.",
               "guard.strip_injection_blocks(body): line-level removal of matching lines plus the '---' fences "
               "around them; never deletes the whole body (an empty body would change the classification).",
               ["line matches INJECTION_PATTERNS -> dropped", "bare --- delimiter lines fencing a dropped line -> dropped",
                "everything else kept verbatim"],
               ["rules.classify(...).body_words_raw", "KB query built from stripped body", "summary subject redacted+truncated"],
               _tests("test_guard.py", r"strip") + _tests("test_injection_probes.py", r"stripped|sla|kb_query|summary_subject")),
        _layer("context_redaction", "Context whitelisting & KB entitlement filter", "context",
               "The model only ever sees a whitelisted view of the account and entitlements (no emails, no internal "
               "rate limits or timestamps) and only KB articles that apply to this customer's plan.",
               "redact.account_view / entitlements_view are explicit field whitelists; redact.filter_applies_to drops "
               "KB hits whose applies_to keys do not match the account; redact_for_trace masks emails and 6+ digit "
               "runs before anything is written to the trace.",
               ["forbidden reply values: %s" % ", ".join(redact.FORBIDDEN_REPLY_VALUES),
                "emails masked in traces", "digit runs >= 6 masked in traces",
                "kb hit dropped unless every applies_to key matches the account"],
               ["event kb_filter_disabled (ablation only)", "ctx.kb_visible", "trace attributes never contain raw emails"],
               _tests("test_redact.py") + _tests("test_tools_gate.py", r"filters_kb|kb_filter")),
        _layer("output_guard", "Output guard (commitments, leaks, citations)", "output",
               "Checks what the model wrote before a human or customer sees it: no refund/approval/severity "
               "promises, no internal fields or emails, no KB article it was not shown, no feature claims while "
               "entitlements are down. A bad LLM draft is swapped for the rules template; if that is also bad the "
               "run fails closed with a fixed safe reply.",
               "guard.scan_output(draft, ctx) -> violations kb_not_visible:<id> | forbidden_commitment:<phrase> | "
               "leak:<field> | feature_while_degraded:<feature>. loop: LLM draft -> rules draft -> SAFE_REPLY.",
               ["forbidden commitment: %s -> %s" % (k, p) for k, p in guard.FORBIDDEN_COMMITMENT_PATTERNS]
               + ["leak check: contact email / any email", "leak check: rate_limit_rpm (name or value)",
                  "leak check: updated_at (name or value)", "leak check: upstream error detail in reply",
                  "kb_not_visible: cited id not in ctx.kb_visible"]
               + ["feature_while_degraded: %s -> %s" % (k, p) for k, p in guard.FEATURE_PATTERNS.items()],
               ["event guard_fallback{violations, source}", "event guard_failed{violations}", "summary.guard_violations",
                "summary.draft_source = rules_fallback", "outcome failed + error guard_failed"],
               _tests("test_guard.py", r"kb_not_visible|commitment|severity|leak|degraded|clean_draft")
               + _tests("test_loop.py", r"guard|injection_is_contained")),
        _layer("write_gate", "Escalation write gate", "write",
               "The only thing the agent can ever write is one escalation, and only when policy says so: dry-run by "
               "default, explicit confirm, check-before-write to avoid duplicates, never after the deadline, never "
               "for another ticket, and never when the ticket was stripped to nothing.",
               "tools.escalate is the single POST in the codebase (statically enforced by a test). It GETs existing "
               "escalations first, refuses intent for a different ticket_id, returns pending_manual past the "
               "deadline or on unparseable 201s, and is skipped in dry_run. Writes are never retried.",
               ["dry_run=true by default -> write_skipped_dry_run", "confirm=true required", "GET before POST -> write_skipped_duplicate",
                "deadline re-checked after the duplicate GET", "intent.ticket_id must equal the run's ticket",
                "stripped-to-nothing tickets -> human review, no write (%s)" % STRIPPED_REVIEW_REASON,
                "RetryPolicy: writes are never retried"],
               ["events write_filed / write_skipped_dry_run / write_skipped_duplicate / write_rejected / write_degraded",
                "summary.write_status", "result.write.escalation_id"],
               _tests("test_tools_gate.py", r"escalate|only_escalate_posts") + _tests("test_injection_probes.py", r"write|post")
               + _tests("test_loop.py", r"write|dry_run") + _tests("test_e2e_write.py")),
        _layer("identity", "Identity fail-closed", "identity",
               "Every read is checked against the ticket's own account. If the account or entitlements record "
               "belongs to someone else the run stops instead of answering with the wrong customer's data.",
               "tools.read_* take only the ticket; account_id is derived server-side and compared on every response "
               "(ADR-001). Mismatch raises and the run ends failed with no draft and no write.",
               ["tools never accept account_id as an argument", "account.account_id must equal ticket.account_id",
                "entitlements.account_id must equal ticket.account_id"],
               ["outcome failed + error identity_mismatch", "no draft, no write"],
               _tests("test_tools_gate.py", r"identity|account_id") + _tests("test_loop.py", r"identity")
               + _tests("test_adversarial.py", r"cross_account")),
        _layer("data_hygiene", "Data hygiene & eval isolation", "data",
               "The agent can never peek at the answers: golden labels live outside anything the agent imports, "
               "adversarial tickets live outside data/, and the fixture data stays byte-identical.",
               "Static tests grep the agent/ and scripts/run_agent.py source for golden references and direct data/ "
               "file access; make data-clean-check hashes data/.",
               ["agent/ never mentions golden", "agent/ never opens data/* directly", "adversarial cases under tests/adversarial/"],
               ["eval header unknown_golden_fields", "eval case_set golden|holdout"],
               _tests("test_no_golden_access.py") + _tests("test_holdout.py") + _tests("test_adversarial.py", r"full_loop")),
    ]
    adversarial: List[dict] = []
    try:
        with open(os.path.join(TESTS_DIR, "adversarial", "cases.json"), encoding="utf-8") as fh:
            for c in json.load(fh):
                adversarial.append({"name": c.get("name"), "subject": c.get("subject"),
                                    "must_not_contain": c.get("must_not_contain") or [],
                                    "expected_category_hint": c.get("expected_category_hint")})
    except (OSError, ValueError):
        pass
    return {
        "layers": layers,
        "adversarial_cases": adversarial,
        "totals": {"layers": len(layers), "tests": sum(l["test_count"] for l in layers),
                   "rules": sum(len(l["rules"]) for l in layers)},
        "test_status": dict(_TEST_STATUS),
    }


GUARDRAIL_TEST_FILES = ("tests/test_guard.py", "tests/test_adversarial.py", "tests/test_injection_probes.py",
                        "tests/test_redact.py", "tests/test_tools_gate.py", "tests/test_no_golden_access.py",
                        "tests/test_policy.py")
_TEST_STATUS: Dict[str, Any] = {"state": "idle", "passed": None, "failed": None, "duration_s": None,
                                "started_ms": None, "finished_ms": None, "output_tail": ""}
_TEST_LOCK = threading.Lock()
_SUMMARY_RE = re.compile(r"(\d+) passed|(\d+) failed|(\d+) error")


def _run_guardrail_tests() -> None:
    started = time.time()
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *GUARDRAIL_TEST_FILES]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        out = (proc.stdout or "") + (proc.stderr or "")
        passed = failed = 0
        for m in _SUMMARY_RE.finditer(out.splitlines()[-1] if out.strip() else ""):
            if m.group(1):
                passed = int(m.group(1))
            elif m.group(2):
                failed += int(m.group(2))
            elif m.group(3):
                failed += int(m.group(3))
        state = "done" if proc.returncode in (0, 1) else "error"
    except (subprocess.TimeoutExpired, OSError) as e:
        out, passed, failed, state = "pytest could not run: %s" % e, None, None, "error"
    with _TEST_LOCK:
        _TEST_STATUS.update({"state": state, "passed": passed, "failed": failed,
                             "duration_s": round(time.time() - started, 1), "finished_ms": int(time.time() * 1000),
                             "output_tail": out[-2000:]})


def start_guardrail_tests() -> bool:
    """Kick off the guardrail test files in a background thread. False if already running."""
    with _TEST_LOCK:
        if _TEST_STATUS["state"] == "running":
            return False
        _TEST_STATUS.update({"state": "running", "passed": None, "failed": None, "duration_s": None,
                             "started_ms": int(time.time() * 1000), "finished_ms": None, "output_tail": ""})
    threading.Thread(target=_run_guardrail_tests, name="guardrail-tests", daemon=True).start()
    return True


# ---- models catalog ------------------------------------------------------------------------------------------
def models_catalog(runs_dir: str) -> dict:
    """dashboard/models.json (probe snapshot) joined with the price table, run counts and latest golden eval."""
    try:
        with open(MODELS_JSON, encoding="utf-8") as fh:
            cat = json.load(fh)
    except (OSError, ValueError):
        cat = {"accessible": [], "inaccessible": [], "probed_at": None}
    runs_by_model = Counter(str(r.get("model_id") or "none") for r in load_run_summaries(runs_dir))
    golden = {m["model_id"]: m["eval_id"] for m in golden_comparison(runs_dir)["models"]}
    models = []
    for m in cat.get("accessible", []):
        mid = m["model_id"]
        price = m.get("price_per_1k") or _price(mid)
        if price.get("input") is None:
            price = _price(mid)
        models.append({**m, "price_per_1k": price, "accessible": True, "runs": runs_by_model.get(mid, 0),
                       "golden_eval_id": golden.get(mid)})
    models.sort(key=lambda m: (-m["runs"], m["model_id"]))
    return {"probed_at": cat.get("probed_at"), "endpoint": cat.get("endpoint"), "note": cat.get("note"),
            "models": models, "inaccessible": cat.get("inaccessible", [])}
