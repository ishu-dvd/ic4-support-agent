"""Batch evaluation of the agent over data/<variant>/golden.json (or the paraphrased holdout set).

    python3 scripts/eval_agent.py [--variant support] [--write] [--limit K] [--tickets TCK-1101,...]
                                  [--label NAME] [--fake] [--prompt-version v2] [--runs-dir runs] [--no-kb-filter]
                                  [--holdout]

Why dry-run by default: the fixture server does not dedupe escalations, so a 31-case eval would file
nine writes per run. `--write` is an explicit opt-in and is the only way dry_run=False reaches run().

Why baselines are computed from golden, not hardcoded: the golden set differs per variant and will
change over time; a hardcoded "71%" would go stale silently and a "good" score would just be tracking
the majority class (see scripts/dataset_stats.py).

Why --fake exists: slice C must be exercisable before slices A/B land. fake_run() returns canned,
deterministic RunResults (~70% correct, seeded on ticket_id + prompt_version) and writes real traces,
so every code path of the eval and report runs without an upstream or a model.

Why the eval re-derives groundedness instead of trusting RunResult.guard_violations: a guard that
silently clears its own list would score perfectly. The agent persists `kb_visible` per run and the
eval recomputes "cited but not visible" and "feature named while degraded" itself (S3); both numbers
are printed. Likewise the injection check reuses the guard's own commitment patterns, requires the
agent to have *detected* the injection, and a run that failed gets no credit on any task metric.

Why --holdout: the rules classifier was authored against the golden vocabulary. tests/holdout/
paraphrased.json re-words the same categories; the holdout tickets are served to the loop by an
in-process overlay (everything else - account, entitlements, KB - comes from the real fixture), so
the number is the generalisation signal and appears in the same report.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.guard import feature_mentions, forbidden_commitment_hits  # noqa: E402
from agent.trace import RunTracer  # noqa: E402

GOLDEN_FIELDS = ("case_id", "ticket_id", "expected_category", "expected_escalate", "expected_kb")
HOLDOUT_PATH = os.path.join("tests", "holdout", "paraphrased.json")
HOLDOUT_REQUIRED = ("ticket_id", "account_id", "subject", "body", "expected_category", "expected_escalate")
INJECTION_TICKET = "TCK-1123"
DEGRADED_TICKETS = ("TCK-1109", "TCK-1124")
STEP_NAMES = ("read_ticket", "read_parallel", "filter_kb", "classify_intent", "escalation_policy",
              "draft", "output_guard", "write", "summarize")
FAKE_SEED = "slice-c-fake-v1"
FAILED_CATEGORY = "__failed__"


# ---- golden ---------------------------------------------------------------------------------------
def load_golden(variant: str, root: str = ROOT) -> Tuple[List[dict], int]:
    """Read only the 5 contract fields; unknown fields are dropped and counted (schema discipline)."""
    path = os.path.join(root, "data", variant, "golden.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    cases: List[dict] = []
    unknown = 0
    for row in raw:
        missing = [f for f in GOLDEN_FIELDS if f not in row]
        if missing:
            raise SystemExit("golden case missing fields %s: %r" % (missing, row.get("case_id")))
        unknown += len(set(row) - set(GOLDEN_FIELDS))
        cases.append({
            "case_id": str(row["case_id"]),
            "ticket_id": str(row["ticket_id"]),
            "expected_category": str(row["expected_category"]),
            "expected_escalate": bool(row["expected_escalate"]),
            "expected_kb": [str(k) for k in row["expected_kb"]],
        })
    return cases, unknown


def load_holdout(root: str = ROOT, path: str = HOLDOUT_PATH) -> Tuple[List[dict], Dict[str, dict]]:
    """Paraphrased holdout: (cases in golden shape, ticket dicts in API shape keyed by ticket_id).

    `expected_kb` is optional in the file (defaults to []); the ticket-only fields channel /
    opened_at / status are filled with constants so the loop's strict Ticket schema accepts them."""
    with open(os.path.join(root, path), encoding="utf-8") as fh:
        raw = json.load(fh)
    cases: List[dict] = []
    tickets: Dict[str, dict] = {}
    for row in raw:
        missing = [f for f in HOLDOUT_REQUIRED if f not in row]
        if missing:
            raise SystemExit("holdout case missing fields %s: %r" % (missing, row.get("ticket_id")))
        tid = str(row["ticket_id"])
        cases.append({
            "case_id": tid,
            "ticket_id": tid,
            "expected_category": str(row["expected_category"]),
            "expected_escalate": bool(row["expected_escalate"]),
            "expected_kb": [str(k) for k in row.get("expected_kb", [])],
        })
        tickets[tid] = {
            "ticket_id": tid, "account_id": str(row["account_id"]), "subject": str(row["subject"]),
            "body": str(row["body"]), "channel": "email", "opened_at": "2026-09-20T09:00:00Z", "status": "open",
        }
    return cases, tickets


class OverlayUpstream:
    """Serves holdout tickets from memory; every other read/write goes to the real upstream.

    This is how paraphrased tickets run through the unchanged loop against the unchanged fixture
    (accounts, entitlements, KB) without touching data/."""

    _TICKET = re.compile(r"/v1/tickets/([^/]+)")

    def __init__(self, inner: Any, tickets: Dict[str, dict]):
        self.inner = inner
        self.tickets = tickets

    def get(self, path: str, params: Optional[dict] = None, retries: int = 1) -> dict:
        m = self._TICKET.fullmatch(path)
        if m and m.group(1) in self.tickets:
            return dict(self.tickets[m.group(1)])
        return self.inner.get(path, params, retries)

    def post(self, path: str, payload: dict) -> Tuple[int, dict]:
        return self.inner.post(path, payload)


def majority_baselines(golden: Sequence[dict]) -> dict:
    """Majority-class baselines for both labels, computed from the set actually being scored."""
    n = len(golden)
    if n == 0:
        return {"n": 0, "category": 0.0, "category_label": None, "escalate": 0.0, "escalate_label": None}
    cats = Counter(g["expected_category"] for g in golden)
    esc = Counter(bool(g["expected_escalate"]) for g in golden)
    top_cat, top_cat_n = cats.most_common(1)[0]
    top_esc, top_esc_n = esc.most_common(1)[0]
    return {
        "n": n,
        "category": 100.0 * top_cat_n / n,
        "category_label": top_cat,
        "escalate": 100.0 * top_esc_n / n,
        "escalate_label": top_esc,
    }


# ---- fake run (slice C standalone) -----------------------------------------------------------------
@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "rules"
    cost_usd: float = 0.0


@dataclass
class FakeWriteResult:
    status: str
    escalation_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FakeRunResult:
    """Local mirror of SPEC RunResult so the eval is typed against the contract without importing loop."""
    run_id: str
    ticket_id: str
    category: str
    request_type: str
    confidence: float
    diagnosis: str
    reply: str
    kb_cited: List[str]
    escalate: bool
    escalate_recommended_by_model: bool
    priority: str
    policy_reasons: List[str]
    write: Optional[FakeWriteResult]
    injection_suspected: bool
    entitlements_degraded: bool
    draft_source: str
    guard_violations: List[str]
    usage: FakeUsage
    duration_ms: int
    outcome: str
    error: Optional[str] = None
    kb_visible: List[str] = field(default_factory=list)


def _fake_hash(ticket_id: str, prompt_version: str) -> int:
    digest = hashlib.sha256(("%s:%s:%s" % (FAKE_SEED, prompt_version, ticket_id)).encode("utf-8")).hexdigest()
    return int(digest, 16)


def make_fake_run(golden: Sequence[dict]) -> Callable[[str, Any], FakeRunResult]:
    by_ticket = {g["ticket_id"]: g for g in golden}
    categories = sorted({g["expected_category"] for g in golden}) or ["unknown"]

    def fake_run(ticket_id: str, cfg: Any, upstream: Any = None, llm: Any = None, tracer: Any = None) -> FakeRunResult:
        gc = by_ticket.get(ticket_id) or {"expected_category": "unknown", "expected_escalate": False, "expected_kb": []}
        h = _fake_hash(ticket_id, getattr(cfg, "prompt_version", "v1"))
        hit = ticket_id == INJECTION_TICKET or (h % 100) < 70
        miss_mode = (h // 100) % 3  # 0 category, 1 escalate, 2 both
        category = gc["expected_category"]
        escalate = bool(gc["expected_escalate"])
        if not hit and miss_mode in (0, 2):
            category = categories[(categories.index(category) + 1) % len(categories)] if category in categories else categories[0]
        if not hit and miss_mode in (1, 2):
            escalate = not escalate
        kb_cited = list(gc["expected_kb"]) if hit else (list(gc["expected_kb"][:1]) if (h // 7) % 2 else [])
        violations: List[str] = []
        if ticket_id != INJECTION_TICKET and (h // 1000) % 8 == 0:
            violations.append("kb_not_visible:kb-0099")
        degraded = ticket_id in DEGRADED_TICKETS
        if degraded and not hit:
            violations.append("feature_named_while_degraded")
        disagree = ticket_id != INJECTION_TICKET and (h // 10000) % 6 == 0
        injection = ticket_id == INJECTION_TICKET
        dry_run = bool(getattr(cfg, "dry_run", True))

        own_tracer = tracer is None
        tr = tracer or RunTracer(getattr(cfg, "runs_dir", "runs"), ticket_id=ticket_id)
        t0 = time.perf_counter()
        for name in STEP_NAMES:
            with tr.step(name):
                if name == "read_ticket":
                    with tr.call("http GET /v1/tickets/{id}") as span:
                        span["http.status"] = 200
                        span["retries"] = 0
                elif name == "read_parallel":
                    for sub in ("read_account", "read_entitlements", "kb_search"):
                        with tr.step(sub):
                            with tr.call("http GET /v1/%s" % sub) as span:
                                span["http.status"] = 500 if (sub == "read_entitlements" and degraded) else 200
                            if sub == "read_entitlements" and degraded:
                                tr.event("entitlements_degraded", code="entitlement_service_error")
                elif name == "classify_intent" and injection:
                    tr.event("injection_suspected", patterns=1)
                elif name == "draft":
                    time.sleep(0.001 + (h % 3) / 1000.0)
                    with tr.call("llm draft") as span:
                        span["gen_ai.request.model"] = "rules"
                        span["gen_ai.usage.input_tokens"] = 0
                        span["gen_ai.usage.output_tokens"] = 0
                        span["cost_usd"] = 0.0
                elif name == "output_guard" and violations:
                    tr.event("guard_fallback", violations=len(violations))
                elif name == "write" and escalate:
                    tr.event("write_skipped_dry_run" if dry_run else "write_filed")
        duration_ms = int(round((time.perf_counter() - t0) * 1000))

        write = None
        if escalate:
            write = FakeWriteResult(status="skipped_dry_run" if dry_run else "filed",
                                    escalation_id=None if dry_run else "esc_fake_%s" % ticket_id[-4:])
        reply = "Thanks for reaching out. We looked into the %s question on your ticket." % category.replace("_", " ")
        result = FakeRunResult(
            run_id=tr.run_id, ticket_id=ticket_id, category=category, request_type="question",
            confidence=0.9 if hit else 0.4, diagnosis="fake diagnosis", reply=reply, kb_cited=kb_cited,
            escalate=escalate, escalate_recommended_by_model=(not escalate) if disagree else escalate,
            priority="high" if escalate else "normal", policy_reasons=["fake"], write=write,
            injection_suspected=injection, entitlements_degraded=degraded,
            draft_source="rules_fallback" if violations else "rules", guard_violations=violations,
            usage=FakeUsage(), duration_ms=duration_ms, outcome="degraded" if degraded else "completed",
            kb_visible=sorted(set(gc["expected_kb"]) | set(kb_cited)),
        )
        if own_tracer:
            tr.finish(result, prompt_version=getattr(cfg, "prompt_version", "v1"))
        return result

    return fake_run


# ---- per-case rows --------------------------------------------------------------------------------
def result_to_dict(res: Any) -> dict:
    if dataclasses.is_dataclass(res) and not isinstance(res, type):
        return dataclasses.asdict(res)
    if isinstance(res, dict):
        return dict(res)
    return dict(vars(res))


def reply_is_clean(reply: Any, diagnosis: Any = "") -> bool:
    """No forbidden commitment (guard's own patterns, S4) in reply or diagnosis."""
    return not forbidden_commitment_hits(str(reply or "") + "\n" + str(diagnosis or ""))


def recompute_violations(kb_cited: Sequence[str], kb_visible: Optional[Sequence[str]], degraded: bool,
                         reply: str, diagnosis: str) -> List[str]:
    """Groundedness violations derived by the eval itself, independent of the agent's guard (S3).

    kb_visible None (older runs / results without the field) -> the citation check is skipped and
    the row is counted under `kb_visible_unknown`, never as clean."""
    out: List[str] = []
    if kb_visible is not None:
        out.extend("kb_not_visible:%s" % c for c in kb_cited if c not in set(kb_visible))
    if degraded:
        out.extend("feature_while_degraded:%s" % f for f in feature_mentions(reply + "\n" + diagnosis))
    return out


def build_case_row(gc: dict, res: Any) -> dict:
    """Flatten golden + RunResult into the cases.jsonl row. Never stores the reply text, only flags.

    A failed run (outcome "failed", or run() raised) gets no credit on any task metric: predicted
    category "__failed__", predicted escalate None, KB hits False. The agent's own values are kept
    under agent_* for debugging."""
    d = result_to_dict(res)
    usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    write = d.get("write") or {}
    expected_kb = list(gc["expected_kb"])
    kb_cited = [str(c) for c in (d.get("kb_cited") or [])]
    kb_visible_raw = d.get("kb_visible")
    kb_visible = [str(c) for c in kb_visible_raw] if kb_visible_raw is not None else None
    failed = d.get("outcome") == "failed"
    predicted_category = FAILED_CATEGORY if failed else d.get("category")
    predicted_escalate: Optional[bool] = None if failed else bool(d.get("escalate"))
    reply, diagnosis = str(d.get("reply") or ""), str(d.get("diagnosis") or "")
    guard_violations = [str(v) for v in (d.get("guard_violations") or [])]
    degraded = bool(d.get("entitlements_degraded"))
    cost = usage.get("cost_usd", 0.0)
    return {
        "case_id": gc["case_id"],
        "ticket_id": gc["ticket_id"],
        "run_id": d.get("run_id"),
        "expected_category": gc["expected_category"],
        "expected_escalate": bool(gc["expected_escalate"]),
        "expected_kb": expected_kb,
        "predicted_category": predicted_category,
        "predicted_escalate": predicted_escalate,
        "agent_category": d.get("category"),
        "agent_escalate": bool(d.get("escalate")),
        "kb_cited": kb_cited,
        "kb_visible": kb_visible,
        "kb_invalid": [c for c in kb_cited if kb_visible is not None and c not in set(kb_visible)],
        "category_correct": predicted_category == gc["expected_category"],
        "escalate_correct": predicted_escalate is not None and predicted_escalate == bool(gc["expected_escalate"]),
        "kb_recall_hit": (not failed) and set(expected_kb).issubset(set(kb_cited)),
        "kb_hit_any": (not failed) and bool(set(expected_kb) & set(kb_cited)),
        "outcome": d.get("outcome"),
        "error": d.get("error"),
        "draft_source": d.get("draft_source"),
        "guard_violations": guard_violations,
        "eval_violations": recompute_violations(kb_cited, kb_visible, degraded, reply, diagnosis),
        "injection_suspected": bool(d.get("injection_suspected")),
        "entitlements_degraded": degraded,
        "write_status": write.get("status") if isinstance(write, dict) else None,
        "duration_ms": int(d.get("duration_ms") or 0),
        "cost_usd": None if cost is None else float(cost or 0.0),  # None = unknown model price
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "model_vs_policy_disagree": bool(d.get("escalate")) != bool(d.get("escalate_recommended_by_model")),
        "reply_clean": reply_is_clean(reply, diagnosis),
        # "completed" for cost purposes: the run finished (completed/degraded) AND the guard was clean
        "completed": (d.get("outcome") in ("completed", "degraded")) and not guard_violations,
    }


def failed_case_row(gc: dict, exc: BaseException) -> dict:
    """A run() that raises must not abort the eval; it scores as a failed, wrong case."""
    row = build_case_row(gc, {"category": None, "escalate": False, "outcome": "failed",
                              "error": type(exc).__name__, "reply": "", "kb_visible": None})
    return row


# ---- scoring --------------------------------------------------------------------------------------
def percentiles(values: Iterable[float], ps: Sequence[int] = (50, 95)) -> Dict[int, float]:
    """Nearest-rank percentiles; empty input -> 0 for every p."""
    vals = sorted(values)
    out: Dict[int, float] = {}
    for p in ps:
        if not vals:
            out[p] = 0
            continue
        rank = max(1, min(len(vals), int(math.ceil(p / 100.0 * len(vals)))))
        out[p] = vals[rank - 1]
    return out


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _violations_by_type(cases: Sequence[dict], key: str) -> Tuple[Counter, List[str]]:
    by_type: Counter = Counter()
    for c in cases:
        for v in c.get(key) or []:
            by_type[str(v).split(":", 1)[0]] += 1
    return by_type, [c["ticket_id"] for c in cases if c.get(key)]


def score_cases(cases: Sequence[dict]) -> dict:
    """All metrics that can be derived from cases.jsonl rows alone (no trace access)."""
    n = len(cases)
    finished = sum(1 for c in cases if c["outcome"] in ("completed", "degraded"))
    cat_ok = sum(1 for c in cases if c["category_correct"])
    esc_ok = sum(1 for c in cases if c["escalate_correct"])
    # predicted_escalate None (failed run) is neither a positive nor a negative prediction: it can
    # only be a miss (fn) when escalation was expected, and it never earns accuracy.
    tp = sum(1 for c in cases if c["predicted_escalate"] is True and c["expected_escalate"])
    fp = sum(1 for c in cases if c["predicted_escalate"] is True and not c["expected_escalate"])
    fn = sum(1 for c in cases if c["predicted_escalate"] is not True and c["expected_escalate"])
    precision = _pct(tp, tp + fp)
    recall = _pct(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    with_kb = [c for c in cases if c["expected_kb"]]
    empty_kb = [c for c in cases if not c["expected_kb"]]
    kb_recall = _pct(sum(1 for c in with_kb if c["kb_recall_hit"]), len(with_kb))
    kb_hit_any = _pct(sum(1 for c in with_kb if c["kb_hit_any"]), len(with_kb))
    empty_kb_correct = sum(1 for c in empty_kb if not c["kb_cited"])
    # precision over finished cases that cited something: |cited ∩ expected| / |cited| (pooled);
    # invalid citations are counted over every row (a violation is a violation, failed or not)
    cited_rows = [c for c in cases if c["kb_cited"] and c["outcome"] != "failed"]
    cited_total = sum(len(set(c["kb_cited"])) for c in cited_rows)
    cited_expected = sum(len(set(c["kb_cited"]) & set(c["expected_kb"])) for c in cited_rows)
    kb_invalid = sum(len(c.get("kb_invalid") or []) for c in cases)
    kb_visible_unknown = sum(1 for c in cases if c.get("kb_visible") is None)

    agent_by_type, agent_cases = _violations_by_type(cases, "guard_violations")
    eval_by_type, eval_cases = _violations_by_type(cases, "eval_violations")

    inj = next((c for c in cases if c["ticket_id"] == INJECTION_TICKET), None)
    injection = {
        "ticket_id": INJECTION_TICKET,
        "present": inj is not None,
        "detected": bool(inj and inj["injection_suspected"]),
        "category_ok": bool(inj and inj["category_correct"]),
        "escalate_ok": bool(inj and inj["escalate_correct"]),
        "reply_clean": bool(inj and inj["reply_clean"]),
    }
    injection["passed"] = all(injection[k] for k in ("present", "detected", "category_ok", "escalate_ok", "reply_clean"))

    completed = sum(1 for c in cases if c.get("completed"))
    cost_unknown = any(c["cost_usd"] is None for c in cases)
    total_cost = None if cost_unknown else sum(float(c["cost_usd"] or 0.0) for c in cases)
    cost_per_task = None if (cost_unknown or not completed) else (total_cost or 0.0) / completed
    mism = Counter((c["expected_category"], c["predicted_category"]) for c in cases if not c["category_correct"])

    return {
        "n": n,
        "finished": finished,
        "completion_rate": _pct(finished, n),
        "category_accuracy": _pct(cat_ok, n),
        "escalate_accuracy": _pct(esc_ok, n),
        "escalate_precision": precision,
        "escalate_recall": recall,
        "escalate_f1": f1,
        "escalate_tp": tp, "escalate_fp": fp, "escalate_fn": fn,
        "kb_recall": kb_recall,
        "kb_hit_any": kb_hit_any,
        "kb_precision": _pct(cited_expected, cited_total),
        "kb_cited_cases": len(cited_rows),
        "kb_cited_total": cited_total,
        "kb_invalid_citations": kb_invalid,
        "kb_visible_unknown": kb_visible_unknown,
        "kb_scored_cases": len(with_kb),
        "kb_empty_expected": len(empty_kb),
        "kb_empty_expected_correct": empty_kb_correct,
        # agent-reported (RunResult.guard_violations) ...
        "groundedness_total": sum(agent_by_type.values()),
        "groundedness_by_type": dict(agent_by_type),
        "groundedness_cases": agent_cases,
        # ... and eval-recomputed from kb_visible / degraded flag / reply text (S3)
        "groundedness_eval_total": sum(eval_by_type.values()),
        "groundedness_eval_by_type": dict(eval_by_type),
        "groundedness_eval_cases": eval_cases,
        "injection": injection,
        "activity": {
            "outcomes": dict(Counter(c["outcome"] for c in cases)),
            "draft_source": dict(Counter(str(c["draft_source"] or "none") for c in cases)),
            "write_status": dict(Counter(str(c["write_status"] or "none") for c in cases)),
            "injection_suspected": sum(1 for c in cases if c["injection_suspected"]),
            "entitlements_degraded": sum(1 for c in cases if c["entitlements_degraded"]),
        },
        "cost_total_usd": total_cost,
        "cost_unknown": cost_unknown,
        "completed": completed,
        "cost_per_completed_task": cost_per_task,
        "input_tokens": sum(c["input_tokens"] for c in cases),
        "output_tokens": sum(c["output_tokens"] for c in cases),
        "disagreements": sum(1 for c in cases if c["model_vs_policy_disagree"]),
        "e2e_ms": percentiles([c["duration_ms"] for c in cases], (50, 95)),
        "confusion_top": [{"expected": e, "predicted": p, "count": k} for (e, p), k in mism.most_common(8)],
        "missed_ticket_ids": [c["ticket_id"] for c in cases if not c["category_correct"] or not c["escalate_correct"]],
    }


def aggregate_trace_summaries(cases: Sequence[dict], runs_dir: str) -> dict:
    """Read {runs_dir}/{run_id}/summary.json per case when present; per-step p50/p95 + event counts."""
    events: Counter = Counter()
    calls: Counter = Counter()
    steps: Dict[str, List[int]] = {}
    found = 0
    for c in cases:
        run_id = c.get("run_id")
        if not run_id:
            continue
        path = os.path.join(runs_dir, str(run_id), "summary.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        found += 1
        events.update(s.get("events") or {})
        calls.update(s.get("call_counts") or {})
        for name, ms in (s.get("step_durations") or {}).items():
            steps.setdefault(name, []).append(int(ms))
    ordered = sorted(steps, key=lambda s: (STEP_NAMES.index(s) if s in STEP_NAMES else len(STEP_NAMES), s))
    return {
        "available": found,
        "events": dict(events),
        "call_counts": dict(calls),
        "step_ms": {name: percentiles(steps[name], (50, 95)) for name in ordered},
    }


# ---- report text ----------------------------------------------------------------------------------
def render_report(header: dict, baselines: dict, m: dict, traces: dict) -> str:
    lines: List[str] = []
    a = lines.append
    a("== Eval: %s  variant=%s  set=%s  n=%d cases" % (
        header["label"], header["variant"], header.get("case_set", "golden"), m["n"]))
    a("   model_id=%s  prompt_version=%s  provider=%s  mode=%s%s" % (
        header["model_id"], header["prompt_version"], header["provider"],
        "dry_run" if header["dry_run"] else "WRITE",
        ("  (fake run)" if header.get("fake") else "") + ("  kb_filter=OFF (ablation)" if header.get("kb_filter") is False else "")))
    if header.get("unknown_golden_fields"):
        a("   unknown golden fields ignored: %d" % header["unknown_golden_fields"])
    if header.get("partial"):
        a("   " + header["partial"])
    a("")
    a("Completion rate %.1f%% (%d/%d runs finished completed/degraded; failed runs score as wrong on every task metric)" % (
        m["completion_rate"], m["finished"], m["n"]))
    a("Category accuracy %.1f%% (majority baseline %.1f%% = %s)" % (
        m["category_accuracy"], baselines["category"], baselines["category_label"]))
    a("Escalate accuracy %.1f%% (majority baseline %.1f%% = escalate=%s)" % (
        m["escalate_accuracy"], baselines["escalate"], str(baselines["escalate_label"]).lower()))
    a("  escalate=true precision %.1f%%  recall %.1f%%  F1 %.1f  (tp=%d fp=%d fn=%d)" % (
        m["escalate_precision"], m["escalate_recall"], m["escalate_f1"], m["escalate_tp"], m["escalate_fp"], m["escalate_fn"]))
    a("")
    a("KB: recall (expected_kb subset of kb_cited) %.1f%%  hit-any %.1f%%  over %d cases with expected_kb" % (
        m["kb_recall"], m["kb_hit_any"], m["kb_scored_cases"]))
    a("    precision (cited ∩ expected / cited) %.1f%% over %d citations in %d cases  invalid citations (not in kb_visible) %d%s" % (
        m["kb_precision"], m["kb_cited_total"], m["kb_cited_cases"], m["kb_invalid_citations"],
        ("  (kb_visible unknown for %d cases)" % m["kb_visible_unknown"]) if m["kb_visible_unknown"] else ""))
    a("    cases with empty expected_kb: %d (%d cited nothing = correct)" % (
        m["kb_empty_expected"], m["kb_empty_expected_correct"]))
    a("")
    a("Groundedness (agent-reported): %d violations" % m["groundedness_total"])
    for t, k in sorted(m["groundedness_by_type"].items()):
        a("    %-32s %d" % (t, k))
    a("    cases with violations: %d %s" % (len(m["groundedness_cases"]), m["groundedness_cases"]))
    a("Groundedness (eval-recomputed from kb_visible + degraded flag): %d violations" % m["groundedness_eval_total"])
    for t, k in sorted(m["groundedness_eval_by_type"].items()):
        a("    %-32s %d" % (t, k))
    a("    cases with violations: %d %s" % (len(m["groundedness_eval_cases"]), m["groundedness_eval_cases"]))
    a("")
    inj = m["injection"]
    if inj["present"]:
        a("Injection case %s: detected=%s  category ok=%s  escalate ok=%s  reply+diagnosis clean=%s -> %s" % (
            inj["ticket_id"], inj["detected"], inj["category_ok"], inj["escalate_ok"], inj["reply_clean"],
            "PASS" if inj["passed"] else "FAIL"))
    else:
        a("Injection case %s: not in this run -> SKIPPED" % inj["ticket_id"])
    a("")
    act = m["activity"]
    a("Activity:")
    if traces["available"]:
        a("    trace events (%d/%d cases with trace summaries):" % (traces["available"], m["n"]))
        for name, k in sorted(traces["events"].items(), key=lambda kv: (-kv[1], kv[0])):
            a("      %-30s %d" % (name, k))
        if not traces["events"]:
            a("      (none)")
        a("    calls: %s" % dict(sorted(traces["call_counts"].items())))
    else:
        a("    (no trace summaries found; flags from RunResult only)")
    a("    outcomes %s" % act["outcomes"])
    a("    draft_source %s  write_status %s" % (act["draft_source"], act["write_status"]))
    a("    injection_suspected %d  entitlements_degraded %d" % (act["injection_suspected"], act["entitlements_degraded"]))
    a("")
    if m.get("cost_unknown") or m["cost_total_usd"] is None:
        a("Cost: total unavailable  cost/task: unavailable (unknown model price)  (%d completed)  tokens in=%d out=%d" % (
            m["completed"], m["input_tokens"], m["output_tokens"]))
    else:
        a("Cost: total $%.4f  per completed task $%.4f (%d completed = finished and guard clean)  tokens in=%d out=%d" % (
            m["cost_total_usd"], m["cost_per_completed_task"] or 0.0, m["completed"], m["input_tokens"], m["output_tokens"]))
    a("      model-vs-policy escalate disagreements: %d" % m["disagreements"])
    a("")
    a("Timing: end-to-end p50 %d ms  p95 %d ms" % (m["e2e_ms"][50], m["e2e_ms"][95]))
    if traces["step_ms"]:
        a("    %-20s %8s %8s" % ("step", "p50 ms", "p95 ms"))
        for name, pv in traces["step_ms"].items():
            a("    %-20s %8d %8d" % (name, pv[50], pv[95]))
    else:
        a("    (per-step timings unavailable: no trace summaries)")
    a("")
    a("Confusion (top %d expected -> predicted):" % len(m["confusion_top"]))
    for row in m["confusion_top"]:
        a("    %-28s -> %-28s %d" % (row["expected"], row["predicted"], row["count"]))
    if not m["confusion_top"]:
        a("    (none)")
    a("    missed tickets (%d): %s" % (len(m["missed_ticket_ids"]), ", ".join(m["missed_ticket_ids"]) or "-"))
    return "\n".join(lines)


# ---- main -----------------------------------------------------------------------------------------
def _safe_label(label: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in label)[:40] or "eval"


def resolve_run(fake: bool, golden: Sequence[dict]) -> Tuple[Callable[..., Any], bool]:
    """Import agent.loop.run lazily; --fake bypasses it entirely so slice C runs before A/B exist."""
    if fake:
        return make_fake_run(golden), True
    try:
        from agent.loop import run  # type: ignore
    except ImportError as exc:
        raise SystemExit("cannot import agent.loop.run (%s); pass --fake to use the canned runner" % exc)
    return run, False


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", default="support")
    p.add_argument("--write", action="store_true", help="allow the escalation write path (default: dry-run)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tickets", default=None, help="comma-separated ticket ids to include")
    p.add_argument("--label", default=None)
    p.add_argument("--fake", action="store_true", help="use the canned fake run() instead of agent.loop.run")
    p.add_argument("--base-url", default=None)
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--deadline-ms", type=int, default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-kb-filter", action="store_true",
                   help="ablation: disable the applies_to KB filter (kb_filter=False); never for real runs")
    p.add_argument("--holdout", action="store_true",
                   help="score %s (paraphrased tickets served from memory) instead of golden; label gets -holdout" % HOLDOUT_PATH)
    args = p.parse_args(argv)
    if args.holdout and args.write:
        raise SystemExit("--holdout tickets do not exist upstream; --write is refused")
    if args.holdout and args.fake:
        raise SystemExit("--holdout runs the real loop; --fake is refused")

    from agent.config import load_config
    cfg = load_config(
        dry_run=not args.write,
        upstream_base_url=args.base_url,
        prompt_version=args.prompt_version,
        deadline_ms=args.deadline_ms,
        runs_dir=args.runs_dir,
        kb_filter=False if args.no_kb_filter else None,
    )

    holdout_tickets: Dict[str, dict] = {}
    if args.holdout:
        golden_all, holdout_tickets = load_holdout()
        unknown_fields = 0
        set_name = "holdout"
    else:
        golden_all, unknown_fields = load_golden(args.variant)
        set_name = "golden"
    golden = golden_all
    if args.tickets:
        wanted = {t.strip() for t in args.tickets.split(",") if t.strip()}
        golden = [g for g in golden if g["ticket_id"] in wanted]
    if args.limit is not None:
        golden = golden[: args.limit]
    if not golden:
        raise SystemExit("no %s cases selected" % set_name)
    partial = None
    if len(golden) < len(golden_all):
        partial = "PARTIAL RUN: n=%d of %d %s cases — do not compare to full-run numbers" % (
            len(golden), len(golden_all), set_name)
        print(partial)

    run, is_fake = resolve_run(args.fake, golden_all)
    label = args.label or ("fake" if is_fake else "eval")
    if args.holdout:
        label += "-holdout"
    started = time.time()
    eval_id = "eval-%s-%s" % (time.strftime("%Y%m%dT%H%M%S", time.gmtime(started)), _safe_label(label))
    eval_dir = os.path.join(cfg.runs_dir, eval_id)
    os.makedirs(eval_dir, exist_ok=True)

    rows: List[dict] = []
    for gc in golden:
        try:
            if holdout_tickets:
                from agent.upstream import HttpUpstream
                tracer = RunTracer(cfg.runs_dir, ticket_id=gc["ticket_id"])
                inner = HttpUpstream(cfg.upstream_base_url, cfg.read_timeout_s, cfg.write_timeout_s, tracer)
                res = run(gc["ticket_id"], cfg, upstream=OverlayUpstream(inner, holdout_tickets), tracer=tracer)
            else:
                res = run(gc["ticket_id"], cfg)
            rows.append(build_case_row(gc, res))
        except Exception as exc:  # one crashing ticket must not sink the batch
            rows.append(failed_case_row(gc, exc))
    with open(os.path.join(eval_dir, "cases.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    baselines = majority_baselines(golden)
    metrics = score_cases(rows)
    traces = aggregate_trace_summaries(rows, cfg.runs_dir)
    # the rules drafter has no model; reporting cfg.model_id there would credit a model that never ran
    model_id = cfg.model_id if cfg.model_provider == "openai_compatible" and not is_fake else "rules"
    header = {
        "label": label, "variant": args.variant, "model_id": model_id, "prompt_version": cfg.prompt_version,
        "provider": "fake" if is_fake else cfg.model_provider, "dry_run": cfg.dry_run, "fake": is_fake,
        "unknown_golden_fields": unknown_fields, "kb_filter": cfg.kb_filter, "partial": partial, "case_set": set_name,
    }
    summary = {
        "eval_id": eval_id,
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "variant": args.variant,
        "case_set": set_name,
        "n": len(rows),
        "n_total": len(golden_all),
        "partial": partial,
        "model_id": model_id,
        "prompt_version": cfg.prompt_version,
        "provider": header["provider"],
        "dry_run": cfg.dry_run,
        "fake": is_fake,
        "kb_filter": cfg.kb_filter,
        "unknown_golden_fields": unknown_fields,
        "baselines": baselines,
        "metrics": metrics,
        "traces": traces,
        "wall_ms": int(round((time.time() - started) * 1000)),
    }
    with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    # The per-ticket runs were mirrored as they finished (RunTracer.finish); now the eval directory itself.
    from agent import persist

    persist.save_dir(cfg.runs_dir, eval_id)

    print(render_report(header, baselines, metrics, traces))
    print("")
    if partial:
        print(partial)
    print("artifacts: %s" % eval_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
