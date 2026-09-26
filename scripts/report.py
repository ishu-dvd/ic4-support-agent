"""Render eval summaries under a runs dir as markdown.

    python3 scripts/report.py runs [--out reports/latest.md]

Why read summary.json files instead of recomputing from cases.jsonl: the eval already scored each run
with the code that was current at the time; the report is a view, not a second scorer, so two evals
with different scoring code never get silently re-scored against each other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


def load_eval_summaries(runs_dir: str) -> List[dict]:
    """Every {runs_dir}/*/summary.json that has an eval_id (per-ticket run summaries do not)."""
    out: List[dict] = []
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "summary.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(s, dict) and "eval_id" in s:
            s.setdefault("_dir", os.path.join(runs_dir, name))
            out.append(s)
    out.sort(key=lambda s: (str(s.get("timestamp", "")), str(s.get("eval_id", ""))))
    return out


def _p(d: Any, key: int) -> Any:
    """Percentile dicts round-trip through JSON with string keys."""
    if not isinstance(d, dict):
        return 0
    return d.get(str(key), d.get(key, 0))


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return lines


def _cost(m: dict) -> str:
    """cost/task cell: None (unknown model price) renders as "unavailable", never as $0."""
    if m.get("cost_unknown") or m.get("cost_per_completed_task") is None:
        return "unavailable"
    return "$%.4f" % float(m.get("cost_per_completed_task") or 0.0)


def _n_cell(s: dict, m: dict) -> str:
    n = int(s.get("n", m.get("n", 0)))
    total = s.get("n_total")
    if total and int(total) != n:
        return "%d/%d (partial)" % (n, int(total))
    return str(n)


def comparison_rows(summaries: Sequence[dict]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for s in summaries:
        m = s.get("metrics") or {}
        inj = m.get("injection") or {}
        rows.append([
            s.get("label", ""),
            s.get("timestamp", ""),
            s.get("case_set", "golden"),
            s.get("prompt_version", ""),
            s.get("model_id", ""),
            s.get("provider", ""),
            _n_cell(s, m),
            "%.1f%%" % float(m.get("completion_rate", 100.0 if m else 0.0)),
            "%.1f%%" % float(m.get("category_accuracy", 0.0)),
            "%.1f%%" % float(m.get("escalate_accuracy", 0.0)),
            "%.1f" % float(m.get("escalate_f1", 0.0)),
            "%.1f%%" % float(m.get("kb_recall", 0.0)),
            ("%.1f%%" % float(m["kb_precision"])) if "kb_precision" in m else "n/a",
            "%d / %s" % (int(m.get("groundedness_total", 0)),
                         str(m["groundedness_eval_total"]) if "groundedness_eval_total" in m else "n/a"),
            ("PASS" if inj.get("passed") else ("n/a" if not inj.get("present") else "FAIL")),
            _cost(m),
            int(_p(m.get("e2e_ms"), 50)),
            int(_p(m.get("e2e_ms"), 95)),
        ])
    return rows


def delta_line(prev: dict, latest: dict) -> str:
    pm, lm = prev.get("metrics") or {}, latest.get("metrics") or {}

    def d(key: str, fmt: str, scale: str = "") -> str:
        a, b = pm.get(key, 0.0), lm.get(key, 0.0)
        if a is None or b is None:
            return "unavailable"
        return (fmt % (float(b) - float(a))) + scale

    return ("Delta vs previous (`%s` -> `%s`): category acc %s, escalate acc %s, cost/task %s" % (
        prev.get("label", "?"), latest.get("label", "?"),
        d("category_accuracy", "%+.1f", " pts"), d("escalate_accuracy", "%+.1f", " pts"),
        d("cost_per_completed_task", "$%+.4f")))


def render_markdown(summaries: Sequence[dict], runs_dir: str) -> str:
    lines: List[str] = ["# Eval report", "", "Source: `%s` (%d eval%s)" % (runs_dir, len(summaries), "" if len(summaries) == 1 else "s"), ""]
    if not summaries:
        lines.append("No eval summaries found (looking for `*/summary.json` containing `eval_id`).")
        return "\n".join(lines) + "\n"

    lines.append("## Run comparison")
    lines.append("")
    lines += _md_table(
        ["label", "timestamp", "set", "prompt_version", "model_id", "provider", "n", "completion", "category acc",
         "escalate acc", "escalate F1", "kb recall", "kb precision", "groundedness viol. (agent / eval)", "injection",
         "cost/task", "e2e p50 ms", "e2e p95 ms"],
        comparison_rows(summaries))
    lines.append("")
    lines.append("Groundedness is shown as agent-reported / eval-recomputed (from persisted `kb_visible` and the degraded flag). "
                 "`set=holdout` rows score the paraphrased tickets in `tests/holdout/paraphrased.json`; they are the "
                 "generalisation signal, not comparable to golden rows. Partial runs are marked in `n`.")
    lines.append("")
    # delta only between the two most recent runs on the SAME case set (golden vs holdout is not a delta)
    latest_set = (summaries[-1].get("case_set") or "golden")
    same_set = [s for s in summaries if (s.get("case_set") or "golden") == latest_set and not s.get("partial")]
    if len(same_set) >= 2:
        lines.append(delta_line(same_set[-2], same_set[-1]))
        lines.append("")

    latest = summaries[-1]
    traces = latest.get("traces") or {}
    lines.append("## Per-step timing (latest: `%s`)" % latest.get("label", ""))
    lines.append("")
    step_ms = traces.get("step_ms") or {}
    if step_ms:
        lines += _md_table(["step", "p50 ms", "p95 ms"],
                           [[name, int(_p(pv, 50)), int(_p(pv, 95))] for name, pv in step_ms.items()])
    else:
        lines.append("No per-step timings (no trace summaries were available to the eval).")
    lines.append("")

    lines.append("## Activity (latest: `%s`)" % latest.get("label", ""))
    lines.append("")
    act = (latest.get("metrics") or {}).get("activity") or {}
    rows: List[List[Any]] = []
    for name, k in sorted((traces.get("events") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append(["event", name, k])
    for name, k in sorted((traces.get("call_counts") or {}).items()):
        rows.append(["call", name, k])
    for group in ("outcomes", "draft_source", "write_status"):
        for name, k in sorted((act.get(group) or {}).items()):
            rows.append([group, name, k])
    rows.append(["flag", "injection_suspected", act.get("injection_suspected", 0)])
    rows.append(["flag", "entitlements_degraded", act.get("entitlements_degraded", 0)])
    rows.append(["flag", "model_vs_policy_disagree", (latest.get("metrics") or {}).get("disagreements", 0)])
    lines += _md_table(["kind", "name", "count"], rows)
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Render eval summaries as markdown.")
    p.add_argument("runs_dir")
    p.add_argument("--out", default=None, help="also write the markdown here (dirs are created)")
    args = p.parse_args(argv)

    md = render_markdown(load_eval_summaries(args.runs_dir), args.runs_dir)
    sys.stdout.write(md)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
