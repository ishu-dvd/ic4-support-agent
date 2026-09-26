#!/usr/bin/env python3
"""Run the agent on one ticket and print the result.

Dry-run by default: nothing is written to the escalations API unless --write is passed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

_IC4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IC4 not in sys.path:
    sys.path.insert(0, _IC4)

from agent.config import load_config  # noqa: E402
from agent.loop import run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the support agent on a single ticket.")
    ap.add_argument("ticket_id")
    ap.add_argument("--write", action="store_true", help="allow the escalation write (default: dry run)")
    ap.add_argument("--json", action="store_true", help="print the RunResult as JSON")
    ap.add_argument("--base-url", default=None, help="upstream base URL (default: UPSTREAM_BASE_URL)")
    ap.add_argument("--deadline-ms", type=int, default=None)
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--no-kb-filter", action="store_true",
                    help="ablation: disable the applies_to KB filter (kb_filter=False); never for real runs")
    args = ap.parse_args(argv)

    cfg = load_config(
        upstream_base_url=args.base_url.rstrip("/") if args.base_url else None,
        deadline_ms=args.deadline_ms,
        prompt_version=args.prompt_version,
        dry_run=False if args.write else None,
        kb_filter=False if args.no_kb_filter else None,
    )
    result = run(args.ticket_id, cfg)

    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2))
        return 0 if result.outcome != "failed" else 1

    write = result.write
    write_status = "%s%s%s" % (
        write.status if write else "none",
        (" id=%s" % write.escalation_id) if write and write.escalation_id else "",
        (" error=%s" % write.error) if write and write.error else "",
    )
    lines = [
        "ticket:        %s" % result.ticket_id,
        "outcome:       %s%s" % (result.outcome, (" (%s)" % result.error) if result.error else ""),
        "category:      %s (confidence %.2f)" % (result.category, result.confidence),
        "request_type:  %s" % result.request_type,
        "escalate:      %s  priority=%s  reasons=%s" % (result.escalate, result.priority, "; ".join(result.policy_reasons) or "-"),
        "write:         %s" % write_status,
        "diagnosis:     %s" % result.diagnosis,
        "reply:         %s" % result.reply,
        "kb_cited:      %s" % (", ".join(result.kb_cited) or "-"),
        "draft_source:  %s  (model recommended escalate=%s)" % (result.draft_source, result.escalate_recommended_by_model),
        "guard:         %s" % (", ".join(result.guard_violations) or "no violations"),
        "injection:     %s" % result.injection_suspected,
        "degraded:      %s" % result.entitlements_degraded,
        "duration:      %d ms" % result.duration_ms,
        "cost:          %s (%s, %d in / %d out tokens)" % (
            ("$%.6f" % result.usage.cost_usd) if result.usage.cost_usd is not None else "unavailable (unknown model price)",
            result.usage.model_id, result.usage.input_tokens, result.usage.output_tokens,
        ),
        "kb_visible:    %s" % (", ".join(result.kb_visible) or "-"),
        "run_id:        %s" % result.run_id,
    ]
    print("\n".join(lines))
    return 0 if result.outcome != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
