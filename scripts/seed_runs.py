#!/usr/bin/env python3
"""Build seed/runs from runs/: every full golden-set eval (runs/eval-*-golden-*) plus the per-ticket run
directories its cases.jsonl links to. start.sh copies seed/runs into an empty RUNS_DIR so a freshly
deployed App Platform instance (ephemeral disk) shows the multi-model comparison immediately.

Stdlib only.   python3 scripts/seed_runs.py [--runs-dir runs] [--out seed/runs] [--pattern golden]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default=os.path.join("seed", "runs"))
    p.add_argument("--pattern", default="golden", help="substring an eval dir name must contain")
    p.add_argument("--extra-runs", type=int, default=0, help="also copy the N most recent non-eval runs")
    args = p.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    evals = sorted(d for d in os.listdir(args.runs_dir)
                   if d.startswith("eval-") and args.pattern in d
                   and os.path.isfile(os.path.join(args.runs_dir, d, "summary.json")))
    run_ids = set()
    for ev in evals:
        shutil.copytree(os.path.join(args.runs_dir, ev), os.path.join(args.out, ev))
        try:
            with open(os.path.join(args.runs_dir, ev, "cases.jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rid = json.loads(line).get("run_id")
                        if rid:
                            run_ids.add(rid)
        except OSError:
            pass
    if args.extra_runs:
        recent = sorted((d for d in os.listdir(args.runs_dir) if not d.startswith("eval-")), reverse=True)
        run_ids.update(recent[:args.extra_runs])
    copied = 0
    for rid in sorted(run_ids):
        src = os.path.join(args.runs_dir, rid)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(args.out, rid))
            copied += 1
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out) for f in fs)
    print(json.dumps({"evals": len(evals), "runs": copied, "out": args.out, "size_kb": size // 1024}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
