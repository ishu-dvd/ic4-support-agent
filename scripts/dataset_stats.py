"""Prints the label distribution of a golden dataset.

    python3 scripts/dataset_stats.py --variant support

Useful before you trust a score: a metric that looks good may only be tracking the
majority class.
"""
import argparse
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="support", choices=("support", "access"))
    a = p.parse_args()

    golden = json.load(open(os.path.join(ROOT, "data", a.variant, "golden.json"), encoding="utf-8"))
    n = len(golden)
    esc = sum(1 for g in golden if g["expected_escalate"])
    cats = Counter(g["expected_category"] for g in golden)
    no_article = [g["ticket_id"] for g in golden if not g["expected_kb"]]

    print("variant: %s" % a.variant)
    print("cases:   %d" % n)
    print()
    print("escalate = true   %2d  (%d%%)" % (esc, round(100 * esc / n)))
    print("escalate = false  %2d  (%d%%)" % (n - esc, round(100 * (n - esc) / n)))
    print("  always answering the majority scores %d%% on this field alone" %
          round(100 * max(esc, n - esc) / n))
    print()
    print("categories: %d" % len(cats))
    for cat, c in cats.most_common():
        print("  %-34s %2d  (%d%%)" % (cat, c, round(100 * c / n)))
    print()
    print("  always answering the largest category scores %d%% on this field alone" %
          round(100 * cats.most_common(1)[0][1] / n))
    print()
    print("cases where no knowledge-base article applies: %d %s" % (len(no_article), no_article))


if __name__ == "__main__":
    main()
