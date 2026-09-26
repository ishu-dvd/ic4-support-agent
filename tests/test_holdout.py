"""Held-out paraphrases: the generalisation signal for the rules classifier (S11).

The rules in agent/rules.py were authored against the golden vocabulary, so 31/31 on golden says
little. tests/holdout/paraphrased.json re-words the same categories with fresh text (no fixture
body is copied). Each holdout ticket is classified with the *real* fixture context for its
account (account, entitlements, applies_to-filtered KB via tests/_fakes), then run through the
policy. The test prints every case and the accuracy and asserts a floor of 0.6 - it is meant to
be read, not just to pass. `scripts/eval_agent.py --holdout` runs the same file through the full
loop so the number lands in reports/latest.md.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Optional

from agent.policy import decide
from agent.rules import CATEGORIES, classify
from agent.schema import Account, Entitlements, KbHit, Ticket
from tests._fakes import FixtureUpstream, filter_applies_to_local, kb_search_local, load

HERE = os.path.dirname(os.path.abspath(__file__))
HOLDOUT_PATH = os.path.join(HERE, "holdout", "paraphrased.json")
MIN_ACCURACY = 0.6
REQUIRED = ("ticket_id", "account_id", "subject", "body", "expected_category", "expected_escalate")


def _holdout():
    with open(HOLDOUT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


HOLDOUT = _holdout()
FIXTURE_BODIES = {t["body"].strip().lower() for t in load("tickets.json")}
FIXTURE_SUBJECTS = {t["subject"].strip().lower() for t in load("tickets.json")}


def _context(up: FixtureUpstream, case: dict):
    """Account / entitlements / kb from the fixture (via the in-process fake); ticket from the holdout file."""
    ticket = Ticket(ticket_id=case["ticket_id"], account_id=case["account_id"], subject=case["subject"],
                    body=case["body"], channel="email", opened_at="2026-09-20T09:00:00Z", status="open")
    acct_raw = up.accounts[case["account_id"]]
    account = Account(**acct_raw)
    ents: Optional[Entitlements] = None
    error: Optional[str] = None
    if case["account_id"] in up.degraded:
        error = "entitlement_service_error"
    else:
        ents = Entitlements(**up.entitlements[case["account_id"]])
    hits = filter_applies_to_local(kb_search_local(ticket.subject + " " + ticket.body, up.kb), acct_raw)
    kb_visible = [KbHit(**h) for h in hits]
    ctx = SimpleNamespace(ticket_id=ticket.ticket_id, ticket=ticket, account=account, entitlements=ents,
                          entitlements_error=error, kb_visible=kb_visible)
    return ticket, account, ents, error, kb_visible, ctx


def test_holdout_file_shape():
    assert len(HOLDOUT) == 10
    ids = [c["ticket_id"] for c in HOLDOUT]
    assert len(set(ids)) == 10 and all(i.startswith("HOLD-") for i in ids)
    up = FixtureUpstream()
    for c in HOLDOUT:
        for f in REQUIRED:
            assert f in c, (c["ticket_id"], f)
        assert c["expected_category"] in CATEGORIES, c["ticket_id"]
        assert c["account_id"] in up.accounts, c["ticket_id"]
        # new wording, not fixture text
        assert c["body"].strip().lower() not in FIXTURE_BODIES, c["ticket_id"]
        assert c["subject"].strip().lower() not in FIXTURE_SUBJECTS, c["ticket_id"]
    cats = [c["expected_category"] for c in HOLDOUT]
    assert cats.count("credential_rotation") == 2
    for cat in ("billing_proration", "rate_limit_increase", "export_stalled", "feature_not_entitled",
                "invoice_dispute", "identity_unverified", "insufficient_information", "webhook_replay"):
        assert cat in cats, cat


def test_holdout_rules_accuracy():
    up = FixtureUpstream()
    rows = []
    for case in HOLDOUT:
        ticket, account, ents, error, kb_visible, ctx = _context(up, case)
        cls = classify(ticket, account, ents, error, kb_visible)
        d = decide(cls, ctx)
        rows.append({
            "ticket_id": case["ticket_id"],
            "expected": case["expected_category"], "predicted": cls.category,
            "category_ok": cls.category == case["expected_category"],
            "expected_escalate": bool(case["expected_escalate"]), "escalate": d.escalate,
            "escalate_ok": d.escalate == bool(case["expected_escalate"]),
            "confidence": cls.confidence,
            "kb_first": kb_visible[0].id if kb_visible else None,
            "expected_kb": case.get("expected_kb", []),
        })
    cat_acc = sum(r["category_ok"] for r in rows) / float(len(rows))
    esc_acc = sum(r["escalate_ok"] for r in rows) / float(len(rows))
    print("\nholdout (paraphrased) rules accuracy: category %.2f  escalate %.2f  (n=%d)" % (cat_acc, esc_acc, len(rows)))
    for r in rows:
        print("  %-9s %-4s category %-24s -> %-24s conf=%.2f | escalate exp=%s got=%s %s | kb first=%s exp=%s" % (
            r["ticket_id"], "ok" if r["category_ok"] else "MISS", r["expected"], r["predicted"], r["confidence"],
            r["expected_escalate"], r["escalate"], "ok" if r["escalate_ok"] else "MISS", r["kb_first"], r["expected_kb"]))
    assert cat_acc >= MIN_ACCURACY, [(r["ticket_id"], r["expected"], r["predicted"]) for r in rows if not r["category_ok"]]


def test_holdout_credential_rotation_cites_by_auth_model():
    """The two credential paraphrases sit on a legacy and a unified account; the visible article must follow."""
    up = FixtureUpstream()
    by_id = {c["ticket_id"]: c for c in HOLDOUT}
    legacy = by_id["HOLD-002"]
    unified = by_id["HOLD-003"]
    assert up.accounts[legacy["account_id"]]["auth_model"] == "legacy_auth"
    assert up.accounts[unified["account_id"]]["auth_model"] == "unified_auth"
    for case, wanted, forbidden in ((legacy, "kb-0002", "kb-0003"), (unified, "kb-0003", "kb-0002")):
        _, _, _, _, kb_visible, _ = _context(up, case)
        ids = [h.id for h in kb_visible]
        assert forbidden not in ids, (case["ticket_id"], ids)
        if wanted in {a["id"] for a in kb_search_local(case["subject"] + " " + case["body"], up.kb)}:
            assert wanted in ids
