"""Rules classifier against the support fixture (read-only) and the golden labels."""
from __future__ import annotations

from typing import Optional

import pytest

from agent.rules import CATEGORIES, REQUEST_TYPES, classify
from agent.schema import Account, Entitlements, KbHit, Ticket
from tests._fakes import filter_applies_to_local, kb_search_local, load

TICKETS = {t["ticket_id"]: t for t in load("tickets.json")}
ACCOUNTS = {a["account_id"]: a for a in load("accounts.json")}
ENTITLEMENTS = {e["account_id"]: e for e in load("entitlements.json")}
KB = load("kb.json")
GOLDEN = load("golden.json")
DEGRADED = {"acct_1009"}  # seats="unlimited" -> 500 entitlement_service_error upstream


def _inputs(ticket_id: str):
    t = TICKETS[ticket_id]
    a = ACCOUNTS[t["account_id"]]
    ticket = Ticket(**t)
    account = Account(**a)
    error: Optional[str] = None
    ents: Optional[Entitlements] = None
    if a["account_id"] in DEGRADED:
        error = "entitlement_service_error"
    else:
        ents = Entitlements(**ENTITLEMENTS[a["account_id"]])
    hits = filter_applies_to_local(kb_search_local(t["subject"] + " " + t["body"], KB), a)
    kb_visible = [KbHit(**h) for h in hits]
    return ticket, account, ents, error, kb_visible


def test_vocabulary_matches_golden():
    assert len(CATEGORIES) == 19
    assert set(CATEGORIES) == {g["expected_category"] for g in GOLDEN}
    assert REQUEST_TYPES == ("question", "how_to", "dispute", "change_request", "incident")


@pytest.mark.parametrize(
    "ticket_id,expected",
    [
        ("TCK-1101", "billing_proration"),
        ("TCK-1102", "credential_rotation"),
        ("TCK-1103", "rate_limit_increase"),
        ("TCK-1104", "credential_rotation"),
        ("TCK-1109", "entitlement_unknown"),
        ("TCK-1115", "feature_not_entitled"),
        ("TCK-1121", "rate_limit_backoff"),
        ("TCK-1123", "invoice_dispute"),
        ("TCK-1128", "identity_unverified"),
        ("TCK-1130", "insufficient_information"),
    ],
)
def test_expected_category(ticket_id, expected):
    cls = classify(*_inputs(ticket_id))
    assert cls.category == expected, cls.reasons
    assert cls.request_type in REQUEST_TYPES
    assert 0.0 <= cls.confidence <= 1.0
    assert cls.reasons


def test_injection_flag_and_sanitized_classification():
    cls = classify(*_inputs("TCK-1123"))
    assert cls.injection_suspected is True
    assert cls.category == "invoice_dispute"
    assert any("injection_suspected" in r for r in cls.reasons)
    clean = classify(*_inputs("TCK-1101"))
    assert clean.injection_suspected is False


def test_degraded_only_matters_for_plan_questions():
    ticket, account, _, _, kb = _inputs("TCK-1102")  # credential rotation, nothing about plans
    cls = classify(ticket, account, None, "entitlement_service_error", kb)
    assert cls.category == "credential_rotation"


def test_auth_model_reason_for_credential_rotation():
    legacy = classify(*_inputs("TCK-1104"))
    unified = classify(*_inputs("TCK-1102"))
    assert any("legacy_auth" in r for r in legacy.reasons)
    assert any("unified_auth" in r for r in unified.reasons)


def test_golden_accuracy():
    misses = []
    for g in GOLDEN:
        cls = classify(*_inputs(g["ticket_id"]))
        if cls.category != g["expected_category"]:
            misses.append((g["ticket_id"], g["expected_category"], cls.category, round(cls.confidence, 2)))
    acc = 1.0 - len(misses) / float(len(GOLDEN))
    print("\nrules accuracy over golden: %.3f (%d/%d)" % (acc, len(GOLDEN) - len(misses), len(GOLDEN)))
    for m in misses:
        print("  miss %s expected=%s got=%s conf=%s" % m)
    assert acc >= 0.7, misses
