"""Deterministic escalation policy."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm import Draft
from agent.policy import ESCALATE_CATEGORIES, PolicyDecision, decide, hours_mentioned, region_unavailable_needs_human
from agent.rules import Classification
from agent.schema import Entitlements, Ticket, WriteIntent

_MARKER = "ZZMODELTEXTZZ refund has been approved"


def _ticket(subject="Subject line", body="Body text"):
    return Ticket(ticket_id="TCK-1", account_id="acct_1", subject=subject, body=body, channel="email", opened_at="", status="open")


def _ents(tier="standard", sla_hours=12):
    return Entitlements(account_id="acct_1", support_tier=tier, seats=5, features=["data_export"], sla_hours=sla_hours, rate_limit_rpm=300, updated_at="x")


def _ctx(ticket=None, ents=None, error=None):
    return SimpleNamespace(ticket_id="TCK-1", ticket=ticket or _ticket(), account=None, entitlements=ents, entitlements_error=error, kb_visible=[])


def _cls(category, request_type="question", injection=False, reasons=None):
    return Classification(category=category, request_type=request_type, confidence=0.9, reasons=reasons or ["signal"], injection_suspected=injection)


def test_escalate_categories_are_the_spec_set():
    assert ESCALATE_CATEGORIES == {
        "rate_limit_increase", "region_unavailable", "invoice_dispute", "identity_unverified",
        "insufficient_information", "entitlement_unknown",
    }


@pytest.mark.parametrize("category", sorted(ESCALATE_CATEGORIES - {"region_unavailable"}))
def test_escalate_for_each_category(category):
    d = decide(_cls(category), _ctx(ents=_ents()))
    assert isinstance(d, PolicyDecision)
    assert d.escalate is True
    assert isinstance(d.intent, WriteIntent)
    assert d.intent.reason == category
    assert d.intent.ticket_id == "TCK-1"
    assert "category:%s" % category in d.reasons


# region_unavailable is conditional: escalate only when the ticket describes a stuck/pending request.
def test_region_unavailable_stuck_request_escalates():
    # mirrors TCK-1105: "queued for three days", "still showing as queued"
    t = _ticket("Private networking request has been queued for three days",
                "We requested private networking on Monday and the request is still showing as queued. Is something wrong?")
    d = decide(_cls("region_unavailable", "incident"), _ctx(ticket=t, ents=_ents()))
    assert d.escalate is True and isinstance(d.intent, WriteIntent)
    assert d.intent.reason == "region_unavailable"
    assert "category:region_unavailable" in d.reasons and "stuck_request" in d.reasons


def test_region_unavailable_availability_question_does_not_escalate():
    # mirrors TCK-1118: a question kb-0008 answers; nothing for a human to unblock
    t = _ticket("Private networking not available in our region?",
                "We tried to enable private networking and nothing happened. We are in ap-south.")
    d = decide(_cls("region_unavailable"), _ctx(ticket=t, ents=_ents()))
    assert d.escalate is False and d.intent is None
    assert "availability_question:answered_from_kb" in d.reasons


@pytest.mark.parametrize("body", [
    "The request has been pending since last week.",
    "Our VPC request is stuck.",
    "Enabled it two days ago, still showing queued.",
])
def test_region_unavailable_needs_human_cues(body):
    assert region_unavailable_needs_human(body) is True


def test_region_unavailable_no_cues():
    assert region_unavailable_needs_human("Is private networking available in eu-central?") is False


def test_benign_category_does_not_escalate():
    d = decide(_cls("billing_proration"), _ctx(ents=_ents()))
    assert d.escalate is False and d.intent is None and d.sla_breach is False


def test_entitlements_error_escalates_benign_category():
    d = decide(_cls("credential_rotation"), _ctx(ents=None, error="entitlement_service_error"))
    assert d.escalate is True
    assert "entitlements_unavailable:entitlement_service_error" in d.reasons
    assert d.intent is not None and "entitlement_service_error" in d.intent.summary


@pytest.mark.parametrize("body", ["our last ticket took eleven hours for a first response", "First response took 11 hours."])
def test_sla_breach(body):
    d = decide(_cls("sla_expectation"), _ctx(ticket=_ticket("First response", body), ents=_ents("premium", 4)))
    assert d.escalate and d.sla_breach and d.priority == "high"
    assert any(r.startswith("sla_breach:") for r in d.reasons)


def test_sla_within_target_no_breach():
    d = decide(_cls("sla_expectation"), _ctx(ticket=_ticket("SLA", "took 3 hours"), ents=_ents("standard", 12)))
    assert not d.escalate and not d.sla_breach


def test_hours_mentioned():
    assert hours_mentioned("eleven hours") == 11
    assert hours_mentioned("about 7 hrs and then three days") == 72
    assert hours_mentioned("no durations") is None


def test_priority_rules():
    assert decide(_cls("billing_proration"), _ctx(ents=_ents("premium"))).priority == "high"
    assert decide(_cls("billing_proration"), _ctx(ents=_ents("basic"))).priority == "low"
    assert decide(_cls("invoice_dispute"), _ctx(ents=_ents("basic"))).priority == "normal"
    assert decide(_cls("billing_proration"), _ctx(ents=_ents("standard"))).priority == "normal"
    assert decide(_cls("billing_proration", injection=True), _ctx(ents=_ents("basic"))).priority == "high"


def test_summary_never_contains_model_text_and_flags_injection():
    draft = Draft(category="invoice_dispute", diagnosis=_MARKER, reply=_MARKER, kb_cited=[], escalate_recommended=True, source="llm")
    cls = _cls("invoice_dispute", "dispute", injection=True)
    ctx = _ctx(ticket=_ticket("Disputed charge on INV-8850213 from ops@example.com", "body " + _MARKER), ents=_ents("standard"))
    d = decide(cls, ctx)
    assert d.intent is not None
    assert "ZZMODELTEXTZZ" not in d.intent.summary
    assert draft.reply not in d.intent.summary
    assert "injection_suspected" in d.intent.summary
    assert "invoice_dispute" in d.intent.summary and "dispute" in d.intent.summary
    assert "tier=standard" in d.intent.summary and "sla_hours=12" in d.intent.summary
    assert "ops@example.com" not in d.intent.summary  # subject is redacted
    assert "<num>" in d.intent.summary and "8850213" not in d.intent.summary
    assert d.intent.priority == "high"


def test_non_escalate_has_no_intent():
    d = decide(_cls("webhook_replay"), _ctx(ents=_ents()))
    assert d.intent is None
