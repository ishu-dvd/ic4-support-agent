from agent.redact import (
    FORBIDDEN_REPLY_VALUES,
    account_view,
    entitlements_view,
    filter_applies_to,
    kb_view,
    redact_for_trace,
)
from agent.schema import Account, Entitlements, KbHit


def _account(auth_model):
    return Account(
        account_id="acct",
        name="Name",
        plan_tier="business",
        region="us-east",
        auth_model=auth_model,
        customer_since="2020-01-01",
        primary_contact={"name": "Ops", "email": "ops@example.test"},
    )


def _hit(hid, applies_to):
    return KbHit(id=hid, title=hid, body="body", tags=[], score=1.0, applies_to=applies_to)


GENERAL = _hit("kb-0001", {})
LEGACY = _hit("kb-0002", {"auth_model": ["legacy_auth"]})
UNIFIED = _hit("kb-0003", {"auth_model": ["unified_auth"]})
UNKNOWN_KEY = _hit("kb-0099", {"contract_type": ["msa"]})
HITS = [GENERAL, LEGACY, UNIFIED, UNKNOWN_KEY]


def _ids(hits):
    return [h.id for h in hits]


def test_legacy_auth_keeps_legacy_article_drops_unified():
    assert _ids(filter_applies_to(HITS, _account("legacy_auth"))) == ["kb-0001", "kb-0002"]


def test_unified_auth_keeps_unified_article_drops_legacy():
    assert _ids(filter_applies_to(HITS, _account("unified_auth"))) == ["kb-0001", "kb-0003"]


def test_empty_applies_to_always_kept_and_none_account_keeps_only_general():
    assert _ids(filter_applies_to(HITS, None)) == ["kb-0001"]


def test_unknown_attribute_key_drops_the_hit():
    assert "kb-0099" not in _ids(filter_applies_to(HITS, _account("legacy_auth")))


def test_multi_key_applies_to_requires_all_keys():
    both = _hit("kb-both", {"auth_model": ["legacy_auth"], "region": ["us-east"]})
    other_region = _hit("kb-other", {"auth_model": ["legacy_auth"], "region": ["eu-west"]})
    assert _ids(filter_applies_to([both, other_region], _account("legacy_auth"))) == ["kb-both"]


def test_account_view_is_whitelist_only():
    view = account_view(_account("unified_auth"))
    assert view == {"plan_tier": "business", "region": "us-east", "auth_model": "unified_auth"}
    assert account_view(None) == {}


def test_entitlements_view_never_leaks_internal_fields():
    ent = Entitlements(
        account_id="acct",
        support_tier="premium",
        seats=40,
        features=["sso"],
        sla_hours=4,
        rate_limit_rpm=1200,
        updated_at="2026-08-02T09:14:00Z",
    )
    view = entitlements_view(ent, None)
    assert view == {"support_tier": "premium", "features": ["sso"], "sla_hours": 4, "seats": 40}
    assert "rate_limit_rpm" not in view and "updated_at" not in view


def test_entitlements_view_degraded():
    view = entitlements_view(None, "entitlement_service_error")
    assert view["unavailable"] is True
    assert view["reason"] == "entitlement_service_error"
    assert "rate_limit_rpm" not in view and "updated_at" not in view and "detail" not in view


def test_kb_view_fields():
    assert kb_view([LEGACY]) == [{"id": "kb-0002", "title": "kb-0002", "body": "body"}]


def test_redact_for_trace_masks_emails_and_long_digit_runs():
    text = "Mail ops@northwind.example about invoice 1234567890 (5 seats, card 4111 1111)."
    out = redact_for_trace(text)
    assert "ops@northwind.example" not in out
    assert "1234567890" not in out
    assert "<email>" in out and "<num>" in out
    assert "5 seats" in out and "4111 1111" in out  # short digit runs stay
    assert redact_for_trace("") == ""


def test_forbidden_reply_values():
    assert FORBIDDEN_REPLY_VALUES == ("rate_limit_rpm", "updated_at", "@")
