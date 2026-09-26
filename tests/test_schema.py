import pytest

from agent.schema import Account, Entitlements, KbHit, SchemaError, Ticket, WriteResult, parse

TICKET = {
    "ticket_id": "TCK-1",
    "account_id": "acct_1",
    "subject": "s",
    "body": "b",
    "channel": "email",
    "opened_at": "2026-01-01T00:00:00Z",
    "status": "open",
}


def test_parse_exact_payload_has_no_unknowns():
    obj, unknown = parse(Ticket, TICKET)
    assert unknown == []
    assert obj == Ticket(**TICKET)


def test_parse_drops_and_reports_unknown_fields():
    raw = dict(TICKET, priority_hint="p1", internal_notes="x")
    obj, unknown = parse(Ticket, raw)
    assert sorted(unknown) == ["internal_notes", "priority_hint"]
    assert obj == Ticket(**TICKET)
    assert not hasattr(obj, "priority_hint")


def test_from_api_matches_parse():
    obj, unknown = Ticket.from_api(dict(TICKET, extra=1))
    assert obj == Ticket(**TICKET)
    assert unknown == ["extra"]


def test_missing_required_raises_schema_error_naming_the_field():
    raw = dict(TICKET)
    del raw["body"]
    with pytest.raises(SchemaError) as excinfo:
        parse(Ticket, raw)
    assert excinfo.value.field == "body"
    assert excinfo.value.args[0] == "body"


def test_no_coercion_of_values():
    raw = {
        "account_id": "acct_1009",
        "support_tier": "premium",
        "seats": "unlimited",
        "features": [],
        "sla_hours": 4,
        "rate_limit_rpm": 1,
        "updated_at": "x",
    }
    ent, _ = parse(Entitlements, raw)
    assert ent.seats == "unlimited"


def test_defaults_are_not_required():
    obj, unknown = parse(WriteResult, {"status": "filed"})
    assert obj.escalation_id is None and obj.error is None and unknown == []


def test_non_dict_root_raises():
    with pytest.raises(SchemaError):
        parse(KbHit, ["not", "a", "dict"])


def test_account_nested_contact_kept_as_is():
    raw = {
        "account_id": "a",
        "name": "n",
        "plan_tier": "business",
        "region": "us-east",
        "auth_model": "unified_auth",
        "customer_since": "2020",
        "primary_contact": {"name": "Ops", "email": "ops@example.test"},
    }
    acct, unknown = parse(Account, raw)
    assert acct.primary_contact["email"] == "ops@example.test"
    assert unknown == []
