"""Model-output parsing and pricing (N3, S9, B4 wording).

The model is an untrusted input like the ticket: its JSON is parsed strictly, its advisory flag
only counts when it is a real JSON boolean, and a model we cannot price reports None, never $0.
"""
from __future__ import annotations

import json

import pytest

from agent.llm import (
    ESCALATION_RECOMMENDED,
    PRICE_PER_1K,
    LLMError,
    LLMUsage,
    RulesLLM,
    _parse_draft,
    cost_usd,
)

CTX = {"category": "invoice_dispute"}


def _doc(**over):
    base = {"category": "invoice_dispute", "diagnosis": "d", "reply": "r", "kb_cited": ["kb-0009"],
            "escalate_recommended": True}
    base.update(over)
    return json.dumps(base)


def test_parse_happy_path_and_fenced_json():
    d = _parse_draft(_doc(), CTX)
    assert d.category == "invoice_dispute" and d.kb_cited == ["kb-0009"] and d.escalate_recommended is True
    assert d.source == "llm"
    fenced = "```json\n" + _doc() + "\n```"
    assert _parse_draft(fenced, CTX).reply == "r"


def test_parse_rejects_unknown_keys():
    with pytest.raises(LLMError):
        _parse_draft(_doc(tool_call="escalate"), CTX)
    with pytest.raises(LLMError):
        _parse_draft(_doc(confirm=True), CTX)


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("true", False), ("false", False), (1, False), (None, False), ("yes", False),
])
def test_escalate_recommended_only_json_true_counts(value, expected):
    assert _parse_draft(_doc(escalate_recommended=value), CTX).escalate_recommended is expected


def test_parse_rejects_non_object_and_non_json():
    for bad in ("not json", "[1,2]", '"string"', ""):
        with pytest.raises(LLMError):
            _parse_draft(bad, CTX)


def test_kb_cited_must_be_a_list():
    with pytest.raises(LLMError):
        _parse_draft(_doc(kb_cited={"id": "kb-0009"}), CTX)
    assert _parse_draft(_doc(kb_cited="kb-0009"), CTX).kb_cited == ["kb-0009"]  # single string tolerated
    assert _parse_draft(_doc(kb_cited=None), CTX).kb_cited == []


def test_unknown_category_falls_back_to_rules_category():
    assert _parse_draft(_doc(category="made_up"), CTX).category == "invoice_dispute"


def test_cost_none_for_unknown_model_zero_for_rules():
    assert cost_usd("rules", 1000, 1000) == 0.0
    assert cost_usd("some-new-model", 1000, 1000) is None
    assert "unknown" not in PRICE_PER_1K
    assert cost_usd("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert LLMUsage().cost_usd == 0.0


def test_rules_drafter_never_claims_filed():
    llm = RulesLLM()
    ctx = {
        "category": "invoice_dispute", "escalate": True, "policy_reasons": ["category invoice_dispute"],
        "ticket": {"subject": "s", "body": "b"}, "account": {"plan": "standard"},
        "entitlements": {"features": []}, "kb": [], "degraded": False,
    }
    draft, usage = llm.draft(ctx, None)
    assert ESCALATION_RECOMMENDED in draft.reply
    assert "has been escalated" not in draft.reply and "has now been filed" not in draft.reply
    assert usage.model_id == "rules" and usage.cost_usd == 0.0
