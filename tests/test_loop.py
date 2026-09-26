"""loop.run end to end against an in-process fixture upstream (no sockets, no writes)."""
from __future__ import annotations

import json
from typing import List

from agent.config import Config
from agent.llm import Draft, LLMError, LLMUsage, RulesLLM
from agent.loop import RunResult, run
from agent.upstream import UpstreamDegraded
from tests._fakes import FixtureUpstream

STEP_NAMES = ["read_ticket", "read_parallel", "filter_kb", "classify_intent", "escalation_policy", "draft", "output_guard", "write", "summarize"]


class RecordingTracer:
    """Minimal tracer that records step names and events (replaces RunTracer in tests)."""

    run_id = "test-run"
    trace_id = "t"

    def __init__(self):
        self.steps: List[str] = []
        self.events: List[dict] = []
        self.finished = None

    def step(self, name, **attrs):
        self.steps.append(name)
        return _Span()

    def call(self, name, **attrs):
        return _Span()

    def event(self, name, **attrs):
        self.events.append(dict(name=name, **attrs))

    def finish(self, result, **attrs):
        self.finished = result


class _Span:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


class BadLLM:
    """Cites an invisible article and makes a forbidden commitment."""

    def draft(self, prompt_ctx, budget):
        d = Draft(category=prompt_ctx["category"], diagnosis="model diag", reply="Your refund has been approved.", kb_cited=["kb-9999"], escalate_recommended=False, source="llm")
        return d, LLMUsage(100, 50, "gpt-4o-mini", 0.000045)


class FailingLLM:
    def draft(self, prompt_ctx, budget):
        raise LLMError("boom")


def _cfg(**over) -> Config:
    base = dict(dry_run=True, deadline_ms=20000, runs_dir="/tmp/ic4-test-runs")
    base.update(over)
    return Config(**base)


def _run(ticket_id, llm=None, upstream=None, tracer=None, **cfg):
    tracer = tracer or RecordingTracer()
    res = run(ticket_id, _cfg(**cfg), upstream=upstream or FixtureUpstream(), llm=llm or RulesLLM(), tracer=tracer)
    return res, tracer


def _names(tracer):
    return [e["name"] for e in tracer.events]


def test_1101_completed_no_write():
    res, tr = _run("TCK-1101")
    assert isinstance(res, RunResult)
    assert res.outcome == "completed" and res.error is None
    assert res.category == "billing_proration" and res.escalate is False
    assert res.write is None
    assert res.kb_cited == ["kb-0001"]
    assert res.draft_source == "rules" and res.guard_violations == []
    assert res.escalate_recommended_by_model is False
    assert res.duration_ms >= 0 and res.run_id == "test-run"
    assert tr.finished is res
    top_level = [s for s in tr.steps if s not in ("read_account", "read_entitlements", "kb_search")]
    assert top_level == STEP_NAMES
    assert {"read_account", "read_entitlements", "kb_search"} <= set(tr.steps)


def test_1109_degraded_escalates_dry_run():
    res, tr = _run("TCK-1109")
    assert res.outcome == "degraded"
    assert res.entitlements_degraded is True
    assert res.category == "entitlement_unknown"
    assert res.escalate is True and res.priority in ("normal", "high")
    assert res.write is not None and res.write.status == "skipped_dry_run"
    assert res.guard_violations == []
    low = res.reply.lower()
    for feature in ("private networking", "sso", "audit log", "custom sla"):
        assert feature not in low
    assert "could not verify" in low and "recommended this ticket for escalation" in low
    assert "has been escalated" not in low  # dry run: nothing was filed, the reply must not claim it
    assert "entitlements_degraded" in _names(tr)
    assert res.kb_visible and set(res.kb_cited) <= set(res.kb_visible)


def test_1123_injection_is_contained():
    res, tr = _run("TCK-1123")
    assert res.category == "invoice_dispute"
    assert res.escalate is True and res.injection_suspected is True and res.priority == "high"
    assert res.write is not None and res.write.status == "skipped_dry_run"
    low = res.reply.lower()
    for phrase in ("refund has been approved", "approved", "administrator", "sev1", "internal pricing"):
        assert phrase not in low
    assert "recommended this ticket for escalation" in low and "has been escalated" not in low
    assert "injection_suspected" in _names(tr)
    assert res.guard_violations == []


def test_guard_fallback_to_rules():
    res, tr = _run("TCK-1123", llm=BadLLM())
    fallback = [e for e in tr.events if e["name"] == "guard_fallback"]
    assert len(fallback) == 1
    assert "kb_not_visible:kb-9999" in fallback[0]["violations"]
    assert "forbidden_commitment:refund approved" in fallback[0]["violations"]
    assert res.draft_source == "rules_fallback"
    assert res.guard_violations == []
    assert "approved" not in res.reply.lower()
    assert res.escalate is True  # policy decided, model's False is only recorded
    assert res.escalate_recommended_by_model is True  # fallback draft mirrors policy
    assert res.usage.model_id == "gpt-4o-mini" and res.usage.input_tokens == 100


def test_llm_error_falls_back():
    res, tr = _run("TCK-1102", llm=FailingLLM())
    assert "llm_error" in _names(tr)
    assert res.draft_source == "rules_fallback" and res.outcome == "completed"
    assert res.kb_cited == ["kb-0003"]  # unified_auth account sees the scoped-key article only


def test_ticket_not_found():
    up = FixtureUpstream()
    res, tr = _run("TCK-9999", upstream=up)
    assert res.outcome == "failed" and res.error == "ticket_not_found"
    assert res.write is None and up.posts == []
    assert tr.finished is res


def test_write_path_when_not_dry_run():
    up = FixtureUpstream()
    res, tr = _run("TCK-1103", upstream=up, dry_run=False)
    assert res.escalate and res.write is not None and res.write.status == "filed"
    assert res.write.escalation_id == "ESC-0001"
    path, payload = up.posts[0]
    assert path == "/v1/escalations" and payload["confirm"] is True
    assert payload["reason"] == "rate_limit_increase"
    assert "ZZ" not in payload["summary"]
    # B4: the filed confirmation is appended only after the write succeeded, with the reference
    assert res.reply.endswith("This escalation has now been filed (reference ESC-0001).")
    assert "recommended this ticket for escalation" in res.reply
    assert "reply_finalised" in _names(tr)
    # second run is a duplicate -> no second POST, but the reply still confirms the existing filing
    res2, _ = _run("TCK-1103", upstream=up, dry_run=False)
    assert res2.write.status == "skipped_duplicate" and len(up.posts) == 1
    assert "This escalation has now been filed (reference ESC-0001)." in res2.reply


def test_dry_run_reply_never_claims_filed():
    res, tr = _run("TCK-1103")
    assert res.write is not None and res.write.status == "skipped_dry_run"
    assert "has now been filed" not in res.reply and "has been escalated" not in res.reply.lower()
    assert "reply_finalised" not in _names(tr)


class UnsafeRules:
    """Stands in for the RulesLLM fallback and is itself unsafe: forces the fail-closed path."""

    def draft(self, prompt_ctx, budget=None):
        d = Draft(category=prompt_ctx["category"], diagnosis="d", reply="Your refund has been approved.",
                  kb_cited=[], escalate_recommended=False, source="rules")
        return d, LLMUsage(0, 0, "rules", 0.0)


def test_guard_fails_closed_when_fallback_is_unsafe(monkeypatch):
    import agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "RulesLLM", UnsafeRules)
    up = FixtureUpstream()
    res, tr = _run("TCK-1123", llm=BadLLM(), upstream=up, dry_run=False)
    assert "guard_fallback" in _names(tr) and "guard_failed" in _names(tr)
    assert res.outcome == "failed" and res.error == "guard_failed"
    assert res.draft_source == "safe_stub"
    assert res.reply.startswith(loop_mod.SAFE_REPLY)
    assert res.diagnosis == loop_mod.SAFE_DIAGNOSIS
    assert res.kb_cited == []
    assert "forbidden_commitment:refund approved" in res.guard_violations  # remaining violations are reported
    assert "approved" not in res.reply.lower()
    # the policy decision and the write never depended on the draft: invoice_dispute still escalates
    assert res.escalate is True and res.write is not None and res.write.status == "filed"
    assert len(up.posts) == 1 and res.reply not in up.posts[0][1]["summary"]
    assert res.reply.endswith("(reference ESC-0001).")


class TimingOutUpstream(FixtureUpstream):
    """Raises the post-retry timeout error for the given path regexes; everything else is served."""

    def __init__(self, slow_paths):
        super().__init__()
        self.slow_paths = list(slow_paths)

    def get(self, path, params=None, retries=1):
        import re

        if any(re.fullmatch(p, path) for p in self.slow_paths):
            self.calls.append(path)
            raise UpstreamDegraded(None, "timeout", "read timeout after 2.0s")
        return super().get(path, params, retries)


def test_ticket_read_timeout_fails_cleanly():
    res, tr = _run("TCK-1101", upstream=TimingOutUpstream([r"/v1/tickets/.*"]))
    assert res.outcome == "failed" and res.error == "upstream_degraded:timeout"
    assert res.write is None and res.category == "unknown"
    assert {"name": "upstream_degraded", "endpoint": "ticket", "code": "timeout"} in tr.events
    assert tr.finished is res


def test_account_timeout_degrades_but_run_continues():
    up = TimingOutUpstream([r"/v1/accounts/[^/]+"])  # account only; entitlements and kb still answer
    res, tr = _run("TCK-1102", upstream=up)
    assert res.outcome == "degraded" and res.error is None
    assert res.category == "credential_rotation"
    # without the account only universally applicable articles survive the applies_to filter
    assert all(h not in ("kb-0002", "kb-0003") for h in res.kb_cited)
    assert {"name": "upstream_degraded", "endpoint": "account", "code": "timeout"} in tr.events
    assert res.guard_violations == []


def test_kb_timeout_degrades_with_no_citations():
    res, tr = _run("TCK-1103", upstream=TimingOutUpstream([r"/v1/kb/search"]))
    assert res.outcome == "degraded" and res.kb_cited == []
    assert res.escalate is True and res.write is not None and res.write.status == "skipped_dry_run"
    assert {"name": "upstream_degraded", "endpoint": "kb_search", "code": "timeout"} in tr.events


def test_deadline_skip_uses_rules():
    res, tr = _run("TCK-1101", llm=BadLLM(), deadline_ms=0)
    assert "deadline_skip" in _names(tr)
    assert res.draft_source == "rules"
    assert res.usage.model_id == "rules"


def test_unknown_model_cost_event():
    class OddModel:
        def draft(self, prompt_ctx, budget):
            return Draft(prompt_ctx["category"], "d", "Thanks for reaching out; see the linked article.", ["kb-0001"], False, "llm"), LLMUsage(1, 1, "mystery-9b", 0.0)

    res, tr = _run("TCK-1101", llm=OddModel())
    assert "cost_unknown_model" in _names(tr)
    assert res.usage.cost_usd is None  # unknown price is "unavailable", never $0
    assert _run("TCK-1101")[0].usage.cost_usd == 0.0  # the rules drafter is genuinely free


def test_error_strings_never_carry_exception_messages():
    class Boom(FixtureUpstream):
        def get(self, path, params=None, retries=1):
            if path.startswith("/v1/accounts/") and not path.endswith("/entitlements"):
                raise RuntimeError("secret detail ops@example.com 1234567890")
            return super().get(path, params, retries)

    res, _ = _run("TCK-1101", upstream=Boom())
    assert res.outcome == "failed" and res.error == "RuntimeError"
    assert "secret" not in res.error and "@" not in res.error


def test_identity_mismatch_on_account_fails_closed():
    class Swapped(FixtureUpstream):
        def get(self, path, params=None, retries=1):
            if path == "/v1/accounts/acct_1001":
                return dict(self.accounts["acct_1003"])  # another customer's record
            return super().get(path, params, retries)

    res, tr = _run("TCK-1101", upstream=Swapped())
    # the wrong record is dropped, never used: the run degrades exactly like an account timeout
    assert res.outcome == "degraded"
    assert {"name": "upstream_degraded", "endpoint": "account", "code": "identity_mismatch"} in tr.events
    assert all(h not in ("kb-0002", "kb-0003") for h in res.kb_cited)


def test_llm_error_event_has_no_detail():
    class LeakyFail:
        def draft(self, prompt_ctx, budget):
            raise LLMError("HTTP 500 body echoing ops@example.com")

    _, tr = _run("TCK-1101", llm=LeakyFail())
    ev = next(e for e in tr.events if e["name"] == "llm_error")
    # type name + the safe reason code only; the message (which may echo a response body) never lands
    assert ev == {"name": "llm_error", "error": "LLMError", "code": "error"}
    assert "@" not in json.dumps(ev)
