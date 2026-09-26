from contextlib import contextmanager

import pytest

from agent.budget import Budget
from agent.context import TicketContext
from agent.schema import WriteIntent
from agent.tools import Tools
from agent.upstream import HttpUpstream, NotFound


class RecordingTracer:
    def __init__(self):
        self.events = []
        self.calls = []

    @contextmanager
    def step(self, name, **attrs):
        yield {}

    @contextmanager
    def call(self, name, **attrs):
        span = dict(attrs)
        self.calls.append((name, span))
        yield span

    def event(self, name, **attrs):
        self.events.append((name, attrs))


def _event_names(tracer):
    return [name for name, _ in tracer.events]


def _build(base_url, ticket_id, dry_run=True, budget=None):
    tracer = RecordingTracer()
    budget = budget or Budget(deadline_ms=5000, max_llm_calls=2, max_tool_calls=8)
    ctx = TicketContext(ticket_id=ticket_id, budget=budget, tracer=tracer)
    upstream = HttpUpstream(base_url, read_timeout_s=1.0, write_timeout_s=1.0, tracer=tracer)
    return Tools(upstream, ctx, dry_run=dry_run), ctx, tracer


def _intent(ticket_id):
    return WriteIntent(ticket_id=ticket_id, reason="rate_limit_increase", summary="capacity review", priority="normal")


# ---- reads ----

def test_read_all_parallel_populates_context_and_filters_kb(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-L")
    tools.read_all_parallel()

    assert ctx.ticket.ticket_id == "TCK-L"
    assert ctx.account_id == "acct_legacy"
    assert ctx.account.auth_model == "legacy_auth"
    assert ctx.entitlements.seats == 10 and ctx.entitlements_error is None
    assert [h.id for h in ctx.kb_hits] == ["kb-0001", "kb-0002", "kb-0003"]
    assert [h.id for h in ctx.kb_visible] == ["kb-0001", "kb-0002"]
    # unknown fields are recorded per endpoint, not silently dropped
    assert ctx.unknown_fields["ticket"] == ["priority_hint"]
    assert ctx.unknown_fields["kb_search"] == ["debug_rank"]
    # four reads charged
    assert ctx.budget.tool_calls == 4
    assert state.calls["/v1/tickets/TCK-L"] == 1
    assert state.calls["/v1/accounts/acct_legacy"] == 1
    assert state.calls["/v1/accounts/acct_legacy/entitlements"] == 1
    assert state.calls["/v1/kb/search"] == 1


def test_kb_query_is_subject_plus_body(fake_upstream, monkeypatch):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-U")
    seen = {}
    original = tools.kb_search

    def spy(query, limit=5):
        seen["query"] = query
        seen["limit"] = limit
        return original(query, limit)

    monkeypatch.setattr(tools, "kb_search", spy)
    tools.read_all_parallel()
    assert seen["query"] == "Rotate scoped API key We need to rotate a key."
    assert seen["limit"] == 5
    assert [h.id for h in ctx.kb_visible] == ["kb-0001", "kb-0003"]
    assert ctx.unknown_fields["entitlements"] == ["shadow_field"]


def test_kb_filter_disabled_keeps_legacy_article_visible_for_unified_account(fake_upstream):
    """Ablation switch: with kb_filter=False the applies_to filter is skipped and the legacy-only
    article (kb-0002) stays visible to a unified_auth account. Default behaviour is unchanged."""
    base, state = fake_upstream
    tracer = RecordingTracer()
    ctx = TicketContext(ticket_id="TCK-U", budget=Budget(5000, 2, 8), tracer=tracer)
    tools = Tools(HttpUpstream(base, 1.0, 1.0, tracer=tracer), ctx, dry_run=True, kb_filter=False)
    tools.read_all_parallel()
    assert ctx.account.auth_model == "unified_auth"
    assert [h.id for h in ctx.kb_hits] == ["kb-0001", "kb-0002", "kb-0003"]
    assert [h.id for h in ctx.kb_visible] == ["kb-0001", "kb-0002", "kb-0003"]  # kb-0002 is legacy-only
    assert ("kb_filter_disabled", {"hits": 3}) in tracer.events

    # default: the filter drops kb-0002 for this account
    tools_default, ctx_default, tracer_default = _build(base, "TCK-U")
    tools_default.read_all_parallel()
    assert [h.id for h in ctx_default.kb_visible] == ["kb-0001", "kb-0003"]
    assert "kb_filter_disabled" not in _event_names(tracer_default)


def test_read_entitlements_degraded_sets_error_and_does_not_raise(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-D")
    tools.read_ticket()
    result = tools.read_entitlements()
    assert result is None
    assert ctx.entitlements is None
    assert ctx.entitlements_error == "entitlement_service_error"
    assert state.calls["/v1/accounts/acct_degraded/entitlements"] == 1  # not retried
    assert ("upstream_degraded", {"endpoint": "entitlements", "code": "entitlement_service_error"}) in tracer.events


def test_tools_tolerate_none_tracer_on_degraded_path(fake_upstream):
    base, state = fake_upstream
    ctx = TicketContext(ticket_id="TCK-D", budget=Budget(5000, 2, 8), tracer=None)
    tools = Tools(HttpUpstream(base, 1.0, 1.0), ctx)
    tools.read_all_parallel()
    assert ctx.entitlements_error == "entitlement_service_error"
    assert tools.escalate(_intent("TCK-D")).status == "skipped_dry_run"


def test_read_all_parallel_with_degraded_entitlements_still_completes(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-D")
    tools.read_all_parallel()
    assert ctx.account.account_id == "acct_degraded"
    assert ctx.entitlements is None and ctx.entitlements_error == "entitlement_service_error"
    assert [h.id for h in ctx.kb_visible] == ["kb-0001", "kb-0003"]


def test_unknown_ticket_raises_not_found(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-NOPE")
    with pytest.raises(NotFound) as excinfo:
        tools.read_all_parallel()
    assert excinfo.value.code == "ticket_not_found"
    assert ctx.ticket is None


def test_tools_do_not_accept_account_id():
    import inspect

    for name in ("read_ticket", "read_account", "read_entitlements", "read_all_parallel", "filter_kb"):
        params = inspect.signature(getattr(Tools, name)).parameters
        assert "account_id" not in params, name


# ---- the write gate ----

def test_escalate_dry_run_skips_and_sends_nothing(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-L", dry_run=True)
    result = tools.escalate(_intent("TCK-L"))
    assert result.status == "skipped_dry_run"
    assert result.escalation_id is None
    assert state.posts == []
    assert "POST /v1/escalations" not in state.calls
    assert "write_skipped_dry_run" in _event_names(tracer)


def test_escalate_live_files_with_confirm_true(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-L", dry_run=False)
    result = tools.escalate(_intent("TCK-L"))
    assert result.status == "filed"
    assert result.escalation_id == "ESC-0001"
    assert len(state.posts) == 1
    path, payload = state.posts[0]
    assert path == "/v1/escalations"
    assert payload == {
        "ticket_id": "TCK-L",
        "reason": "rate_limit_increase",
        "summary": "capacity review",
        "confirm": True,
    }
    assert payload["confirm"] is True
    assert "write_filed" in _event_names(tracer)
    # duplicate check happened before the write
    assert state.calls["/v1/escalations"] == 1


def test_escalate_second_time_is_duplicate_and_stores_only_one(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-L", dry_run=False)
    first = tools.escalate(_intent("TCK-L"))
    second = tools.escalate(_intent("TCK-L"))
    assert first.status == "filed"
    assert second.status == "skipped_duplicate"
    assert second.escalation_id == first.escalation_id == "ESC-0001"
    assert len(state.escalations) == 1
    assert len(state.posts) == 1
    assert "write_skipped_duplicate" in _event_names(tracer)


def test_escalate_past_deadline_is_pending_manual_and_sends_nothing(fake_upstream):
    base, state = fake_upstream
    clock = [100.0]
    budget = Budget(deadline_ms=1000, max_llm_calls=2, max_tool_calls=8, now=lambda: clock[0])
    tools, ctx, tracer = _build(base, "TCK-L", dry_run=False, budget=budget)
    assert not budget.past_deadline()
    clock[0] += 5.0  # five seconds later, one-second deadline
    assert budget.past_deadline()
    assert budget.remaining_ms() == 0 and budget.fraction_remaining() == 0.0

    result = tools.escalate(_intent("TCK-L"))
    assert result.status == "pending_manual"
    assert state.posts == []
    assert "/v1/escalations" not in state.calls
    assert "deadline_skip" in _event_names(tracer)


def test_escalate_rejected_when_upstream_says_no(fake_upstream):
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-NOPE", dry_run=False)  # the ticket the context is bound to
    result = tools.escalate(_intent("TCK-NOPE"))  # 422 unknown_ticket
    assert result.status == "rejected"
    assert result.error == "unknown_ticket"
    assert len(state.escalations) == 0
    assert "write_rejected" in _event_names(tracer)


def test_escalate_refuses_intent_for_another_ticket(fake_upstream):
    """B1: the intent is bound to the context; a mismatched ticket_id never reaches the upstream."""
    base, state = fake_upstream
    tools, ctx, tracer = _build(base, "TCK-L", dry_run=False)
    for bad in ("TCK-U", ""):
        result = tools.escalate(_intent(bad))
        assert result.status == "rejected" and result.error == "ticket_mismatch"
    assert state.posts == []
    assert "/v1/escalations" not in state.calls  # not even the duplicate GET
    assert "POST /v1/escalations" not in state.calls
    assert ("write_rejected", {"ticket_id": "TCK-L", "code": "ticket_mismatch"}) in tracer.events
    assert ctx.budget.tool_calls == 0


def test_escalate_rechecks_deadline_after_duplicate_get():
    """B2: the clock advances during the duplicate GET; the POST must not happen."""
    from tests._fakes import FixtureUpstream

    clock = [100.0]

    class SlowGet(FixtureUpstream):
        def get(self, path, params=None, retries=1):
            out = super().get(path, params, retries)
            if path == "/v1/escalations":
                clock[0] += 5.0  # the GET took five seconds; deadline is one second
            return out

    up = SlowGet()
    tracer = RecordingTracer()
    budget = Budget(deadline_ms=1000, max_llm_calls=2, max_tool_calls=8, now=lambda: clock[0])
    ctx = TicketContext(ticket_id="TCK-1103", budget=budget, tracer=tracer)
    tools = Tools(up, ctx, dry_run=False)
    assert not budget.past_deadline()
    result = tools.escalate(_intent("TCK-1103"))
    assert result.status == "pending_manual" and result.error == "past_deadline"
    assert up.posts == [] and up.calls.count("/v1/escalations") == 1
    assert ("deadline_skip", {"step": "write", "ticket_id": "TCK-1103", "phase": "pre_post"}) in tracer.events


def test_escalate_201_without_parseable_id_is_pending_manual_never_reposted():
    """S8: a 201 whose body cannot be read as a row is treated like a timed-out write."""
    from agent.upstream import UpstreamDegraded
    from tests._fakes import FixtureUpstream

    class OddBody(FixtureUpstream):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def post(self, path, payload):
            self.posts.append((path, dict(payload)))
            if self.mode == "raise":
                raise UpstreamDegraded(201, "invalid_response", "unparseable body")
            return 201, {"status": "filed"}  # object without an escalation_id

    for mode in ("raise", "noid"):
        up = OddBody(mode)
        tracer = RecordingTracer()
        ctx = TicketContext(ticket_id="TCK-1103", budget=Budget(5000, 2, 8), tracer=tracer)
        result = Tools(up, ctx, dry_run=False).escalate(_intent("TCK-1103"))
        assert result.status == "pending_manual" and result.error == "invalid_response", mode
        assert len(up.posts) == 1, mode
        assert any(name == "write_degraded" for name, _ in tracer.events)


def test_reads_fail_closed_on_identity_mismatch():
    """S7: a record whose id differs from the one requested is never used."""
    from agent.upstream import UpstreamDegraded
    from tests._fakes import FixtureUpstream

    class Swapped(FixtureUpstream):
        def get(self, path, params=None, retries=1):
            if path == "/v1/tickets/TCK-1101":
                return dict(self.tickets["TCK-1102"])
            if path == "/v1/accounts/acct_1002/entitlements":
                return dict(self.entitlements["acct_1003"])
            return super().get(path, params, retries)

    up = Swapped()
    tracer = RecordingTracer()
    ctx = TicketContext(ticket_id="TCK-1101", budget=Budget(5000, 2, 8), tracer=tracer)
    tools = Tools(up, ctx)
    with pytest.raises(UpstreamDegraded) as excinfo:
        tools.read_ticket()
    assert excinfo.value.code == "identity_mismatch" and ctx.ticket is None

    ctx2 = TicketContext(ticket_id="TCK-1102", budget=Budget(5000, 2, 8), tracer=tracer)
    tools2 = Tools(up, ctx2)
    tools2.read_ticket()
    assert tools2.read_entitlements() is None
    assert ctx2.entitlements is None and ctx2.entitlements_error == "identity_mismatch"


def test_only_escalate_posts_to_escalations():
    """Static guard: the write endpoint string must appear only inside Tools.escalate."""
    import inspect
    import os

    import agent

    pkg_dir = os.path.dirname(agent.__file__)
    offenders = []
    for fname in os.listdir(pkg_dir):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(pkg_dir, fname), encoding="utf-8") as fh:
            src = fh.read()
        if ".post(" in src and fname != "tools.py" and fname != "upstream.py":
            offenders.append(fname)
    assert offenders == []
    escalate_src = inspect.getsource(Tools.escalate)
    assert '"confirm": True' in escalate_src
    assert 'self.upstream.post("/v1/escalations"' in escalate_src
    tools_src = inspect.getsource(Tools)
    assert tools_src.count(".post(") == 1
