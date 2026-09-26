"""Ticket-scoped tools: the only code that talks to the upstream on the agent's behalf.

Why no account_id parameters: every read is keyed off `ctx.ticket_id`, and the account is
whatever the ticket says it is. A prompt injection therefore has no lever to make the agent
read (or escalate) a different customer's data. `escalate` is the single write path.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .budget import BudgetExceeded
from .context import TicketContext
from .guard import strip_injection_blocks
from .redact import filter_applies_to
from .schema import Account, Entitlements, KbHit, Ticket, WriteIntent, WriteResult, parse
from .trace import NoopTracer
from .upstream import HttpUpstream, UpstreamDegraded, WriteRejected


class Tools:
    def __init__(self, upstream: HttpUpstream, ctx: TicketContext, dry_run: bool = True, kb_filter: bool = True):
        self.upstream = upstream
        self.ctx = ctx
        self.dry_run = dry_run
        # kb_filter=False is an ablation switch for the eval (measures what the applies_to filter buys);
        # it is never the default and the loop records the fact with a `kb_filter_disabled` event.
        self.kb_filter = kb_filter
        self._lock = threading.Lock()
        if self.ctx.tracer is None:
            # Slice B may build a context without a tracer (unit tests, --fake); events are then dropped.
            self.ctx.tracer = NoopTracer()

    # ---- helpers ----
    def _note_unknown(self, endpoint: str, unknown: List[str]) -> None:
        if not unknown:
            return
        with self._lock:
            bucket = self.ctx.unknown_fields.setdefault(endpoint, [])
            for k in unknown:
                if k not in bucket:
                    bucket.append(k)

    @staticmethod
    def _check_identity(endpoint: str, expected: str, actual: str) -> None:
        """A record whose id is not the one we asked for is treated as a degraded upstream, not
        as data: a proxy or cache returning another customer's row must fail closed."""
        if str(actual) != str(expected):
            raise UpstreamDegraded(None, "identity_mismatch", "%s: asked for %s, got %s" % (endpoint, expected, actual))

    # ---- reads ----
    def read_ticket(self) -> Ticket:
        self.ctx.budget.charge_tool()
        raw = self.upstream.get("/v1/tickets/%s" % self.ctx.ticket_id)
        ticket, unknown = parse(Ticket, raw)
        self._check_identity("ticket", self.ctx.ticket_id, ticket.ticket_id)
        self._note_unknown("ticket", unknown)
        self.ctx.ticket = ticket
        self.ctx.account_id = ticket.account_id
        return ticket

    def read_account(self) -> Account:
        if not self.ctx.account_id:
            raise RuntimeError("read_ticket must run before read_account")
        self.ctx.budget.charge_tool()
        raw = self.upstream.get("/v1/accounts/%s" % self.ctx.account_id)
        account, unknown = parse(Account, raw)
        self._check_identity("account", self.ctx.account_id, account.account_id)
        self._note_unknown("account", unknown)
        self.ctx.account = account
        return account

    def read_entitlements(self) -> Optional[Entitlements]:
        """Degradation is a normal outcome here, not an exception: acct_1009's record is
        malformed upstream and the agent must still answer the ticket (with entitlements marked
        unavailable), so UpstreamDegraded is turned into `ctx.entitlements_error`."""
        if not self.ctx.account_id:
            raise RuntimeError("read_ticket must run before read_entitlements")
        self.ctx.budget.charge_tool()
        try:
            raw = self.upstream.get("/v1/accounts/%s/entitlements" % self.ctx.account_id)
            ent, unknown = parse(Entitlements, raw)
            self._check_identity("entitlements", self.ctx.account_id, ent.account_id)
        except UpstreamDegraded as exc:
            self.ctx.entitlements = None
            self.ctx.entitlements_error = exc.code  # the code only; `detail` never enters the context
            self.ctx.tracer.event("upstream_degraded", endpoint="entitlements", code=exc.code)
            return None
        self._note_unknown("entitlements", unknown)
        self.ctx.entitlements = ent
        self.ctx.entitlements_error = None
        return ent

    def kb_search(self, query: str, limit: int = 5) -> List[KbHit]:
        self.ctx.budget.charge_tool()
        raw = self.upstream.get("/v1/kb/search", params={"q": query, "limit": limit})
        hits: List[KbHit] = []
        unknown_all: List[str] = []
        for item in raw.get("results", []) or []:
            hit, unknown = parse(KbHit, item)
            hits.append(hit)
            for k in unknown:
                if k not in unknown_all:
                    unknown_all.append(k)
        self._note_unknown("kb_search", unknown_all)
        self.ctx.kb_hits = hits
        return hits

    def read_all_parallel(self) -> None:
        """read_ticket first (it yields account_id and the kb query), then the three independent
        reads concurrently; failures are collected so every thread finishes before we raise."""
        ticket = self.read_ticket()
        # The KB query never carries injected lines upstream (they would poison the lexical
        # ranking and end up in the server's logs).
        query = ticket.subject + " " + strip_injection_blocks(ticket.body)
        errors: List[Exception] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(self.read_account),
                pool.submit(self.read_entitlements),
                pool.submit(self.kb_search, query, 5),
            ]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:  # re-raised below once every worker has finished
                    errors.append(exc)
                    for pending in futures:
                        pending.cancel()
        if errors:
            raise errors[0]
        self.filter_kb()

    def filter_kb(self) -> List[KbHit]:
        if not self.kb_filter:
            self.ctx.kb_visible = list(self.ctx.kb_hits)
            self.ctx.tracer.event("kb_filter_disabled", hits=len(self.ctx.kb_hits))
            return self.ctx.kb_visible
        self.ctx.kb_visible = filter_applies_to(self.ctx.kb_hits, self.ctx.account)
        return self.ctx.kb_visible

    # ---- the only write ----
    def escalate(self, intent: WriteIntent) -> WriteResult:
        """File an escalation, in this order: deadline -> dry_run -> duplicate check -> POST.

        `confirm` is a hardcoded literal here, not a parameter: the server treats it as the
        "I really mean it" flag, and the only place allowed to mean it is this function after
        the policy has produced a WriteIntent. Nothing model-generated can set it.
        """
        ctx = self.ctx
        tracer = ctx.tracer

        # The intent must be about *this* ticket. Policy always builds it from ctx.ticket_id, so
        # a mismatch means a caller (or a future refactor) is trying to file against another
        # customer's ticket: refuse before any network call.
        if not intent.ticket_id or intent.ticket_id != ctx.ticket_id:
            tracer.event("write_rejected", ticket_id=ctx.ticket_id, code="ticket_mismatch")
            return WriteResult(status="rejected", error="ticket_mismatch")

        if ctx.budget.past_deadline():
            tracer.event("deadline_skip", step="write", ticket_id=intent.ticket_id)
            return WriteResult(status="pending_manual", error="past_deadline")

        if self.dry_run:
            tracer.event("write_skipped_dry_run", ticket_id=intent.ticket_id, reason=intent.reason)
            return WriteResult(status="skipped_dry_run")

        try:
            ctx.budget.charge_tool()
            existing = self.upstream.get("/v1/escalations")
        except BudgetExceeded:
            tracer.event("deadline_skip", step="write", ticket_id=intent.ticket_id, reason="budget_exceeded")
            return WriteResult(status="pending_manual", error="budget_exceeded")
        except UpstreamDegraded as exc:
            # Cannot prove there is no duplicate; a human should file rather than risk a double write.
            tracer.event("write_degraded", ticket_id=intent.ticket_id, code=exc.code, phase="duplicate_check")
            return WriteResult(status="pending_manual", error=exc.code)
        for row in existing.get("escalations", []) or []:
            if row.get("ticket_id") == intent.ticket_id:
                eid = row.get("escalation_id")
                tracer.event("write_skipped_duplicate", ticket_id=intent.ticket_id, escalation_id=eid)
                return WriteResult(status="skipped_duplicate", escalation_id=eid)

        # The duplicate GET took time; re-check the deadline right before the irreversible step.
        if ctx.budget.past_deadline():
            tracer.event("deadline_skip", step="write", ticket_id=intent.ticket_id, phase="pre_post")
            return WriteResult(status="pending_manual", error="past_deadline")

        payload = {
            "ticket_id": ctx.ticket_id,
            "reason": intent.reason,
            "summary": intent.summary,
            "confirm": True,
        }
        try:
            ctx.budget.charge_tool()
            status, body = self.upstream.post("/v1/escalations", payload)
        except BudgetExceeded:
            tracer.event("deadline_skip", step="write", ticket_id=intent.ticket_id, reason="budget_exceeded")
            return WriteResult(status="pending_manual", error="budget_exceeded")
        except WriteRejected as exc:
            tracer.event("write_rejected", ticket_id=intent.ticket_id, code=exc.code)
            return WriteResult(status="rejected", error=exc.code)
        except UpstreamDegraded as exc:
            # Unknown whether the write landed; never retry a write, hand it to a human.
            tracer.event("write_degraded", ticket_id=intent.ticket_id, code=exc.code, phase="post")
            return WriteResult(status="pending_manual", error=exc.code)

        if status != 201:
            tracer.event("write_rejected", ticket_id=intent.ticket_id, code="unexpected_status_%s" % status)
            return WriteResult(status="rejected", error="unexpected_status_%s" % status)
        eid = body.get("escalation_id") if isinstance(body, dict) else None
        if not eid:
            # 201 without an id we can quote: the row probably exists, so never re-POST.
            tracer.event("write_degraded", ticket_id=intent.ticket_id, code="invalid_response", phase="post")
            return WriteResult(status="pending_manual", error="invalid_response")
        tracer.event("write_filed", ticket_id=intent.ticket_id, escalation_id=eid, reason=intent.reason)
        return WriteResult(status="filed", escalation_id=eid)
