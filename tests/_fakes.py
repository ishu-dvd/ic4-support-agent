"""Test doubles for slice B: an in-process upstream that serves data/support/*.json read-only.

No HTTP, no sockets: `FixtureUpstream` implements the `get`/`post` contract of
agent.upstream.HttpUpstream so loop.run can be exercised end to end in a unit test.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agent.upstream import NotFound, UpstreamDegraded, WriteRejected

IC4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(IC4_ROOT, "data", "support")


def load(name: str) -> Any:
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


# --- a local copy of the fixture server's lexical ranking (server/search.py weights) ---
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "for", "in", "on", "at",
    "by", "with", "from", "how", "do", "does", "did", "my", "our", "we", "i", "it", "this", "that", "can", "not",
}


def _tok(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def kb_search_local(query: str, articles: List[dict], limit: int = 5) -> List[dict]:
    q = Counter(_tok(query))
    scored = []
    for a in articles:
        title, body, tags = Counter(_tok(a["title"])), Counter(_tok(a["body"])), Counter(_tok(" ".join(a["tags"])))
        s = 0.0
        for term, qn in q.items():
            s += 3.0 * qn * min(title.get(term, 0), 3) + 1.0 * qn * min(body.get(term, 0), 4) + 1.5 * qn * min(tags.get(term, 0), 2)
        if s > 0:
            scored.append((round(s, 4), a))
    scored.sort(key=lambda p: (-p[0], p[1]["id"]))
    out = []
    for s, a in scored[:limit]:
        item = {k: a[k] for k in ("id", "title", "body", "tags")}
        item["score"] = s
        item["applies_to"] = a.get("applies_to", {})
        out.append(item)
    return out


def filter_applies_to_local(hits: List[dict], account: Optional[dict]) -> List[dict]:
    out = []
    for h in hits:
        cond = h.get("applies_to") or {}
        if not cond:
            out.append(h)
        elif account is not None and all(account.get(k) in v for k, v in cond.items()):
            out.append(h)
    return out


class FixtureUpstream:
    """Serves the support fixture in-process. `degraded` account ids raise UpstreamDegraded on entitlements."""

    def __init__(self, degraded: Tuple[str, ...] = ("acct_1009",)):
        self.tickets = {t["ticket_id"]: t for t in load("tickets.json")}
        self.accounts = {a["account_id"]: a for a in load("accounts.json")}
        self.entitlements = {e["account_id"]: e for e in load("entitlements.json")}
        self.kb = load("kb.json")
        self.degraded = set(degraded)
        self.escalations: List[dict] = []
        self.calls: List[str] = []
        self.posts: List[Tuple[str, dict]] = []

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 1) -> dict:
        self.calls.append(path)
        m = re.fullmatch(r"/v1/tickets/([A-Za-z0-9_\-]+)", path)
        if m:
            t = self.tickets.get(m.group(1))
            if t is None:
                raise NotFound(404, "ticket_not_found", "No ticket with id '%s'" % m.group(1))
            return dict(t)
        m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)/entitlements", path)
        if m:
            aid = m.group(1)
            if aid not in self.accounts:
                raise NotFound(404, "account_not_found", "No account")
            if aid in self.degraded:
                raise UpstreamDegraded(500, "entitlement_service_error", "field 'seats' expected int, got str ('unlimited')")
            return dict(self.entitlements[aid])
        m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)", path)
        if m:
            a = self.accounts.get(m.group(1))
            if a is None:
                raise NotFound(404, "account_not_found", "No account")
            return dict(a)
        if path == "/v1/kb/search":
            params = params or {}
            return {"query": params.get("q", ""), "results": kb_search_local(params.get("q", ""), self.kb, int(params.get("limit", 5)))}
        if path == "/v1/escalations":
            return {"escalations": list(self.escalations)}
        raise NotFound(404, "not_found", "No route %s" % path)

    def post(self, path: str, payload: dict) -> Tuple[int, dict]:
        self.posts.append((path, dict(payload)))
        if path != "/v1/escalations":
            raise WriteRejected(404, "not_found", "No route")
        missing = [f for f in ("ticket_id", "reason", "summary") if not payload.get(f)]
        if missing:
            raise WriteRejected(422, "missing_fields", ",".join(missing))
        if payload["ticket_id"] not in self.tickets:
            raise WriteRejected(422, "unknown_ticket", "no such ticket")
        if payload.get("confirm") is not True:
            raise WriteRejected(409, "confirmation_required", "confirm must be true")
        entry = dict(payload)
        entry["escalation_id"] = "ESC-%04d" % (len(self.escalations) + 1)
        entry["status"] = "filed"
        self.escalations.append(entry)
        return 201, entry
