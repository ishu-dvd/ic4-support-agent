"""Ticket intake against the *real* fixture server (server/), not the test double.

A tester creates a ticket for an existing account, then runs the agent on it. The agent must pick the
new ticket up over HTTP exactly like a shipped one: nothing in agent/ reads data/ directly, so a real
upstream that returns tickets the fixture never had is handled by the same path.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agent.config import Config
from agent.loop import run
from agent.upstream import HttpUpstream, NotFound
from server.app import Handler
from server.store import Store


class _Quiet(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


@pytest.fixture
def fixture_server():
    """The actual mock systems-of-record API on a free port, support variant, fresh Store per test."""

    class H(Handler):
        store = Store("support")

        def log_message(self, fmt, *args):
            pass

    srv = _Quiet(("127.0.0.1", 0), H)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1], H.store
    finally:
        srv.shutdown()
        srv.server_close()


def _call(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


# ---- fixture routes ----

def test_accounts_listing_matches_individual_reads(fixture_server):
    base, store = fixture_server
    status, body = _call(base, "/v1/accounts")
    assert status == 200
    ids = [a["account_id"] for a in body["accounts"]]
    assert ids == sorted(ids) and len(ids) == len(store.accounts) > 0
    for a in body["accounts"][:3]:
        assert _call(base, "/v1/accounts/%s" % a["account_id"])[1] == a


def test_create_ticket_is_readable_through_every_read_path(fixture_server):
    base, store = fixture_server
    before = set(_call(base, "/v1/tickets")[1]["ticket_ids"])
    status, t = _call(base, "/v1/tickets", {
        "account_id": "acct_1001", "subject": "Invoice question", "body": "Why is line 3 prorated?", "channel": "email",
    })
    assert status == 201
    assert t["ticket_id"] not in before
    assert t["ticket_id"].startswith("TCK-") and t["account_id"] == "acct_1001"
    assert t["status"] == "open" and t["channel"] == "email" and t["opened_at"].endswith("Z")
    # every field the agent's Ticket schema requires is present
    assert set(t) >= {"ticket_id", "account_id", "subject", "body", "channel", "opened_at", "status"}

    assert _call(base, "/v1/tickets/%s" % t["ticket_id"]) == (200, t)
    assert t["ticket_id"] in _call(base, "/v1/tickets")[1]["ticket_ids"]
    # the new ticket is a legitimate escalation target too
    status, esc = _call(base, "/v1/escalations", {"ticket_id": t["ticket_id"], "reason": "r", "summary": "s", "confirm": True})
    assert status == 201 and esc["ticket_id"] == t["ticket_id"]


def test_create_ticket_numbering_continues_the_fixture_sequence(fixture_server):
    base, store = fixture_server
    highest = max(int(t.split("-")[1]) for t in store.tickets if t.startswith("TCK-"))
    _, a = _call(base, "/v1/tickets", {"account_id": "acct_1002", "subject": "s", "body": "b"})
    _, b = _call(base, "/v1/tickets", {"account_id": "acct_1002", "subject": "s", "body": "b"})
    assert a["ticket_id"] == "TCK-%04d" % (highest + 1)
    assert b["ticket_id"] == "TCK-%04d" % (highest + 2)


def test_create_ticket_validation(fixture_server):
    base, _ = fixture_server
    status, body = _call(base, "/v1/tickets", {"account_id": "acct_1001"})
    assert status == 422 and body["error"]["code"] == "missing_fields"
    assert body["error"]["fields"] == ["subject", "body"]

    status, body = _call(base, "/v1/tickets", {"account_id": "acct_nope", "subject": "s", "body": "b"})
    assert status == 422 and body["error"]["code"] == "unknown_account"

    status, body = _call(base, "/v1/tickets", {"account_id": "acct_1001", "subject": "s", "body": "b", "ticket_id": "../x"})
    assert status == 422 and body["error"]["code"] == "invalid_ticket_id"

    status, body = _call(base, "/v1/tickets", {"account_id": "acct_1001", "subject": "s", "body": "b", "ticket_id": "TCK-1101"})
    assert status == 409 and body["error"]["code"] == "ticket_exists"

    status, body = _call(base, "/v1/tickets", {"account_id": "acct_1001", "subject": "s", "body": "b", "ticket_id": "TCK-CUSTOM"})
    assert status == 201 and body["ticket_id"] == "TCK-CUSTOM"
    status, body = _call(base, "/v1/tickets", {"account_id": "acct_1001", "subject": "s", "body": "b", "ticket_id": "TCK-CUSTOM"})
    assert status == 409


def test_shipped_data_is_never_mutated(fixture_server):
    """The write lands in memory only; a second Store loads the pristine fixture."""
    base, store = fixture_server
    _call(base, "/v1/tickets", {"account_id": "acct_1001", "subject": "s", "body": "b", "ticket_id": "TCK-TEMP"})
    assert "TCK-TEMP" in store.tickets
    assert "TCK-TEMP" not in Store("support").tickets


# ---- the agent, end to end, on a ticket that did not exist at startup ----

def _cfg(base, runs_dir):
    return Config(upstream_base_url=base, model_provider="rules", runs_dir=str(runs_dir), dry_run=True,
                  read_timeout_s=2.0, write_timeout_s=2.0, deadline_ms=10000)


def test_agent_runs_a_ticket_created_at_runtime(fixture_server, tmp_path):
    base, _ = fixture_server
    cfg = _cfg(base, tmp_path)

    # not there yet: the agent fails closed exactly as for any unknown id
    missing = run("TCK-RUNTIME", cfg)
    assert missing.outcome == "failed" and missing.error == "ticket_not_found"

    status, t = _call(base, "/v1/tickets", {
        "ticket_id": "TCK-RUNTIME", "account_id": "acct_1001", "channel": "web",
        "subject": "First invoice after upgrade is higher than the plan price",
        "body": "We upgraded to Business on the 14th and the invoice is higher than the plan price. Invoice INV-99999.",
    })
    assert status == 201

    result = run("TCK-RUNTIME", cfg)
    assert result.outcome in ("completed", "degraded"), result.error
    assert result.ticket_id == "TCK-RUNTIME"
    assert result.category  # classified from the body the upstream served, not from any local file
    assert result.write is None or result.write.status == "skipped_dry_run"


def test_agent_client_create_ticket_helper_roundtrip(fixture_server):
    base, _ = fixture_server
    up = HttpUpstream(base, read_timeout_s=2.0, write_timeout_s=2.0)
    rec = up.create_ticket({"account_id": "acct_1003", "subject": "s", "body": "b"})
    assert rec["account_id"] == "acct_1003"
    assert up.get("/v1/tickets/%s" % rec["ticket_id"]) == rec
    assert rec["ticket_id"] in up.get("/v1/tickets")["ticket_ids"]
    with pytest.raises(NotFound):
        up.get("/v1/tickets/TCK-STILL-MISSING")
