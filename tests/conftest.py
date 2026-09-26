"""Test wiring: make `import agent` work and provide an in-memory HTTP test double.

The double mirrors the mock API's routes and error envelope closely enough to exercise the
retry / degraded / write-gate paths without touching the real fixture server.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

IC4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if IC4_ROOT not in sys.path:
    sys.path.insert(0, IC4_ROOT)

SLOW_SLEEP_S = 1.0  # longer than any read_timeout used in tests

TICKETS = {
    "TCK-L": {
        "ticket_id": "TCK-L",
        "account_id": "acct_legacy",
        "subject": "Reset our API token",
        "body": "How do we rotate the token? Contact ops@example.test, ref 1234567890.",
        "channel": "email",
        "opened_at": "2026-09-01T00:00:00Z",
        "status": "open",
        "priority_hint": "unknown-field",
    },
    "TCK-U": {
        "ticket_id": "TCK-U",
        "account_id": "acct_unified",
        "subject": "Rotate scoped API key",
        "body": "We need to rotate a key.",
        "channel": "web",
        "opened_at": "2026-09-01T00:00:00Z",
        "status": "open",
    },
    "TCK-D": {
        "ticket_id": "TCK-D",
        "account_id": "acct_degraded",
        "subject": "How many seats do we have?",
        "body": "Please confirm our seat count.",
        "channel": "email",
        "opened_at": "2026-09-01T00:00:00Z",
        "status": "open",
    },
}

ACCOUNTS = {
    "acct_legacy": {
        "account_id": "acct_legacy",
        "name": "Legacy Co",
        "plan_tier": "business",
        "region": "us-west",
        "auth_model": "legacy_auth",
        "customer_since": "2020-01-01",
        "primary_contact": {"name": "Ops", "email": "ops@legacy.example"},
    },
    "acct_unified": {
        "account_id": "acct_unified",
        "name": "Unified Co",
        "plan_tier": "startup",
        "region": "eu-west",
        "auth_model": "unified_auth",
        "customer_since": "2024-01-01",
        "primary_contact": {"name": "Ops", "email": "ops@unified.example"},
    },
    "acct_degraded": {
        "account_id": "acct_degraded",
        "name": "Degraded Co",
        "plan_tier": "enterprise",
        "region": "us-east",
        "auth_model": "unified_auth",
        "customer_since": "2021-01-01",
        "primary_contact": {"name": "Ops", "email": "ops@degraded.example"},
    },
}

ENTITLEMENTS = {
    "acct_legacy": {
        "account_id": "acct_legacy",
        "support_tier": "standard",
        "seats": 10,
        "features": ["sso"],
        "sla_hours": 24,
        "rate_limit_rpm": 600,
        "updated_at": "2026-08-01T00:00:00Z",
    },
    "acct_unified": {
        "account_id": "acct_unified",
        "support_tier": "premium",
        "seats": 5,
        "features": ["audit_log"],
        "sla_hours": 4,
        "rate_limit_rpm": 1200,
        "updated_at": "2026-08-01T00:00:00Z",
        "shadow_field": True,
    },
}

KB = [
    {"id": "kb-0001", "title": "General article", "body": "Applies to everyone.", "tags": ["general"], "applies_to": {}},
    {"id": "kb-0002", "title": "Legacy token reset", "body": "Legacy only.", "tags": ["legacy"], "applies_to": {"auth_model": ["legacy_auth"]}},
    {"id": "kb-0003", "title": "Scoped keys", "body": "Unified only.", "tags": ["unified"], "applies_to": {"auth_model": ["unified_auth"]}},
]


class FakeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = {}  # path -> count
        self.posts = []  # (path, payload)
        self.escalations = []
        self.created_tickets = {}  # runtime intake (POST /v1/tickets); read back via GET /v1/tickets/{id}
        self.intake_enabled = True  # False -> the fake behaves like an upstream with no accounts/intake routes

    def hit(self, path):
        with self.lock:
            self.calls[path] = self.calls.get(path, 0) + 1
            return self.calls[path]


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client gave up (timeout test); nothing to do

        def _error(self, status, code, message, **extra):
            env = {"code": code, "message": message}
            env.update(extra)
            self._send(status, {"error": env})

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            n = state.hit(path)

            if path == "/ok":
                return self._send(200, {"ok": True, "attempt": n})
            if path == "/missing":
                return self._error(404, "thing_not_found", "No thing")
            if path == "/degraded":
                return self._error(500, "entitlement_service_error", "cannot serve", detail="seats is a string")
            if path == "/flaky":
                if n == 1:
                    return self._error(500, "internal_error", "transient")
                return self._send(200, {"ok": True, "attempt": n})
            if path == "/slow":
                time.sleep(SLOW_SLEEP_S)
                return self._send(200, {"ok": True})

            if path == "/v1/tickets":
                if not state.intake_enabled:
                    return self._error(404, "not_found", "No route")
                with state.lock:
                    return self._send(200, {"ticket_ids": sorted(list(TICKETS) + list(state.created_tickets))})
            if path == "/v1/accounts":
                if not state.intake_enabled:
                    return self._error(404, "not_found", "No route")
                return self._send(200, {"accounts": [ACCOUNTS[k] for k in sorted(ACCOUNTS)]})
            m = re.fullmatch(r"/v1/tickets/([A-Za-z0-9_\-]+)", path)
            if m:
                with state.lock:
                    t = TICKETS.get(m.group(1)) or state.created_tickets.get(m.group(1))
                if t is None:
                    return self._error(404, "ticket_not_found", "No ticket")
                return self._send(200, t)
            m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)/entitlements", path)
            if m:
                aid = m.group(1)
                if aid not in ACCOUNTS:
                    return self._error(404, "account_not_found", "No account")
                if aid == "acct_degraded":
                    return self._error(
                        500, "entitlement_service_error", "cannot serve",
                        detail="field 'seats' expected int, got str ('unlimited')",
                    )
                return self._send(200, ENTITLEMENTS[aid])
            m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)", path)
            if m:
                a = ACCOUNTS.get(m.group(1))
                if a is None:
                    return self._error(404, "account_not_found", "No account")
                return self._send(200, a)
            if path == "/v1/kb/search":
                q = (query.get("q") or [""])[0]
                if not q.strip():
                    return self._error(400, "missing_query", "q required")
                limit = int((query.get("limit") or ["5"])[0])
                results = []
                for i, a in enumerate(KB[:limit]):
                    item = dict(a)
                    item["score"] = float(10 - i)
                    item["debug_rank"] = i  # unknown field on purpose
                    results.append(item)
                return self._send(200, {"query": q, "results": results})
            if path == "/v1/escalations":
                with state.lock:
                    return self._send(200, {"escalations": list(state.escalations)})
            return self._error(404, "not_found", "No route")

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            state.hit("POST " + path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                return self._error(400, "invalid_json", "bad json")
            if not isinstance(payload, dict):
                return self._error(400, "invalid_body", "not an object")
            with state.lock:
                state.posts.append((path, payload))
            if path == "/write-garbage":  # 201 with a body that is not JSON (S8)
                raw_body = b"<html>created</html>"
                self.send_response(201)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(raw_body)))
                self.end_headers()
                self.wfile.write(raw_body)
                return None
            if path == "/write-list":  # 201 with a JSON array instead of an object (S8)
                return self._send(201, ["filed"])
            if path == "/v1/tickets":
                if not state.intake_enabled:
                    return self._error(404, "not_found", "No route")
                missing = [f for f in ("account_id", "subject", "body") if not payload.get(f)]
                if missing:
                    return self._error(422, "missing_fields", "missing", fields=missing)
                if payload["account_id"] not in ACCOUNTS:
                    return self._error(422, "unknown_account", "no such account")
                with state.lock:
                    tid = payload.get("ticket_id") or "TCK-N%d" % (len(state.created_tickets) + 1)
                    if tid in TICKETS or tid in state.created_tickets:
                        return self._error(409, "ticket_exists", "taken")
                    rec = {"ticket_id": tid, "account_id": payload["account_id"], "subject": payload["subject"],
                           "body": payload["body"], "channel": payload.get("channel") or "web",
                           "opened_at": "2026-09-26T00:00:00Z", "status": "open"}
                    state.created_tickets[tid] = rec
                return self._send(201, rec)
            if path not in ("/write", "/v1/escalations"):
                return self._error(404, "not_found", "No route")
            missing = [f for f in ("ticket_id", "reason", "summary") if not payload.get(f)]
            if path == "/v1/escalations" and missing:
                return self._error(422, "missing_fields", "missing", fields=missing)
            if path == "/v1/escalations" and payload["ticket_id"] not in TICKETS and payload["ticket_id"] not in state.created_tickets:
                return self._error(422, "unknown_ticket", "no such ticket")
            if payload.get("confirm") is not True:
                return self._error(409, "confirmation_required", "re-send with confirm true")
            with state.lock:
                entry = dict(payload)
                entry["escalation_id"] = "ESC-%04d" % (len(state.escalations) + 1)
                entry["status"] = "filed"
                if path == "/v1/escalations":
                    state.escalations.append(entry)
            return self._send(201, entry)

    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # client disconnects are expected in the timeout tests


@pytest.fixture
def fake_upstream():
    """Yields (base_url, FakeState). Port 0 lets the OS pick a free port."""
    state = FakeState()
    server = _QuietServer(("127.0.0.1", 0), _make_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % port, state
    finally:
        server.shutdown()
        server.server_close()
