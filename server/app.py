"""Mock systems-of-record API.

Stdlib only: no dependencies to install. Start with

    python3 -m server

Endpoints are documented in docs/API.md.
"""
import json
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .search import search
from .store import RecordExists, RecordNotFound, Store, UpstreamError

LATENCY_MS = 0  # set via --latency-ms to simulate a slower upstream


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-sor/1.0"
    protocol_version = "HTTP/1.1"

    store = None  # injected by serve()

    # ---- plumbing ----
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _send(self, status, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message, **extra):
        payload = {"error": {"code": code, "message": message}}
        payload["error"].update(extra)
        self._send(status, payload)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(str(exc))

    # ---- routing ----
    def do_GET(self):
        if LATENCY_MS:
            time.sleep(LATENCY_MS / 1000.0)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/healthz":
            return self._send(200, {"status": "ok", "variant": self.store.variant})

        if path == "/v1/tickets":
            return self._send(200, {"ticket_ids": self.store.ticket_ids()})

        m = re.fullmatch(r"/v1/tickets/([A-Za-z0-9_\-]+)", path)
        if m:
            try:
                return self._send(200, self.store.ticket(m.group(1)))
            except RecordNotFound:
                return self._error(404, "ticket_not_found", "No ticket with id %r" % m.group(1))

        m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)/entitlements", path)
        if m:
            account_id = m.group(1)
            try:
                return self._send(200, self.store.entitlements_for(account_id))
            except RecordNotFound:
                return self._error(404, "account_not_found", "No account with id %r" % account_id)
            except UpstreamError as exc:
                return self._error(
                    500,
                    "entitlement_service_error",
                    "The entitlement service could not produce a response for this account.",
                    detail=str(exc),
                )

        m = re.fullmatch(r"/v1/accounts/([A-Za-z0-9_\-]+)", path)
        if m:
            try:
                return self._send(200, self.store.account(m.group(1)))
            except RecordNotFound:
                return self._error(404, "account_not_found", "No account with id %r" % m.group(1))

        if path == "/v1/accounts":
            return self._send(200, {"accounts": self.store.account_list()})

        if path == "/v1/kb/search":
            q = (query.get("q") or [""])[0]
            if not q.strip():
                return self._error(400, "missing_query", "Query parameter 'q' is required.")
            try:
                limit = int((query.get("limit") or ["5"])[0])
            except ValueError:
                return self._error(400, "invalid_limit", "Query parameter 'limit' must be an integer.")
            limit = max(1, min(limit, 25))
            return self._send(200, {"query": q, "results": search(q, self.store.kb, limit)})

        if path == "/v1/escalations":
            return self._send(200, {"escalations": self.store.escalations})

        return self._error(404, "not_found", "No route for GET %s" % path)

    def do_POST(self):
        if LATENCY_MS:
            time.sleep(LATENCY_MS / 1000.0)
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/v1/escalations", "/v1/tickets"):
            return self._error(404, "not_found", "No route for POST %s" % path)

        try:
            payload = self._read_json()
        except ValueError as exc:
            return self._error(400, "invalid_json", "Request body is not valid JSON.", detail=str(exc))
        if not isinstance(payload, dict):
            return self._error(400, "invalid_body", "Request body must be a JSON object.")

        if path == "/v1/tickets":
            return self._create_ticket(payload)

        missing = [f for f in ("ticket_id", "reason", "summary") if not payload.get(f)]
        if missing:
            return self._error(
                422, "missing_fields", "Missing required field(s): %s" % ", ".join(missing), fields=missing
            )

        ticket_id = payload["ticket_id"]
        try:
            self.store.ticket(ticket_id)
        except RecordNotFound:
            return self._error(422, "unknown_ticket", "No ticket with id %r" % ticket_id)

        if payload.get("confirm") is not True:
            return self._error(
                409,
                "confirmation_required",
                "Filing an escalation is a write. Re-send with \"confirm\": true to commit it.",
            )

        return self._send(201, self.store.file_escalation(payload))

    def _create_ticket(self, payload):
        """Ticket intake. Unlike the escalation write this is a test-harness convenience, so no
        `confirm` handshake: the record only lives in memory and vanishes on restart."""
        missing = [f for f in ("account_id", "subject", "body") if not payload.get(f)]
        if missing:
            return self._error(
                422, "missing_fields", "Missing required field(s): %s" % ", ".join(missing), fields=missing
            )
        try:
            return self._send(201, self.store.create_ticket(payload))
        except RecordNotFound:
            return self._error(422, "unknown_account", "No account with id %r" % payload["account_id"])
        except ValueError as exc:
            return self._error(422, "invalid_ticket_id", str(exc))
        except RecordExists as exc:
            return self._error(409, "ticket_exists", "A ticket with id %r already exists" % str(exc))


def serve(variant="support", host="127.0.0.1", port=8080, latency_ms=0):
    global LATENCY_MS
    LATENCY_MS = latency_ms
    Handler.store = Store(variant)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        "mock systems-of-record API | variant=%s | %d tickets | http://%s:%d"
        % (variant, len(Handler.store.tickets), host, port),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        httpd.server_close()
