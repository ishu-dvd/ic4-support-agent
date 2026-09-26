"""HTTP client for the systems-of-record API. Stdlib urllib only.

Reads are idempotent, so `get` may retry. Writes are not: a POST that timed out may have
been committed upstream, so `post` never retries and leaves the "did it land?" question to
the caller (Tools.escalate checks GET /v1/escalations before writing for exactly this reason).
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .trace import NoopTracer


@dataclass(frozen=True)
class RetryPolicy:
    """Status-based retry decisions for idempotent calls. Default: 3 retries (4 attempts).

    Retry:  timeouts, connection errors, 429, and 5xx (they are transient by definition).
    Never:  404 (the thing does not exist), other 4xx (our request is wrong; a retry cannot fix it),
            `entitlement_service_error` (the stored record is malformed; every attempt returns the same 500),
            and any attempt whose backoff + timeout would run past the run deadline.
    Backoff: exponential from `backoff_base_s`, capped at `backoff_max_s`, plus uniform jitter.
    """

    max_retries: int = 3
    retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504)
    never_retry_codes: Tuple[str, ...] = ("entitlement_service_error",)
    backoff_base_s: float = 0.2
    backoff_max_s: float = 2.0
    jitter_s: float = 0.2

    def backoff_s(self, attempt: int) -> float:
        return min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt)) + random.uniform(0, self.jitter_s)


def decide_retry(
    policy: RetryPolicy,
    attempt: int,
    max_retries: int,
    status: Optional[int] = None,
    code: Optional[str] = None,
    exc_kind: Optional[str] = None,
    remaining_ms: Optional[int] = None,
    next_cost_ms: float = 0.0,
) -> Tuple[bool, str]:
    """(retry?, reason). `attempt` is the 0-based attempt that just failed. Reasons are stable strings
    (`retry:status_503`, `stop:max_retries`, ...) so the dashboard can group on them."""
    if code and code in policy.never_retry_codes:
        return False, "stop:code_%s" % code
    if status is not None and not (status in policy.retry_statuses or status >= 500):
        return False, "stop:status_%d" % status
    if attempt >= max_retries:
        return False, "stop:max_retries"
    if remaining_ms is not None and next_cost_ms >= remaining_ms:
        return False, "stop:deadline"
    if status is not None:
        return True, "retry:status_%d" % status
    return True, "retry:%s" % (exc_kind or "error")


class UpstreamError(Exception):
    def __init__(self, status: Optional[int], code: str, detail: str = ""):
        super().__init__("%s %s: %s" % (status, code, detail))
        self.status = status
        self.code = code
        self.detail = detail


class NotFound(UpstreamError):
    """404 on a read."""


class UpstreamDegraded(UpstreamError):
    """The upstream exists but cannot answer: entitlement_service_error, or timeout after retries."""


class WriteRejected(UpstreamError):
    """4xx on the write (400 / 409 / 422): nothing was committed."""


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.timeout, TimeoutError))


def _parse_json(raw: bytes) -> Any:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _error_parts(err: urllib.error.HTTPError) -> Tuple[str, str]:
    """Pull (code, detail) out of the {"error": {...}} envelope; tolerate non-JSON bodies."""
    try:
        body = _parse_json(err.read())
    except (ValueError, UnicodeDecodeError):
        body = {}
    env = body.get("error", {}) if isinstance(body, dict) else {}
    code = str(env.get("code") or "http_%d" % err.code)
    detail = str(env.get("detail") or env.get("message") or "")
    return code, detail


class HttpUpstream:
    def __init__(
        self,
        base_url: str,
        read_timeout_s: float,
        write_timeout_s: float,
        tracer: Any = None,
        retry_policy: Optional[RetryPolicy] = None,
        remaining_ms: Optional[Callable[[], int]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.read_timeout_s = float(read_timeout_s)
        self.write_timeout_s = float(write_timeout_s)
        self.tracer = tracer if tracer is not None else NoopTracer()
        self.retry_policy = retry_policy or RetryPolicy()
        # Optional view of the run budget: a retry is skipped when it cannot finish before the deadline.
        self.remaining_ms = remaining_ms

    # ---- reads ----
    def get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: Optional[int] = None) -> dict:
        """GET with status-based retry (see RetryPolicy). `retries=None` -> policy.max_retries (default 3).

        `entitlement_service_error` is deliberately *not* retried: the server returns it when
        the stored record itself is malformed (e.g. seats="unlimited"), so a second attempt
        would return the same 500 and only burn the deadline. We surface it at once as
        UpstreamDegraded and let the agent continue without entitlements.
        Query params are not put on the span: the kb query is derived from the ticket body.
        Every attempt is its own `call` span (attribute `attempt`); each decision to retry is a
        `retry` event and the final give-up a `retries_exhausted` event, so the dashboard can count
        retries and group by reason without parsing anything.
        """
        policy = self.retry_policy
        max_retries = policy.max_retries if retries is None else int(retries)
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        last_error: Optional[UpstreamError] = None
        attempt = 0
        while True:
            status: Optional[int] = None
            code: Optional[str] = None
            exc_kind: Optional[str] = None
            with self.tracer.call(name="http GET %s" % path, attempt=attempt) as span:
                span["retries"] = attempt
                req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=self.read_timeout_s) as resp:
                        status = resp.status
                        raw = resp.read()
                    span["http.status"] = status
                    try:
                        body = _parse_json(raw)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise UpstreamError(status, "invalid_json", str(exc))
                    if not isinstance(body, dict):
                        raise UpstreamError(status, "invalid_body", "expected a JSON object")
                    return body
                except urllib.error.HTTPError as err:
                    status = err.code
                    span["http.status"] = err.code
                    code, detail = _error_parts(err)
                    if err.code == 404:
                        raise NotFound(404, code, detail)
                    if err.code >= 500 or err.code == 429:
                        last_error = UpstreamDegraded(err.code, code, detail)
                    else:
                        raise UpstreamError(err.code, code, detail)
                except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                    span["http.status"] = None
                    if _is_timeout(exc):
                        exc_kind = "timeout"
                        last_error = UpstreamDegraded(None, "timeout", "read timeout after %.1fs" % self.read_timeout_s)
                    else:
                        exc_kind = "unreachable"
                        last_error = UpstreamDegraded(None, "unreachable", str(getattr(exc, "reason", exc)))
                backoff = policy.backoff_s(attempt)
                remaining = self.remaining_ms() if self.remaining_ms is not None else None
                do_retry, reason = decide_retry(
                    policy, attempt, max_retries, status=status, code=code, exc_kind=exc_kind,
                    remaining_ms=remaining, next_cost_ms=(backoff + self.read_timeout_s) * 1000.0,
                )
                span["retry_decision"] = reason
            assert last_error is not None
            if not do_retry:
                if attempt > 0 or reason == "stop:deadline":
                    self.tracer.event("retries_exhausted", path=path, attempts=attempt + 1, reason=reason, code=last_error.code)
                raise last_error
            self.tracer.event("retry", path=path, attempt=attempt + 1, reason=reason, backoff_ms=int(backoff * 1000),
                              code=last_error.code)
            time.sleep(backoff)
            attempt += 1

    # ---- writes ----
    def post(self, path: str, payload: dict) -> Tuple[int, dict]:
        """POST once. Never retried: a timed-out write may already have been committed.

        A 2xx whose body cannot be parsed as a JSON object is raised as UpstreamDegraded
        (`invalid_response`), not UpstreamError: the write may well have landed, and the caller
        must treat it like a timeout (hand to a human, never re-POST)."""
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        with self.tracer.call(name="http POST %s" % path) as span:
            span["retries"] = 0
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.write_timeout_s) as resp:
                    status = resp.status
                    raw = resp.read()
                span["http.status"] = status
                try:
                    body = _parse_json(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise UpstreamDegraded(status, "invalid_response", "unparseable body: %s" % exc)
                if not isinstance(body, dict):
                    raise UpstreamDegraded(status, "invalid_response", "expected a JSON object")
                return status, body
            except urllib.error.HTTPError as err:
                span["http.status"] = err.code
                code, detail = _error_parts(err)
                if 400 <= err.code < 500:
                    raise WriteRejected(err.code, code, detail)
                raise UpstreamDegraded(err.code, code, detail)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                span["http.status"] = None
                if _is_timeout(exc):
                    raise UpstreamDegraded(None, "timeout", "write timeout after %.1fs" % self.write_timeout_s)
                raise UpstreamDegraded(None, "unreachable", str(getattr(exc, "reason", exc)))
