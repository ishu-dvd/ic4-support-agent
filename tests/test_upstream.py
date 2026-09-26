import time
from contextlib import contextmanager

import pytest

from agent.upstream import HttpUpstream, NotFound, UpstreamDegraded, UpstreamError, WriteRejected

READ_TIMEOUT = 0.2


class RecordingTracer:
    def __init__(self):
        self.calls = []
        self.events = []

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


def _client(base_url, tracer=None):
    return HttpUpstream(base_url, read_timeout_s=READ_TIMEOUT, write_timeout_s=READ_TIMEOUT, tracer=tracer)


def test_get_ok_returns_json_and_records_span(fake_upstream):
    base, state = fake_upstream
    tracer = RecordingTracer()
    body = _client(base, tracer).get("/ok")
    assert body["ok"] is True
    assert state.calls["/ok"] == 1
    name, span = tracer.calls[-1]
    assert name == "http GET /ok"
    assert span["http.status"] == 200 and span["retries"] == 0


def test_get_with_params_encodes_query(fake_upstream):
    base, state = fake_upstream
    body = _client(base).get("/v1/kb/search", params={"q": "reset token & more", "limit": 2})
    assert body["query"] == "reset token & more"
    assert len(body["results"]) == 2


def test_404_raises_not_found_with_code_from_body(fake_upstream):
    base, state = fake_upstream
    with pytest.raises(NotFound) as excinfo:
        _client(base).get("/missing")
    assert excinfo.value.status == 404
    assert excinfo.value.code == "thing_not_found"
    assert state.calls["/missing"] == 1  # never retried


def test_entitlement_service_error_is_not_retried(fake_upstream):
    base, state = fake_upstream
    with pytest.raises(UpstreamDegraded) as excinfo:
        _client(base).get("/degraded", retries=3)
    assert excinfo.value.status == 500
    assert excinfo.value.code == "entitlement_service_error"
    assert "seats" in excinfo.value.detail
    assert state.calls["/degraded"] == 1


def test_generic_5xx_is_retried_once_and_succeeds(fake_upstream):
    base, state = fake_upstream
    tracer = RecordingTracer()
    body = _client(base, tracer).get("/flaky", retries=1)
    assert body["ok"] is True and body["attempt"] == 2
    assert state.calls["/flaky"] == 2
    statuses = [span["http.status"] for name, span in tracer.calls if name == "http GET /flaky"]
    assert statuses == [500, 200]
    assert tracer.calls[-1][1]["retries"] == 1


def test_timeout_after_retries_raises_degraded_timeout(fake_upstream):
    base, state = fake_upstream
    retries = 1
    started = time.monotonic()
    with pytest.raises(UpstreamDegraded) as excinfo:
        _client(base).get("/slow", retries=retries)
    elapsed = time.monotonic() - started
    assert excinfo.value.code == "timeout"
    assert excinfo.value.status is None
    # (retries + 1) timeouts plus one jittered back-off of 0.2-0.4s; never as long as the server sleep.
    assert (retries + 1) * READ_TIMEOUT <= elapsed < (retries + 1) * READ_TIMEOUT + 0.4 * retries + 0.5


def test_zero_retries_times_out_once(fake_upstream):
    base, state = fake_upstream
    started = time.monotonic()
    with pytest.raises(UpstreamDegraded):
        _client(base).get("/slow", retries=0)
    elapsed = time.monotonic() - started
    assert READ_TIMEOUT <= elapsed < READ_TIMEOUT + 0.3


def test_unreachable_host_is_degraded_not_crash():
    client = HttpUpstream("http://127.0.0.1:9", read_timeout_s=0.2, write_timeout_s=0.2)
    with pytest.raises(UpstreamDegraded) as excinfo:
        client.get("/ok", retries=0)
    assert excinfo.value.code in ("unreachable", "timeout")


def test_post_without_confirm_is_rejected_409(fake_upstream):
    base, state = fake_upstream
    with pytest.raises(WriteRejected) as excinfo:
        _client(base).post("/write", {"ticket_id": "TCK-L", "reason": "r", "summary": "s"})
    assert excinfo.value.status == 409
    assert excinfo.value.code == "confirmation_required"
    assert state.calls["POST /write"] == 1  # never retried


def test_post_with_confirm_returns_201(fake_upstream):
    base, state = fake_upstream
    tracer = RecordingTracer()
    status, body = _client(base, tracer).post(
        "/write", {"ticket_id": "TCK-L", "reason": "r", "summary": "s", "confirm": True}
    )
    assert status == 201
    assert body["status"] == "filed" and body["escalation_id"].startswith("ESC-")
    assert state.posts == [("/write", {"ticket_id": "TCK-L", "reason": "r", "summary": "s", "confirm": True})]
    assert tracer.calls[-1][0] == "http POST /write"
    assert tracer.calls[-1][1]["http.status"] == 201


@pytest.mark.parametrize("path", ["/write-garbage", "/write-list"])
def test_post_201_with_unparseable_body_is_degraded_not_retried(fake_upstream, path):
    """S8: the write may have landed; surface it as UpstreamDegraded so the caller hands it to a human."""
    base, state = fake_upstream
    with pytest.raises(UpstreamDegraded) as excinfo:
        _client(base).post(path, {"ticket_id": "TCK-L", "confirm": True})
    assert excinfo.value.code == "invalid_response" and excinfo.value.status == 201
    assert state.calls["POST " + path] == 1


def test_error_hierarchy():
    assert issubclass(NotFound, UpstreamError)
    assert issubclass(UpstreamDegraded, UpstreamError)
    assert issubclass(WriteRejected, UpstreamError)
