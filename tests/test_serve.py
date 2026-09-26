import os
import sys

import pytest

pytest.importorskip("fastapi")

_IC4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IC4 not in sys.path:
    sys.path.insert(0, _IC4)

from fastapi.testclient import TestClient  # noqa: E402

import agent.serve as serve  # noqa: E402
from agent.llm import LLMUsage  # noqa: E402
from agent.loop import RunResult  # noqa: E402
from agent.trace import RunTracer  # noqa: E402


@pytest.fixture
def client():
    return TestClient(serve.app)


def test_healthz_never_leaks_key(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["key_set"] is True
    assert "sk-secret" not in r.text
    assert "X-Request-ID" in r.headers
    r2 = client.get("/api/config")
    assert r2.json() == body


def _fake_result(ticket_id, request_id, correlation_id, outcome="completed", error=None):
    return RunResult(
        run_id="20260101T000000-abc123",
        ticket_id=ticket_id,
        category="billing_proration",
        request_type="question",
        confidence=0.9,
        diagnosis="d",
        reply="r",
        kb_cited=["KB-1"],
        escalate=False,
        escalate_recommended_by_model=False,
        priority="normal",
        policy_reasons=[],
        write=None,
        injection_suspected=False,
        entitlements_degraded=False,
        draft_source="rules",
        guard_violations=[],
        usage=LLMUsage(input_tokens=10, output_tokens=5, model_id="rules", cost_usd=0.0),
        duration_ms=12,
        outcome=outcome,
        error=error,
        request_id=request_id or "",
        correlation_id=correlation_id or request_id or "",
    )


def test_post_run_propagates_ids(monkeypatch, client):
    seen = {}

    def fake_run(ticket_id, cfg, upstream=None, llm=None, tracer=None, request_id=None, correlation_id=None):
        seen["cfg"] = cfg
        return _fake_result(ticket_id, request_id, correlation_id)

    monkeypatch.setattr(serve, "run", fake_run)
    r = client.post("/run", json={"ticket_id": "TCK-1101", "correlation_id": "corr-body"},
                    headers={"X-Request-ID": "req-test-1", "X-Correlation-ID": "corr-header"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticket_id"] == "TCK-1101"
    assert body["outcome"] == "completed"
    assert body["usage"]["model_id"] == "rules"
    assert r.headers["X-Request-ID"] == "req-test-1"
    assert r.headers["X-Correlation-ID"] == "corr-body"
    assert seen["cfg"].dry_run is True


def test_post_run_failed_outcome_still_200(monkeypatch, client):
    monkeypatch.setattr(serve, "run", lambda t, cfg, **kw: _fake_result(t, kw.get("request_id"), kw.get("correlation_id"),
                                                                        outcome="failed", error="ticket_not_found"))
    r = client.post("/run", json={"ticket_id": "TCK-404"})
    assert r.status_code == 200
    assert r.json()["outcome"] == "failed"
    assert r.json()["error"] == "ticket_not_found"


def test_post_run_bad_ticket_id(client):
    r = client.post("/run", json={"ticket_id": "../etc"})
    assert r.status_code == 422


def _populate(runs_dir):
    t1 = RunTracer(str(runs_dir), ticket_id="TCK-1")
    with t1.step("draft"):
        with t1.call("llm draft"):
            pass
    t1.finish({"outcome": "completed", "category": "billing_proration", "escalate": False,
               "usage": {"model_id": "openai-gpt-4o-mini", "cost_usd": 0.001, "input_tokens": 10,
                         "output_tokens": 5, "ttft_ms": 120},
               "duration_ms": 50, "draft_source": "llm"})
    t2 = RunTracer(str(runs_dir), ticket_id="TCK-2")
    with t2.step("read_ticket"):
        pass
    t2.finish({"outcome": "failed", "error": "ticket_not_found", "category": None, "escalate": False,
               "usage": {"model_id": "rules", "cost_usd": 0.0}, "duration_ms": 5, "draft_source": "none"})
    return t1.run_id, t2.run_id


def test_read_api_over_tmp_runs_dir(monkeypatch, tmp_path, client):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    run_ok, run_failed = _populate(tmp_path)

    r = client.get("/api/runs")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["runs"]) == 2
    assert set(body["facets"]["outcomes"]) == {"completed", "failed"}

    r = client.get("/api/runs", params={"outcome": "failed"})
    assert r.json()["total"] == 1
    assert r.json()["runs"][0]["run_id"] == run_failed

    r = client.get("/api/runs", params={"limit": 1, "offset": 1, "sort": "ticket_id", "order": "asc"})
    assert r.json()["total"] == 2
    assert len(r.json()["runs"]) == 1
    assert r.json()["runs"][0]["ticket_id"] == "TCK-2"

    r = client.get("/api/metrics")
    assert r.status_code == 200
    agg = r.json()
    assert agg["n"] == 2
    assert agg["outcomes"] == {"completed": 1, "failed": 1}
    assert agg["cost"]["total"]["usd"] == pytest.approx(0.001)
    assert "facets" in agg

    r = client.get("/api/metrics", params={"outcome": "completed"})
    assert r.json()["n"] == 1

    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "ic4_runs_total" in r.text
    assert "ic4_cost_usd_total" in r.text

    r = client.get("/api/runs/%s" % run_ok)
    assert r.status_code == 200
    detail = r.json()
    assert detail["summary"]["run_id"] == run_ok
    assert isinstance(detail["trace"], list) and detail["trace"]
    assert detail["result"]["outcome"] == "completed"

    r = client.get("/runs/%s" % run_ok)
    assert r.status_code == 200
    assert r.json()["run_id"] == run_ok
    assert r.json()["cost_usd"] == pytest.approx(0.001)

    assert client.get("/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/does-not-exist").status_code == 404

    r = client.get("/api/evals")
    assert r.status_code == 200
    assert r.json() == {"evals": []}
    assert client.get("/api/evals/eval-nope").status_code == 404


# ---- intake pass-through: the dashboard's accounts/tickets come from the upstream, never from files ----

def test_intake_routes_proxy_the_upstream(monkeypatch, fake_upstream, client):
    base, state = fake_upstream
    monkeypatch.setenv("UPSTREAM_BASE_URL", base)

    r = client.get("/api/accounts")
    assert r.status_code == 200
    ids = [a["account_id"] for a in r.json()["accounts"]]
    assert ids == ["acct_degraded", "acct_legacy", "acct_unified"]
    assert state.calls["/v1/accounts"] == 1

    r = client.get("/api/tickets")
    assert r.status_code == 200 and set(r.json()["ticket_ids"]) == {"TCK-D", "TCK-L", "TCK-U"}

    r = client.post("/api/tickets", json={"account_id": "acct_legacy", "subject": "s", "body": "b", "channel": "email"})
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["account_id"] == "acct_legacy" and created["channel"] == "email"
    assert state.posts[-1] == ("/v1/tickets", {"account_id": "acct_legacy", "subject": "s", "body": "b", "channel": "email"})
    # written upstream, readable back through the ordinary listing
    assert created["ticket_id"] in client.get("/api/tickets").json()["ticket_ids"]
    # nothing was cached in the agent layer: no runs/ artefact, no local state
    assert state.calls["POST /v1/tickets"] == 1


def test_intake_forwards_upstream_rejections_verbatim(monkeypatch, fake_upstream, client):
    base, state = fake_upstream
    monkeypatch.setenv("UPSTREAM_BASE_URL", base)

    r = client.post("/api/tickets", json={"account_id": "acct_nope", "subject": "s", "body": "b"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_account"

    r = client.post("/api/tickets", json={"account_id": "acct_legacy", "subject": "s", "body": "b", "ticket_id": "TCK-L"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "ticket_exists"

    # shape validation is ours (422 before anything reaches the upstream)
    posts_before = len(state.posts)
    assert client.post("/api/tickets", json={"account_id": "../x", "subject": "s", "body": "b"}).status_code == 422
    assert client.post("/api/tickets", json={"account_id": "acct_legacy", "subject": "", "body": "b"}).status_code == 422
    assert len(state.posts) == posts_before


def test_intake_reports_unsupported_upstream_as_501(monkeypatch, fake_upstream, client):
    """A real systems-of-record API may have no accounts listing or ticket intake. That must surface as
    an explicit 501, not as an empty dropdown or a misleading 'not found'."""
    base, state = fake_upstream
    state.intake_enabled = False
    monkeypatch.setenv("UPSTREAM_BASE_URL", base)

    for call in (lambda: client.get("/api/accounts"), lambda: client.get("/api/tickets"),
                 lambda: client.post("/api/tickets", json={"account_id": "acct_legacy", "subject": "s", "body": "b"})):
        r = call()
        assert r.status_code == 501, r.text
        assert r.json()["detail"]["code"] == "upstream_unsupported"


def test_intake_unreachable_upstream_is_503(monkeypatch, client):
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.setenv("MAX_RETRIES", "0")
    r = client.get("/api/accounts")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "unreachable"


def test_root_redirects_and_dashboard_served(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/dashboard"
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
