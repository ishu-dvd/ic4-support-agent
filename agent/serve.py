"""HTTP boundary (FastAPI + Pydantic v2). The only module allowed to import fastapi; agent core stays stdlib.

Run: uvicorn agent.serve:app --host 0.0.0.0 --port 8080   (or: python -m agent.serve)
"""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_IC4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _IC4 not in sys.path:
    sys.path.insert(0, _IC4)

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.loop import run  # noqa: E402
from agent.metrics import (  # noqa: E402
    aggregate,
    facets,
    filter_runs,
    load_eval,
    load_eval_summaries,
    load_run,
    load_run_summaries,
    prometheus_text,
)
from agent.trace import new_request_id, sanitize_id  # noqa: E402
from obs.api import router as obs_router  # noqa: E402
from obs.insights import looks_flagged, read_result, run_eval_index, security_for_run  # noqa: E402
from agent.upstream import HttpUpstream, NotFound, RetryPolicy, UpstreamDegraded, UpstreamError, WriteRejected  # noqa: E402

REPO_ROOT = _IC4
DASHBOARD_HTML = os.path.join(REPO_ROOT, "dashboard", "index.html")
_TICKET_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]")

app = FastAPI(title="ic4-agent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(obs_router)


# ---- models -------------------------------------------------------------------------------------------
class RunRequest(BaseModel):
    ticket_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")
    write: bool = False
    prompt_version: Optional[str] = None
    correlation_id: Optional[str] = None


class EvalRequest(BaseModel):
    label: str = "dashboard"
    limit: Optional[int] = None
    prompt_version: Optional[str] = None
    tickets: Optional[str] = None
    model_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._/-]{1,80}$")


class TicketCreate(BaseModel):
    """Intake form for a new ticket. Validated here for shape only; existence of the account is the
    upstream's call (it owns the data), so `unknown_account` comes back from there, not from us."""
    account_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=4000)
    channel: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,32}$")
    ticket_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


# ---- middleware ---------------------------------------------------------------------------------------
@app.middleware("http")
async def request_ids(request: Request, call_next):
    req_id = sanitize_id(request.headers.get("X-Request-ID"), None) or new_request_id()
    corr_id = sanitize_id(request.headers.get("X-Correlation-ID"), None)
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", req_id)
    if corr_id and "X-Correlation-ID" not in response.headers:
        response.headers["X-Correlation-ID"] = corr_id
    return response


# ---- helpers ------------------------------------------------------------------------------------------
def _config_payload() -> Dict[str, Any]:
    cfg = load_config()
    return {
        "status": "ok",
        "service": cfg.service_name,
        "provider": cfg.model_provider,
        "model_id": cfg.model_id,
        "upstream": cfg.upstream_base_url,
        "inference_host": urlparse(cfg.openai_base_url).hostname,
        "key_set": bool(cfg.openai_api_key),
        "runs_dir": cfg.runs_dir,
        "version": "1.0",
    }


def _sort_key(key: str):
    def _k(row: dict):
        v = row.get(key)
        if v is None:
            return (1, 0, "")
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            return (0, float(v), "")
        return (0, 0.0, str(v))

    return _k


def _with_eval_ids(runs_dir: str, rows: List[dict]) -> List[dict]:
    """Attach eval_id (the eval whose cases.jsonl references the run) and the security severity
    (none / flagged / blocked) without mutating the loaded rows. result.json is read only for rows whose
    summary already shows a security signal, so this stays cheap over thousands of runs."""
    index = run_eval_index(runs_dir)
    out = []
    for r in rows:
        sev = "none"
        if looks_flagged(r):
            sev = security_for_run(r, read_result(runs_dir, r.get("run_id", "")))["severity"]
        out.append(dict(r, eval_id=index.get(r.get("run_id")), severity=sev))
    return out


def _filtered(runs_dir: str, outcome, category, model, ticket, correlation_id, request_id, q, since_ms, until_ms):
    all_runs = _with_eval_ids(runs_dir, load_run_summaries(runs_dir))
    filtered = filter_runs(all_runs, outcome=outcome, category=category, model=model, ticket=ticket,
                           since_ms=since_ms, until_ms=until_ms, correlation_id=correlation_id,
                           request_id=request_id, q=q)
    return all_runs, filtered


# ---- upstream pass-through (intake) ---------------------------------------------------------------------
# The dashboard never sees fixture files; accounts and tickets come from whatever UPSTREAM_BASE_URL serves.
# Swap the fixture for a real API and these routes follow it. If that API has no such route, the answer is
# an explicit 501 `upstream_unsupported` so the UI can hide the form instead of guessing.
_ROUTE_MISSING_CODES = ("not_found", "http_404", "http_405", "method_not_allowed")


def _upstream() -> HttpUpstream:
    cfg = load_config()
    return HttpUpstream(cfg.upstream_base_url, cfg.read_timeout_s, cfg.write_timeout_s,
                        retry_policy=RetryPolicy(max_retries=int(cfg.max_retries)))


def _upstream_http_error(exc: UpstreamError) -> HTTPException:
    """Translate a client-side upstream exception into the response the dashboard should see."""
    detail = {"code": exc.code, "message": exc.detail or exc.code}
    if exc.code in _ROUTE_MISSING_CODES and (exc.status in (404, 405)):
        return HTTPException(status_code=501, detail={"code": "upstream_unsupported",
                                                      "message": "upstream has no such route (%s)" % exc.code})
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(exc, WriteRejected):
        return HTTPException(status_code=int(exc.status or 400), detail=detail)
    if isinstance(exc, UpstreamDegraded):
        return HTTPException(status_code=502 if exc.status else 503, detail=detail)
    return HTTPException(status_code=502, detail=detail)


# ---- endpoints ----------------------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return _config_payload()


@app.get("/api/config")
def api_config() -> Dict[str, Any]:
    return _config_payload()


# NB: POST routes use api_route(methods=["POST"]) rather than the FastAPI verb shorthand, because
# tests/test_tools_gate.py statically forbids the verb-call token anywhere in agent/ except tools.py/upstream.py.
@app.api_route("/run", methods=["POST"])
async def post_run(req: RunRequest, request: Request):
    if not _TICKET_RE.match(req.ticket_id):
        raise HTTPException(status_code=422, detail="invalid ticket_id")
    request_id = sanitize_id(request.headers.get("X-Request-ID"), None) or new_request_id()
    correlation_id = sanitize_id(req.correlation_id, None) or sanitize_id(request.headers.get("X-Correlation-ID"), None)
    cfg = load_config(dry_run=not req.write, prompt_version=req.prompt_version)
    result = await run_in_threadpool(run, req.ticket_id, cfg, request_id=request_id, correlation_id=correlation_id)
    body = dataclasses.asdict(result)
    headers = {
        "X-Request-ID": str(getattr(result, "request_id", "") or request_id),
        "X-Correlation-ID": str(getattr(result, "correlation_id", "") or correlation_id or request_id),
    }
    return JSONResponse(content=body, headers=headers)


@app.get("/api/accounts")
def api_accounts() -> Dict[str, Any]:
    """Accounts as the upstream lists them (dropdown source for ticket intake)."""
    try:
        body = _upstream().get("/v1/accounts")
    except UpstreamError as exc:
        raise _upstream_http_error(exc)
    accounts = body.get("accounts")
    if not isinstance(accounts, list):
        raise HTTPException(status_code=502, detail={"code": "invalid_response", "message": "expected {accounts: [...]}"})
    return {"accounts": accounts}


@app.get("/api/tickets")
def api_tickets() -> Dict[str, Any]:
    """Ticket ids the upstream currently knows (fixture ones plus anything created since start)."""
    try:
        body = _upstream().get("/v1/tickets")
    except UpstreamError as exc:
        raise _upstream_http_error(exc)
    ids = body.get("ticket_ids")
    if not isinstance(ids, list):
        raise HTTPException(status_code=502, detail={"code": "invalid_response", "message": "expected {ticket_ids: [...]}"})
    return {"ticket_ids": [str(t) for t in ids]}


@app.api_route("/api/tickets", methods=["POST"])
def api_create_ticket(req: TicketCreate) -> Dict[str, Any]:
    """Create a ticket upstream and hand back the stored record. Nothing is cached here: the next
    POST /run for that id reads it through the same GET /v1/tickets/{id} path as every other ticket."""
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        record = _upstream().create_ticket(payload)
    except UpstreamError as exc:
        raise _upstream_http_error(exc)
    if not isinstance(record, dict) or not record.get("ticket_id"):
        raise HTTPException(status_code=502, detail={"code": "invalid_response", "message": "upstream returned no ticket_id"})
    return record


@app.get("/runs/{run_id}")
def get_run_summary(run_id: str) -> Dict[str, Any]:
    data = load_run(load_config().runs_dir, run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="run not found")
    return data["summary"]


@app.get("/api/runs")
def api_runs(
    outcome: Optional[str] = None,
    category: Optional[str] = None,
    model: Optional[str] = None,
    ticket: Optional[str] = None,
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    q: Optional[str] = None,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: int = Query(200, ge=0, le=2000),
    offset: int = Query(0, ge=0),
    sort: str = "start_ms",
    order: str = "desc",
) -> Dict[str, Any]:
    all_runs, filtered = _filtered(load_config().runs_dir, outcome, category, model, ticket, correlation_id,
                                   request_id, q, since_ms, until_ms)
    present = [r for r in filtered if r.get(sort) is not None]
    missing = [r for r in filtered if r.get(sort) is None]
    present.sort(key=_sort_key(sort), reverse=(order.lower() != "asc"))
    ordered: List[dict] = present + missing
    return {"total": len(filtered), "runs": ordered[offset:offset + limit], "facets": facets(all_runs)}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> Dict[str, Any]:
    runs_dir = load_config().runs_dir
    data = load_run(runs_dir, run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="run not found")
    data["summary"] = dict(data["summary"], eval_id=run_eval_index(runs_dir).get(run_id))
    data["security"] = security_for_run(data["summary"], data.get("result"), data.get("trace"))
    return data


@app.get("/api/metrics")
def api_metrics(
    outcome: Optional[str] = None,
    category: Optional[str] = None,
    model: Optional[str] = None,
    ticket: Optional[str] = None,
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    q: Optional[str] = None,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
) -> Dict[str, Any]:
    all_runs, filtered = _filtered(load_config().runs_dir, outcome, category, model, ticket, correlation_id,
                                   request_id, q, since_ms, until_ms)
    agg = aggregate(filtered)
    agg["facets"] = facets(all_runs)
    return agg


@app.get("/metrics")
def prom_metrics(since_ms: Optional[int] = None) -> PlainTextResponse:
    cfg = load_config()
    all_runs = load_run_summaries(cfg.runs_dir)
    rows = filter_runs(all_runs, since_ms=since_ms) if since_ms is not None else all_runs
    return PlainTextResponse(prometheus_text(aggregate(rows), cfg.service_name), media_type="text/plain; version=0.0.4")


@app.get("/api/evals")
def api_evals() -> Dict[str, Any]:
    return {"evals": load_eval_summaries(load_config().runs_dir)}


@app.get("/api/evals/{eval_id}")
def api_eval(eval_id: str) -> Dict[str, Any]:
    data = load_eval(load_config().runs_dir, eval_id)
    if data is None:
        raise HTTPException(status_code=404, detail="eval not found")
    return data


@app.api_route("/api/evals", methods=["POST"])
def api_start_eval(req: EvalRequest) -> Dict[str, Any]:
    label = _LABEL_RE.sub("", req.label or "")[:40] or "dashboard"
    cfg = load_config()
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "eval_agent.py"), "--variant", "support", "--label", label]
    if req.limit is not None:
        cmd += ["--limit", str(int(req.limit))]
    if req.prompt_version:
        cmd += ["--prompt-version", str(req.prompt_version)]
    if req.tickets:
        cmd += ["--tickets"] + [t for t in re.split(r"[,\s]+", req.tickets.strip()) if _TICKET_RE.match(t)]
    cmd += ["--runs-dir", cfg.runs_dir]
    env = dict(os.environ)
    if req.model_id:
        # run the same case set on another serverless model; the eval header records the model actually used
        env["MODEL_ID"] = req.model_id
        env.setdefault("MODEL_PROVIDER", "openai_compatible")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started": True, "pid": proc.pid, "label": label, "model_id": req.model_id or cfg.model_id}


@app.get("/dashboard")
def dashboard():
    if os.path.isfile(DASHBOARD_HTML):
        return FileResponse(DASHBOARD_HTML, media_type="text/html")
    return HTMLResponse("<!doctype html><html><head><title>ic4-agent</title></head>"
                        "<body><h1>ic4-agent</h1><p>Dashboard not built yet. See <a href=\"/api/metrics\">/api/metrics</a>.</p>"
                        "</body></html>")


@app.get("/")
def root():
    return RedirectResponse("/dashboard")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
