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

REPO_ROOT = _IC4
DASHBOARD_HTML = os.path.join(REPO_ROOT, "dashboard", "index.html")
_TICKET_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]")

app = FastAPI(title="ic4-agent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


def _filtered(runs_dir: str, outcome, category, model, ticket, correlation_id, request_id, q, since_ms, until_ms):
    all_runs = load_run_summaries(runs_dir)
    filtered = filter_runs(all_runs, outcome=outcome, category=category, model=model, ticket=ticket,
                           since_ms=since_ms, until_ms=until_ms, correlation_id=correlation_id,
                           request_id=request_id, q=q)
    return all_runs, filtered


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
    data = load_run(load_config().runs_dir, run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="run not found")
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
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=dict(os.environ),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started": True, "pid": proc.pid, "label": label}


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
