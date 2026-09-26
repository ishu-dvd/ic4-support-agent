"""Dashboard-only endpoints (FastAPI router), mounted by agent/serve.py.

Everything here is read-side over runs/ except two job starters (guardrail tests, DB sync) that never
touch the agent's own write path. Kept out of agent/ so the agent package stays free of eval vocabulary.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from agent.config import load_config
from agent.metrics import filter_runs, load_run_summaries
from obs.insights import (
    glossary,
    golden_comparison,
    guardrails_inventory,
    models_catalog,
    run_eval_index,
    security_overview,
    start_guardrail_tests,
)

router = APIRouter()
_MODEL_RE = re.compile(r"^[A-Za-z0-9._/-]{1,80}$")


def _runs_dir() -> str:
    return load_config().runs_dir


@router.get("/api/glossary")
def api_glossary() -> Dict[str, Any]:
    return glossary()


@router.get("/api/security")
def api_security(
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
    runs_dir = _runs_dir()
    rows = filter_runs(load_run_summaries(runs_dir), outcome=outcome, category=category, model=model, ticket=ticket,
                       since_ms=since_ms, until_ms=until_ms, correlation_id=correlation_id, request_id=request_id, q=q)
    return security_overview(runs_dir, rows, run_eval_index(runs_dir))


@router.get("/api/golden")
def api_golden() -> Dict[str, Any]:
    return golden_comparison(_runs_dir())


@router.get("/api/guardrails")
def api_guardrails() -> Dict[str, Any]:
    return guardrails_inventory()


@router.post("/api/guardrails/test")
def api_guardrails_test() -> Dict[str, Any]:
    if not start_guardrail_tests():
        raise HTTPException(status_code=409, detail="guardrail tests already running")
    return {"started": True}


@router.get("/api/models")
def api_models() -> Dict[str, Any]:
    return models_catalog(_runs_dir())


# ---- persistent store (agent/store.py, SQLite locally / Postgres export for DigitalOcean) -------------------
def _store():
    try:
        from agent import store  # noqa: WPS433 - optional module, built separately
    except ImportError:
        return None
    return store


@router.get("/api/db/status")
def api_db_status() -> Dict[str, Any]:
    store = _store()
    runs_dir = _runs_dir()
    if store is None:
        return {"sqlite_path": os.path.join(runs_dir, "index.sqlite"), "exists": False,
                "rows": {"runs": 0, "evals": 0, "cases": 0}, "last_sync_ms": None,
                "postgres": {"configured": bool(os.environ.get("DATABASE_URL")), "driver": None},
                "error": "agent.store not available"}
    return store.status(store.default_db_path(runs_dir))


@router.post("/api/db/sync")
def api_db_sync() -> Dict[str, Any]:
    store = _store()
    if store is None:
        raise HTTPException(status_code=501, detail="agent.store not available on this build")
    runs_dir = _runs_dir()
    return store.sync(runs_dir, store.default_db_path(runs_dir))
