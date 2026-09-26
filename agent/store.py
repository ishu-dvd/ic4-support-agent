"""Persistent run store: runs/ on disk -> SQLite index (local) -> optional Postgres export (DigitalOcean).

Stdlib only (sqlite3, json, os, time). The files under runs/ stay the source of truth; this module
indexes them so run/eval history survives redeploys (App Platform's filesystem is ephemeral) and can be
queried by run_id / ticket_id / model_id and compared across past runs without walking ~1000 directories.

Three tables mirror the three on-disk shapes (see agent/metrics.py loaders, which are reused here):

  runs        one row per runs/<run_id>/summary.json  (flat scalars + JSON blobs for the nested maps)
  evals       one row per runs/eval-*/summary.json    (headline metrics lifted out of metrics{})
  eval_cases  one row per line of runs/eval-*/cases.jsonl

plus `meta` (key/value; holds last_sync_ms). Every write is `INSERT ... ON CONFLICT DO UPDATE`, so sync()
is idempotent and re-running it after new runs land only adds the new rows.

Column kinds drive both the SQLite DDL and the Postgres export:
  text -> TEXT / text          int  -> INTEGER / bigint        real -> REAL / double precision
  bool -> INTEGER 0/1 / boolean   json -> TEXT / jsonb (serialised with json.dumps, sort_keys=True)

NOTE on eval metric columns (completion_rate, category_accuracy, escalate_*, kb_*): scripts/eval_agent.py
already writes them as PERCENTAGES in 0..100 (e.g. kb_recall = 93.33). They are stored AS-IS - do not divide
by 100 again when reading them back, and do not compare them against 0..1 ratios.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from agent.metrics import load_eval, load_eval_summaries, load_run_summaries

SCHEMA_VERSION = 1

# ---- column specs ------------------------------------------------------------------------------------
# (column, kind). Order here is the column order in CREATE TABLE and in the Postgres export.
RUN_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("run_id", "text"),
    ("trace_id", "text"),
    ("request_id", "text"),
    ("correlation_id", "text"),
    ("ticket_id", "text"),
    ("start_ms", "int"),
    ("outcome", "text"),
    ("error", "text"),
    ("category", "text"),
    ("escalate", "bool"),
    ("write_status", "text"),
    ("draft_source", "text"),
    ("injection_suspected", "bool"),
    ("entitlements_degraded", "bool"),
    ("guard_violations", "int"),
    ("model_id", "text"),
    ("prompt_version", "text"),
    ("cost_usd", "real"),
    ("cost_input_usd", "real"),
    ("cost_output_usd", "real"),
    ("cost_diagnosis_usd", "real"),
    ("cost_reply_usd", "real"),
    ("input_tokens", "int"),
    ("output_tokens", "int"),
    ("usage_estimated", "bool"),
    ("ttft_ms", "int"),
    ("tbt_ms_avg", "real"),
    ("tbt_ms_p50", "real"),
    ("tbt_ms_p95", "real"),
    ("llm_latency_ms", "int"),
    ("llm_attempts", "int"),
    ("duration_ms", "int"),
    ("retries_http", "int"),
    ("retries_llm", "int"),
    ("call_counts_http", "int"),
    ("call_counts_llm", "int"),
    ("eval_id", "text"),
    ("step_durations_json", "json"),
    ("events_json", "json"),
    ("summary_json", "json"),
)

EVAL_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("eval_id", "text"),
    ("label", "text"),
    ("timestamp", "text"),
    ("variant", "text"),
    ("model_id", "text"),
    ("prompt_version", "text"),
    ("case_set", "text"),
    ("n", "int"),
    ("n_total", "int"),
    ("partial", "bool"),
    ("provider", "text"),
    ("dry_run", "bool"),
    ("kb_filter", "bool"),
    # percentages 0..100, stored as written by the eval (see module docstring)
    ("completion_rate", "real"),
    ("category_accuracy", "real"),
    ("escalate_accuracy", "real"),
    ("escalate_precision", "real"),
    ("escalate_recall", "real"),
    ("escalate_f1", "real"),
    ("kb_recall", "real"),
    ("kb_hit_any", "real"),
    ("kb_precision", "real"),
    ("injection_passed", "bool"),
    ("wall_ms", "int"),
    ("metrics_json", "json"),
    ("summary_json", "json"),
)

CASE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("eval_id", "text"),
    ("case_id", "text"),
    ("ticket_id", "text"),
    ("run_id", "text"),
    ("expected_category", "text"),
    ("predicted_category", "text"),
    ("expected_escalate", "bool"),
    ("predicted_escalate", "bool"),
    ("category_correct", "bool"),
    ("escalate_correct", "bool"),
    ("kb_recall_hit", "bool"),
    ("kb_hit_any", "bool"),
    ("outcome", "text"),
    ("draft_source", "text"),
    ("injection_suspected", "bool"),
    ("guard_violations_json", "json"),
    ("cost_usd", "real"),
    ("duration_ms", "int"),
)

META_COLUMNS: Tuple[Tuple[str, str], ...] = (("key", "text"), ("value", "text"))

TABLES: Dict[str, Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]] = {
    # table -> (columns, primary key columns)
    "runs": (RUN_COLUMNS, ("run_id",)),
    "evals": (EVAL_COLUMNS, ("eval_id",)),
    "eval_cases": (CASE_COLUMNS, ("eval_id", "case_id")),
    "meta": (META_COLUMNS, ("key",)),
}

INDEXES: Tuple[Tuple[str, str, str], ...] = (
    ("idx_runs_ticket_id", "runs", "ticket_id"),
    ("idx_runs_model_id", "runs", "model_id"),
    ("idx_runs_start_ms", "runs", "start_ms"),
    ("idx_runs_correlation_id", "runs", "correlation_id"),
    ("idx_runs_request_id", "runs", "request_id"),
    ("idx_runs_eval_id", "runs", "eval_id"),
    ("idx_eval_cases_run_id", "eval_cases", "run_id"),
)

_SQLITE_TYPE = {"text": "TEXT", "int": "INTEGER", "real": "REAL", "bool": "INTEGER", "json": "TEXT"}


# ---- paths / connections -----------------------------------------------------------------------------
def default_db_path(runs_dir: str) -> str:
    return os.path.join(runs_dir, "index.sqlite")


def _connect(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _create_sql(table: str) -> str:
    columns, pk = TABLES[table]
    cols = ", ".join("%s %s" % (name, _SQLITE_TYPE[kind]) for name, kind in columns)
    return "CREATE TABLE IF NOT EXISTS %s (%s, PRIMARY KEY (%s))" % (table, cols, ", ".join(pk))


def _ensure_schema(conn: sqlite3.Connection) -> None:
    for table in TABLES:
        conn.execute(_create_sql(table))
        # forward-compatible: an index created by an older build gets any new columns added in place
        existing = {row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)}
        for name, kind in TABLES[table][0]:
            if name not in existing:
                conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, _SQLITE_TYPE[kind]))
    for idx, table, column in INDEXES:
        conn.execute("CREATE INDEX IF NOT EXISTS %s ON %s(%s)" % (idx, table, column))
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (str(SCHEMA_VERSION),))


def _upsert_sql(table: str) -> str:
    columns, pk = TABLES[table]
    names = [name for name, _ in columns]
    updates = ", ".join("%s = excluded.%s" % (n, n) for n in names if n not in pk)
    return "INSERT INTO %s (%s) VALUES (%s) ON CONFLICT(%s) DO UPDATE SET %s" % (
        table, ", ".join(names), ", ".join("?" for _ in names), ", ".join(pk), updates)


# ---- value coercion (tolerant: anything odd becomes NULL rather than aborting the sync) ---------------
def _coerce(value: Any, kind: str) -> Any:
    if value is None:
        return None
    try:
        if kind == "text":
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
        if kind == "int":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return int(value)
        if kind == "real":
            f = float(value)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if kind == "bool":
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes"):
                    return 1
                if low in ("false", "0", "no", ""):
                    return 0
                return None
            return 1 if value else 0
        if kind == "json":
            return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return None


def _row(columns: Sequence[Tuple[str, str]], values: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(_coerce(values.get(name), kind) for name, kind in columns)


def _run_values(s: dict, eval_by_run: Dict[str, str]) -> Dict[str, Any]:
    retries = s.get("retries") if isinstance(s.get("retries"), dict) else {}
    calls = s.get("call_counts") if isinstance(s.get("call_counts"), dict) else {}
    v = dict(s)
    v["retries_http"] = retries.get("http")
    v["retries_llm"] = retries.get("llm")
    v["call_counts_http"] = calls.get("http")
    v["call_counts_llm"] = calls.get("llm")
    v["eval_id"] = eval_by_run.get(s["run_id"])
    v["step_durations_json"] = s.get("step_durations")
    v["events_json"] = s.get("events")
    v["summary_json"] = s
    # legacy rows: guard_violations may be absent or a list (the eval case shape)
    gv = s.get("guard_violations")
    if isinstance(gv, (list, tuple)):
        v["guard_violations"] = len(gv)
    return v


def _eval_values(s: dict) -> Dict[str, Any]:
    metrics = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
    v = dict(s)
    for key in ("completion_rate", "category_accuracy", "escalate_accuracy", "escalate_precision", "escalate_recall",
                "escalate_f1", "kb_recall", "kb_hit_any", "kb_precision"):
        v[key] = metrics.get(key)
    injection = metrics.get("injection") if isinstance(metrics.get("injection"), dict) else {}
    v["injection_passed"] = injection.get("passed")
    v["metrics_json"] = metrics
    v["summary_json"] = s
    return v


def _case_values(eval_id: str, c: dict) -> Dict[str, Any]:
    v = dict(c)
    v["eval_id"] = eval_id
    v["guard_violations_json"] = c.get("guard_violations")
    return v


def _case_id(c: dict) -> Optional[str]:
    """case_id, falling back to ticket_id for case rows written before case ids existed."""
    cid = c.get("case_id") or c.get("ticket_id")
    return None if cid is None else str(cid)


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0])
    except sqlite3.Error:
        return 0


# ---- public API -----------------------------------------------------------------------------------------
def sync(runs_dir: str, db_path: Optional[str] = None) -> dict:
    """Index every run summary, eval summary and eval case under runs_dir into SQLite. Idempotent."""
    t0 = time.monotonic()
    db_path = db_path or default_db_path(runs_dir)
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        before = {t: _count(conn, t) for t in ("runs", "evals", "eval_cases")}

        eval_rows: List[Tuple[Any, ...]] = []
        case_rows: List[Tuple[Any, ...]] = []
        eval_by_run: Dict[str, str] = {}
        for es in load_eval_summaries(runs_dir):
            eval_id = es["eval_id"]
            loaded = load_eval(runs_dir, eval_id) or {"summary": es, "cases": []}
            eval_rows.append(_row(EVAL_COLUMNS, _eval_values(loaded["summary"] or es)))
            seen: set = set()
            for c in loaded["cases"]:
                if not isinstance(c, dict):
                    continue
                cid = _case_id(c)
                if cid is None or cid in seen:
                    continue
                seen.add(cid)
                c = dict(c, case_id=cid)
                case_rows.append(_row(CASE_COLUMNS, _case_values(eval_id, c)))
                if c.get("run_id"):
                    # newest eval wins if a run somehow belongs to two (summaries are sorted newest first)
                    eval_by_run.setdefault(str(c["run_id"]), eval_id)

        run_rows = [_row(RUN_COLUMNS, _run_values(s, eval_by_run)) for s in load_run_summaries(runs_dir)]

        with conn:
            conn.executemany(_upsert_sql("evals"), eval_rows)
            conn.executemany(_upsert_sql("eval_cases"), case_rows)
            conn.executemany(_upsert_sql("runs"), run_rows)
            # backfill eval_id onto runs indexed before their eval's cases existed (or vice versa)
            conn.execute(
                "UPDATE runs SET eval_id = (SELECT c.eval_id FROM eval_cases c WHERE c.run_id = runs.run_id "
                "ORDER BY c.eval_id DESC LIMIT 1) WHERE eval_id IS NULL AND run_id IN (SELECT run_id FROM eval_cases)")
            now_ms = int(time.time() * 1000)
            conn.execute(_upsert_sql("meta"), ("last_sync_ms", str(now_ms)))
            conn.execute(_upsert_sql("meta"), ("runs_dir", os.path.abspath(runs_dir)))

        after = {t: _count(conn, t) for t in ("runs", "evals", "eval_cases")}
    finally:
        conn.close()
    return {
        "ok": True,
        "db": "sqlite:%s" % db_path,
        "runs": after["runs"],
        "evals": after["evals"],
        "cases": after["eval_cases"],
        "inserted": {
            "runs": after["runs"] - before["runs"],
            "evals": after["evals"] - before["evals"],
            "cases": after["eval_cases"] - before["eval_cases"],
        },
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }


def _postgres_driver() -> Optional[str]:
    import importlib.util

    for name in ("psycopg", "psycopg2"):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


def status(db_path: str) -> dict:
    """Shape of the index without touching runs/: row counts, last sync time, Postgres readiness."""
    exists = os.path.isfile(db_path)
    rows = {"runs": 0, "evals": 0, "cases": 0}
    last_sync_ms: Optional[int] = None
    if exists:
        conn = _connect(db_path)
        try:
            rows = {"runs": _count(conn, "runs"), "evals": _count(conn, "evals"), "cases": _count(conn, "eval_cases")}
            try:
                got = conn.execute("SELECT value FROM meta WHERE key = 'last_sync_ms'").fetchone()
                last_sync_ms = int(got[0]) if got and got[0] is not None else None
            except (sqlite3.Error, TypeError, ValueError):
                last_sync_ms = None
        finally:
            conn.close()
    return {
        "sqlite_path": db_path,
        "exists": exists,
        "rows": rows,
        "last_sync_ms": last_sync_ms,
        "postgres": {"configured": bool(os.environ.get("DATABASE_URL")), "driver": _postgres_driver()},
    }


def _decode_row(row: sqlite3.Row, columns: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """sqlite row -> dict: bools back to bool, *_json columns decoded under the un-suffixed key."""
    out: Dict[str, Any] = {}
    keys = set(row.keys())
    for name, kind in columns:
        if name not in keys:
            continue
        value = row[name]
        if kind == "bool" and value is not None:
            value = bool(value)
        if kind == "json":
            if value is not None:
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            out[name[: -len("_json")] if name.endswith("_json") else name] = value
            continue
        out[name] = value
    return out


def query_runs(db_path: str, ticket_id: Optional[str] = None, model_id: Optional[str] = None,
               limit: int = 200) -> List[dict]:
    """Newest-first list of indexed runs, optionally narrowed by ticket_id / model_id."""
    if not os.path.isfile(db_path):
        return []
    where: List[str] = []
    params: List[Any] = []
    if ticket_id:
        where.append("ticket_id = ?")
        params.append(ticket_id)
    if model_id:
        where.append("model_id = ?")
        params.append(model_id)
    sql = "SELECT * FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY start_ms DESC, run_id DESC LIMIT ?"
    params.append(max(1, min(int(limit or 200), 10000)))
    conn = _connect(db_path)
    try:
        return [_decode_row(r, RUN_COLUMNS) for r in conn.execute(sql, params)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def get_run(db_path: str, run_id: str) -> Optional[dict]:
    if not run_id or not os.path.isfile(db_path):
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _decode_row(row, RUN_COLUMNS) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# ---- Postgres export ----------------------------------------------------------------------------------------
def _pg_literal(value: Any, kind: str) -> str:
    """One SQL literal for Postgres. Strings double their single quotes; no backslash escapes are used
    (standard_conforming_strings is on by default), NUL bytes are dropped because text cannot hold them."""
    if value is None:
        return "NULL"
    if kind == "bool":
        return "TRUE" if value else "FALSE"
    if kind == "int":
        return str(int(value))
    if kind == "real":
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return "NULL"
        return repr(f)
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\x00", "").replace("'", "''")
    lit = "'" + text + "'"
    return lit + "::jsonb" if kind == "json" else lit


def _pg_upsert(table: str, row: sqlite3.Row) -> str:
    columns, pk = TABLES[table]
    keys = set(row.keys())
    present = [(n, k) for n, k in columns if n in keys]
    names = ", ".join(n for n, _ in present)
    values = ", ".join(_pg_literal(row[n], k) for n, k in present)
    updates = ", ".join("%s = EXCLUDED.%s" % (n, n) for n, _ in present if n not in pk)
    return "INSERT INTO %s (%s) VALUES (%s) ON CONFLICT (%s) DO UPDATE SET %s;" % (
        table, names, values, ", ".join(pk), updates)


def export_postgres_sql(db_path: str) -> Iterator[str]:
    """Yield one Postgres upsert statement per indexed row (runs, evals, eval_cases, then meta).

    Apply with `psql "$DATABASE_URL" -f runs/export.postgres.sql` after db/schema.postgres.sql. Only
    statements are yielded (no comments, no BEGIN/COMMIT) so a driver can execute them one by one inside
    its own transaction; scripts/db_sync.py adds the wrapper when writing a file.
    """
    if not os.path.isfile(db_path):
        return
    conn = _connect(db_path)
    try:
        for table, order in (("runs", "start_ms, run_id"), ("evals", "timestamp, eval_id"),
                             ("eval_cases", "eval_id, case_id"), ("meta", "key")):
            try:
                cursor = conn.execute("SELECT * FROM %s ORDER BY %s" % (table, order))
            except sqlite3.Error:
                continue
            for row in cursor:
                yield _pg_upsert(table, row)
    finally:
        conn.close()
