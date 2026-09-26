"""Durable mirror of runs/ so a redeploy loses nothing. Stdlib + optional psycopg.

App Platform's disk is ephemeral: every deploy or restart starts with an empty RUNS_DIR, taking the
dashboard history, the traces and the eval comparisons with it. This module keeps a byte-for-byte copy
of every file under runs/<dir>/ in a database and puts it back on boot:

    write-through   RunTracer.finish() and scripts/eval_agent.py call save_dir() the moment a run / eval
                    directory is complete (summary.json written). One upsert per file, one transaction.
    restore         `python3 -m agent.persist restore` (scripts/start.sh, before uvicorn) writes every
                    stored file back into RUNS_DIR that is missing or differs. The file readers in
                    agent/metrics.py and the dashboard are untouched: they keep reading runs/.
    backfill        `python3 -m agent.persist backfill` pushes whatever is on disk (first cut-over,
                    local history) - idempotent.

Backend is chosen from DATABASE_URL:
    postgresql://user:pw@host:25060/db?sslmode=require   DigitalOcean Managed PostgreSQL (psycopg 3 or psycopg2)
    sqlite:///relative/path.sqlite  |  sqlite:////abs.sqlite   local / tests (stdlib sqlite3)
    unset                                                  disabled: every call is a cheap no-op

Failure policy: persistence must never break a run. Every public function catches everything, prints one
JSON line on stderr (`{"persist": ..., "ok": false, "error": ...}`), and returns a dict with ok=False.
The database URL is never printed.

Relationship to agent/store.py (SQLite index / Postgres export of *columns*): that is the queryable
analytics view; this is the durability layer the view can be rebuilt from. They share DATABASE_URL and
use disjoint tables (`run_files` here).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

TABLE = "run_files"
FILE_NAMES = ("summary.json", "result.json", "trace.jsonl", "cases.jsonl")
MAX_FILE_BYTES = 32 * 1024 * 1024  # a trace bigger than this is almost certainly a bug, not history
CONNECT_TIMEOUT_S = 5

_DDL = (
    "CREATE TABLE IF NOT EXISTS %s ("
    " dir_name text NOT NULL,"       # runs/<dir_name>: a run_id or an eval-... id
    " file_name text NOT NULL,"      # summary.json | result.json | trace.jsonl | cases.jsonl
    " kind text NOT NULL,"           # run | eval
    " body text NOT NULL,"           # file contents, verbatim
    " sha256 text NOT NULL,"
    " size_bytes bigint NOT NULL,"
    " updated_ms bigint NOT NULL,"
    " PRIMARY KEY (dir_name, file_name))" % TABLE,
    "CREATE INDEX IF NOT EXISTS %s_kind_updated ON %s (kind, updated_ms)" % (TABLE, TABLE),
)


# ---- backend ------------------------------------------------------------------------------------------
class _Backend:
    """Thin DB-API wrapper hiding the two placeholder styles. One connection per instance; use as a context."""

    def __init__(self, url: str):
        self.url = url
        self.conn: Any = None
        self.param = "?"
        self.name = "disabled"

    def __enter__(self) -> "_Backend":
        if self.url.startswith("sqlite:"):
            path = self.url[len("sqlite:"):]
            path = path[3:] if path.startswith("///") else path  # sqlite:///rel -> rel ; sqlite:////abs -> /abs
            if path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=CONNECT_TIMEOUT_S)
            self.param, self.name = "?", "sqlite"
        else:
            self.conn = _pg_connect(self.url)
            self.param, self.name = "%s", "postgres"
        for stmt in _DDL:
            self._cur().execute(stmt)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()

    def _cur(self):
        return self.conn.cursor()

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> Any:
        cur = self._cur()
        cur.execute(sql.replace("?", self.param) if self.param != "?" else sql, params)
        return cur

    def fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[tuple]:
        return list(self.execute(sql, params).fetchall())


def _pg_connect(url: str):
    try:
        import psycopg  # type: ignore

        return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S)
    except ImportError:
        pass
    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(url, connect_timeout=CONNECT_TIMEOUT_S)
    except ImportError:
        raise RuntimeError("no_driver: install psycopg[binary] (requirements-serve.txt) for postgresql:// URLs")


def database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def enabled() -> bool:
    return bool(database_url())


def _log(event: str, **fields: Any) -> None:
    rec = {"persist": event, "ts_ms": int(time.time() * 1000)}
    rec.update(fields)
    try:
        sys.stderr.write(json.dumps(rec, default=str) + "\n")
        sys.stderr.flush()
    except Exception:  # pragma: no cover - stderr gone
        pass


def _redact(exc: BaseException) -> str:
    msg = "%s: %s" % (type(exc).__name__, exc)
    url = database_url()
    return msg.replace(url, "<DATABASE_URL>") if url else msg


def _kind(dir_name: str) -> str:
    return "eval" if dir_name.startswith("eval-") else "run"


def _sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_dir(runs_dir: str, dir_name: str) -> List[Tuple[str, str]]:
    """(file_name, body) for every known file present in runs/<dir_name>/."""
    out: List[Tuple[str, str]] = []
    d = os.path.join(runs_dir, dir_name)
    for name in FILE_NAMES:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        if os.path.getsize(p) > MAX_FILE_BYTES:
            _log("skip_large_file", dir=dir_name, file=name, size=os.path.getsize(p))
            continue
        with open(p, encoding="utf-8") as fh:
            out.append((name, fh.read()))
    return out


def _upsert(db: _Backend, dir_name: str, files: Iterable[Tuple[str, str]], now_ms: int) -> int:
    n = 0
    kind = _kind(dir_name)
    for file_name, body in files:
        db.execute(
            "INSERT INTO %s (dir_name, file_name, kind, body, sha256, size_bytes, updated_ms) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (dir_name, file_name) DO UPDATE SET"
            " kind = excluded.kind, body = excluded.body, sha256 = excluded.sha256,"
            " size_bytes = excluded.size_bytes, updated_ms = excluded.updated_ms" % TABLE,
            (dir_name, file_name, kind, body, _sha(body), len(body.encode("utf-8")), now_ms),
        )
        n += 1
    return n


# ---- public API (never raise) ---------------------------------------------------------------------------
def save_dir(runs_dir: str, dir_name: str) -> Dict[str, Any]:
    """Write-through for one finished run or eval directory. Called from RunTracer.finish / eval_agent."""
    if not enabled():
        return {"ok": True, "enabled": False, "files": 0}
    if not dir_name or "/" in dir_name or dir_name.startswith("."):
        return {"ok": False, "enabled": True, "error": "bad_dir_name"}
    t0 = time.perf_counter()
    try:
        files = _read_dir(runs_dir, dir_name)
        if not files:
            return {"ok": True, "enabled": True, "files": 0}
        with _Backend(database_url()) as db:
            n = _upsert(db, dir_name, files, int(time.time() * 1000))
        return {"ok": True, "enabled": True, "files": n, "ms": int((time.perf_counter() - t0) * 1000)}
    except Exception as exc:  # noqa: BLE001 - persistence must never sink a run
        _log("save_failed", dir=dir_name, error=_redact(exc))
        return {"ok": False, "enabled": True, "error": _redact(exc)}


def backfill(runs_dir: str) -> Dict[str, Any]:
    """Push every run/eval directory currently on disk. Idempotent; one transaction."""
    if not enabled():
        return {"ok": True, "enabled": False, "dirs": 0, "files": 0}
    t0 = time.perf_counter()
    dirs = files = 0
    try:
        names = sorted(n for n in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, n)) and not n.startswith("."))
        now_ms = int(time.time() * 1000)
        with _Backend(database_url()) as db:
            for name in names:
                fl = _read_dir(runs_dir, name)
                if fl:
                    files += _upsert(db, name, fl, now_ms)
                    dirs += 1
        out = {"ok": True, "enabled": True, "dirs": dirs, "files": files, "ms": int((time.perf_counter() - t0) * 1000)}
        _log("backfill", **out)
        return out
    except Exception as exc:  # noqa: BLE001
        _log("backfill_failed", error=_redact(exc), dirs=dirs, files=files)
        return {"ok": False, "enabled": True, "error": _redact(exc), "dirs": dirs, "files": files}


def restore(runs_dir: str) -> Dict[str, Any]:
    """Materialise every stored file into runs_dir when missing or different. Existing identical files are
    left alone (so a seeded or partially-populated disk is only ever completed, never clobbered)."""
    if not enabled():
        return {"ok": True, "enabled": False, "written": 0, "skipped": 0}
    t0 = time.perf_counter()
    written = skipped = 0
    try:
        os.makedirs(runs_dir, exist_ok=True)
        with _Backend(database_url()) as db:
            rows = db.fetchall("SELECT dir_name, file_name, body, sha256 FROM %s ORDER BY dir_name, file_name" % TABLE)
        for dir_name, file_name, body, sha in rows:
            if "/" in dir_name or dir_name.startswith(".") or file_name not in FILE_NAMES:
                continue  # never let a row pick a path outside runs/<dir>/<known file>
            p = os.path.join(runs_dir, dir_name, file_name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    if _sha(fh.read()) == sha:
                        skipped += 1
                        continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, p)
            written += 1
        out = {"ok": True, "enabled": True, "written": written, "skipped": skipped, "rows": len(rows),
               "ms": int((time.perf_counter() - t0) * 1000)}
        _log("restore", **out)
        return out
    except Exception as exc:  # noqa: BLE001
        _log("restore_failed", error=_redact(exc), written=written)
        return {"ok": False, "enabled": True, "error": _redact(exc), "written": written, "skipped": skipped}


def status() -> Dict[str, Any]:
    """Backend kind, reachability and row counts. Safe to expose: never includes the URL."""
    url = database_url()
    if not url:
        return {"enabled": False, "backend": None}
    backend = "sqlite" if url.startswith("sqlite:") else "postgres"
    try:
        with _Backend(url) as db:
            (files,) = db.fetchall("SELECT COUNT(*) FROM %s" % TABLE)[0]
            (dirs,) = db.fetchall("SELECT COUNT(DISTINCT dir_name) FROM %s" % TABLE)[0]
            (evals,) = db.fetchall("SELECT COUNT(DISTINCT dir_name) FROM %s WHERE kind = 'eval'" % TABLE)[0]
            (latest,) = db.fetchall("SELECT MAX(updated_ms) FROM %s" % TABLE)[0]
        return {"enabled": True, "backend": backend, "reachable": True, "dirs": dirs, "evals": evals,
                "runs": dirs - evals, "files": files, "last_write_ms": latest}
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "backend": backend, "reachable": False, "error": _redact(exc)}


# ---- CLI (scripts/start.sh, Makefile) ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="runs/ durability: restore | backfill | status (uses $DATABASE_URL)")
    ap.add_argument("command", choices=("restore", "backfill", "status"))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "runs"))
    args = ap.parse_args(argv)
    if args.command == "restore":
        out = restore(args.runs_dir)
    elif args.command == "backfill":
        out = backfill(args.runs_dir)
    else:
        out = status()
    print(json.dumps(out, default=str))
    return 0 if out.get("ok", out.get("reachable", True)) is not False else 1


if __name__ == "__main__":
    sys.exit(main())
