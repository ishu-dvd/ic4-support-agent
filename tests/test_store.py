"""agent/store.py: runs/ -> SQLite index -> Postgres export."""
from __future__ import annotations

import json
import os
import re
import sqlite3

import pytest

from agent import store

IC4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL_RUN = {
    "run_id": "20260926T082625-bc9847", "trace_id": "a9721a2d56d844b5bd070ab4927a4214",
    "request_id": "req_f850", "correlation_id": "demo-1", "ticket_id": "TCK-1102", "start_ms": 1790411185701,
    "outcome": "completed", "error": None, "category": "credential_rotation", "escalate": False,
    "write_status": None, "draft_source": "llm", "injection_suspected": False, "entitlements_degraded": False,
    "guard_violations": 0, "model_id": "deepseek-v4-pro", "prompt_version": "v1",
    "cost_usd": 0.002076, "cost_input_usd": 0.001425, "cost_output_usd": 0.000651,
    "cost_diagnosis_usd": 0.000184, "cost_reply_usd": 0.000467, "input_tokens": 819, "output_tokens": 187,
    "usage_estimated": False, "ttft_ms": 960, "tbt_ms_avg": 24.45, "tbt_ms_p50": 4.82, "tbt_ms_p95": 31.76,
    "llm_latency_ms": 3086, "llm_attempts": 1, "duration_ms": 3091, "retries": {"http": 0, "llm": 2},
    "step_durations": {"read_ticket": 1, "draft": 3087}, "call_counts": {"http": 4, "llm": 2},
    "events": {"policy_decided": 1},
}

# what the first summaries looked like: 13 keys, no start_ms / model_id / trace ids
LEGACY_RUN = {
    "run_id": "20260926T064707-03c8d1", "ticket_id": "TCK-1106", "outcome": "completed",
    "category": "feature_not_entitled", "escalate": False, "write_status": None, "cost_usd": 0.0,
    "input_tokens": 0, "output_tokens": 0, "duration_ms": 6, "step_durations": {"draft": 6},
    "call_counts": {"http": 4, "llm": 1}, "events": {},
}

# a run that belongs to the eval below; the error text carries a quote to exercise escaping
EVAL_MEMBER_RUN = dict(FULL_RUN, run_id="20260926T074129-d00b9a", ticket_id="TCK-1101", start_ms=1790408489000,
                       outcome="failed", error="upstream said 'no'", model_id="rules", draft_source="rules",
                       escalate=True, write_status="skipped_dry_run", cost_usd=0.0, retries={"http": 1, "llm": 0})

EVAL_ID = "eval-20260926T074129-rules-v1"
EVAL_SUMMARY = {
    "eval_id": EVAL_ID, "label": "rules-v1", "timestamp": "2026-09-26T07:41:29Z", "variant": "support",
    "case_set": "golden", "n": 2, "n_total": 31, "partial": None, "model_id": "rules", "prompt_version": "v1",
    "provider": "rules", "dry_run": True, "kb_filter": True, "baselines": {"n": 31},
    "metrics": {"n": 2, "completion_rate": 100.0, "category_accuracy": 50.0, "escalate_accuracy": 100.0,
                "escalate_precision": 100.0, "escalate_recall": 100.0, "escalate_f1": 100.0,
                "kb_recall": 93.33333333333333, "kb_hit_any": 100.0, "kb_precision": 100.0,
                "injection": {"ticket_id": "TCK-1123", "passed": True}, "cost_total_usd": 0.0},
    "traces": {"available": 2}, "wall_ms": 84,
}
CASES = [
    {"case_id": "sup-001", "ticket_id": "TCK-1101", "run_id": EVAL_MEMBER_RUN["run_id"],
     "expected_category": "billing_proration", "predicted_category": "billing_proration",
     "expected_escalate": True, "predicted_escalate": True, "kb_cited": ["kb-0001"], "expected_kb": ["kb-0001"],
     "category_correct": True, "escalate_correct": True, "kb_recall_hit": True, "kb_hit_any": True,
     "outcome": "failed", "error": None, "draft_source": "rules", "guard_violations": ["pii_email"],
     "eval_violations": [], "injection_suspected": False, "entitlements_degraded": False, "write_status": None,
     "duration_ms": 8, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "reply_clean": True, "completed": True},
    {"case_id": "sup-002", "ticket_id": "TCK-1102", "run_id": "20260926T074129-ffffff",  # run dir not on disk
     "expected_category": "credential_rotation", "predicted_category": "other", "expected_escalate": False,
     "predicted_escalate": False, "category_correct": False, "escalate_correct": True, "kb_recall_hit": False,
     "kb_hit_any": False, "outcome": "completed", "draft_source": "rules", "guard_violations": [],
     "injection_suspected": True, "duration_ms": 5, "cost_usd": 0.0},
]


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


@pytest.fixture
def runs_dir(tmp_path):
    root = tmp_path / "runs"
    for run in (FULL_RUN, LEGACY_RUN, EVAL_MEMBER_RUN):
        _write_json(str(root / run["run_id"] / "summary.json"), run)
        (root / run["run_id"] / "trace.jsonl").write_text("{}\n")
    # in-flight run (trace only, no summary yet) and an in-progress eval dir: both must be skipped quietly
    (root / "20260926T082628-d51abe").mkdir()
    (root / "20260926T082628-d51abe" / "trace.jsonl").write_text("{}\n")
    (root / "eval-20260926T090000-inflight").mkdir()
    _write_json(str(root / EVAL_ID / "summary.json"), EVAL_SUMMARY)
    with open(root / EVAL_ID / "cases.jsonl", "w", encoding="utf-8") as fh:
        for c in CASES:
            fh.write(json.dumps(c) + "\n")
        fh.write("\n{not json\n")  # a torn line must not abort the sync
    return str(root)


def _counts(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {t: conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0] for t in ("runs", "evals", "eval_cases")}
    finally:
        conn.close()


def test_default_db_path():
    assert store.default_db_path("runs") == os.path.join("runs", "index.sqlite")


def test_sync_creates_tables_indexes_and_rows(runs_dir):
    out = store.sync(runs_dir)
    db = store.default_db_path(runs_dir)
    assert os.path.isfile(db)
    assert out["ok"] is True and out["db"] == "sqlite:" + db
    assert (out["runs"], out["evals"], out["cases"]) == (3, 1, 2)
    assert out["inserted"] == {"runs": 3, "evals": 1, "cases": 2}
    assert isinstance(out["duration_ms"], int) and out["duration_ms"] >= 0

    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"runs", "evals", "eval_cases", "meta"} <= tables
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"idx_runs_ticket_id", "idx_runs_model_id", "idx_runs_start_ms", "idx_runs_correlation_id",
                "idx_runs_request_id", "idx_eval_cases_run_id"} <= indexes
        cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
        assert cols == [name for name, _ in store.RUN_COLUMNS]
        # typed scalars, flattened retries, JSON blobs
        row = dict(zip(cols, conn.execute("SELECT * FROM runs WHERE run_id=?", (FULL_RUN["run_id"],)).fetchone()))
        assert row["retries_http"] == 0 and row["retries_llm"] == 2
        assert row["escalate"] == 0 and row["cost_usd"] == pytest.approx(0.002076) and row["input_tokens"] == 819
        assert json.loads(row["step_durations_json"]) == {"read_ticket": 1, "draft": 3087}
        assert json.loads(row["summary_json"])["trace_id"] == FULL_RUN["trace_id"]
        assert row["eval_id"] is None
        # eval metrics are lifted out of metrics{} and kept as 0..100 percentages
        ev = dict(zip([r[1] for r in conn.execute("PRAGMA table_info(evals)")],
                      conn.execute("SELECT * FROM evals").fetchone()))
        assert ev["kb_recall"] == pytest.approx(93.3333333)
        assert ev["category_accuracy"] == 50.0 and ev["injection_passed"] == 1 and ev["partial"] is None
        assert ev["dry_run"] == 1 and ev["n_total"] == 31 and json.loads(ev["metrics_json"])["n"] == 2
        case = dict(zip([r[1] for r in conn.execute("PRAGMA table_info(eval_cases)")],
                        conn.execute("SELECT * FROM eval_cases WHERE case_id='sup-001'").fetchone()))
        assert json.loads(case["guard_violations_json"]) == ["pii_email"] and case["expected_escalate"] == 1
    finally:
        conn.close()


def test_legacy_summary_missing_fields_become_null(runs_dir):
    store.sync(runs_dir)
    row = store.get_run(store.default_db_path(runs_dir), LEGACY_RUN["run_id"])
    assert row is not None
    for key in ("trace_id", "model_id", "prompt_version", "ttft_ms", "retries_http", "draft_source"):
        assert row[key] is None, key
    assert row["cost_usd"] == 0.0 and row["duration_ms"] == 6
    # load_run_summaries derives start_ms from the run_id prefix for legacy rows
    assert row["start_ms"] == 1790405227000
    assert row["escalate"] is False and row["events"] == {}


def test_eval_id_backfilled_onto_runs_from_cases(runs_dir):
    db = store.default_db_path(runs_dir)
    store.sync(runs_dir)
    assert store.get_run(db, EVAL_MEMBER_RUN["run_id"])["eval_id"] == EVAL_ID
    assert store.get_run(db, FULL_RUN["run_id"])["eval_id"] is None
    # the SQL backfill path: wipe eval_id and re-sync only the UPDATE should restore it
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET eval_id = NULL")
    conn.commit()
    conn.close()
    store.sync(runs_dir)
    assert store.get_run(db, EVAL_MEMBER_RUN["run_id"])["eval_id"] == EVAL_ID


def test_resync_is_idempotent_and_replaces_updated_values(runs_dir):
    db = store.default_db_path(runs_dir)
    first = store.sync(runs_dir)
    second = store.sync(runs_dir)
    assert _counts(db) == {"runs": 3, "evals": 1, "eval_cases": 2}
    assert second["inserted"] == {"runs": 0, "evals": 0, "cases": 0}
    assert (second["runs"], second["evals"], second["cases"]) == (first["runs"], first["evals"], first["cases"])

    # the file changed on disk (e.g. summary rewritten): the row is replaced, not duplicated
    changed = dict(FULL_RUN, outcome="degraded", cost_usd=0.5, events={"upstream_degraded": 1})
    _write_json(os.path.join(runs_dir, FULL_RUN["run_id"], "summary.json"), changed)
    changed_eval = dict(EVAL_SUMMARY, metrics=dict(EVAL_SUMMARY["metrics"], kb_recall=12.5))
    _write_json(os.path.join(runs_dir, EVAL_ID, "summary.json"), changed_eval)
    third = store.sync(runs_dir)
    assert third["inserted"] == {"runs": 0, "evals": 0, "cases": 0}
    assert _counts(db) == {"runs": 3, "evals": 1, "eval_cases": 2}
    row = store.get_run(db, FULL_RUN["run_id"])
    assert row["outcome"] == "degraded" and row["cost_usd"] == 0.5 and row["events"] == {"upstream_degraded": 1}
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT kb_recall FROM evals").fetchone()[0] == 12.5
    conn.close()

    # a new run appearing later is picked up incrementally
    _write_json(os.path.join(runs_dir, "20260926T090000-aaaaaa", "summary.json"),
                dict(LEGACY_RUN, run_id="20260926T090000-aaaaaa"))
    fourth = store.sync(runs_dir)
    assert fourth["inserted"] == {"runs": 1, "evals": 0, "cases": 0} and fourth["runs"] == 4


def test_status_shape(runs_dir, monkeypatch):
    db = store.default_db_path(runs_dir)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    before = store.status(db)
    assert before == {"sqlite_path": db, "exists": False, "rows": {"runs": 0, "evals": 0, "cases": 0},
                      "last_sync_ms": None, "postgres": {"configured": False, "driver": before["postgres"]["driver"]}}
    assert before["postgres"]["driver"] in (None, "psycopg", "psycopg2")

    store.sync(runs_dir)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:secret@host/db")
    after = store.status(db)
    assert after["exists"] is True and after["rows"] == {"runs": 3, "evals": 1, "cases": 2}
    assert isinstance(after["last_sync_ms"], int) and after["last_sync_ms"] > 1_700_000_000_000
    assert after["postgres"]["configured"] is True


def test_get_run_and_query_runs(runs_dir):
    db = store.default_db_path(runs_dir)
    assert store.get_run(db, FULL_RUN["run_id"]) is None  # no index yet
    assert store.query_runs(db) == []
    store.sync(runs_dir)

    row = store.get_run(db, FULL_RUN["run_id"])
    assert row["ticket_id"] == "TCK-1102" and row["model_id"] == "deepseek-v4-pro"
    assert row["escalate"] is False and row["usage_estimated"] is False  # bools come back as bools
    assert row["step_durations"] == {"read_ticket": 1, "draft": 3087}
    assert row["summary"]["cost_reply_usd"] == pytest.approx(0.000467)
    assert "summary_json" not in row and "step_durations_json" not in row
    assert store.get_run(db, "nope") is None and store.get_run(db, "") is None

    all_runs = store.query_runs(db)
    assert [r["run_id"] for r in all_runs] == [FULL_RUN["run_id"], EVAL_MEMBER_RUN["run_id"], LEGACY_RUN["run_id"]]
    assert [r["run_id"] for r in store.query_runs(db, ticket_id="TCK-1101")] == [EVAL_MEMBER_RUN["run_id"]]
    assert [r["run_id"] for r in store.query_runs(db, model_id="rules")] == [EVAL_MEMBER_RUN["run_id"]]
    assert store.query_runs(db, ticket_id="TCK-1101", model_id="deepseek-v4-pro") == []
    assert len(store.query_runs(db, limit=2)) == 2


def test_export_postgres_sql(runs_dir):
    db = store.default_db_path(runs_dir)
    assert list(store.export_postgres_sql(db)) == []  # nothing indexed yet -> nothing to export
    store.sync(runs_dir)
    stmts = list(store.export_postgres_sql(db))
    by_table = {}
    for s in stmts:
        m = re.match(r"INSERT INTO (\w+) \(", s)
        assert m, s
        by_table.setdefault(m.group(1), []).append(s)
        assert " ON CONFLICT (" in s and ") DO UPDATE SET " in s and s.endswith(";")
        assert "?" not in s.split(" VALUES ")[0]  # no sqlite placeholders leaked
    assert len(by_table["runs"]) == 3 and len(by_table["evals"]) == 1 and len(by_table["eval_cases"]) == 2
    assert len(by_table["meta"]) >= 1

    member = next(s for s in by_table["runs"] if EVAL_MEMBER_RUN["run_id"] in s)
    assert "'upstream said ''no'''" in member  # single quote doubled
    assert "ON CONFLICT (run_id) DO UPDATE SET" in member
    assert " TRUE" in member and "'" + EVAL_ID + "'" in member
    assert "::jsonb" in member
    assert re.search(r"cost_usd = EXCLUDED\.cost_usd", member)
    # error=None / write_status=None -> NULL; no backslash escaping through the JSON blob either
    full = next(s for s in by_table["runs"] if FULL_RUN["run_id"] in s)
    assert re.search(r"'completed', NULL, 'credential_rotation', FALSE, NULL, 'llm'", full)
    assert "\\'" not in full
    case = by_table["eval_cases"][0]
    assert "ON CONFLICT (eval_id, case_id) DO UPDATE SET" in case
    assert "eval_id = EXCLUDED" not in case and "case_id = EXCLUDED" not in case  # pk columns are not updated
    assert "guard_violations_json = EXCLUDED.guard_violations_json" in case
    ev = by_table["evals"][0]
    assert "ON CONFLICT (eval_id) DO UPDATE SET" in ev and "93.33333333333333" in ev  # percentage kept as-is
    assert re.search(r"partial, provider.*VALUES \(.*NULL, 'rules'", ev)  # partial=None -> NULL


def test_pg_literal_escaping():
    assert store._pg_literal(None, "text") == "NULL"
    assert store._pg_literal("it's", "text") == "'it''s'"
    assert store._pg_literal("a\\b", "text") == "'a\\b'"  # no backslash escaping (standard_conforming_strings)
    assert store._pg_literal("nul\x00byte", "text") == "'nulbyte'"
    assert store._pg_literal(1, "bool") == "TRUE" and store._pg_literal(0, "bool") == "FALSE"
    assert store._pg_literal(3, "int") == "3" and store._pg_literal(2.5, "real") == "2.5"
    assert store._pg_literal(float("nan"), "real") == "NULL"
    assert store._pg_literal('{"a": "x\'y"}', "json") == "'{\"a\": \"x''y\"}'::jsonb"


def test_postgres_schema_file_matches_store_columns():
    """db/schema.postgres.sql is the documentation the dashboard team reads; keep it in step with the code."""
    with open(os.path.join(IC4_ROOT, "db", "schema.postgres.sql"), encoding="utf-8") as fh:
        ddl = fh.read()
    for table, (columns, _pk) in store.TABLES.items():
        assert re.search(r"CREATE TABLE IF NOT EXISTS %s \(" % table, ddl), table
        body = ddl.split("CREATE TABLE IF NOT EXISTS %s (" % table, 1)[1].split("\n);", 1)[0]
        for name, kind in columns:
            m = re.search(r"^\s*%s\s+(\w+(?: precision)?)" % re.escape(name), body, re.M)
            assert m, "%s.%s missing from schema.postgres.sql" % (table, name)
            expected = {"text": "text", "int": ("bigint", "integer"), "real": "double precision",
                        "bool": "boolean", "json": "jsonb"}[kind]
            assert m.group(1) in (expected if isinstance(expected, tuple) else (expected,)), (table, name, m.group(1))
    for idx, table, column in store.INDEXES:
        assert re.search(r"CREATE INDEX IF NOT EXISTS %s\s+ON %s \(%s\)" % (idx, table, column), ddl), idx


def test_cli_sync_and_export(runs_dir, tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    out_sql = tmp_path / "export.sql"
    env = dict(os.environ)
    env.pop("DATABASE_URL", None)
    proc = subprocess.run([sys.executable, os.path.join(IC4_ROOT, "scripts", "db_sync.py"), "--runs-dir", runs_dir,
                           "--export-postgres", str(out_sql)], capture_output=True, text=True, env=env, cwd=IC4_ROOT)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["runs"] == 3 and summary["evals"] == 1 and summary["cases"] == 2
    assert summary["export"]["statements"] == 3 + 1 + 2 + 3  # + meta: schema_version, last_sync_ms, runs_dir
    text = out_sql.read_text(encoding="utf-8")
    assert text.startswith("-- generated by") and "BEGIN;\n" in text and text.rstrip().endswith("COMMIT;")
    assert text.count("INSERT INTO runs (") == 3

    # --push without DATABASE_URL: instructions on stderr, exit 2, no URL echoed
    proc2 = subprocess.run([sys.executable, os.path.join(IC4_ROOT, "scripts", "db_sync.py"), "--runs-dir", runs_dir,
                            "--push", "--no-sync"], capture_output=True, text=True, env=env, cwd=IC4_ROOT)
    assert proc2.returncode == 2
    assert "DATABASE_URL is not set" in proc2.stderr and "schema.postgres.sql" in proc2.stderr
