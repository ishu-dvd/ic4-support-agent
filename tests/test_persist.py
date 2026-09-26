"""agent/persist.py: runs/ survives a redeploy.

Exercised on the sqlite backend (stdlib). The SQL is identical for PostgreSQL apart from the placeholder
style, which _Backend rewrites; the postgres path itself needs a live cluster and is covered by
`make persist-status` against DATABASE_URL.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3

import pytest

from agent import persist
from agent.trace import RunTracer


def _db_url(tmp_path) -> str:
    return "sqlite:///" + str(tmp_path / "persist.sqlite")


def _finish_run(runs_dir, ticket="TCK-1", **attrs) -> str:
    t = RunTracer(str(runs_dir), ticket_id=ticket)
    with t.step("read_ticket"):
        with t.call("http GET /v1/tickets/x"):
            pass
    t.finish({"outcome": "completed", "category": "billing_proration", "escalate": False,
              "usage": {"model_id": "rules", "cost_usd": 0.0}, "duration_ms": 3, "draft_source": "rules"}, **attrs)
    return t.run_id


def _write_eval(runs_dir, eval_id="eval-20260926T000000-t") -> str:
    d = runs_dir / eval_id
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"eval_id": eval_id, "n": 2, "metrics": {"kb_recall": 50.0}}))
    (d / "cases.jsonl").write_text('{"case_id": "a"}\n{"case_id": "b"}\n')
    return eval_id


def _tree(root) -> dict:
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, root)] = open(p, encoding="utf-8").read()
    return out


# ---- disabled by default ----

def test_disabled_without_database_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert persist.enabled() is False
    assert persist.save_dir(str(tmp_path), "x") == {"ok": True, "enabled": False, "files": 0}
    assert persist.restore(str(tmp_path))["enabled"] is False
    assert persist.backfill(str(tmp_path))["enabled"] is False
    assert persist.status() == {"enabled": False, "backend": None}
    # a tracer finishing with persistence off writes files exactly as before and touches no database
    runs = tmp_path / "runs"
    run_id = _finish_run(runs)
    assert sorted(os.listdir(runs / run_id)) == ["result.json", "summary.json", "trace.jsonl"]
    assert not (tmp_path / "persist.sqlite").exists()


# ---- write-through + restore ----

def test_tracer_finish_mirrors_the_run_and_restore_rebuilds_it_byte_for_byte(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    runs = tmp_path / "runs"
    run_id = _finish_run(runs, prompt_version="v1")
    before = _tree(runs)
    assert set(before) == {f"{run_id}/result.json", f"{run_id}/summary.json", f"{run_id}/trace.jsonl"}

    st = persist.status()
    assert st["reachable"] and st["runs"] == 1 and st["evals"] == 0 and st["files"] == 3

    # the redeploy: disk is gone
    shutil.rmtree(runs)
    out = persist.restore(str(runs))
    assert out["ok"] and out["written"] == 3 and out["skipped"] == 0
    assert _tree(runs) == before

    # a second restore over an intact disk touches nothing
    out = persist.restore(str(runs))
    assert out["written"] == 0 and out["skipped"] == 3


def test_eval_dirs_are_mirrored_with_kind_eval(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    runs = tmp_path / "runs"
    runs.mkdir()
    eval_id = _write_eval(runs)
    out = persist.save_dir(str(runs), eval_id)
    assert out["ok"] and out["files"] == 2
    con = sqlite3.connect(str(tmp_path / "persist.sqlite"))
    rows = con.execute("SELECT dir_name, file_name, kind, size_bytes FROM run_files ORDER BY file_name").fetchall()
    assert rows == [(eval_id, "cases.jsonl", "eval", len('{"case_id": "a"}\n{"case_id": "b"}\n')),
                    (eval_id, "summary.json", "eval", len(json.dumps({"eval_id": eval_id, "n": 2, "metrics": {"kb_recall": 50.0}})))]
    assert persist.status()["evals"] == 1


def test_save_is_an_upsert_and_restore_prefers_the_database_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    runs = tmp_path / "runs"
    run_id = _finish_run(runs)
    summary = runs / run_id / "summary.json"
    updated = json.loads(summary.read_text())
    updated["outcome"] = "degraded"
    summary.write_text(json.dumps(updated))
    assert persist.save_dir(str(runs), run_id)["files"] == 3  # second save: same keys, new body
    assert persist.status()["files"] == 3

    # a seeded / stale copy on disk differs from what the database holds -> database wins
    summary.write_text("{}")
    out = persist.restore(str(runs))
    assert out["written"] == 1 and out["skipped"] == 2
    assert json.loads(summary.read_text())["outcome"] == "degraded"


def test_backfill_pushes_existing_history_once(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runs = tmp_path / "runs"
    ids = [_finish_run(runs, ticket=t) for t in ("TCK-1", "TCK-2")]
    ids.append(_write_eval(runs))
    (runs / "index.sqlite").write_text("not a run dir")  # sibling files at the root are ignored
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    out = persist.backfill(str(runs))
    assert out == {**out, "ok": True, "dirs": 3, "files": 8}
    assert persist.backfill(str(runs))["files"] == 8  # idempotent
    fresh = tmp_path / "fresh"
    assert persist.restore(str(fresh))["written"] == 8
    assert set(os.listdir(fresh)) == set(ids)


# ---- failure isolation ----

def test_database_failure_never_breaks_the_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "no-such-dir-file.sqlite" / "x"))
    (tmp_path / "no-such-dir-file.sqlite").write_text("a file where a directory is needed")
    runs = tmp_path / "runs"
    run_id = _finish_run(runs)  # must not raise
    assert (runs / run_id / "summary.json").exists()
    err = capsys.readouterr().err
    line = [json.loads(l) for l in err.splitlines() if l.startswith('{"persist"')][-1]
    assert line["persist"] == "save_failed" and line["dir"] == run_id
    assert "x" in line["error"]
    assert persist.restore(str(runs))["ok"] is False
    assert persist.status()["reachable"] is False


def test_error_messages_never_contain_the_url(monkeypatch, tmp_path):
    secret = "postgresql://doadmin:s3cret@db.example:25060/defaultdb?sslmode=require"
    monkeypatch.setenv("DATABASE_URL", secret)
    monkeypatch.setattr(persist, "_pg_connect", lambda url: (_ for _ in ()).throw(RuntimeError("cannot reach " + url)))
    out = persist.save_dir(str(tmp_path), "20260101T000000-abc123")
    assert out["ok"] is True and out["files"] == 0  # nothing on disk to save -> no connection attempted
    runs = tmp_path / "runs"
    run_id = _finish_run(runs)
    out = persist.save_dir(str(runs), run_id)
    assert out["ok"] is False and "s3cret" not in out["error"] and "<DATABASE_URL>" in out["error"]
    st = persist.status()
    assert st["backend"] == "postgres" and st["reachable"] is False and "s3cret" not in json.dumps(st)


def test_restore_ignores_rows_that_escape_the_runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    persist.status()  # creates the table
    con = sqlite3.connect(str(tmp_path / "persist.sqlite"))
    con.execute("INSERT INTO run_files VALUES ('../evil', 'summary.json', 'run', '{}', 'x', 2, 1)")
    con.execute("INSERT INTO run_files VALUES ('ok-run', 'passwd', 'run', 'root', 'x', 4, 1)")
    con.commit()
    out = persist.restore(str(tmp_path / "runs"))
    assert out["ok"] and out["written"] == 0
    assert not (tmp_path / "evil").exists() and not (tmp_path / "runs" / "ok-run").exists()


def test_bad_dir_name_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    assert persist.save_dir(str(tmp_path), "../x")["error"] == "bad_dir_name"
    assert persist.save_dir(str(tmp_path), ".hidden")["error"] == "bad_dir_name"


# ---- CLI (what scripts/start.sh and the Makefile call) ----

def test_cli_restore_and_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_URL", _db_url(tmp_path))
    runs = tmp_path / "runs"
    _finish_run(runs)
    shutil.rmtree(runs)
    assert persist.main(["restore", "--runs-dir", str(runs)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["written"] == 3
    assert persist.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["runs"] == 1
    monkeypatch.delenv("DATABASE_URL")
    assert persist.main(["restore", "--runs-dir", str(runs)]) == 0  # disabled is not an error
