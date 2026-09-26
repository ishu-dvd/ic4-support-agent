#!/usr/bin/env bash
# Container entrypoint: optionally start the fixture upstream API in the background, wait for it,
# then exec uvicorn (PID 1 semantics -> signals reach the agent process directly).
#
# Env:
#   PORT               agent HTTP port (default 8080)
#   START_FIXTURE      "false" to skip the embedded fixture (point UPSTREAM_BASE_URL at a real API instead)
#   FIXTURE_PORT       fixture port (default 8081; must match UPSTREAM_BASE_URL)
#   FIXTURE_VARIANT    fixture variant (default support)
#   RUNS_DIR           where per-run traces are written (created if missing)
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8080}"
RUNS_DIR="${RUNS_DIR:-runs}"
FIXTURE_PORT="${FIXTURE_PORT:-8081}"
FIXTURE_VARIANT="${FIXTURE_VARIANT:-support}"

mkdir -p "$RUNS_DIR"

# App Platform's disk is ephemeral: every deploy starts with an empty RUNS_DIR. If the image ships a
# seed (scripts/seed_runs.py -> seed/runs: the golden-set evals for every model plus the runs they
# link to), copy it in once so the dashboard's history/comparison views are populated from the first
# request. Set SEED_RUNS=false to start empty. Never overwrites an existing run.
if [ "${SEED_RUNS:-true}" != "false" ] && [ -d seed/runs ] && [ -z "$(ls -A "$RUNS_DIR" 2>/dev/null)" ]; then
  cp -rn seed/runs/. "$RUNS_DIR"/
  echo "[start] seeded $(ls "$RUNS_DIR" | wc -l) run/eval directories into ${RUNS_DIR}" >&2
fi

# Durability (agent/persist.py): with DATABASE_URL set, every run/eval directory written since the first
# deploy is mirrored to the database as it finishes; pull them all back before serving so history,
# traces and eval comparisons survive the redeploy. No-op without DATABASE_URL; never blocks startup on
# a DB error (it logs and continues with whatever is on disk).
if [ -n "${DATABASE_URL:-}" ]; then
  python3 -m agent.persist restore --runs-dir "$RUNS_DIR" >&2 || echo "[start] restore from database failed; continuing with local runs/" >&2
fi

if [ "${START_FIXTURE:-true}" != "false" ]; then
  echo "[start] fixture: python3 -m server --port ${FIXTURE_PORT} --variant ${FIXTURE_VARIANT}" >&2
  python3 -m server --host 127.0.0.1 --port "$FIXTURE_PORT" --variant "$FIXTURE_VARIANT" &
  FIXTURE_PID=$!

  # Wait (up to ~15s) for the fixture's /healthz before accepting traffic.
  for _ in $(seq 1 30); do
    if python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${FIXTURE_PORT}/healthz', timeout=1).status == 200 else 1)" 2>/dev/null; then
      echo "[start] fixture ready on :${FIXTURE_PORT}" >&2
      break
    fi
    if ! kill -0 "$FIXTURE_PID" 2>/dev/null; then
      echo "[start] fixture exited before becoming healthy" >&2
      exit 1
    fi
    sleep 0.5
  done
fi

echo "[start] agent: uvicorn agent.serve:app on 0.0.0.0:${PORT} (upstream ${UPSTREAM_BASE_URL:-default})" >&2
exec python3 -m uvicorn agent.serve:app --host 0.0.0.0 --port "$PORT" --workers 1 --proxy-headers
