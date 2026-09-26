.PHONY: run run-access smoke smoke-access clean \
        agent-run agent-eval agent-eval-write agent-test agent-report data-clean-check venv \
        serve docker-build docker-run do-validate do-create do-update do-logs prices-refresh

run:
	python3 -m server

run-access:
	python3 -m server --variant access --port 8081

smoke:
	python3 scripts/smoke_test.py

smoke-access:
	python3 scripts/smoke_test.py --base-url http://127.0.0.1:8081 --variant access

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# ---- agent (solution) ----
PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
T ?= TCK-1101

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt

agent-run:
	$(PY) scripts/run_agent.py $(T)

agent-eval:
	$(PY) scripts/eval_agent.py --variant support

agent-eval-write:
	$(PY) scripts/eval_agent.py --variant support --write

agent-test: data-clean-check
	$(PY) -m pytest -q tests

agent-report:
	$(PY) scripts/report.py runs --out reports/latest.md

data-clean-check:
	@test -z "$$(git status --porcelain -- data/)" && echo "data/ clean" || (echo "data/ has modifications - refusing" && exit 1)

# ---- serving / deploy ----
# Local HTTP layer with reload (fixture on :8080 via `make run`, agent on :8090).
serve:
	$(PY) -m uvicorn agent.serve:app --host 0.0.0.0 --port 8090 --reload

docker-build:
	docker build -t ic4-agent .

# Same layout as App Platform: fixture + agent in one container, runs/ in a named volume.
docker-run: docker-build
	docker run --rm -p 8080:8080 $(if $(wildcard .env),--env-file .env,) -v ic4-runs:/data/runs ic4-agent

# DigitalOcean App Platform (doctl reads DIGITALOCEAN_ACCESS_TOKEN; `. scripts/env.sh` loads it).
do-validate:
	doctl apps spec validate .do/app.yaml

do-create:
	doctl apps create --spec .do/app.yaml

do-update:
	@test -n "$(APP_ID)" || (echo "usage: make do-update APP_ID=<id>" && exit 1)
	doctl apps update $(APP_ID) --spec .do/app.yaml

do-logs:
	@test -n "$(APP_ID)" || (echo "usage: make do-logs APP_ID=<id>" && exit 1)
	doctl apps logs $(APP_ID) --type run --follow

# Regenerate the serverless-inference price table (needs DIGITALOCEAN_ACCESS_TOKEN).
prices-refresh:
	$(PY) scripts/refresh_prices.py

# ---- runs/ durability (agent/persist.py; needs DATABASE_URL, see .env.example) ----
.PHONY: persist-status persist-backfill persist-restore
RUNS_DIR ?= runs

# Backend, reachability, how many runs / evals / files are mirrored. Never prints the URL.
persist-status:
	$(PY) -m agent.persist status

# Push everything currently under $(RUNS_DIR) (first cut-over or local history). Idempotent.
persist-backfill:
	$(PY) -m agent.persist backfill --runs-dir $(RUNS_DIR)

# Materialise the mirrored files into $(RUNS_DIR) (what scripts/start.sh does on boot).
persist-restore:
	$(PY) -m agent.persist restore --runs-dir $(RUNS_DIR)
