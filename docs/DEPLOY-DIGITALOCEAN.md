# Deploying the IC4 support agent on DigitalOcean

Team: **Blitz_team1_BLR**. Everything below uses `doctl` + the files in this directory (`Dockerfile`,
`compose.yaml`, `.do/app.yaml`, `scripts/start.sh`, `scripts/env.sh`).

## 1. Hosting options

| Option | Fit | Notes |
|---|---|---|
| **App Platform (blr)** — chosen | Team's existing pattern: 6 apps already deploy a Dockerfile from GitHub on `apps-s-1vcpu-1gb` with SECRET env vars | No persistent disk: `runs/` and the `/dashboard` history reset on every deploy. Traces also go to stdout (`TRACE_STDOUT=true`) for log forwarding. Health check on `/healthz`. |
| **Droplet + `docker compose`** | Same image, `runs-data` named volume | Dashboard history survives deploys/restarts. You own patching, TLS, restarts. Use when durable run history matters (`compose.yaml`). |
| **Gradient AI Agents** (managed) | Not a fit | Our policy engine, guard and write-gate are custom code with deterministic tests; a managed agent runtime would replace exactly the part we need to control. We use Gradient **serverless inference** for the model only. |

One container runs both processes: the fixture systems-of-record API on loopback `:8081`
(`python3 -m server --port 8081 --variant support`, skipped when `START_FIXTURE=false`) and the FastAPI layer
`agent.serve:app` on `$PORT` (8080). `scripts/start.sh` waits for the fixture's `/healthz` before `exec`-ing uvicorn.

## 2. Serverless inference (model)

- Base URL `https://inference.do-ai.run/v1`, OpenAI-compatible (`/chat/completions`, streaming supported).
- Model ids: `openai-gpt-4o-mini` (default), `openai-gpt-5-nano`, `openai-gpt-oss-120b`, `anthropic-claude-haiku-4.5`, …
- Pricing: `GET https://api.digitalocean.com/v2/gen-ai/models` returns `pricing.input_price_per_million` /
  `output_price_per_million` **in USD per token**. `make prices-refresh` (`scripts/refresh_prices.py`) converts to
  USD per 1k and prints a dict literal to paste into the cost table.
- Set `OPENAI_BASE_URL`, `MODEL_ID`, `OPENAI_API_KEY`; `MODEL_PROVIDER` auto-switches to `openai_compatible`.

## 3. Keys and where they live

| Key | Purpose | Create / rotate | Local storage |
|---|---|---|---|
| **DO personal access token** (`DIGITALOCEAN_ACCESS_TOKEN`) | Infra control (`doctl`, REST API, price lookup). Scope to what is needed (apps, gen-ai read). | [cloud.digitalocean.com/account/api/tokens](https://cloud.digitalocean.com/account/api/tokens) | `.env.digitalocean` (gitignored via `.env.*`, `chmod 600`) |
| **Gradient model access key** (`OPENAI_API_KEY`) | Inference only — least privilege | Console: Gradient AI Platform > Serverless inference > Model access keys (API creation is retired) | `.env` (gitignored) |

Rules: never paste either into chat, the repo, a commit, or logs. `scripts/env.sh` loads both files silently and
falls back to the DO token for inference only as a stopgap; prefer a model key.

On App Platform both are **encrypted `SECRET` env vars**. `.do/app.yaml` carries a `${OPENAI_API_KEY}` placeholder;
inject it at apply time, never commit the value:

```bash
. scripts/env.sh                         # loads DIGITALOCEAN_ACCESS_TOKEN (+ OPENAI_API_KEY if in .env)
make do-validate                         # doctl apps spec validate .do/app.yaml
envsubst < .do/app.yaml | doctl apps create --spec -          # first deploy (or set the secret in the console)
envsubst < .do/app.yaml | doctl apps update "$APP_ID" --spec - # re-apply
make do-logs APP_ID=$APP_ID              # doctl apps logs --type run --follow
```

**Rotation:** (1) create the new key/token; (2) update `.env` / `.env.digitalocean` locally and the SECRET in the
app (console or `doctl apps update --spec`, which triggers a redeploy); (3) verify `/healthz` and one `POST /run`;
(4) revoke the old one; (5) check Insights > Logs for auth errors. Rotate the DO token once this exercise ends.

## 4. Observability

- **`/dashboard`** — per-run table (ticket, decision, outcome, cost, latency, retries) with drill-down into the
  trace, tool calls and model I/O. Backed by `RUNS_DIR`; on App Platform that is per-deployment history.
- **`/metrics`** — Prometheus text format (counters for runs/outcomes/retries, histograms for latency, TTFT, cost).
- **Structured trace lines** — with `TRACE_STDOUT=true` every trace record is one JSON line on stdout. On App
  Platform: Insights > Logs, or forward to OpenSearch / Datadog / Papertrail (app settings > Log forwarding).
- **Correlation ids** — send `X-Request-ID` or `X-Correlation-ID`; the value is echoed on the response and stamped
  on every trace line and metric label for that run (one is generated if absent).

### Dashboard tabs and the JSON behind them

| Tab (question it answers) | Endpoint(s) | Notes |
|---|---|---|
| Overview — is the agent healthy? | `GET /api/metrics`, `GET /api/glossary` | every KPI carries a plain-English line + formula from the glossary (`obs/insights.py::GLOSSARY`, one copy) |
| Runs — find one run by id | `GET /api/runs?...`, `GET /api/runs/{run_id}` | rows carry `eval_id` when the run belonged to an eval; the detail carries `security` (severity, flags, explanation) |
| Evals — did it answer correctly? | `GET /api/evals`, `GET /api/evals/{eval_id}`, `POST /api/evals {label, model_id?, limit?}` | each case row has its `run_id`, so a wrong answer is one click from its trace |
| Golden set — which model is best? | `GET /api/golden` | latest **full** golden eval per model (partial ones only in `history`), eval quality joined with the linked runs' latency/cost/retries/fallbacks |
| Guardrails & Security — what protects us, did anything trip? | `GET /api/guardrails`, `POST /api/guardrails/test`, `GET /api/security?...` | inventory is derived from `agent/guard.py` / `agent/redact.py` and the `def test_*` names in `tests/`; the test button runs those files with pytest inside the container |
| Models — what can we run on DO? | `GET /api/models` | `dashboard/models.json` from `scripts/probe_models.py` (one 4-token call per catalog id: 200 / 403 tier / 404 not served) + price table + run counts |
| Data store card | `GET /api/db/status`, `POST /api/db/sync` | SQLite index of runs/evals/cases (`runs/index.sqlite`), Postgres export for a Managed DB — section 7 |

Everything read-side lives in `obs/` (not `agent/`): the agent package must stay free of eval vocabulary
(`tests/test_no_golden_access.py`), the dashboard layer may not.

Two conventions worth knowing when reading raw JSON: **all rates are already percentages (0–100)** —
`success_rate: 99.0`, `kb_recall: 86.7`, `escalate_f1: 100.0` — and **category / escalation accuracy tie
across models by design**: the deterministic rules + policy layer decides both; the model only drafts.
Models therefore differ on KB citation quality, guard fallbacks, latency and cost, and the golden leaderboard
is ordered that way.

### Cost / latency / retry glossary

| Term | Meaning |
|---|---|
| `cost_success` / `cost_failure` | Model spend (USD) summed over runs that ended in success / failure |
| `cost_per_success` | total cost ÷ successful runs — the number to optimise |
| `cost_diagnosis` vs `cost_reply` | one model call produces both; cost is attributed by each section's share of output text |
| `TTFT` | time to first streamed token (needs `LLM_STREAM=true`) |
| `TBT` | mean time between tokens after the first |
| retries | default `MAX_RETRIES=3` / `LLM_MAX_RETRIES=3`, **status-based**: retry on timeout / 429 / 5xx; never on 4xx / 404 / `entitlement_service_error` / deadline exceeded; **writes never retry** (idempotency is not guaranteed upstream) |

## 5. Two-minute interviewer demo

```bash
make run                                  # terminal 1: fixture on :8080 (support variant)
make serve                                # terminal 2: agent on :8090 with reload
open http://localhost:8090/dashboard      # empty history
make agent-eval                           # rules provider: deterministic, zero cost
. scripts/env.sh && make agent-eval       # DO serverless inference (openai-gpt-4o-mini): cost, TTFT, retries populate
```

Refresh the dashboard; open a run to show the drill-down (policy decision -> tool calls -> model I/O -> cost split).
Then one correlated request end to end:

```bash
curl -s -X POST http://localhost:8090/run \
  -H 'Content-Type: application/json' -H 'X-Correlation-ID: demo-1' \
  -d '{"ticket_id":"TCK-1123"}'
```

Point at the `demo-1` line in the uvicorn stdout (JSON trace), the same id in the dashboard, and `/metrics`.

## 6. Local container check

```bash
make docker-build && make docker-run      # or: docker compose up --build
curl -s localhost:8080/healthz
```

## 7. Persisting runs to a database

**Why.** App Platform containers have no persistent disk: every deploy or restart starts with an empty `runs/`,
so the dashboard history, the eval comparisons and any run you want to look up by `run_id` are gone. The files
under `runs/` stay the source of truth while the container lives; the database keeps a copy of the rows that
survives deploys and lets past runs be queried (`by run_id / ticket_id / model_id`) and compared across evals.

**Local: SQLite index.** `agent/store.py` indexes `runs/<run_id>/summary.json`, `runs/eval-*/summary.json` and
`runs/eval-*/cases.jsonl` into three tables (`runs`, `evals`, `eval_cases`, plus `meta`) at
**`$RUNS_DIR/index.sqlite`** (default `runs/index.sqlite`, gitignored with the rest of `runs/`). The sync is
idempotent (`INSERT ... ON CONFLICT DO UPDATE`), takes ~150 ms for ~1200 runs, and tolerates older summaries
that lack fields (missing → NULL). Eval metric columns are percentages 0–100 exactly as the eval wrote them.

```bash
make db-sync                              # -> {"ok": true, "runs": 1187, "evals": 16, "cases": 475, ...}
make db-export                            # also writes runs/export.postgres.sql (Postgres upserts)
```

**DigitalOcean: Managed PostgreSQL.** Steps:

1. Create a cluster in the team's region: Databases > Create > PostgreSQL 16, region **blr1**, smallest node
   (`db-s-1vcpu-1gb`). Add the App Platform app as a *trusted source* so only it can connect. Copy the
   connection string (`postgresql://doadmin:...@...db.ondigitalocean.com:25060/defaultdb?sslmode=require`).
2. Set it as a **SECRET** env var `DATABASE_URL` on the app (console: app > Settings > component > Environment
   Variables > *Encrypt*; or in `.do/app.yaml` with `type: SECRET` and a `${DATABASE_URL}` placeholder filled by
   `envsubst` at apply time, same pattern as `OPENAI_API_KEY`). Never commit or print the value —
   `scripts/db_sync.py` redacts it from error output.
3. Create the tables once (idempotent, safe to re-run): `psql "$DATABASE_URL" -f db/schema.postgres.sql`.
   The file doubles as the column-by-column documentation for the dashboard team.
4. Load the rows. Either of:
   - **In-cluster job** (no `psql` needed): add a *job* component (or a post-deploy hook / cron) to the app
     that runs `make db-sync && python3 scripts/db_sync.py --runs-dir "$RUNS_DIR" --push --no-sync`. `--push`
     applies `db/schema.postgres.sql` and upserts every row in one transaction; it needs `psycopg` or `psycopg2`
     in the image (`pip install 'psycopg[binary]'`). Without `DATABASE_URL` or a driver it prints what is
     missing and exits 2, so it is safe to leave in a startup script.
   - **From a laptop / CI with `psql`**: `make db-export && psql "$DATABASE_URL" -f runs/export.postgres.sql`.
     The export is a plain `BEGIN; INSERT ... ON CONFLICT (pk) DO UPDATE ...; COMMIT;` file, so re-applying it
     after new runs only adds the new rows.

Query examples once loaded (same SQL works against `runs/index.sqlite`):

```sql
SELECT run_id, ticket_id, model_id, outcome, cost_usd, duration_ms FROM runs WHERE run_id = '20260926T082625-bc9847';
SELECT model_id, count(*), avg(cost_usd), percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) FROM runs GROUP BY 1;
SELECT eval_id, model_id, category_accuracy, escalate_f1, kb_recall FROM evals ORDER BY timestamp;
```
