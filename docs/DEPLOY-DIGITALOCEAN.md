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
