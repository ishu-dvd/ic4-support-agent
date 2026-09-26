# Solution — Senior AI/ML Engineer II exercise

Variant: `support`. Language: Python 3.9+, standard library only for the agent (`pytest` is the one
dev dependency). The fixture data (`data/`) is untouched and `make agent-test` refuses to run if it is not; `server/`
gained two additive, in-memory routes for ticket intake (`POST /v1/tickets`, `GET /v1/accounts`, documented in
`docs/API.md`). Architecture and sequence diagrams are in [`README.md`](README.md).

```bash
make run                       # fixture on :8080
make venv                      # .venv with pytest
make agent-run T=TCK-1123      # one ticket, dry-run
make agent-eval                # score against data/support/golden.json, dry-run
make agent-eval-write          # same, files escalations (use a fresh server)
make agent-test                # refuses if data/ is modified, then pytest
make agent-report              # reports/latest.md from runs/
```

No model key is needed. Without one the agent runs a rules-based drafter (`RulesLLM`); set
`OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `MODEL_ID`) to use any OpenAI-compatible model.

## What I built

A fixed nine-step pipeline per ticket, not an open tool-calling loop:

```
read_ticket -> [read_account | read_entitlements | kb_search] -> filter_kb -> classify_intent
            -> escalation_policy -> draft -> output_guard -> write -> summarize
```

- **Ticket-scoped tools.** `TicketContext` binds the account once; no tool accepts an `account_id`.
  See `docs/ADR-001-identity.md`.
- **Entitlement-aware output.** KB hits are filtered by `applies_to` against the account (the search
  endpoint does not do this). The model sees whitelisted views only; the guard scans the output for
  anything outside them.
- **Model proposes, policy decides.** A deterministic `EscalationPolicy` runs before drafting and is
  the only producer of a `WriteIntent`. `Tools.escalate` is the only caller of `POST /v1/escalations`
  and sets `confirm: true` itself, checks for a duplicate, and refuses past the run deadline. Escalation is by
  category plus upstream health and SLA breach; `region_unavailable` escalates only when the ticket describes a
  stuck or pending request, a pure availability question is answered from the KB.
- **Survives upstream failure.** `entitlement_service_error` (acct_1009 in the fixture) is not
  retried: the run degrades to `entitlement_unknown`, escalates, and the reply names no features.
  Timeouts retry once with jitter on reads, never on the write.
- **Injection handling.** Ticket text is scanned; matching lines (and only those lines) are stripped
  before classification, the KB query, the SLA scan and the prompt, so an attacker cannot delete the
  customer's own text by fencing it; the body is delimited as untrusted data; the escalation summary is
  built from tool facts only. A ticket that is mostly injected text is routed to human review without a
  write instead of being escalated as "insufficient information". TCK-1123 is a scored case; the
  probes that got past the first review are regression tests (`tests/test_injection_probes.py`).
- **Fail closed on output.** If the model draft and the rules fallback both fail the output guard,
  the customer gets a fixed holding reply, the run is marked `failed`/`guard_failed`, and the
  escalation still files. A draft never claims a write happened; the loop appends the filed
  reference after the write step succeeds.
- **Eval + traces + cost.** `scripts/eval_agent.py` prints completion rate, category/escalate/KB scores
  (recall and precision) next to the majority baselines, groundedness violations both as the agent
  reported them and as the eval recomputes them, the injection case result, event counts, cost per
  completed task (or "unavailable" for an unpriced model), and per-step p50/p95. `--holdout` scores
  the paraphrased set; `--limit` runs are labelled partial. Traces are JSONL with OpenTelemetry GenAI
  attribute names; `result.json` is redacted like the trace.
- **Budgets.** Per-run caps on LLM calls, tool calls, output tokens and wall-clock; deadline-aware
  degradation (rules draft, no late write).

## Results

From `reports/latest.md`; dry-run against a local fixture server, `RulesLLM` drafter. Golden = the 31 labelled
cases; holdout = 10 paraphrased tickets in `tests/holdout/paraphrased.json` (new wording, real fixture accounts).

| Run | Set | n | Completion | Category acc | Escalate acc / F1 | KB recall / precision | Groundedness viol. (agent / eval) | Injection | Cost/task |
|---|---|---|---|---|---|---|---|---|---|
| rules-v1 | golden | 31 | 100.0% | 100.0% | 100.0% / 100.0 | 93.3% / 100.0% | 0 / 0 | PASS | $0.0000 |
| rules-v1-nokbfilter (`--no-kb-filter`) | golden | 31 | 100.0% | 100.0% | 100.0% / 100.0 | 83.3% / 90.0% | 0 / 0 | PASS | $0.0000 |
| rules-v1-holdout (`--holdout`) | holdout | 10 | 100.0% | 90.0% | 90.0% / 88.9 | 88.9% / 88.9% | 0 / 0 | n/a | $0.0000 |

Baselines from the data itself: always-false escalate scores 71% on golden (60% on holdout); the largest category
scores 16% (20%). KB recall is scored as `expected_kb` being a subset of `kb_cited`; the two golden misses (TCK-1109,
TCK-1112) expect two articles and the rules drafter cites one, so hit-any is 100%. Groundedness is reported twice: as
the agent's own guard saw it, and recomputed by the eval from the persisted `kb_visible` list and degraded flag, so an
agent that under-reports cannot hide it. Failed runs count as wrong on every task metric (completion rate is the first
line of the report for that reason). A prompt version only changes anything once a model is wired in; with the
rules drafter v1 and v2 are identical case by case, so only v1 is reported.

The holdout row is the generalisation signal: the rules were written against the golden vocabulary, so 100% there is
expected, not informative. On the paraphrases one ticket misses (HOLD-001, a billing question worded around "bill" and
"rate" rather than "invoice"/"prorated" lands in `invoice_dispute` and escalates unnecessarily). That is the honest
shape of a keyword classifier and the first thing a model would be measured against.

The no-kb-filter ablation drops KB recall from 93.3% to 83.3% (hit-any 100% to 90%): three unified-auth accounts
(TCK-1102, TCK-1126, TCK-1131) are pointed at the legacy-auth article kb-0002 instead of kb-0003, and the
groundedness count stays at 0 because the guard scores citations against `kb_visible`, which the ablation widened;
the `applies_to` filter is what keeps "grounded" and "correct for this customer" the same thing.

### Evidence

- Write path (`--write` against a fresh fixture on :8093): first eval filed 9 escalations, exactly the 9 cases the policy
  escalated (TCK-1103, 1105, 1109, 1112, 1114, 1123, 1124, 1128, 1130), every stored row `confirm: true` with a
  summary built from ticket subject + category + tool facts (no draft text). Second eval against the same server: 0 new,
  9 `skipped_duplicate`, count still 9. Automated in `tests/test_e2e_write.py` (starts its own server).
- Latency (fixture with `--latency-ms 3000`, `read_timeout_s` 2.0, one retry): TCK-1109 and TCK-1101 end in
  `outcome=failed`, `error=upstream_degraded:timeout` after ~4.3 s with one `upstream_degraded` event, no traceback,
  no write. With `--latency-ms 1500` and `--deadline-ms 1500`, TCK-1103 completes with `deadline_skip` on `draft`
  and on `write`; the write reports `pending_manual` (`past_deadline`) instead of posting. A timeout on the account or
  KB read alone degrades the run (`outcome=degraded`, filter falls back to universal articles) rather than failing it.
- Adversarial set (`tests/test_adversarial.py`, 4 cases substituted into TCK-1101 with `dry_run=False`): all flagged by
  the input scan; the three cases whose body is almost entirely injected text classify as `insufficient_information`
  after stripping and are routed to human review with no write; one classifies as `export_stalled` with no write;
  replies contain none of the forbidden strings; the injected `acct_1003` is never fetched.
- Review probes (`tests/test_injection_probes.py`): a fenced injection wrapped around a genuine proration question still
  classifies `billing_proration` with no write (the fence does not delete the customer's text); an injected "999 hours"
  cannot force an SLA breach; the summary subject drops the injected line and is redacted and truncated.

## What I traded away

- **No agent framework.** The pipeline never branches; a framework would be the largest thing in the
  repo and the hardest to review. LangGraph is the one I would reach for if approvals ever need to be
  resumable across sessions.
- **Rules classify, the model drafts.** Cheap, deterministic, testable, and honest as a baseline.
  On a new domain the rules start weak; the eval will say so.
- **No retrieval beyond the lexical endpoint.** Eighteen short articles; the two recall misses were an
  authorization problem (`applies_to`), not a retrieval one. Hybrid retrieval goes behind the same
  `kb_search` signature when the KB is real.
- **Markdown report first, dashboard later.** In the first hour two runs side by side in `reports/latest.md` was the
  before/after; the dashboard (`/dashboard`, one dependency-free HTML file over the same `runs/` data) came with the
  hosting work and is a view, not a second scorer.
- **Traces without an OTel SDK.** Field names already follow the GenAI semantic conventions; an
  exporter is a small addition.
- **Probable label noise left alone.** TCK-1127 asks about a custom SLA the account has; the label
  says `feature_not_entitled`. I would raise it with the golden set owner rather than fit it.

## Known limits

- **Duplicate check is GET-then-POST.** Two agents on the same ticket can both pass the check and both
  file (TOCTOU). The agent-side check covers the sequential case only; the fix is a server-side
  idempotency key on `POST /v1/escalations`, which the fixture does not offer.
- **A read retry can outlive the deadline.** `HttpUpstream.get` retries once with a fresh timeout and
  does not know the run budget, so a run with a 1.5 s deadline against a 3 s upstream fails after ~4.3 s.
  The write step still refuses past the deadline; only the time-to-fail is wrong.
- **Rules are authored against the golden vocabulary.** 100% on golden is expected. The holdout row
  (90% on 10 paraphrases) is the generalisation signal and the number to compare a model against.
- **No ticket means a hard fail, by design.** `NotFound` on the ticket ends the run as `failed` /
  `ticket_not_found` with no write. There is nothing to act on; a softer outcome would hide it.
- **Cross-account probes are not filtered by the API.** The tools never take an `account_id`, so the
  only account the run can read is the ticket's, but the fixture itself would answer for any id.
- **An unknown model price reports as unavailable, not $0.** Deliberate: a flattering zero is the wrong
  default for a cost column.

## What I would delete

- `ModelRouter`: a hook for tier-based model selection that currently returns the configured model.
  Premature until there is a second model to route to.
- The regenerate-then-fallback branch in the output guard: with the rules drafter it never fires;
  with a real model it should be measured before it is kept. (The fail-closed stub behind it stays.)
- Half of the per-category reply templates in `RulesLLM`, the moment a real model is wired in.

## Controls map

| Concern | Control | Where |
|---|---|---|
| Prompt injection (OWASP LLM01) | input scan, line-level stripping, untrusted delimiters, policy-owned write, review-only routing for stripped tickets | `agent/guard.py`, `agent/policy.py`, `tests/test_guard.py`, `tests/test_injection_probes.py` |
| Insecure output handling (LLM02) | strict JSON parse (unknown keys rejected), output guard over reply and diagnosis, fail-closed safe stub, no execution of model text | `agent/llm.py`, `agent/guard.py`, `agent/loop.py` |
| Eval integrity | agent never sees the golden labels; eval recomputes groundedness from persisted `kb_visible`; failed runs score as wrong; holdout set | `tests/test_no_golden_access.py`, `scripts/eval_agent.py`, `tests/holdout/` |
| Sensitive information disclosure (LLM06) | whitelisted views, `applies_to` filter, trace PII redaction | `agent/redact.py`, `tests/test_redact.py` |
| Excessive agency (LLM08) | no model tool-calling; single non-model write path; no `account_id` parameter | `agent/tools.py`, `tests/test_tools_gate.py` |
| Unbounded consumption (LLM10) | call/token/time budgets, fixed pipeline | `agent/budget.py`, `agent/loop.py` |
| Auditability | every write has a trace with the policy reasons and the intent | `agent/trace.py`, `runs/<id>/trace.jsonl` |
| Data integrity | `data/` never read directly, `make agent-test` refuses if modified | `Makefile` |
| Failure transparency | degraded and unknown states are explicit categories, never silent defaults | `agent/rules.py`, `agent/policy.py` |

## Hosting and observability (built)

The core stays stdlib. `agent/serve.py` (FastAPI, deps in `requirements-serve.txt`) exposes `POST /run`,
`GET /healthz`, `GET /runs/{id}`, a JSON API (`/api/runs`, `/api/runs/{id}`, `/api/metrics`, `/api/evals`),
a Prometheus endpoint (`/metrics`) and a zero-dependency dashboard at `/dashboard` (filterable, sortable,
drill-down to the trace waterfall). One container runs the fixture (:8081) and the agent (:8080);
`Dockerfile`, `compose.yaml`, `.do/app.yaml`. Deployed on DigitalOcean App Platform (region `blr`, the
team's existing pattern) with model calls on DigitalOcean serverless inference (`https://inference.do-ai.run/v1`,
OpenAI-compatible; `gemma-4-31B-it` on this team's tier). See `docs/DEPLOY-DIGITALOCEAN.md`.

Observability model, all in `agent/trace.py` + `agent/metrics.py`:

- Ids on every trace record: `trace_id` (run), `request_id` (`X-Request-ID` or generated),
  `correlation_id` (`X-Correlation-ID`, propagated unchanged, groups runs), `activity_id` (the pipeline step),
  `operation_id` (one HTTP or model attempt). Both request and correlation ids are echoed as response headers.
- Retries are status-based with a default of 3 (`RetryPolicy`): retry on timeout / 429 / 5xx, never on
  404, other 4xx, `entitlement_service_error`, or when the backoff cannot fit in the deadline; writes never
  retry. Each attempt is its own span with `retry_decision`; `retry` / `retries_exhausted` events are counted.
- Model calls stream so time-to-first-token and inter-token gaps (p50/p95) are measured; usage missing
  from the provider is estimated and flagged, never zero. Cost is split input/output and the output part
  attributed to diagnosis vs reply by text share; the dashboard shows cost of successful runs, cost burnt
  on failures, cost per success, and cost by model with DigitalOcean's per-token prices.
- Latency p50/p90/p95/p99 end-to-end and per step; every trace record also goes to stdout as a JSON line
  (`TRACE_STDOUT=true`) for App Platform log forwarding.

Measured on DigitalOcean serverless (`gemma-4-31B-it`, 31 golden cases, dry-run): category 100%, escalate
100%, injection PASS, TTFT p50 ~1.1 s / p95 ~2.0 s, ~$0.00022 per ticket. With the model unreachable
(`gpt-4o-mini` returned 403 "not available for your subscription tier") every run degraded to
`rules_fallback` after one non-retried attempt and still scored 31/31.

**Nothing is lost on a redeploy.** App Platform's disk is ephemeral, so `agent/persist.py` mirrors every
finished run and eval directory (`summary.json`, `result.json`, `trace.jsonl`, `cases.jsonl`, verbatim) to a
database the moment it completes — `RunTracer.finish()` and `scripts/eval_agent.py` call `save_dir()` on the
request path, not in the background, because a write that is still queued when the container is replaced is
exactly the write that goes missing. On boot `scripts/start.sh` runs `python3 -m agent.persist restore`, which
puts every stored file back under `RUNS_DIR`; the dashboard, `/api/*` readers and the Prometheus endpoint keep
reading files and never know a redeploy happened. Backend is chosen from `DATABASE_URL` (`postgresql://…` on
DigitalOcean Managed PostgreSQL via psycopg; `sqlite:///…` locally; unset = disabled, files only). It never
raises into a run: a DB failure is one JSON line on stderr and the run still completes. Drill on the real
history here: 1,332 directories / 3,963 files backfilled in 125 ms, restored byte-identical in 167 ms.
`make persist-status | persist-backfill | persist-restore`. This is the durability layer; the queryable
column index in `agent/store.py` (`runs` / `evals` / `eval_cases`) can always be rebuilt from it.

**Testing on tickets that are not in the fixture.** The dashboard's Evals tab has a *New ticket* card: pick
an account from a dropdown, type subject and body, and the ticket is created **upstream** (`POST /api/tickets`
→ `POST /v1/tickets` on whatever `UPSTREAM_BASE_URL` points at), then run through `POST /run` like any other.
The agent layer holds nothing: accounts (`/api/accounts`) and ticket ids (`/api/tickets`) are read from the
upstream on demand, and the run reads the new ticket back over `GET /v1/tickets/{id}`. Against the fixture
the record lives in memory (gone on restart, `data/` untouched). Against a real systems-of-record API the
same routes follow it; if that API has no listing/intake route the agent answers `501 upstream_unsupported`
and the dashboard disables the card instead of showing an empty dropdown.

## Which model? The golden set on every model this key can reach

`scripts/probe_models.py` asks the inference endpoint for its catalogue and tries one tiny completion per id:
22 of 107 listed models answer on this team's tier (`dashboard/models.json` keeps the snapshot; 403 = not in
the subscription tier, 404 = listed but not served). `POST /api/evals {"model_id": …}` then replays the
31-ticket golden set on a chosen model, and `GET /api/golden` (dashboard tab *Golden set*) puts the latest
full pass of every model side by side, with the deterministic `rules` drafter as the free baseline.

Two findings shaped the read-side code:

- **Category and escalation tie at 100 % on every model, by design.** The rules/policy layer decides both;
  the model only writes the diagnosis and reply. So the comparison axes that actually discriminate are KB
  citation quality (recall / precision), how often the output guard had to fall back to rules, latency and
  cost. The leaderboard sorts on KB recall and the quality-vs-cost chart plots KB recall on a log cost axis.
- **Thinking models need two payload changes.** `kimi-k2.6` rejects `temperature` ("must be 1") and
  `response_format: json_object`; `kimi` and `qwen3.5` stream `reasoning_content` and burn the whole output
  budget before the JSON starts. `agent/llm.py` now sends `chat_template_kwargs.enable_thinking=false`, and
  when a 400 names a parameter it drops that parameter, retries once (`retry_decision=retry:param_unsupported`,
  event `llm_param_dropped`) and remembers the rejection per model for the rest of the process. Mistral, which
  rejects `chat_template_kwargs` itself, is covered by the same path. A plain 400 is still never retried.

Latest full pass per model (31 tickets, dry-run, DigitalOcean serverless, 26 Sep 2026):

| model | KB recall / precision | LLM drafts | cost / ticket | TTFT p50 | e2e p50 |
|---|---|---|---|---|---|
| llama-4-maverick | 96.7 / 88.6 | 29 / 31 | $0.00030 | 0.9 s | 3.6 s |
| kimi-k2.6 | 96.7 / 79.5 | 29 / 31 | $0.00144 | 1.1 s | 5.3 s |
| deepseek-v4.1-flash | 96.7 / 75.6 | 29 / 31 | $0.00077 | 2.9 s | 7.9 s |
| mistral-3-14B | 93.3 / 85.7 | 29 / 31 | $0.00021 | 0.9 s | 2.1 s |
| openai-gpt-oss-120b | 93.3 / 88.2 | 28 / 31 | $0.00037 | 4.3 s | 7.4 s |
| gemma-4-31B-it (default) | 83.3 / 96.4 | 29 / 31 | $0.00022 | 1.0 s | 4.2 s |
| kimi-k3 | 93.3 / 93.8 | 15 / 31 | $0.00468 | 6.9 s | 27 s |
| rules (baseline) | 83.3 / 90.0 | 0 / 31 | $0 | — | 2 ms |

The two non-LLM drafts on most models are the injection ticket and the empty-after-strip ticket, which the
policy routes to rules regardless. `kimi-k3` and `nemotron-3-nano-omni` fall back on half the set (timeouts
at 60 s); `openai-gpt-4o-mini` appears with *no LLM drafts* because the tier returns 403 — the row is the
rules layer under another name, and the leaderboard says so. On this evidence `llama-4-maverick` or
`mistral-3-14B` would replace `gemma-4-31B-it` as the default: same price, better recall, same latency.

The dashboard's other new tabs read the same artefacts: *Evals* lists every eval with its run ids, *Guardrails
& Security* enumerates the seven guard layers with their patterns, the tests that cover each one (runnable
from the page via `POST /api/guardrails/test`) and every run that carried an injection or guard flag, with
the matched-pattern count now written on the `injection_suspected` trace event; *Models* is the probe
snapshot joined with prices and run counts. `GET /api/glossary` backs the ⓘ tooltips so the KPIs are
defined in one place. `obs/` holds this read-side code so the `agent/` package keeps its "never sees the
golden labels" guarantee (`tests/test_no_golden_access.py`).

## Next 30 minutes with a model key

Set `OPENAI_API_KEY`, run `make agent-eval` twice (`PROMPT_VERSION=v1`, then `v2`), `make agent-report`,
and read the model-vs-policy disagreement count before anything else.
