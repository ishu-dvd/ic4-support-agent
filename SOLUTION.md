# Solution — Senior AI/ML Engineer II exercise

Variant: `support`. Language: Python 3.9+, standard library only for the agent (`pytest` is the one
dev dependency). The fixture (`server/`, `data/`, `docs/API.md`) is untouched.

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
- **Markdown report, not a dashboard.** Two runs side by side is the before/after.
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

## Hosting path

The core is stdlib so it can be reviewed and run with nothing installed. For hosting, `specs/SPEC.md`
defines `agent/serve.py` as a FastAPI app (`POST /run`, `GET /healthz`, `GET /runs/{id}`) using
Pydantic v2 `TypeAdapter` over the existing dataclasses, so no domain type changes are needed; deps are
isolated in `requirements-serve.txt`. Container: `python:3.14-slim` + uvicorn, compose with the fixture as
a second service, endpoints swapped by `UPSTREAM_BASE_URL` / `OPENAI_BASE_URL` only.

## Next 30 minutes with a model key

Set `OPENAI_API_KEY`, run `make agent-eval` twice (`PROMPT_VERSION=v1`, then `v2`), `make agent-report`,
and read the model-vs-policy disagreement count before anything else.
