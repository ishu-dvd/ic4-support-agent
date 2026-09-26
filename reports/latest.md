# Eval report

Source: `runs` (3 evals)

## Run comparison

| label | timestamp | set | prompt_version | model_id | provider | n | completion | category acc | escalate acc | escalate F1 | kb recall | kb precision | groundedness viol. (agent / eval) | injection | cost/task | e2e p50 ms | e2e p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules-v1 | 2026-09-26T07:41:29Z | golden | v1 | rules | rules | 31 | 100.0% | 100.0% | 100.0% | 100.0 | 93.3% | 100.0% | 0 / 0 | PASS | $0.0000 | 2 | 2 |
| rules-v1-nokbfilter | 2026-09-26T07:41:29Z | golden | v1 | rules | rules | 31 | 100.0% | 100.0% | 100.0% | 100.0 | 83.3% | 90.0% | 0 / 0 | PASS | $0.0000 | 2 | 2 |
| rules-v1-holdout | 2026-09-26T07:41:30Z | holdout | v1 | rules | rules | 10 | 100.0% | 90.0% | 90.0% | 88.9 | 88.9% | 88.9% | 0 / 0 | n/a | $0.0000 | 2 | 7 |

Groundedness is shown as agent-reported / eval-recomputed (from persisted `kb_visible` and the degraded flag). `set=holdout` rows score the paraphrased tickets in `tests/holdout/paraphrased.json`; they are the generalisation signal, not comparable to golden rows. Partial runs are marked in `n`.

## Per-step timing (latest: `rules-v1-holdout`)

| step | p50 ms | p95 ms |
|---|---|---|
| read_ticket | 0 | 0 |
| read_parallel | 1 | 6 |
| filter_kb | 0 | 0 |
| classify_intent | 0 | 0 |
| escalation_policy | 0 | 0 |
| draft | 0 | 0 |
| output_guard | 0 | 0 |
| write | 0 | 0 |
| summarize | 0 | 0 |
| kb_search | 1 | 5 |
| read_account | 1 | 6 |
| read_entitlements | 1 | 5 |

## Activity (latest: `rules-v1-holdout`)

| kind | name | count |
|---|---|---|
| event | policy_decided | 10 |
| event | write_skipped_dry_run | 5 |
| call | http | 30 |
| call | llm | 10 |
| outcomes | completed | 10 |
| draft_source | rules | 10 |
| write_status | none | 5 |
| write_status | skipped_dry_run | 5 |
| flag | injection_suspected | 0 |
| flag | entitlements_degraded | 0 |
| flag | model_vs_policy_disagree | 0 |

