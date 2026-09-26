-- IC4 support agent: run / eval history schema for PostgreSQL (DigitalOcean Managed PostgreSQL).
--
-- Mirrors the SQLite index built by agent/store.py (runs/index.sqlite). Column order and names are the
-- same in both, so `scripts/db_sync.py --export-postgres FILE` produces INSERT ... ON CONFLICT statements
-- that load straight into these tables. Idempotent: every statement is CREATE ... IF NOT EXISTS.
--
-- Apply:   psql "$DATABASE_URL" -f db/schema.postgres.sql
-- Load:    psql "$DATABASE_URL" -f runs/export.postgres.sql      (or scripts/db_sync.py --push)
--
-- Type conventions: *_ms and token counts are bigint; cost_* (USD) and eval percentages are double
-- precision; flags are boolean; *_json columns are jsonb copies of the nested maps from the source files.
-- Anything the source file did not have (older summaries have far fewer fields) is NULL, never 0.

CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,          -- setting name, e.g. 'last_sync_ms', 'schema_version', 'runs_dir'
    value text                       -- setting value as text (last_sync_ms is epoch milliseconds)
);

-- One row per agent run (one ticket handled once) = runs/<run_id>/summary.json.
CREATE TABLE IF NOT EXISTS runs (
    run_id                text PRIMARY KEY,   -- directory name under runs/, "YYYYmmddTHHMMSS-<6 hex>" (UTC start + random suffix)
    trace_id              text,               -- 32-hex trace id stamped on every trace line of this run
    request_id            text,               -- id of the HTTP request that started the run (generated if the caller sent none)
    correlation_id        text,               -- caller-supplied X-Correlation-ID / X-Request-ID, echoed on the response; equals request_id when absent
    ticket_id             text,               -- support ticket handled, e.g. TCK-1101
    start_ms              bigint,             -- run start, epoch milliseconds (derived from run_id for legacy rows)
    outcome               text,               -- completed | degraded | failed
    error                 text,               -- error code when outcome = failed, else NULL
    category              text,               -- predicted intent category (billing_proration, credential_rotation, ...)
    escalate              boolean,            -- did the policy decide to escalate
    write_status          text,               -- escalation write result: filed | skipped_dry_run | skipped_duplicate | rejected | pending_manual | NULL (no write)
    draft_source          text,               -- who wrote the reply: llm | rules | fallback
    injection_suspected   boolean,            -- prompt-injection heuristics fired on the ticket text
    entitlements_degraded boolean,            -- entitlements upstream failed; run continued without it
    guard_violations      integer,            -- number of output-guard violations found in the reply (0 = clean)
    model_id              text,               -- model used ("rules" when no LLM call was made)
    prompt_version        text,               -- prompt template version, e.g. v1
    cost_usd              double precision,   -- total model cost in USD (NULL = price unknown, never 0)
    cost_input_usd        double precision,   -- share of cost_usd from input tokens
    cost_output_usd       double precision,   -- share of cost_usd from output tokens
    cost_diagnosis_usd    double precision,   -- output cost attributed to the diagnosis section (by text share)
    cost_reply_usd        double precision,   -- output cost attributed to the customer reply section
    input_tokens          bigint,             -- prompt tokens billed
    output_tokens         bigint,             -- completion tokens billed
    usage_estimated       boolean,            -- TRUE when the provider returned no usage and tokens were estimated locally
    ttft_ms               bigint,             -- time to first streamed token (NULL when not streaming)
    tbt_ms_avg            double precision,   -- mean time between streamed tokens
    tbt_ms_p50            double precision,   -- median time between tokens
    tbt_ms_p95            double precision,   -- p95 time between tokens
    llm_latency_ms        bigint,             -- wall time of the model call(s) including retries
    llm_attempts          integer,            -- model call attempts (1 = no retry)
    duration_ms           bigint,             -- end-to-end run duration
    retries_http          integer,            -- retried upstream HTTP reads (writes never retry)
    retries_llm           integer,            -- retried model calls
    call_counts_http      integer,            -- upstream HTTP calls made
    call_counts_llm       integer,            -- model calls made
    eval_id               text,               -- eval this run was part of (backfilled from eval_cases), NULL for ad-hoc runs
    step_durations_json   jsonb,              -- {step_name: ms} per pipeline step (read_ticket, kb_search, draft, output_guard, ...)
    events_json           jsonb,              -- {event_name: count} of trace events (policy_decided, upstream_degraded, ...)
    summary_json          jsonb               -- the full summary.json row as written, for fields not lifted into columns
);

CREATE INDEX IF NOT EXISTS idx_runs_ticket_id      ON runs (ticket_id);
CREATE INDEX IF NOT EXISTS idx_runs_model_id       ON runs (model_id);
CREATE INDEX IF NOT EXISTS idx_runs_start_ms       ON runs (start_ms);
CREATE INDEX IF NOT EXISTS idx_runs_correlation_id ON runs (correlation_id);
CREATE INDEX IF NOT EXISTS idx_runs_request_id     ON runs (request_id);
CREATE INDEX IF NOT EXISTS idx_runs_eval_id        ON runs (eval_id);

-- One row per evaluation (a batch of runs scored against golden labels) = runs/eval-*/summary.json.
-- All *_rate / *_accuracy / *_precision / *_recall / *_f1 / kb_* columns are PERCENTAGES 0..100 exactly as
-- the eval wrote them (kb_recall = 93.33 means 93.33 %). Do not divide by 100 again.
CREATE TABLE IF NOT EXISTS evals (
    eval_id            text PRIMARY KEY,      -- directory name, "eval-YYYYmmddTHHMMSS-<label>"
    label              text,                  -- human label given on the command line (e.g. golden-gpt4omini-v1)
    timestamp          text,                  -- ISO-8601 UTC start time of the eval
    variant            text,                  -- fixture variant evaluated against (support | access)
    model_id           text,                  -- model under test ("rules" = deterministic baseline, no LLM)
    prompt_version     text,                  -- prompt template version under test
    case_set           text,                  -- which labelled set: golden | holdout | adversarial
    n                  integer,               -- cases actually scored
    n_total            integer,               -- cases in the set (n < n_total when the eval was cut short)
    partial            boolean,               -- TRUE when the eval stopped early (budget/interrupt); NULL in older summaries
    provider           text,                  -- model provider: rules | openai_compatible | ...
    dry_run            boolean,               -- TRUE = escalations were not written upstream
    kb_filter          boolean,               -- TRUE = KB results filtered by account applies_to (FALSE = ablation)
    completion_rate    double precision,      -- % of cases whose run finished (completed or degraded)
    category_accuracy  double precision,      -- % of cases with the expected category
    escalate_accuracy  double precision,      -- % of cases with the expected escalate decision
    escalate_precision double precision,      -- % of predicted escalations that were expected
    escalate_recall    double precision,      -- % of expected escalations that were predicted
    escalate_f1        double precision,      -- harmonic mean of escalate precision/recall, in %
    kb_recall          double precision,      -- % of expected KB articles that were cited
    kb_hit_any         double precision,      -- % of cases citing at least one expected article
    kb_precision       double precision,      -- % of cited articles that were expected (or visible)
    injection_passed   boolean,               -- the prompt-injection probe case was detected AND handled correctly
    wall_ms            bigint,                -- wall time of the whole eval
    metrics_json       jsonb,                 -- full metrics{} block (confusion, groundedness, activity, cost totals, ...)
    summary_json       jsonb                  -- the full eval summary.json as written
);

-- One row per scored case inside an eval = one line of runs/eval-*/cases.jsonl.
CREATE TABLE IF NOT EXISTS eval_cases (
    eval_id               text NOT NULL,      -- owning eval (evals.eval_id)
    case_id               text NOT NULL,      -- case label from the golden set, e.g. sup-001
    ticket_id             text,               -- ticket the case exercises
    run_id                text,               -- the run that produced the prediction (runs.run_id)
    expected_category     text,               -- golden category
    predicted_category    text,               -- category the agent produced
    expected_escalate     boolean,            -- golden escalate decision
    predicted_escalate    boolean,            -- decision the agent produced
    category_correct      boolean,            -- predicted_category = expected_category
    escalate_correct      boolean,            -- predicted_escalate = expected_escalate
    kb_recall_hit         boolean,            -- every expected KB article was cited
    kb_hit_any            boolean,            -- at least one expected KB article was cited
    outcome               text,               -- run outcome: completed | degraded | failed
    draft_source          text,               -- llm | rules | fallback
    injection_suspected   boolean,            -- injection heuristics fired for this case
    guard_violations_json jsonb,              -- list of output-guard violation codes found in the reply
    cost_usd              double precision,   -- model cost of this case's run
    duration_ms           bigint,             -- end-to-end duration of this case's run
    PRIMARY KEY (eval_id, case_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_cases_run_id ON eval_cases (run_id);
