"""The run loop: read -> classify -> policy -> draft -> guard -> write -> summarize.

WHY this order: every decision that has consequences (category, escalation, the write) is
made from tool facts before the model is consulted, so a slow, failing or manipulated model
degrades the prose but never the outcome. Each step is a trace span with a fixed name so
runs are comparable across prompt versions and models.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent.budget import Budget, BudgetExceeded
from agent.config import Config
from agent.context import TicketContext
from agent.guard import leak_hits, scan_output, strip_injection_blocks
from agent.llm import PRICE_PER_1K, Draft, LLMError, LLMUsage, RulesLLM, attribute_cost, make_llm
from agent.policy import decide
from agent.prompts import build_prompt_ctx
from agent.rules import classify
from agent.schema import WriteResult
from agent.tools import Tools
from agent.trace import RunTracer
from agent.upstream import HttpUpstream, NotFound, RetryPolicy, UpstreamDegraded, UpstreamError

# Fail-closed output: when even the rules fallback trips the guard, the customer gets this and
# nothing else. Constant text, no citations, no feature names, no claims about what was filed.
SAFE_REPLY = (
    "Thank you for contacting support. Your ticket has been received and is being reviewed by our team; "
    "we will follow up shortly."
)
SAFE_DIAGNOSIS = "Automated drafting withheld: output guard failed."


@dataclass
class RunResult:
    run_id: str
    ticket_id: str
    category: str
    request_type: str
    confidence: float
    diagnosis: str
    reply: str
    kb_cited: List[str]
    escalate: bool
    escalate_recommended_by_model: bool
    priority: str
    policy_reasons: List[str]
    write: Optional[WriteResult]
    injection_suspected: bool
    entitlements_degraded: bool
    draft_source: str  # "rules" | "llm" | "rules_fallback" | "safe_stub" | "none"
    guard_violations: List[str]
    usage: LLMUsage
    duration_ms: int
    outcome: str  # "completed" | "degraded" | "failed"
    error: Optional[str] = None
    kb_visible: List[str] = field(default_factory=list)  # ids the drafter was allowed to cite (eval re-checks kb_cited against it)
    request_id: str = ""  # inbound request id (X-Request-ID or generated); see trace.py for the id model
    correlation_id: str = ""  # caller-supplied, propagated unchanged; defaults to request_id


class _Abort(Exception):
    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


def _read_parallel(tools: Any, ctx: Any, tracer: Any) -> None:
    """account + entitlements + kb_search concurrently, each as its own nested step.

    WHY not tools.read_all_parallel(): that helper re-reads the ticket, which the read_ticket
    step already did (and where NotFound is handled); reading it twice would double a tool
    charge and a span for no information.

    WHY a timeout on one read does not fail the run: the ticket is already in hand. Without the
    account the applies_to filter keeps only universally applicable articles; without KB hits
    the draft cites nothing. Both are safe degradations, recorded in ctx.degraded_reads and as
    `upstream_degraded` events, and the run's outcome becomes "degraded". Anything other than
    UpstreamDegraded (schema errors, budget exhaustion, bugs) still propagates.

    The KB query is built from the *stripped* body: injected lines must not reach the search
    endpoint (they would poison the lexical ranking and end up in upstream logs).
    """
    ticket = ctx.ticket
    query = (ticket.subject + " " + strip_injection_blocks(ticket.body)) if ticket else ctx.ticket_id

    def _in_step(name: str, fn, *args):
        with tracer.step(name):
            return fn(*args)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "account": pool.submit(_in_step, "read_account", tools.read_account),
            "entitlements": pool.submit(_in_step, "read_entitlements", tools.read_entitlements),
            "kb_search": pool.submit(_in_step, "kb_search", tools.kb_search, query, 5),
        }
        errors: List[Exception] = []
        for endpoint, fut in futures.items():
            try:
                fut.result()
            except UpstreamDegraded as exc:  # timeout / 5xx after retries on account or kb -> degrade, not fail
                ctx.degraded_reads[endpoint] = exc.code
                tracer.event("upstream_degraded", endpoint=endpoint, code=exc.code)
            except Exception as exc:  # collect so every running thread finishes before we raise
                errors.append(exc)
                for pending in futures.values():
                    pending.cancel()
    if errors:
        raise errors[0]


def _finalise_reply(draft: Draft, write: Optional[WriteResult], ctx: Any, tracer: Any) -> Draft:
    """Post-write reply finalisation.

    WHY after the write: the draft may only *recommend* escalation (nothing has been filed
    when it is written). Once the write step reports filed / already-filed we append one
    confirmation sentence. It goes through the same leak scan as the draft; an id that looks
    like a leaked value is dropped from the sentence rather than quoted.
    """
    if write is None or write.status not in ("filed", "skipped_duplicate"):
        return draft
    ref = " (reference %s)" % write.escalation_id if write.escalation_id else ""
    sentence = " This escalation has now been filed%s." % ref
    if leak_hits(sentence, ctx):
        sentence = " This escalation has now been filed."
    draft.reply = (draft.reply or "").rstrip() + sentence
    tracer.event("reply_finalised", write_status=write.status)
    return draft


def run(
    ticket_id: str,
    cfg: Config,
    upstream: Any = None,
    llm: Any = None,
    tracer: Any = None,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> RunResult:
    t0 = time.monotonic()
    tracer = tracer or RunTracer(
        cfg.runs_dir, ticket_id=ticket_id, request_id=request_id, correlation_id=correlation_id,
        stdout=bool(getattr(cfg, "trace_stdout", False)), service=getattr(cfg, "service_name", "ic4-agent"),
    )
    budget = Budget(cfg.deadline_ms, cfg.max_llm_calls, cfg.max_tool_calls)
    ctx = TicketContext(ticket_id=ticket_id, budget=budget, tracer=tracer)
    upstream = upstream or HttpUpstream(
        cfg.upstream_base_url, cfg.read_timeout_s, cfg.write_timeout_s, tracer,
        retry_policy=RetryPolicy(max_retries=int(getattr(cfg, "max_retries", 3))),
        remaining_ms=budget.remaining_ms,
    )
    tools = Tools(upstream, ctx, dry_run=cfg.dry_run, kb_filter=getattr(cfg, "kb_filter", True))
    llm = llm or make_llm(cfg)
    if hasattr(llm, "tracer"):
        llm.tracer = tracer  # per-attempt spans nest under the draft step

    cls: Any = None
    decision: Any = None
    draft: Optional[Draft] = None
    usage = LLMUsage(0, 0, "none", 0.0)
    violations: List[str] = []
    write: Optional[WriteResult] = None
    outcome = "completed"
    error: Optional[str] = None
    guard_failed = False

    try:
        with tracer.step("read_ticket"):
            try:
                tools.read_ticket()
            except NotFound:
                raise _Abort("ticket_not_found")
            except UpstreamDegraded as exc:
                # Nothing can be done without the ticket; fail with the code, not a stack trace.
                tracer.event("upstream_degraded", endpoint="ticket", code=exc.code)
                raise _Abort("upstream_degraded:%s" % exc.code)

        with tracer.step("read_parallel"):
            _read_parallel(tools, ctx, tracer)

        with tracer.step("filter_kb"):
            if getattr(cfg, "kb_filter", True):
                tools.filter_kb()
            else:
                # Ablation: the model may cite anything the search returned; groundedness is scored the same way.
                ctx.kb_visible = list(ctx.kb_hits)
                tracer.event("kb_filter_disabled", hits=len(ctx.kb_hits))

        with tracer.step("classify_intent"):
            cls = classify(ctx.ticket, ctx.account, ctx.entitlements, ctx.entitlements_error, ctx.kb_visible)
            ctx.injection_suspected = cls.injection_suspected
            if cls.injection_suspected:
                tracer.event("injection_suspected", category=cls.category)
            if ctx.entitlements_error:
                tracer.event("entitlements_degraded", code=ctx.entitlements_error)

        with tracer.step("escalation_policy"):
            decision = decide(cls, ctx)
            tracer.event("policy_decided", escalate=decision.escalate, priority=decision.priority, sla_breach=decision.sla_breach)

        with tracer.step("draft"):
            prompt_ctx = build_prompt_ctx(ctx, cls, decision, cfg.prompt_version)
            router = getattr(llm, "router", None)
            if router is not None:
                prompt_ctx["model_id"] = router.pick(ctx)
            drafter = llm
            if budget.fraction_remaining() < 0.3:
                tracer.event("deadline_skip", step="draft", remaining_ms=budget.remaining_ms())
                drafter = RulesLLM()
            with tracer.call("llm draft") as attrs:
                try:
                    draft, usage = drafter.draft(prompt_ctx, budget)
                except (LLMError, BudgetExceeded) as exc:
                    # Type + safe code only: an LLMError message may carry an HTTP body that echoes the prompt.
                    tracer.event("llm_error", error=type(exc).__name__, code=getattr(exc, "code", None))
                    draft, usage = RulesLLM().draft(prompt_ctx, budget)
                    draft.source = "rules_fallback"
                if usage.model_id not in PRICE_PER_1K:
                    # No price on file: the cost is unknown, and "unknown" must never read as free.
                    tracer.event("cost_unknown_model", model_id=usage.model_id)
                    usage.cost_usd = None
                usage = attribute_cost(usage, draft)
                if isinstance(attrs, dict):
                    attrs.update({
                        "gen_ai.request.model": usage.model_id,
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                        "cost_usd": usage.cost_usd,
                        "cost_input_usd": usage.cost_input_usd,
                        "cost_output_usd": usage.cost_output_usd,
                        "cost_diagnosis_usd": usage.cost_diagnosis_usd,
                        "cost_reply_usd": usage.cost_reply_usd,
                        "ttft_ms": usage.ttft_ms,
                        "tbt_ms_avg": usage.tbt_ms_avg,
                        "tbt_ms_p95": usage.tbt_ms_p95,
                        "attempts": usage.attempts,
                        "retries": usage.retries,
                        "usage_estimated": usage.usage_estimated,
                    })

        with tracer.step("output_guard"):
            result = scan_output(draft, ctx)
            violations = list(result.violations)
            if not result.ok:
                tracer.event("guard_fallback", violations=violations, source=draft.source)
                draft, _ = RulesLLM().draft(prompt_ctx, budget)
                draft.source = "rules_fallback"
                violations = list(scan_output(draft, ctx).violations)
                if violations:
                    # Fail closed: the fallback is unsafe too, so no generated text leaves the run.
                    # The policy decision (and the write below) never depended on draft text.
                    tracer.event("guard_failed", violations=violations)
                    draft = Draft(
                        category=cls.category,
                        diagnosis=SAFE_DIAGNOSIS,
                        reply=SAFE_REPLY,
                        kb_cited=[],
                        escalate_recommended=bool(decision.escalate),
                        source="safe_stub",
                    )
                    guard_failed = True

        with tracer.step("write"):
            if decision.escalate and decision.intent is not None:
                write = tools.escalate(decision.intent)
            draft = _finalise_reply(draft, write, ctx, tracer)

        with tracer.step("summarize"):
            degraded = bool(ctx.entitlements_error) or bool(getattr(ctx, "degraded_reads", None))
            outcome = "degraded" if degraded else "completed"
            if guard_failed:
                outcome, error = "failed", "guard_failed"

    except _Abort as exc:
        outcome, error = "failed", exc.error
    except UpstreamDegraded as exc:  # e.g. a read that is not tolerated above; code only, never the detail
        outcome, error = "failed", "upstream_degraded:%s" % exc.code
    except UpstreamError as exc:  # 4xx on a read, invalid JSON, ...; the code is safe, the detail is not
        outcome, error = "failed", "upstream_error:%s" % exc.code
    except Exception as exc:  # any other failure still yields a result and a finished trace
        # Exception type only: messages can carry upstream detail or ticket text (never persisted).
        outcome, error = "failed", type(exc).__name__

    duration_ms = int((time.monotonic() - t0) * 1000)
    result = RunResult(
        run_id=getattr(tracer, "run_id", "") or "",
        ticket_id=ticket_id,
        category=cls.category if cls else "unknown",
        request_type=cls.request_type if cls else "question",
        confidence=cls.confidence if cls else 0.0,
        diagnosis=draft.diagnosis if draft else "",
        reply=draft.reply if draft else "",
        kb_cited=list(draft.kb_cited) if draft else [],
        escalate=bool(decision.escalate) if decision else False,
        escalate_recommended_by_model=bool(draft.escalate_recommended) if draft else False,
        priority=decision.priority if decision else "normal",
        policy_reasons=list(decision.reasons) if decision else [],
        write=write,
        injection_suspected=bool(cls.injection_suspected) if cls else False,
        entitlements_degraded=bool(ctx.entitlements_error),
        draft_source=draft.source if draft else "none",
        guard_violations=violations,
        usage=usage,
        duration_ms=duration_ms,
        outcome=outcome,
        error=error,
        kb_visible=[getattr(h, "id", "") for h in (ctx.kb_visible or [])],
        request_id=getattr(tracer, "request_id", "") or "",
        correlation_id=getattr(tracer, "correlation_id", "") or "",
    )
    try:
        tracer.finish(result, prompt_version=cfg.prompt_version, model_id=usage.model_id)
    except Exception:  # tracing must never turn a completed run into a failure
        pass
    return result
