"""Draft generators: a deterministic template baseline and an OpenAI-compatible client.

WHY a rules baseline: it gives a zero-cost, always-available fallback for deadline and
guard failures, and a reference point for the eval so model gains are measured, not assumed.
"""
from __future__ import annotations

import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:  # Protocol exists on 3.8+; keep a plain-class fallback for exotic interpreters
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from agent.config import Config
from agent.rules import CATEGORIES, features_mentioned
from agent.upstream import RetryPolicy, decide_retry

# (input, output) USD per 1K tokens. Unknown model -> cost None ("unavailable", never $0) and the
# loop emits cost_unknown_model. The rules drafter is genuinely free: model_id "rules" -> 0.0.
# DigitalOcean Gradient serverless ids (openai-*, anthropic-*, ...) taken from GET /v2/gen-ai/models
# pricing on 2026-09-26 (per-token USD x 1000). Refresh with scripts/refresh_prices.py.
PRICE_PER_1K: Dict[str, Tuple[float, float]] = {
    # OpenAI direct
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    # DigitalOcean serverless inference (https://inference.do-ai.run/v1)
    "openai-gpt-4o-mini": (0.00015, 0.0006),
    "openai-gpt-4o": (0.0025, 0.01),
    "openai-gpt-4.1": (0.002, 0.008),
    "openai-gpt-5": (0.00125, 0.01),
    "openai-gpt-5-mini": (0.00025, 0.002),
    "openai-gpt-5-nano": (0.00005, 0.0004),
    "openai-gpt-5.4-mini": (0.00075, 0.0045),
    "openai-gpt-5.4-nano": (0.0002, 0.00125),
    "openai-gpt-5.6-luna": (0.0002, 0.0012),
    "openai-gpt-6-luna": (0.0001, 0.0005),
    "openai-gpt-oss-120b": (0.0001, 0.0007),
    "openai-gpt-oss-20b": (0.00005, 0.00045),
    "openai-o3-mini": (0.0011, 0.0044),
    "anthropic-claude-haiku-4.5": (0.001, 0.005),
    "anthropic-claude-4.5-sonnet": (0.003, 0.015),
    "anthropic-claude-5-sonnet": (0.002, 0.01),
    "gemma-4-31B-it": (0.00018, 0.0005),
    "mistral-3-14B": (0.0002, 0.0002),
    "nemotron-nano-12b-v2-vl": (0.0002, 0.0006),
    "rules": (0.0, 0.0),
    "none": (0.0, 0.0),
}


class LLMError(Exception):
    """Any failure of a model call (network, HTTP, parsing). The loop falls back to rules.

    `code` is a safe, stable reason (`http_429`, `timeout`, `bad_output`, ...) that may be traced;
    the message may carry response text and must not be."""

    def __init__(self, message: str = "", code: str = "error"):
        super().__init__(message or code)
        self.code = code


@dataclass
class Draft:
    category: str
    diagnosis: str
    reply: str
    kb_cited: List[str] = field(default_factory=list)
    escalate_recommended: bool = False
    source: str = "rules"  # "rules" | "llm" | "rules_fallback"


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "rules"
    cost_usd: Optional[float] = 0.0  # None = price unknown for this model (reported as "unavailable")
    # ---- timing (streaming) ----
    ttft_ms: Optional[int] = None  # time to first content token; None when not streamed / no model
    tbt_ms_avg: Optional[float] = None  # mean gap between consecutive content chunks
    tbt_ms_p50: Optional[float] = None
    tbt_ms_p95: Optional[float] = None
    stream_chunks: int = 0
    latency_ms: int = 0  # wall-clock of the successful attempt
    # ---- attempts ----
    attempts: int = 0  # 0 for the rules drafter (no model call)
    retries: int = 0
    usage_estimated: bool = False  # token counts estimated (provider sent no usage block)
    # ---- cost split (filled by attribute_cost) ----
    cost_input_usd: Optional[float] = None
    cost_output_usd: Optional[float] = None
    cost_diagnosis_usd: Optional[float] = None  # output cost attributed to the diagnosis text
    cost_reply_usd: Optional[float] = None  # output cost attributed to the customer reply


class LLM(Protocol):
    def draft(self, prompt_ctx: dict, budget: Any) -> Tuple[Draft, LLMUsage]: ...


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD for the call, or None when the model is not in PRICE_PER_1K: an unknown price must
    surface as "unavailable" in the eval, not as a flattering $0."""
    price = PRICE_PER_1K.get(model_id)
    if price is None:
        return None
    pin, pout = price
    return round(input_tokens / 1000.0 * pin + output_tokens / 1000.0 * pout, 6)


def attribute_cost(usage: LLMUsage, draft: Optional[Draft]) -> LLMUsage:
    """Split the call's cost into input / output, and the output part into diagnosis vs reply.

    One model call produces both texts, so the split is by character share of the output: honest
    as an *attribution* (it is labelled so on the dashboard), not a measurement. Input cost is the
    shared prompt and is reported separately rather than assigned to either. Unknown price -> all None.
    """
    price = PRICE_PER_1K.get(usage.model_id)
    if price is None or usage.cost_usd is None:
        usage.cost_input_usd = usage.cost_output_usd = None
        usage.cost_diagnosis_usd = usage.cost_reply_usd = None
        return usage
    pin, pout = price
    usage.cost_input_usd = round(usage.input_tokens / 1000.0 * pin, 6)
    usage.cost_output_usd = round(usage.output_tokens / 1000.0 * pout, 6)
    diag = len((draft.diagnosis or "")) if draft else 0
    reply = len((draft.reply or "")) if draft else 0
    total = diag + reply
    if total <= 0:
        usage.cost_diagnosis_usd = 0.0
        usage.cost_reply_usd = usage.cost_output_usd
    else:
        usage.cost_diagnosis_usd = round(usage.cost_output_usd * diag / total, 6)
        usage.cost_reply_usd = round(usage.cost_output_usd - usage.cost_diagnosis_usd, 6)
    return usage


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    rank = max(1, min(len(vals), int(math.ceil(p / 100.0 * len(vals)))))
    return round(vals[rank - 1], 2)


class ModelRouter:
    """Hook for per-ticket model selection. Default: always cfg.model_id.
    Subclass and override pick() to e.g. route premium tickets to a larger model."""

    def __init__(self, default_model_id: str):
        self.default_model_id = default_model_id

    def pick(self, ctx: Any = None) -> str:
        return self.default_model_id


# --------------------------------------------------------------------------- RulesLLM

_FEATURE_LABELS = {
    "sso": "SSO",
    "audit_log": "audit log",
    "private_networking": "private networking",
    "data_export": "data export",
    "priority_routing": "priority routing",
    "custom_sla": "custom SLA",
}

# category -> (diagnosis, reply body). Replies are written for a customer; no promises.
_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "billing_proration": (
        "First invoice after a mid-cycle upgrade; the difference is the proration credit and charge lines.",
        "The first invoice after an upgrade is prorated: you see a credit for the unused days on the previous plan "
        "followed by a charge for the new plan, so the total differs from the list price for that one invoice only. "
        "Subsequent invoices match the plan price.",
    ),
    "credential_rotation": (
        "Customer needs to rotate an API credential; procedure depends on the account auth model.",
        "Here is the safe procedure for rotating your API credential: create the replacement first, move your "
        "integration onto it, then revoke the old one. The linked article walks through each step for your account type.",
    ),
    "rate_limit_increase": (
        "Sustained 429s at the plan ceiling; customer asks for a higher limit. Needs capacity review.",
        "Sustained 429 responses across the whole window indicate the plan ceiling rather than a burst. Increases above "
        "the ceiling go through a capacity review; please keep the peak sustained requests per minute and the window you "
        "measured it over handy, as the reviewer will ask for them.",
    ),
    "region_unavailable": (
        "Private networking requested in a region where it is not yet offered; request stays queued.",
        "Private networking is available in a subset of regions today, and requests in other regions remain queued rather "
        "than failing. The linked article lists current availability by region.",
    ),
    "export_stalled": (
        "Export job running well past the expected duration; treat as stalled.",
        "Export jobs of this size normally finish within an hour, and a job that reports running for more than six hours "
        "has stalled. Please cancel the job and resubmit it; if the resubmitted job also stalls, reply here with the new "
        "job id and we will investigate the regional queue.",
    ),
    "sso_metadata_mismatch": (
        "IdP certificate rotated without updating metadata; assertion errors on sign-in.",
        "The assertion error after rotating your identity provider certificate means the metadata we hold no longer "
        "matches. Re-upload the current metadata, test sign-in with a single account, and only then re-enable enforcement "
        "for everyone.",
    ),
    "audit_retention_limit": (
        "Customer needs audit history beyond the retention window of their tier.",
        "Audit log retention depends on your support tier, and entries older than the retention window are no longer "
        "available to export. The linked article explains the retention periods and the NDJSON export option for the "
        "entries that are still within the window.",
    ),
    "entitlement_unknown": (
        "Entitlement lookup unavailable; the plan question cannot be answered from tool data.",
        "We could not verify your entitlement details at the moment, so we are not able to confirm what your plan "
        "includes in this reply. A support engineer will confirm your entitlements and follow up on this ticket.",
    ),
    "sandbox_limit": (
        "Sandbox environment limits apply regardless of plan.",
        "Sandbox environments have their own fixed limits and reset schedule that apply regardless of your plan, so the "
        "cap you are seeing there is expected and does not reflect your production limits. The linked article has the details.",
    ),
    "webhook_replay": (
        "Missed webhook deliveries; customer asks for a replay within the replay window.",
        "Failed webhook deliveries are retried automatically for about an hour, after which each event stays available "
        "for manual replay for a limited window. The linked article describes how to request a replay and how long events "
        "remain replayable.",
    ),
    "invoice_dispute": (
        "Disputed invoice line; needs the invoice id, disputed line and expected amount for review. Held from collections.",
        "We have logged the invoice line you do not recognise. To investigate we use the invoice id, the disputed "
        "line and the amount you expected, and the invoice is held from collections while it is under review. Please do "
        "not take any action on the invoice in the meantime.",
    ),
    "deprecated_endpoint": (
        "Customer depends on the v1 reporting endpoints, which have a published sunset date.",
        "The v1 reporting endpoints have a published retirement date and the v2 endpoints return the same fields under a "
        "paginated envelope. The linked article has the exact timeline and migration notes for your dashboard.",
    ),
    "sla_expectation": (
        "Customer asks about the first-response target after a slow response.",
        "We are sorry the first response took longer than you expected.",
    ),
    "feature_not_entitled": (
        "Customer asks whether a feature is available on their plan.",
        "Feature availability depends on your plan entitlements.",
    ),
    "seat_billing": (
        "Question about how seat changes are billed.",
        "Seat additions are billed pro rata from the day they are added, and seat removals take effect at the next "
        "renewal rather than being credited mid-cycle. The linked article covers both cases.",
    ),
    "auth_migration": (
        "Customer on Legacy Auth planning the migration to Unified Auth.",
        "Migration replaces account-wide tokens with scoped keys. Plan for a dual-running window: create scoped keys, "
        "cut traffic over, then disable the legacy token. Please note the migration cannot be reversed once legacy tokens "
        "are disabled, so keep the dual-running window until you are confident.",
    ),
    "rate_limit_backoff": (
        "Client retries immediately on 429 instead of honouring Retry-After.",
        "Each 429 response carries a Retry-After header in seconds. Your client should wait at least that long, adding "
        "jitter, before retrying. If 429s persist across a whole window that indicates the plan ceiling rather than a burst.",
    ),
    "identity_unverified": (
        "Billing/contact/refund change requested from an address we cannot verify as the account owner.",
        "Changes to billing details, contacts, entitlements or credentials require confirmation from a named account owner "
        "on record, so we cannot action this request from this address. A colleague will need to complete identity "
        "confirmation with the account owner before anything changes.",
    ),
    "insufficient_information": (
        "Ticket does not say what is broken; needs details before it can be routed.",
        "We want to get this fixed, but the ticket does not tell us which product area or error you are seeing. Could you "
        "reply with what stopped working, when it started, and any error message or id you have? A colleague will pick "
        "this up as soon as those details arrive.",
    ),
}

_DEGRADED_BODY = (
    "We could not verify your entitlement details at the moment, so we are not able to confirm plan-specific "
    "details in this reply. A support engineer will confirm them and follow up on this ticket."
)
# Diagnosis used whenever entitlements are degraded: it must not name a feature (guard rule
# feature_while_degraded now covers the diagnosis too), so the per-category text is not reused.
_DEGRADED_DIAGNOSIS = "Entitlement lookup unavailable; drafted for category %s without plan facts."

# The message could not be read as a support request once injected lines were removed (policy
# reason STRIPPED_REVIEW_REASON): say so, promise nothing, cite nothing.
_REVIEW_ONLY_BODY = (
    "We were not able to process this message automatically because parts of it could not be interpreted as a "
    "support request. A member of our team will review the original message and follow up on this ticket."
)
_REVIEW_ONLY_DIAGNOSIS = "Suspected injection left no classifiable request; routed to human review, no escalation filed."

# Escalation wording before the write: a recommendation, never a claim that something was filed.
# The loop appends the filed confirmation after the write step succeeds (see loop._finalise_reply).
ESCALATION_RECOMMENDED = "I have recommended this ticket for escalation to a specialist."


class RulesLLM:
    """Template drafts. escalate_recommended mirrors the policy decision so the baseline has zero disagreement."""

    def draft(self, prompt_ctx: dict, budget: Any = None) -> Tuple[Draft, LLMUsage]:
        category = prompt_ctx.get("category") or "insufficient_information"
        ents = prompt_ctx.get("entitlements") or {}
        degraded = bool(ents.get("unavailable"))
        tier = ents.get("support_tier") if not degraded else None
        sla_hours = ents.get("sla_hours") if not degraded else None
        features = ents.get("features") if not degraded else None
        kb = prompt_ctx.get("kb") or []
        escalate = bool(prompt_ctx.get("escalate"))
        ticket = prompt_ctx.get("ticket") or {}
        text = "%s %s" % (ticket.get("subject", ""), ticket.get("body", ""))
        review_only = any(
            str(r).startswith("injection_suspected_content_stripped") for r in (prompt_ctx.get("policy_reasons") or [])
        )

        if review_only:
            reply = "Thanks for contacting support. " + _REVIEW_ONLY_BODY
            draft = Draft(category=category, diagnosis=_REVIEW_ONLY_DIAGNOSIS, reply=reply, kb_cited=[],
                          escalate_recommended=escalate, source="rules")
            return draft, LLMUsage(0, 0, "rules", 0.0)

        diagnosis, body = _TEMPLATES.get(category, _TEMPLATES["insufficient_information"])
        if degraded:
            body = _DEGRADED_BODY
            if category != "entitlement_unknown":
                diagnosis = _DEGRADED_DIAGNOSIS % category
        elif category == "region_unavailable" and escalate:
            # Only a stuck request is handed on (policy decided); a pure availability question is answered from the KB.
            body += " We have passed your queued request on so you receive an update on the timeline for your region."
        elif category == "sla_expectation":
            if sla_hours is not None:
                body += " Your plan's first-response target is %s hours" % sla_hours
                body += ", which the response you describe exceeded." if prompt_ctx.get("sla_breach") else "."
            else:
                body += " The first-response target depends on your support tier; the linked article explains the tiers."
        elif category == "feature_not_entitled":
            asked = features_mentioned(text)
            if asked and features is not None:
                lines = []
                for feat in asked:
                    label = _FEATURE_LABELS.get(feat, feat)
                    label = label[0].upper() + label[1:]
                    if feat in features:
                        lines.append("%s is included in your current entitlements, so no plan change is needed to use it." % label)
                    else:
                        lines.append(
                            "%s is not included in your current plan entitlements, which is why access to it is refused. "
                            "Your account team can add it or discuss a plan change." % label
                        )
                body = " ".join(lines)
            else:
                body += " The linked article lists which features come with each tier."

        opening = {
            "premium": "As a premium support customer, your ticket has been placed in the priority queue.",
            "standard": "Thanks for reaching out.",
            "basic": "Thanks for reaching out.",
        }.get(tier or "", "Thanks for contacting support.")

        kb_cited: List[str] = []
        kb_line = ""
        if kb and category != "insufficient_information":  # nothing to ground when the ticket says nothing
            first = kb[0]
            kb_cited = [first.get("id")]
            kb_line = ("Reference: %s." % first.get("id")) if degraded else (
                "See \"%s\" (%s) for the full details." % (first.get("title", ""), first.get("id"))
            )

        footer = ""
        if escalate:
            footer = (
                ESCALATION_RECOMMENDED + " What happens next: a specialist reviews the details above and replies on "
                "this ticket"
            )
            footer += " within your plan's %s-hour first-response target." % sla_hours if sla_hours is not None else "."
            footer += " You do not need to open a new request."

        reply = " ".join(p for p in (opening, body, kb_line, footer) if p)
        draft = Draft(
            category=category,
            diagnosis=diagnosis,
            reply=reply,
            kb_cited=kb_cited,
            escalate_recommended=escalate,
            source="rules",
        )
        return draft, LLMUsage(0, 0, "rules", 0.0)


# ------------------------------------------------------------------ OpenAI-compatible

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


DRAFT_KEYS = frozenset({"category", "diagnosis", "reply", "kb_cited", "escalate_recommended"})


def _parse_draft(content: str, prompt_ctx: dict) -> Draft:
    """Strict parse of the model's JSON object into a Draft.

    Unknown keys are rejected (a model that invents fields is not following the schema and its
    other fields deserve no trust either); `escalate_recommended` is advisory and only a JSON
    `true` counts - the string "false" or any other truthy junk is False.
    """
    text = _FENCE.sub("", content or "").strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LLMError("model returned non-JSON content: %s" % exc, code="bad_output")
    if not isinstance(data, dict):
        raise LLMError("model JSON is not an object", code="bad_output")
    unknown = sorted(set(data) - DRAFT_KEYS)
    if unknown:
        raise LLMError("model JSON has unknown keys: %s" % ", ".join(unknown), code="bad_output")
    cited = data.get("kb_cited") or []
    if isinstance(cited, str):
        cited = [cited]
    if not isinstance(cited, list):
        raise LLMError("kb_cited must be a list of ids", code="bad_output")
    category = str(data.get("category") or prompt_ctx.get("category") or "")
    if category not in CATEGORIES:
        category = prompt_ctx.get("category") or category
    escalate = data.get("escalate_recommended", False)
    return Draft(
        category=category,
        diagnosis=str(data.get("diagnosis") or ""),
        reply=str(data.get("reply") or ""),
        kb_cited=[str(c) for c in cited],
        escalate_recommended=escalate is True,
        source="llm",
    )


class _NoopSpan:
    """Stand-in tracer for an LLM used outside the loop (unit tests, scripts)."""

    def call(self, name: str, **attrs: Any):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield dict(attrs)

        return _cm()

    def event(self, name: str, **attrs: Any) -> None:
        return None


@dataclass
class _Attempt:
    """One HTTP attempt against the model: what came back and how fast."""
    content: str = ""
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    usage_present: bool = False
    ttft_ms: Optional[int] = None
    gaps_ms: List[float] = field(default_factory=list)
    chunks: int = 0
    latency_ms: int = 0
    status: int = 200


def _iter_sse(resp) -> "Any":
    """Yield the JSON object of every `data:` line of an SSE stream; stop at [DONE]."""
    for raw_line in resp:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


class OpenAICompatibleLLM:
    """POST {base_url}/chat/completions, streamed by default so time-to-first-token and the gaps
    between tokens are measured, with status-based retries (default 3) shared with the HTTP client.

    Retry decisions: 429 / 5xx / timeout / connection error -> retry with backoff; 400/401/403/404 ->
    stop (our request or credentials are wrong); model output that is not the JSON schema -> one
    retry (`bad_output`), then stop. Every attempt is a `llm chat.completions` call span with an
    `attempt` attribute, every retry a `llm_retry` event; the loop's `llm draft` span is the total.
    """

    def __init__(
        self,
        cfg: Config,
        router: Optional[ModelRouter] = None,
        timeout_s: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        stream: Optional[bool] = None,
    ):
        self.cfg = cfg
        self.router = router or ModelRouter(cfg.model_id)
        self.timeout_s = float(timeout_s if timeout_s is not None else getattr(cfg, "llm_timeout_s", 30.0))
        self.retry_policy = retry_policy or RetryPolicy(max_retries=int(getattr(cfg, "llm_max_retries", 3)))
        self.stream = bool(getattr(cfg, "llm_stream", True) if stream is None else stream)
        self.bad_output_retries = 1
        self.tracer: Any = None  # the loop sets this so attempts nest under the current step

    # ---- one attempt -------------------------------------------------------------------------------
    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            self.cfg.openai_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
                "Authorization": "Bearer %s" % self.cfg.openai_api_key,
            },
            method="POST",
        )

    def _attempt(self, payload: dict, model: str) -> _Attempt:
        out = _Attempt(model_id=model)
        t0 = time.perf_counter()
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout_s) as resp:
            out.status = resp.status
            if payload.get("stream"):
                parts: List[str] = []
                last_t: Optional[float] = None
                for obj in _iter_sse(resp):
                    if obj.get("model"):
                        out.model_id = str(obj["model"])
                    usage = obj.get("usage")
                    if isinstance(usage, dict) and usage:
                        out.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        out.output_tokens = int(usage.get("completion_tokens", 0) or 0)
                        out.usage_present = True
                    for choice in obj.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if not delta:
                            continue
                        now = time.perf_counter()
                        if out.ttft_ms is None:
                            out.ttft_ms = int(round((now - t0) * 1000))
                        elif last_t is not None:
                            out.gaps_ms.append((now - last_t) * 1000.0)
                        last_t = now
                        out.chunks += 1
                        parts.append(delta)
                out.content = "".join(parts)
            else:
                raw = json.loads(resp.read().decode("utf-8"))
                out.content = str(raw["choices"][0]["message"]["content"])
                out.model_id = str(raw.get("model") or model)
                usage = raw.get("usage") or {}
                if usage:
                    out.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    out.output_tokens = int(usage.get("completion_tokens", 0) or 0)
                    out.usage_present = True
                out.ttft_ms = int(round((time.perf_counter() - t0) * 1000))  # whole response = first token
        out.latency_ms = int(round((time.perf_counter() - t0) * 1000))
        return out

    # ---- the public call ---------------------------------------------------------------------------
    def draft(self, prompt_ctx: dict, budget: Any) -> Tuple[Draft, LLMUsage]:
        if budget is not None:
            budget.charge_llm()  # raises BudgetExceeded before we spend money
        tracer = self.tracer or _NoopSpan()
        model = prompt_ctx.get("model_id") or self.router.pick(None)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_ctx.get("system", "")},
                {"role": "user", "content": prompt_ctx.get("user", "")},
            ],
            "temperature": 0,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        prompt_chars = len(payload["messages"][0]["content"]) + len(payload["messages"][1]["content"])

        policy = self.retry_policy
        attempt = 0
        bad_outputs = 0
        while True:
            status: Optional[int] = None
            exc_kind: Optional[str] = None
            code = "error"
            result: Optional[_Attempt] = None
            with tracer.call("llm chat.completions", attempt=attempt) as span:
                span["gen_ai.request.model"] = model
                span["gen_ai.request.stream"] = bool(payload.get("stream"))
                try:
                    result = self._attempt(payload, model)
                    span["http.status"] = result.status
                    span["gen_ai.usage.input_tokens"] = result.input_tokens
                    span["gen_ai.usage.output_tokens"] = result.output_tokens
                    span["ttft_ms"] = result.ttft_ms
                    try:
                        draft = _parse_draft(result.content, prompt_ctx)
                    except LLMError as exc:
                        bad_outputs += 1
                        code = exc.code
                        span["error"] = code
                        if bad_outputs > self.bad_output_retries or attempt >= policy.max_retries:
                            span["retry_decision"] = "stop:bad_output"
                            raise LLMError("model output rejected", code=code)
                        span["retry_decision"] = "retry:bad_output"
                        tracer.event("llm_retry", attempt=attempt + 1, reason="retry:bad_output", code=code)
                        attempt += 1
                        continue
                    in_tok, out_tok = result.input_tokens, result.output_tokens
                    estimated = False
                    if not result.usage_present:
                        # Provider sent no usage block: estimate so cost is never silently zero.
                        in_tok = max(1, prompt_chars // 4)
                        out_tok = max(result.chunks, len(result.content) // 4, 1)
                        estimated = True
                    model_id = result.model_id or model
                    price_key = model_id if model_id in PRICE_PER_1K else model
                    usage = LLMUsage(
                        in_tok, out_tok, model_id, cost_usd(price_key, in_tok, out_tok),
                        ttft_ms=result.ttft_ms,
                        tbt_ms_avg=(round(sum(result.gaps_ms) / len(result.gaps_ms), 2) if result.gaps_ms else None),
                        tbt_ms_p50=_percentile(result.gaps_ms, 50),
                        tbt_ms_p95=_percentile(result.gaps_ms, 95),
                        stream_chunks=result.chunks,
                        latency_ms=result.latency_ms,
                        attempts=attempt + 1,
                        retries=attempt,
                        usage_estimated=estimated,
                    )
                    span["cost_usd"] = usage.cost_usd
                    return draft, usage
                except LLMError:
                    raise
                except urllib.error.HTTPError as err:
                    status = err.code
                    code = "http_%d" % err.code
                    span["http.status"] = err.code
                    span["error"] = code
                except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                    reason_obj = getattr(exc, "reason", exc)
                    exc_kind = "timeout" if isinstance(exc, (socket.timeout, TimeoutError)) or isinstance(
                        reason_obj, (socket.timeout, TimeoutError)) else "unreachable"
                    code = exc_kind
                    span["http.status"] = None
                    span["error"] = code
                except Exception as exc:  # shape errors etc. - type name only, never the body
                    code = type(exc).__name__
                    span["error"] = code
                    span["retry_decision"] = "stop:%s" % code
                    raise LLMError(code, code=code)
                backoff = policy.backoff_s(attempt)
                remaining = budget.remaining_ms() if budget is not None and hasattr(budget, "remaining_ms") else None
                do_retry, reason = decide_retry(
                    policy, attempt, policy.max_retries, status=status, exc_kind=exc_kind,
                    remaining_ms=remaining, next_cost_ms=backoff * 1000.0,
                )
                span["retry_decision"] = reason
            if not do_retry:
                tracer.event("llm_retries_exhausted", attempts=attempt + 1, reason=reason, code=code)
                raise LLMError(code, code=code)
            tracer.event("llm_retry", attempt=attempt + 1, reason=reason, code=code, backoff_ms=int(backoff * 1000))
            time.sleep(backoff)
            attempt += 1


def make_llm(cfg: Config) -> Any:
    if cfg.model_provider == "openai_compatible":
        return OpenAICompatibleLLM(cfg, ModelRouter(cfg.model_id))
    return RulesLLM()
