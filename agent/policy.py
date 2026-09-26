"""Escalation policy.

WHY policy runs before the draft: whether a ticket is escalated (the only write) is a
business decision made from the classification and tool facts. Running it first means the
model's prose can never change what gets written, and a model failure never blocks it.
The model's `escalate_recommended` is recorded only as a disagreement signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent.guard import strip_injection_blocks
from agent.redact import redact_for_trace as _redact
from agent.rules import Classification
from agent.schema import WriteIntent

ESCALATE_CATEGORIES = {
    "rate_limit_increase",
    "region_unavailable",
    "invoice_dispute",
    "identity_unverified",
    "insufficient_information",
    "entitlement_unknown",
}

# Cues that a region_unavailable ticket is about a request that is *stuck*, not a question.
_STUCK_CUES = re.compile(
    r"\b(queued|pending|still\s+showing|stuck|days?|weeks?|since|waiting|no\s+progress|not\s+moved)\b",
    re.IGNORECASE,
)


def region_unavailable_needs_human(text: str) -> bool:
    """True when a region_unavailable ticket describes a stuck or pending request.

    WHY only escalate the stuck case: the data has two kinds of region_unavailable ticket.
    One is a customer whose feature request has been sitting in a queue ("queued for three
    days", "still showing as queued") - the region cannot serve it and nobody upstream will
    unblock it without a human, so it escalates. The other is a pure availability question
    ("is private networking available in ap-south?") which kb-0008 answers completely; filing
    an escalation there creates a ticket for a human with nothing to do. The distinction is
    made on stuck/pending duration cues in the ticket text (queued, pending, still showing,
    days, week, stuck, since, waiting), not on the scored labels: the cues are what a support
    lead reads to decide whether someone has to go and push a request through.
    """
    return bool(_STUCK_CUES.search(text or ""))


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "forty-eight": 48,
}
_HOURS = re.compile(
    r"\b(\d+(?:\.\d+)?|%s)\s*-?\s*(hours?|hrs?|days?)\b" % "|".join(sorted(_NUM_WORDS, key=len, reverse=True)),
    re.IGNORECASE,
)


@dataclass
class PolicyDecision:
    escalate: bool
    reasons: List[str] = field(default_factory=list)
    priority: str = "normal"
    sla_breach: bool = False
    intent: Optional[WriteIntent] = None


# Reason recorded when a suspected injection left no classifiable request behind; the RulesLLM
# and the loop key their "human will review" handling off this prefix.
STRIPPED_REVIEW_REASON = "injection_suspected_content_stripped: routed to human review without write"
# A body shorter than this before stripping is genuinely empty; longer means content was removed.
STRIPPED_MIN_WORDS = 12
SUMMARY_SUBJECT_CHARS = 120


def hours_mentioned(text: str) -> Optional[float]:
    """Largest duration mentioned in the text, in hours (days are converted)."""
    best: Optional[float] = None
    for num, unit in _HOURS.findall(text or ""):
        low = num.lower()
        value = float(_NUM_WORDS[low]) if low in _NUM_WORDS else float(num)
        if unit.lower().startswith("day"):
            value *= 24
        best = value if best is None or value > best else best
    return best


def _summary(cls: Classification, ctx: Any, reasons: List[str], ent: Any) -> str:
    """Human-facing summary from ticket subject + category + tool facts only.

    WHY never from model text: the summary is written to a system of record. Anything the
    model produced could carry injected instructions or hallucinated commitments.
    """
    ticket = getattr(ctx, "ticket", None)
    # The subject is customer text too: strip injected lines, mask emails / long digit runs,
    # then bound the length so a long subject cannot pad the system of record.
    subject = _redact(strip_injection_blocks(getattr(ticket, "subject", "") or ""))[:SUMMARY_SUBJECT_CHARS]
    tier = getattr(ent, "support_tier", None) if ent is not None else None
    sla = getattr(ent, "sla_hours", None) if ent is not None else None
    parts = [
        "[%s/%s] %s" % (cls.category, cls.request_type, subject),
        "tier=%s" % (tier or "unknown"),
        "sla_hours=%s" % (sla if sla is not None else "unknown"),
        "entitlements_error=%s" % (getattr(ctx, "entitlements_error", None) or "none"),
    ]
    if cls.injection_suspected:
        parts.append("injection_suspected")
    parts.append("signals: " + "; ".join(reasons))
    return " | ".join(parts)


def decide(cls: Classification, ctx: Any) -> PolicyDecision:
    ent = getattr(ctx, "entitlements", None)
    err = getattr(ctx, "entitlements_error", None)
    ticket = getattr(ctx, "ticket", None)
    # Every text-derived signal below (stuck cues, SLA hours) reads the *stripped* text: an
    # injected "999 hours" line must not be able to manufacture an SLA breach.
    body = strip_injection_blocks(
        (getattr(ticket, "subject", "") or "") + "\n" + (getattr(ticket, "body", "") or "")
    )
    reasons: List[str] = []
    escalate = False
    sla_breach = False
    review_only = False

    if cls.category == "region_unavailable":
        # Conditional member of ESCALATE_CATEGORIES: see region_unavailable_needs_human.
        if region_unavailable_needs_human(body):
            escalate = True
            reasons.append("category:%s" % cls.category)
            reasons.append("stuck_request")
        else:
            reasons.append("availability_question:answered_from_kb")
    elif (
        cls.category == "insufficient_information"
        and cls.injection_suspected
        and getattr(cls, "pre_strip_words", 0) > STRIPPED_MIN_WORDS
    ):
        # The customer wrote a non-trivial message and the strip removed enough of it that
        # nothing classifiable is left. That is an attacker's text, not an empty ticket, so it
        # must not be able to force the one write we have: hand it to a human without filing.
        review_only = True
        reasons.append(STRIPPED_REVIEW_REASON)
    elif cls.category in ESCALATE_CATEGORIES:
        escalate = True
        reasons.append("category:%s" % cls.category)
    if err:
        escalate = True
        reasons.append("entitlements_unavailable:%s" % err)
    if cls.category == "sla_expectation" and ent is not None:
        sla_hours = getattr(ent, "sla_hours", None)
        hours = hours_mentioned(body)
        if isinstance(sla_hours, (int, float)) and hours is not None and hours > sla_hours:
            escalate = True
            sla_breach = True
            reasons.append("sla_breach:%gh>%gh" % (hours, sla_hours))
    if cls.injection_suspected:
        reasons.append("injection_suspected")

    tier = getattr(ent, "support_tier", None) if ent is not None else None
    if tier == "premium" or cls.injection_suspected or sla_breach or review_only:
        priority = "high"
    elif tier == "basic" and not escalate:
        priority = "low"
    else:
        priority = "normal"

    intent: Optional[WriteIntent] = None
    if escalate:
        # confirm=True is added by tools.escalate at the moment of the write, never here and
        # never by the model: the confirmation belongs to the code path that owns the write.
        intent = WriteIntent(
            ticket_id=getattr(ctx, "ticket_id", None) or getattr(ticket, "ticket_id", ""),
            reason=cls.category,
            summary=_summary(cls, ctx, reasons + list(cls.reasons), ent),
            priority=priority,
        )
    return PolicyDecision(escalate=escalate, reasons=reasons, priority=priority, sla_breach=sla_breach, intent=intent)
