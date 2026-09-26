"""Input and output guard for the decision plane.

WHY: the ticket body is untrusted data. A requester can type text that tries to steer the
model (prompt injection), and the model can emit text that leaks tool facts or makes
commitments only a human may make. Both edges are checked deterministically here so the
policy, the write path and the trace never depend on the model behaving.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
    r"(administrator|admin|developer|root|god)\s+mode",
    r"you\s+are\s+now\s+(an?\s+)?(administrator|admin|developer|system)",
    r"system\s+prompt",
    r"has\s+been\s+approved",
    r"\bas\s+an\s+ai\b",
    r"\bsev\s*-?\s*1\b",
    r"internal\s+(account\s+)?notes",
    r"internal\s+pricing",
    r"new\s+instructions?\s*:",
    r"act\s+as\s+(an?\s+)?(administrator|admin|system|developer)",
    r"do\s+not\s+(tell|inform)\s+the\s+(user|customer|agent)",
    r"\bjailbreak\b",
    r"override\s+(the\s+)?(policy|policies|instructions|safety|guardrails)",
    r"reveal\s+(the\s+|your\s+)?(hidden|secret|internal|system)",
    # cross-account exfiltration: asking the agent to disclose another account's record or contact
    r"\b(show|give|send|share|confirm|paste|reveal|list|tell)\b[^.\n]{0,80}\baccount\s+acct_[a-z0-9]+",
    r"\b(sister|other|another|different|their)\s+account'?s?\s+(entitlements?|features?|rate\s+limits?|contacts?|plan|details|data|record)",
    r"\b(their|the\s+other\s+account'?s?)\s+primary\s+contact",
    # bypassing the write gate or dumping raw upstream records
    r"skip\s+(the\s+)?confirmation",
    r"raw\s+json\s+of",
    # steering the model's grounding: override claims and instructions to cite a specific article
    r"system\s+override",
    r"\bcite\s+(only\s+|exclusively\s+)?(the\s+)?(article\s+)?kb-\d+",
    r"you\s+must\s+cite",
    r"ignore\s+(the\s+)?applies[_ ]to",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_BLOCK_DELIM_LINE = re.compile(r"^[ \t]*-{3,}[ \t]*$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Money-commitment detector: a commitment verb within ~6 tokens of a money noun, in either
# order ("approved a refund", "money will be returned", "greenlit the reimbursement").
_MONEY_VERB = (
    r"(?:approved?|approving|approves|grant(?:ed|s)?|issued?|issuing|issues|appl(?:y|ied|ies)|"
    r"return(?:ed|s)?|reimburs(?:e|ed|es)|refund(?:ed|s)?|credit(?:ed|s)?|greenlit|greenlight(?:ed)?|"
    r"authori[sz]ed?|process(?:ed)?|waiv(?:e|ed))"
)
_MONEY_NOUN = r"(?:refunds?|credits?|reimbursements?|money|charge\s+reversal|chargeback|compensation)"
_NEAR = r"(?:\W+\w+){0,6}?\W+"
_MONEY_COMMITMENT = r"\b%s\b%s%s\b|\b%s\b%s%s\b" % (_MONEY_VERB, _NEAR, _MONEY_NOUN, _MONEY_NOUN, _NEAR, _MONEY_VERB)

# Phrases the agent is never allowed to state: they are commitments (money, severity,
# queue position) that belong to a human, or disclosures the injection asked for.
# Public so the eval scores the injection case with the same patterns (never a shorter copy).
FORBIDDEN_COMMITMENT_PATTERNS: List[Tuple[str, str]] = [
    ("refund approved", r"refund\s+(has\s+been\s+|is\s+|was\s+)?approved"),
    ("credit issued", r"credit\s+(has\s+been\s+|is\s+|was\s+)?(issued|applied)"),
    ("money commitment", _MONEY_COMMITMENT),
    ("we will refund", r"\bwe\s+(will|shall|are\s+going\s+to|have|'ll)\s+(refund|credit|reimburse|return\s+(the|your)\s+money)"),
    ("approval claim", r"\b(has|have|is|are|was|were)\s+(been\s+)?approved\b"),
    ("sev1", r"\bsev\s*-?\s*1\b"),
    ("severity claim", r"\b(sev|severity)\s*-?\s*[2-4]\b|\b(as|to)\s+p[0-3]\b"),
    ("priority raised", r"priority\s+(has\s+been\s+|was\s+|is\s+)?(raised|escalated|bumped|increased|set\s+to\s+(high|urgent|critical))"),
    ("administrator mode", r"(administrator|admin)\s+mode"),
    ("internal pricing", r"internal\s+pricing"),
    ("internal account notes", r"internal\s+(account\s+)?notes"),
]
_FORBIDDEN_COMPILED = [(k, re.compile(p, re.IGNORECASE)) for k, p in FORBIDDEN_COMMITMENT_PATTERNS]

# Feature names that must not be asserted while the entitlement service is degraded.
FEATURE_PATTERNS = {
    "sso": r"\bsso\b|single\s+sign[- ]?on",
    "audit_log": r"audit[ _-]?logs?\b",
    "private_networking": r"private[ _-]?networking",
    "data_export": r"data[ _-]?exports?\b",
    "priority_routing": r"priority[ _-]?routing",
    "custom_sla": r"custom[ _-]?sla",
}
_FEATURE_COMPILED = {k: re.compile(p, re.IGNORECASE) for k, p in FEATURE_PATTERNS.items()}


@dataclass
class GuardResult:
    ok: bool
    violations: List[str] = field(default_factory=list)


def _matches(text: str) -> List[str]:
    return [p.pattern for p in _COMPILED if p.search(text or "")]


def scan_input(subject: str, body: str) -> Tuple[bool, List[str]]:
    """Return (suspected, matched_patterns) over subject and body."""
    matched = _matches((subject or "") + "\n" + (body or ""))
    return (bool(matched), matched)


def strip_injection_blocks(body: str) -> str:
    """Remove the lines that match an injection pattern, plus the bare `---` delimiter lines
    that fence such a line. Every other line is kept verbatim.

    WHY line-level and not whole fences: an attacker who fences the customer's real request
    together with one injected line must not be able to delete the request (an empty body
    classifies as insufficient_information, which used to escalate - see notes on B7). The
    model and the rules classifier should see exactly the customer's text minus the part
    written for the agent, nothing more and nothing less.
    """
    if not body:
        return body or ""
    lines = body.splitlines()
    delim_idx = [i for i, ln in enumerate(lines) if _BLOCK_DELIM_LINE.match(ln)]
    delims = set(delim_idx)
    drop = {i for i, ln in enumerate(lines) if i not in delims and _matches(ln)}
    # pair delimiters in order; an unpaired last delimiter fences to the end of the text
    bounds = list(zip(delim_idx[0::2], delim_idx[1::2]))
    if len(delim_idx) % 2 == 1:
        bounds.append((delim_idx[-1], len(lines)))
    for start, end in bounds:
        if any(i in drop for i in range(start + 1, end)):
            drop.add(start)
            if end < len(lines):
                drop.add(end)
    text = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str) -> str:
    return re.sub(r"[_-]+", " ", (text or "").lower())


def forbidden_commitment_hits(text: str) -> List[str]:
    """Keys of FORBIDDEN_COMMITMENT_PATTERNS found in `text` (order preserved, no duplicates)."""
    return [key for key, pat in _FORBIDDEN_COMPILED if pat.search(text or "")]


def feature_mentions(text: str) -> List[str]:
    """Feature keys named in `text` (used for the feature-while-degraded rule, in guard and eval)."""
    low = _norm(text)
    return [feat for feat, pat in _FEATURE_COMPILED.items() if pat.search(low)]


def leak_hits(text: str, ctx: Any) -> List[str]:
    """Value leaks in `text` against the context: contact email / any email, rate_limit_rpm
    (name or value), updated_at (name or value). Shared by scan_output and the post-write
    reply finalisation in the loop."""
    text = text or ""
    found: List[str] = []
    account = getattr(ctx, "account", None)
    contact = getattr(account, "primary_contact", None) or {}
    contact_email = str(contact.get("email", "")) if isinstance(contact, dict) else ""
    if (contact_email and contact_email.lower() in text.lower()) or _EMAIL.search(text):
        found.append("leak:email")
    ent = getattr(ctx, "entitlements", None)
    rpm = str(getattr(ent, "rate_limit_rpm", "")) if ent is not None else ""
    if "rate_limit_rpm" in text or (rpm and re.search(r"\b%s\b" % re.escape(rpm), text)):
        found.append("leak:rate_limit_rpm")
    upd = str(getattr(ent, "updated_at", "") or "") if ent is not None else ""
    if "updated_at" in text or (upd and upd in text):
        found.append("leak:updated_at")
    return found


def scan_output(draft: Any, ctx: Any) -> GuardResult:
    """Check a Draft against the context it was produced from.

    Violations are stable strings so the trace and eval can count them:
    kb_not_visible:<id>, forbidden_commitment:<phrase>, leak:<what>, feature_while_degraded:<feature>.
    """
    violations: List[str] = []
    reply = getattr(draft, "reply", "") or ""
    diagnosis = getattr(draft, "diagnosis", "") or ""
    combined = reply + "\n" + diagnosis

    visible_ids = {getattr(h, "id", None) for h in (getattr(ctx, "kb_visible", None) or [])}
    for kb_id in getattr(draft, "kb_cited", None) or []:
        if kb_id not in visible_ids:
            violations.append("kb_not_visible:%s" % kb_id)

    for key in forbidden_commitment_hits(combined):
        violations.append("forbidden_commitment:%s" % key)

    violations.extend(leak_hits(combined, ctx))

    # Upstream error codes are internal; the customer reply must never echo them. (The detail
    # string never reaches the context at all: tools.read_entitlements stores the code only.)
    err_code: Optional[str] = getattr(ctx, "entitlements_error", None)
    if err_code:
        if err_code in reply:
            violations.append("leak:upstream_detail")
        # Feature names are asserted from entitlements; with the service degraded neither the
        # reply nor the engineer-facing diagnosis may name one as a fact.
        for feat in feature_mentions(combined):
            violations.append("feature_while_degraded:%s" % feat)

    # de-duplicate, keep order
    seen = set()
    uniq = [v for v in violations if not (v in seen or seen.add(v))]
    return GuardResult(ok=not uniq, violations=uniq)
