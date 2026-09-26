"""Deterministic ticket classifier.

WHY rules and not the model: the category drives the escalation policy and the write path,
so it must be reproducible, explainable (reasons) and immune to text injected into the
ticket. The model only drafts prose after the decision is made.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.guard import scan_input, strip_injection_blocks

# The 19 labelled categories of the support variant (the label vocabulary of the scored set).
CATEGORIES: List[str] = [
    "billing_proration",
    "credential_rotation",
    "rate_limit_increase",
    "region_unavailable",
    "export_stalled",
    "sso_metadata_mismatch",
    "audit_retention_limit",
    "entitlement_unknown",
    "sandbox_limit",
    "webhook_replay",
    "invoice_dispute",
    "deprecated_endpoint",
    "sla_expectation",
    "feature_not_entitled",
    "seat_billing",
    "auth_migration",
    "rate_limit_backoff",
    "identity_unverified",
    "insufficient_information",
]

REQUEST_TYPES = ("question", "how_to", "dispute", "change_request", "incident")

FEATURE_CUES: Dict[str, str] = {
    "sso": r"\bsso\b|single\s+sign[- ]?on",
    "audit_log": r"audit[ _-]?logs?\b|audit\s+endpoint",
    "private_networking": r"private[ _-]?networking",
    "data_export": r"data[ _-]?exports?\b",
    "priority_routing": r"priority[ _-]?routing",
    "custom_sla": r"custom[ _-]?sla",
}

_REGIONS = r"us-east|us-west|eu-west|eu-central|ap-south(?:east)?"

# (category, [(regex, weight, label)]) - presence-based so long bodies do not dominate.
_CUES: Dict[str, List[Tuple[str, float, str]]] = {
    "billing_proration": [
        (r"upgrad", 3, "upgrade"), (r"prorat", 3, "proration"), (r"plan price|list price", 2, "plan price"),
        (r"higher than|more than the plan", 2, "higher than plan"), (r"first invoice", 2, "first invoice"),
        (r"extra line", 2, "extra lines"), (r"mid[- ]cycle", 2, "mid-cycle"), (r"invoice", 1, "invoice"),
    ],
    "credential_rotation": [
        (r"\breset\b", 2, "reset"), (r"\btokens?\b", 2, "token"), (r"api key", 2, "api key"), (r"rotat", 3, "rotate"),
        (r"revoke", 2, "revoke"), (r"leak", 2, "leaked credential"), (r"credential", 2, "credential"),
        (r"locked out", 1, "locked out"), (r"scoped key", 2, "scoped key"),
    ],
    "rate_limit_increase": [
        (r"raise (the|our) limit", 4, "raise the limit"), (r"\bincrease", 3, "increase"), (r"higher limit", 3, "higher limit"),
        (r"\brpm\b", 1, "rpm"), (r"\b429s?\b", 1, "429"), (r"rate[- ]limit", 1, "rate limit"), (r"sustained", 1, "sustained"),
        (r"\bquota\b", 2, "quota"), (r"ceiling", 1, "ceiling"),
    ],
    "rate_limit_backoff": [
        (r"retry-after", 4, "Retry-After"), (r"back[- ]?off", 4, "backoff"), (r"\bretr(y|ies)\b", 2, "retry"),
        (r"\b429s?\b", 1, "429"), (r"rate[- ]limit", 1, "rate limit"),
    ],
    "export_stalled": [
        (r"export job", 3, "export job"), (r"\bexp-\d+", 3, "EXP job id"), (r"stall", 3, "stalled"),
        (r"running (for|since|about)", 2, "running long"), (r"\bexport", 1, "export"), (r"\bgb\b", 1, "size"),
        (r"take this long", 2, "duration question"), (r"\bhours?\b", 0.5, "hours"),
    ],
    "sso_metadata_mismatch": [
        (r"assertion", 3, "assertion error"), (r"metadata", 3, "metadata"), (r"identity provider|\bidp\b", 3, "IdP"),
        (r"certificate", 2, "certificate"), (r"sign[- ]?in|log[- ]?in", 1, "sign-in"), (r"\bsso\b", 1, "sso"),
        (r"stopped working", 1, "stopped working"),
    ],
    "deprecated_endpoint": [
        (r"deprecat", 3, "deprecated"), (r"going away", 3, "going away"), (r"retire", 3, "retired"), (r"sunset", 3, "sunset"),
        (r"\bv1\b", 2, "v1"), (r"/v1/", 2, "v1 path"), (r"endpoint", 1, "endpoint"), (r"timeline", 1, "timeline"),
    ],
    "region_unavailable": [
        (r"\bregion", 2, "region"), (r"\b(%s)\b" % _REGIONS, 2, "region name"), (r"queued", 3, "queued"),
        (r"not available|unavailable", 1, "not available"), (r"private[ _-]?networking", 1, "private networking"),
        (r"nothing happened", 1, "nothing happened"),
    ],
    "invoice_dispute": [
        (r"disput", 4, "dispute"), (r"not recogni[sz]e|unrecogni[sz]ed", 3, "unrecognised line"), (r"\bcharged\b", 2, "charged"),
        (r"charge on", 2, "charge on invoice"), (r"\bcredit\b", 1, "credit"), (r"refund", 1, "refund"),
        (r"still shows", 2, "still shows"), (r"\binv-\d+", 1, "invoice id"), (r"invoice", 1, "invoice"),
        (r"\bexpected\b", 1, "expected amount"), (r"\$\s?\d", 1, "amount"),
    ],
    "feature_not_entitled": [
        (r"\b403\b", 3, "403"), (r"on our plan", 2, "on our plan"), (r"included in", 2, "included"),
        (r"part of what we", 2, "part of plan"), (r"add-on", 2, "add-on"), (r"available to us", 2, "available to us"),
        (r"change plan|upgrade plan", 2, "change plan"), (r"do we (actually )?have", 2, "do we have"), (r"entitled", 1, "entitled"),
        (r"permissions? issue", 1, "permissions"), (r"sales (mentioned|said|promised)", 2, "sales mentioned"),
        (r"custom[ _-]?sla", 2, "custom sla"),
        (r"\bsso\b|audit[ _-]?log|private[ _-]?networking|data[ _-]?export|priority[ _-]?routing", 1, "feature named"),
    ],
    "seat_billing": [
        (r"\bseats?\b", 2, "seats"), (r"add(ing)? (\d+ )?seats", 2, "adding seats"), (r"mid-month", 2, "mid-month"),
        (r"full month", 2, "full month"), (r"\bbilled\b", 1, "billed"), (r"pro rata", 2, "pro rata"), (r"seat count", 1, "seat count"),
    ],
    "auth_migration": [
        (r"migrat", 3, "migration"), (r"unified auth", 2, "unified auth"), (r"legacy auth", 2, "legacy auth"),
        (r"old auth", 3, "old auth model"), (r"auth model", 2, "auth model"), (r"move to", 2, "move to"), (r"roll (it )?back", 1, "rollback"),
    ],
    "audit_retention_limit": [
        (r"\baudit", 2, "audit"), (r"retention", 3, "retention"), (r"\d+ months", 2, "months of history"), (r"90 days", 2, "90 days"),
        (r"goes back|how far back", 2, "history depth"), (r"last year", 2, "last year"), (r"compliance", 1, "compliance"),
        (r"auditor", 1, "auditor"),
    ],
    "webhook_replay": [
        (r"webhook", 3, "webhook"), (r"replay", 3, "replay"), (r"\bevents?\b", 1, "events"), (r"missed", 2, "missed"),
        (r"deliver", 1, "delivery"),
    ],
    "sla_expectation": [
        (r"first response", 3, "first response"), (r"response time", 3, "response time"), (r"\bsla\b", 2, "sla"),
        (r"\btarget\b", 2, "target"), (r"took .{0,24}hours", 2, "hours taken"), (r"premium", 1, "premium"),
        (r"slow response", 2, "slow response"),
    ],
    "sandbox_limit": [
        (r"sandbox", 4, "sandbox"), (r"test environment", 2, "test environment"), (r"capped", 1, "capped"),
        (r"reset every", 1, "reset cadence"),
    ],
    "identity_unverified": [
        (r"personal (email|address|account)", 3, "personal email/address"), (r"cannot access my work|can't access my work", 3, "no access to work account"),
        (r"not the account owner", 3, "not owner"), (r"on behalf of", 2, "on behalf of"), (r"billing contact", 2, "billing contact"),
        (r"new email", 1, "new email"),
    ],
    "insufficient_information": [
        (r"broken again|it is broken|it's broken", 2, "broken again"), (r"same (thing )?as last time", 3, "same as last time"),
        (r"just fix it", 2, "just fix it"), (r"not working|doesn't work|does not work", 1, "not working"),
    ],
    "entitlement_unknown": [],
}

# KB tag vocabulary per category; the visible hits nudge the score by rank.
_TAGS: Dict[str, set] = {
    "billing_proration": {"proration", "upgrade"},
    "credential_rotation": {"token", "rotate", "revoke", "credentials"},
    "rate_limit_increase": {"increase", "quota"},
    "rate_limit_backoff": {"backoff"},
    "export_stalled": {"stall", "delay", "job"},
    "sso_metadata_mismatch": {"saml", "metadata", "certificate", "signin"},
    "deprecated_endpoint": {"deprecated", "sunset", "endpoint"},
    "region_unavailable": {"region", "vpc", "networking"},
    "invoice_dispute": {"dispute", "refund", "charge"},
    "feature_not_entitled": {"feature", "entitlement", "tier"},
    "seat_billing": {"seat", "seats", "licence"},
    "auth_migration": {"migration"},
    "audit_retention_limit": {"retention", "compliance", "audit"},
    "webhook_replay": {"webhook", "replay", "delivery"},
    "sla_expectation": {"sla", "priority", "premium"},
    "sandbox_limit": {"sandbox", "test"},
    "identity_unverified": {"verification", "owner", "identity", "authorisation"},
    # scoring never uses this entry (entitlement_unknown is an account-fact category); it only
    # orders the visible articles so the human reviewer sees the entitlement/seat article first
    "entitlement_unknown": {"entitlement", "plan", "tier", "seat", "seats"},
}
_RANK_WEIGHTS = (1.5, 1.0, 0.5, 0.25, 0.25)

_PLAN_QUESTION = re.compile(
    r"\bplan\b|entitle|included|\bseats?\b|add-on|pay for|console|invoice|allowed|part of what", re.IGNORECASE
)
_CHANGE_SENSITIVE = re.compile(r"billing|refund|credential|token|password|contact|owner", re.IGNORECASE)
_UNVERIFIABLE = re.compile(
    r"personal (email|address|account)|cannot access my work|can't access my work|not the account owner|on behalf of",
    re.IGNORECASE,
)

_COMPILED_CUES = {
    cat: [(re.compile(p, re.IGNORECASE), w, label) for p, w, label in cues] for cat, cues in _CUES.items()
}
_COMPILED_FEATURES = {k: re.compile(p, re.IGNORECASE) for k, p in FEATURE_CUES.items()}


@dataclass
class Classification:
    category: str
    request_type: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    injection_suspected: bool = False
    # Word count of the body *before* injection lines were stripped. The policy uses it to tell
    # "the customer wrote nothing" from "the customer wrote something we removed".
    pre_strip_words: int = 0


def _request_type(text: str) -> str:
    low = text.lower()
    if re.search(r"disput|\bcharged\b|unrecogni|not recogni", low):
        return "dispute"
    if re.search(r"how (do|to|is|can) |procedure|walk us through|what is the (correct|fastest)", low):
        return "how_to"
    if re.search(r"\b(change|reset|update|refund|remove|revoke|rotate|raise the limit|increase)\b", low):
        return "change_request"
    if re.search(r"failing|broken|\b429|\b403|stall|stopped|not working|error|locked out|nothing happened", low):
        return "incident"
    return "question"


def features_mentioned(text: str) -> List[str]:
    return [f for f, pat in _COMPILED_FEATURES.items() if pat.search(text or "")]


def rank_kb_for_category(category: str, kb_visible: List[Any]) -> List[Any]:
    """Stable re-order of the visible articles by tag affinity to the category.

    WHY: the search endpoint ranks on lexical overlap only ("what we need" pulls the invoice
    dispute article to the top of a token-reset ticket). The first visible article is what the
    baseline cites and what the model sees first, so it should be the one about the category.
    """
    wanted = _TAGS.get(category, set())
    indexed = list(enumerate(kb_visible or []))
    indexed.sort(key=lambda p: (-len({str(t).lower() for t in (getattr(p[1], "tags", None) or [])} & wanted), p[0]))
    return [h for _, h in indexed]


def classify(
    ticket: Any,
    account: Any,
    entitlements: Any,
    entitlements_error: Optional[str],
    kb_visible: List[Any],
) -> Classification:
    subject = getattr(ticket, "subject", "") or ""
    raw_body = getattr(ticket, "body", "") or ""
    injection_suspected, patterns = scan_input(subject, raw_body)
    # Classify on the sanitized body so injected text cannot steer the category. The strip is
    # line-level and always runs: on a clean body it is the identity.
    body = strip_injection_blocks(raw_body)
    pre_strip_words = len(re.findall(r"\w+", raw_body))
    text = subject + "\n" + body

    scores: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
    hits: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for cat, cues in _COMPILED_CUES.items():
        for pat, weight, label in cues:
            if pat.search(text):
                scores[cat] += weight
                hits[cat].append(label)

    for rank, hit in enumerate((kb_visible or [])[: len(_RANK_WEIGHTS)]):
        tags = {str(t).lower() for t in (getattr(hit, "tags", None) or [])}
        for cat, wanted in _TAGS.items():
            if cat == "entitlement_unknown":
                continue
            if tags & wanted:
                scores[cat] += _RANK_WEIGHTS[rank]
                hits[cat].append("kb:%s" % getattr(hit, "id", "?"))

    reasons: List[str] = []
    features = getattr(entitlements, "features", None) if entitlements is not None else None
    asked = features_mentioned(text)

    # Account-fact adjustments.
    if entitlements_error and _PLAN_QUESTION.search(text):
        scores["entitlement_unknown"] += 10
        reasons.append("entitlements unavailable (%s) and ticket asks about plan/entitlements" % entitlements_error)
    if asked and features is not None:
        for feat in asked:
            if feat not in features:
                scores["feature_not_entitled"] += 3
                reasons.append("feature %s not in entitlements" % feat)
                if re.search(r"\b403\b", text):
                    scores["feature_not_entitled"] += 2
            else:
                reasons.append("feature %s is in entitlements" % feat)
    request_type = _request_type(text)
    if request_type == "change_request" and _CHANGE_SENSITIVE.search(text) and _UNVERIFIABLE.search(text):
        scores["identity_unverified"] += 4
        reasons.append("sensitive change requested from an unverifiable sender")

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], CATEGORIES.index(kv[0])))
    top_cat, top = ranked[0]
    words = len(re.findall(r"\w+", body))
    if top < 2 or (words < 12 and top < 3):
        scores["insufficient_information"] += 5
        reasons.append("no concrete signal in a %d-word body" % words)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], CATEGORIES.index(kv[0])))
        top_cat, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = 0.0 if top <= 0 else max(0.0, min(1.0, (top - second) / top))

    auth_model = getattr(account, "auth_model", None)
    if top_cat == "credential_rotation" and auth_model:
        reasons.append("auth_model=%s selects the %s credential article" % (auth_model, "legacy token" if auth_model == "legacy_auth" else "scoped key"))
    if injection_suspected:
        reasons.append("injection_suspected: %d pattern(s) matched; classified on sanitized body" % len(patterns))
    signal = ", ".join(hits[top_cat][:6]) or "none"
    reasons.insert(0, "%s signals: %s (score %.1f vs %.1f)" % (top_cat, signal, top, second))

    return Classification(
        category=top_cat,
        request_type=request_type,
        confidence=round(confidence, 3),
        reasons=reasons,
        injection_suspected=injection_suspected,
        pre_strip_words=pre_strip_words,
    )
