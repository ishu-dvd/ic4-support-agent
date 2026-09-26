"""Whitelist views of upstream data for the prompt, and PII masking for the trace.

Everything here is *allow-list* based: a new upstream field never reaches the model until
someone adds it to a view on purpose.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .schema import Account, Entitlements, KbHit

# Values the guard must never see in a reply: internal entitlement fields and any email.
FORBIDDEN_REPLY_VALUES = ("rate_limit_rpm", "updated_at", "@")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONG_DIGITS_RE = re.compile(r"\d{6,}")


def filter_applies_to(hits: List[KbHit], account: Optional[Account]) -> List[KbHit]:
    """Keep a hit if `applies_to` is empty or every condition matches the account.

    The search endpoint ranks on text only, so a Legacy Auth token article outranks the
    Unified Auth one for a Unified Auth customer. Filtering here (not in the prompt) is what
    makes "cite only kb_visible" enforceable by the guard. An unknown condition key means we
    cannot prove applicability, so the article is dropped rather than guessed at.
    """
    visible: List[KbHit] = []
    for hit in hits:
        cond = hit.applies_to or {}
        if not cond:
            visible.append(hit)
            continue
        if account is None:
            continue
        ok = True
        for key, allowed in cond.items():
            value = getattr(account, key, None)
            if value is None or not isinstance(allowed, (list, tuple, set)) or value not in allowed:
                ok = False
                break
        if ok:
            visible.append(hit)
    return visible


def account_view(a: Optional[Account]) -> Dict[str, str]:
    """Only the fields the model needs to tailor a reply. Never name or primary_contact."""
    if a is None:
        return {}
    return {"plan_tier": a.plan_tier, "region": a.region, "auth_model": a.auth_model}


def entitlements_view(e: Optional[Entitlements], error: Optional[str]) -> Dict[str, object]:
    """Customer-facing entitlement facts, or an explicit "unavailable" marker.

    rate_limit_rpm / updated_at / the upstream `detail` string are internal and are excluded
    so a leaked value is a guard violation, not a judgement call.
    """
    if e is None:
        return {"unavailable": True, "reason": error or "unknown"}
    return {
        "support_tier": e.support_tier,
        "features": list(e.features),
        "sla_hours": e.sla_hours,
        "seats": e.seats,
    }


def kb_view(hits: List[KbHit]) -> List[Dict[str, str]]:
    return [{"id": h.id, "title": h.title, "body": h.body} for h in hits]


def redact_for_trace(text: str) -> str:
    """Mask emails and long digit runs (invoice ids, card fragments, phone numbers).

    Anything derived from the ticket body goes through this before being written to a trace.
    """
    if not text:
        return ""
    out = _EMAIL_RE.sub("<email>", text)
    out = _LONG_DIGITS_RE.sub("<num>", out)
    return out
