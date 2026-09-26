"""Strict dataclasses for everything that crosses the HTTP boundary.

Why strict: the upstream may grow fields at any time. We never want an unknown field to
silently flow into a prompt or a reply, so `parse` drops unknowns and *reports* them; the
caller records them in the trace instead of pretending they do not exist. Missing fields
raise rather than default, because a half-populated Ticket would let the model reason on
data that is not there.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class SchemaError(Exception):
    """A required field is missing from an upstream payload. `args[0]` is the field name."""

    def __init__(self, field_name: str):
        super().__init__(field_name)
        self.field = field_name


def parse(cls: Type[T], d: Dict[str, Any]) -> Tuple[T, List[str]]:
    """Build `cls` from a raw dict. Returns (obj, unknown_keys_dropped).

    No coercion is done on purpose: if the upstream sends seats="unlimited" that is the
    upstream's contract violation and it should surface as-is, not be papered over here.
    """
    if not isinstance(d, dict):
        raise SchemaError("<root>")
    known = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in d:
            kwargs[f.name] = d[f.name]
        elif f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
            raise SchemaError(f.name)
    unknown = [k for k in d.keys() if k not in known]
    return cls(**kwargs), unknown


class _FromApi:
    """Mixin so every model has the `from_api` constructor named in SPEC."""

    @classmethod
    def from_api(cls, d: Dict[str, Any]):
        return parse(cls, d)


@dataclass
class Ticket(_FromApi):
    ticket_id: str
    account_id: str
    subject: str
    body: str
    channel: str
    opened_at: str
    status: str


@dataclass
class Account(_FromApi):
    account_id: str
    name: str
    plan_tier: str
    region: str
    auth_model: str
    customer_since: str
    primary_contact: dict


@dataclass
class Entitlements(_FromApi):
    account_id: str
    support_tier: str
    seats: int
    features: List[str]
    sla_hours: int
    rate_limit_rpm: int
    updated_at: str


@dataclass
class KbHit(_FromApi):
    id: str
    title: str
    body: str
    tags: List[str]
    score: float
    applies_to: dict


# The scored-set row type deliberately does not live here: the agent package must not know the
# shape of the labels it is scored against (see tests/test_no_*_access.py and scripts/eval_agent.py).


@dataclass
class WriteIntent:
    """Produced only by policy (slice B). Tools never invent one."""

    ticket_id: str
    reason: str
    summary: str
    priority: str


@dataclass
class WriteResult:
    status: str  # pending_manual | skipped_dry_run | skipped_duplicate | filed | rejected
    escalation_id: Optional[str] = None
    error: Optional[str] = None
