"""Everything the run knows about one ticket, in one place.

Tools write into it, policy/guard/llm read from it. There is no account_id parameter on any
tool because the account is *derived* from the ticket: that closes the door on a prompt
injection asking the model to "also look up acct_1009".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import Budget
from .schema import Account, Entitlements, KbHit, Ticket


@dataclass
class TicketContext:
    ticket_id: str
    budget: Budget
    tracer: Any  # RunTracer from slice C or a no-op with the same interface
    account_id: Optional[str] = None
    ticket: Optional[Ticket] = None
    account: Optional[Account] = None
    entitlements: Optional[Entitlements] = None  # None when degraded
    entitlements_error: Optional[str] = None  # code when degraded, e.g. "entitlement_service_error"
    kb_hits: List[KbHit] = field(default_factory=list)  # raw search results
    kb_visible: List[KbHit] = field(default_factory=list)  # after applies_to filter -> what the model may cite
    unknown_fields: Dict[str, List[str]] = field(default_factory=dict)  # endpoint -> dropped fields
    injection_suspected: bool = False
    degraded_reads: Dict[str, str] = field(default_factory=dict)  # endpoint -> error code for account/kb reads that timed out
