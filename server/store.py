"""Loads a fixture variant from data/<variant>/ and serves lookups."""
import json
import os

VARIANTS = ("support", "access")
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class RecordNotFound(Exception):
    pass


class UpstreamError(Exception):
    """Raised when a record exists but cannot be served as a valid response."""


def _load(variant, name):
    path = os.path.join(DATA_ROOT, variant, name + ".json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Store:
    def __init__(self, variant="support"):
        if variant not in VARIANTS:
            raise ValueError("unknown variant %r; expected one of %s" % (variant, ", ".join(VARIANTS)))
        self.variant = variant
        self.tickets = {t["ticket_id"]: t for t in _load(variant, "tickets")}
        self.accounts = {a["account_id"]: a for a in _load(variant, "accounts")}
        self.entitlements = {e["account_id"]: e for e in _load(variant, "entitlements")}
        self.kb = _load(variant, "kb")
        self.escalations = []

    # ---- reads ----
    def ticket(self, ticket_id):
        try:
            return self.tickets[ticket_id]
        except KeyError:
            raise RecordNotFound(ticket_id)

    def ticket_ids(self):
        return sorted(self.tickets)

    def account(self, account_id):
        try:
            return self.accounts[account_id]
        except KeyError:
            raise RecordNotFound(account_id)

    def entitlements_for(self, account_id):
        if account_id not in self.accounts:
            raise RecordNotFound(account_id)
        record = self.entitlements.get(account_id)
        if record is None:
            raise RecordNotFound(account_id)
        return _validate_entitlements(record)

    # ---- write ----
    def file_escalation(self, payload):
        eid = "ESC-%04d" % (len(self.escalations) + 1)
        entry = dict(payload)
        entry["escalation_id"] = eid
        entry["status"] = "filed"
        self.escalations.append(entry)
        return entry


_REQUIRED = {
    "account_id": str,
    "support_tier": str,
    "seats": int,
    "features": list,
    "sla_hours": int,
}


def _validate_entitlements(record):
    """The entitlement service contract requires these types.

    Records written by the upstream provisioning system occasionally violate it;
    when that happens the service cannot produce a valid response body.
    """
    for field, expected in _REQUIRED.items():
        if field not in record:
            raise UpstreamError("entitlement record missing required field %r" % field)
        value = record[field]
        if expected is int and isinstance(value, bool):
            raise UpstreamError("field %r has non-numeric value %r" % (field, value))
        if not isinstance(value, expected):
            raise UpstreamError(
                "field %r expected %s, got %s (%r)" % (field, expected.__name__, type(value).__name__, value)
            )
    return record
