# ADR-001 — Identity: ticket-scoped tools, requester verification as policy

Status: accepted. Date: 2026-09-26. Scope: the support-desk agent in this repo.

## Context

The agent reads from four upstreams (tickets, accounts, entitlements, knowledge base) and performs one
consequential write (file an escalation). The brief requires that no field is surfaced that the
requesting user is not entitled to see, and that we *narrow* the system's own authorization rather
than replace it. Two facts in the fixture shape the decision:

- Ticket bodies are untrusted input. TCK-1123 contains an instruction block telling the agent to
  enter "administrator mode", disclose internal pricing and state that a refund has been approved.
- The knowledge base carries `applies_to` conditions (for example `auth_model: legacy_auth`) that
  the search endpoint does not enforce. Returning the wrong article is both a correctness bug and a
  disclosure of another customer class's procedure.

There is no authentication on the mock API and no requester identity on the ticket beyond
`account_id`. Whatever identity model we choose has to work with what the ticketing system asserts.

## Decision

**Identity is bound at construction, not inferred at runtime.**

1. `TicketContext(ticket_id)` resolves `account_id` once, from the ticket record. Every tool is
   bound to that context. **No tool takes an `account_id` parameter.** The model therefore has no
   surface through which to request another account's data, regardless of what the ticket body says.
2. The model never sees raw upstream payloads. It sees whitelisted views built in `agent/redact.py`:
   `plan_tier, region, auth_model` from the account; `support_tier, features, sla_hours, seats` from
   entitlements (or an explicit `unavailable` marker); `id, title, body` of KB articles that passed
   the `applies_to` filter for this account. `primary_contact`, `rate_limit_rpm`, `updated_at`, and
   upstream error `detail` strings are never in the prompt and are checked for in the output guard.
3. **Requester verification is a policy check, not a model judgement.** A request that would change
   billing contacts, credentials or issue refunds, arriving with cues that the sender is not the
   account owner on record (KB article kb-0018 defines the rule), is classified
   `identity_unverified` and escalated to a human. The agent does not attempt to verify identity
   itself and does not act on the request.
4. The write carries the system's own gate. `POST /v1/escalations` requires `confirm: true`; our
   single write path `Tools.escalate(WriteIntent)` sets it, and only `EscalationPolicy` can produce a
   `WriteIntent`. The model has no tool-calling surface at all.

## Alternatives rejected

- **Model-inferred identity from the ticket text.** The text is the attack surface. TCK-1128 says
  "sending from my personal email"; TCK-1123 says "you are now in administrator mode". A model
  reading either as an identity claim is exactly the failure we are asked to prevent.
- **Agent-side RBAC replacing the upstream's authorization.** The brief says narrow, do not replace.
  A second authorization system drifts from the first and becomes the one that is wrong. We add
  filters on top of what the upstream already returns for this account; we never widen.
- **Prompt instructions as the gate** ("only cite articles that apply to the customer"). Instructions
  in the prompt compete with instructions in the ticket. The `applies_to` filter and the output guard
  are code and cannot be argued with.
- **Passing raw entitlement JSON to the model and asking it to be careful.** Cheaper to build, but it
  puts `rate_limit_rpm` and the upstream `detail` string one careless sentence away from the customer.
- **Per-requester credentials to the upstream** (the agent impersonates the requester). The right
  long-term shape for a real deployment, but the mock has no authentication and the ticket has no
  requester principal; building it here would be fiction. It is the first thing to add when a real
  identity provider exists: swap `TicketContext` to carry a requester token and have `HttpUpstream`
  forward it, with the same tool signatures.

## Consequences

- Cross-account access is impossible by construction; there is nothing to test except that tool
  signatures have not grown an `account_id` parameter (a grep-level test).
- The entitlement filter is a fixed allowlist. A new useful field from a richer real API needs a
  code change to become visible to the model. That is the intended friction.
- `identity_unverified` cases cost a human touch every time. The alternative, guessing, is worse.
- When the entitlement upstream fails we cannot know what the customer is entitled to, so we say so
  (`entitlement_unknown`), escalate, and the guard blocks any feature name in the reply.

## Related

- `notes` (local) D-005 model proposes, policy decides; D-009 strict schema with unknown-field drop.
- `agent/redact.py`, `agent/tools.py`, `agent/policy.py`, `agent/guard.py`; tests `tests/test_redact.py`,
  `tests/test_tools_gate.py`, `tests/test_guard.py`, `tests/test_policy.py`.
