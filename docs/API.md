# API reference

Base URL: `http://127.0.0.1:8080` (override the port with `--port`).
All responses are JSON. All reads are idempotent. There is no authentication.

Errors use a consistent envelope:

```json
{ "error": { "code": "ticket_not_found", "message": "No ticket with id 'TCK-9999'" } }
```

---

## `GET /healthz`

Liveness plus which fixture is loaded.

```json
{ "status": "ok", "variant": "support" }
```

---

## `GET /v1/tickets`

Every ticket id in the loaded fixture, sorted.

```json
{ "ticket_ids": ["TCK-1101", "TCK-1102", "..."] }
```

---

## `GET /v1/tickets/{ticket_id}`

The ticket, including the free-text body as submitted by the requester.

```json
{
  "ticket_id": "TCK-1101",
  "account_id": "acct_1001",
  "subject": "First invoice after upgrade is higher than the plan price",
  "body": "We upgraded to Business on the 14th and ...",
  "channel": "email",
  "opened_at": "2026-08-24T08:12:00Z",
  "status": "open"
}
```

`404 ticket_not_found` if the id is unknown.

---

## `POST /v1/tickets`

**This is a write.** Files a new ticket against an existing account so it can be run through the
agent like any shipped one. Intended for testing and demos: the record is held in memory only
and disappears when the server restarts (the fixture files under `data/` are never modified).

Request:

```json
{
  "account_id": "acct_1004",
  "subject": "Cannot enable SSO",
  "body": "SSO toggle is greyed out in admin settings since yesterday.",
  "channel": "chat",
  "ticket_id": "TCK-CUSTOM"
}
```

`account_id`, `subject` and `body` are required. `channel` defaults to `web`. `ticket_id` is
optional; when omitted the server continues the fixture's numbering (`TCK-1132`, `TCK-1133`, …).

Response `201` is the stored ticket, in the same shape `GET /v1/tickets/{ticket_id}` returns:

```json
{
  "ticket_id": "TCK-1132",
  "account_id": "acct_1004",
  "subject": "Cannot enable SSO",
  "body": "SSO toggle is greyed out in admin settings since yesterday.",
  "channel": "chat",
  "opened_at": "2026-09-26T08:27:03Z",
  "status": "open"
}
```

- `400 invalid_json` / `400 invalid_body` — unparseable or non-object body.
- `422 missing_fields` — one or more required fields absent or empty; `fields` lists them.
- `422 unknown_account` — `account_id` does not exist.
- `422 invalid_ticket_id` — `ticket_id` is not `[A-Za-z0-9_-]{1,64}`.
- `409 ticket_exists` — `ticket_id` is already taken. Nothing was written.

The new ticket appears in `GET /v1/tickets` and is a valid target for `POST /v1/escalations`.

---

## `GET /v1/accounts`

Every account in the loaded fixture, sorted by id. Each entry is the full record that
`GET /v1/accounts/{account_id}` returns.

```json
{ "accounts": [ { "account_id": "acct_1001", "name": "Northwind Retail", "...": "..." } ] }
```

---

## `GET /v1/accounts/{account_id}`

The account or person behind a ticket.

```json
{
  "account_id": "acct_1001",
  "name": "Northwind Retail",
  "plan_tier": "business",
  "region": "us-east",
  "auth_model": "unified_auth",
  "customer_since": "2023-04-11",
  "primary_contact": { "name": "Northwind Ops", "email": "ops@northwind.example" }
}
```

In the `access` fixture the same endpoint returns a person record with `department`,
`worker_type`, `level` and `manager_id` instead of plan fields.

`404 account_not_found` if the id is unknown.

---

## `GET /v1/accounts/{account_id}/entitlements`

What the account is entitled to. Served by a different upstream from the account record.

```json
{
  "account_id": "acct_1001",
  "support_tier": "standard",
  "seats": 40,
  "features": ["sso", "audit_log", "data_export"],
  "sla_hours": 12,
  "rate_limit_rpm": 1200,
  "updated_at": "2026-08-02T09:14:00Z"
}
```

- `404 account_not_found` — no such account.
- `500 entitlement_service_error` — the account exists but the entitlement service could
  not produce a valid response body. The `detail` field carries the reason.

---

## `GET /v1/kb/search?q={query}&limit={n}`

Lexical search over the knowledge base. `q` is required; `limit` defaults to 5 and is
capped at 25.

```json
{
  "query": "reset api token",
  "results": [
    {
      "id": "kb-0002",
      "title": "Reset an API token on Legacy Auth accounts",
      "body": "To reset the API token, open the Security tab ...",
      "tags": ["api", "token", "reset", "rotate", "legacy"],
      "score": 23.5,
      "applies_to": { "auth_model": ["legacy_auth"] }
    }
  ]
}
```

Scoring weights title matches above body matches, with a bonus for tag hits. Ties break on
article id, so ranking is stable across runs.

`applies_to` states the conditions under which an article is applicable. An empty object
means it applies generally. The search does **not** filter on it — results are ranked on
text relevance alone.

`400 missing_query` if `q` is absent or blank.

---

## `POST /v1/escalations`

**This is a write.** It records an escalation against a ticket.

Request:

```json
{
  "ticket_id": "TCK-1103",
  "reason": "rate_limit_increase",
  "summary": "Sustained 5,400 rpm exceeds the plan ceiling of 6,000; capacity review needed.",
  "confirm": true
}
```

`ticket_id`, `reason` and `summary` are required. `confirm` must be exactly `true`.

Response `201`:

```json
{
  "ticket_id": "TCK-1103",
  "reason": "rate_limit_increase",
  "summary": "Sustained 5,400 rpm exceeds ...",
  "confirm": true,
  "escalation_id": "ESC-0001",
  "status": "filed"
}
```

- `400 invalid_json` / `400 invalid_body` — unparseable or non-object body.
- `422 missing_fields` — one or more required fields absent or empty; `fields` lists them.
- `422 unknown_ticket` — `ticket_id` does not exist.
- `409 confirmation_required` — `confirm` was not `true`. Nothing was written.

---

## `GET /v1/escalations`

Everything filed since the server started. Useful for asserting in tests that your agent
wrote what you expected — and only what you expected.

```json
{ "escalations": [ { "escalation_id": "ESC-0001", "...": "..." } ] }
```

State is in memory. Restarting the server clears it.
