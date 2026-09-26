# Exercise fixture — Senior AI/ML Engineer II

**Your brief is in [`TASK.md`](TASK.md). Read that first.**

This repo also contains the fixture the exercise runs against: a small HTTP service that
stands in for the internal systems a support or access-management team works from. It lets
you build against realistic reads and one consequential write without touching a real
system.

**No dependencies.** Python 3.9+ and the standard library. Nothing to install.

```bash
python3 -m server                 # http://127.0.0.1:8080, "support" fixture
python3 -m server --variant access --port 8081
python3 scripts/smoke_test.py     # verifies every endpoint responds
```

Full endpoint reference: [`docs/API.md`](docs/API.md).

## What is in here

| Path | What it is |
|---|---|
| `server/` | The mock API. Stdlib only. |
| `data/support/` | Fixture: customer support tickets, accounts, entitlements, knowledge base |
| `data/access/` | Fixture: internal access requests, people, current access, policy library |
| `data/*/golden.json` | Labeled cases you can score an agent against |
| `docs/API.md` | Endpoint reference with example requests and responses |
| `TASK.md` | The exercise brief |
| `examples/minimal_client.py` | A 40-line client showing every endpoint being called |
| `scripts/smoke_test.py` | Checks the server is up and behaving |
| `scripts/dataset_stats.py` | Label distribution of a golden dataset |

## The two fixtures

Both expose exactly the same API, so code written against one works against the other.
They differ only in data.

- **`support`** — a customer support desk. Tickets arrive from customer accounts; the
  knowledge base holds product articles; the write files an escalation.
- **`access`** — an internal access-request desk. Requests arrive from employees and
  contractors; the knowledge base holds access policies; the write files an escalation.

## The golden dataset

`data/<variant>/golden.json` holds one labeled record per ticket:

```json
{
  "case_id": "sup-001",
  "ticket_id": "TCK-1101",
  "expected_category": "billing_proration",
  "expected_escalate": false,
  "expected_kb": ["kb-0001"]
}
```

- `expected_category` — the correct classification of the underlying issue.
- `expected_escalate` — whether this case genuinely warrants the escalation write.
- `expected_kb` — the knowledge-base article that actually applies to *this* account.
  An empty list means no article applies.

The category vocabulary is whatever appears in the file; read it rather than guessing.
How you score against these labels is your decision, and part of what we will discuss.

### Know your baseline

The labels are not evenly distributed. Before trusting a score, find out what a constant
answer would score:

```bash
python3 scripts/dataset_stats.py --variant support
```

Class balance differs between the two fixtures, so a raw percentage from one is not
directly comparable to the other.

## Notes on the fixture

- Data is loaded into memory at startup. Restarting the server resets filed escalations;
  nothing else mutates.
- Knowledge-base search is deterministic lexical scoring — the same query always returns
  the same ranking. It is not a semantic search engine, and it is not tuned for you.
- `--latency-ms` adds a delay to every request if you want to see how your code behaves
  against a slower upstream.
- The service is a stand-in for systems owned by other teams. Treat it the way you would
  treat an upstream you do not control.
