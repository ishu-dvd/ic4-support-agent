"""Verifies the fixture is serving correctly. Run after starting the server.

    python3 -m server &
    python3 scripts/smoke_test.py
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def call(base, path, params=None, payload=None):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}
    except urllib.error.URLError as e:
        print("cannot reach %s (%s)\nis the server running?" % (url, e.reason))
        sys.exit(2)


def check(label, ok, detail=""):
    print("  %s %s%s" % ("PASS" if ok else "FAIL", label, (" - " + detail) if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--variant", default="support")
    a = p.parse_args()
    base = a.base_url.rstrip("/")

    print("checking %s (variant=%s)\n" % (base, a.variant))

    status, health = call(base, "/healthz")
    check("healthz returns 200", status == 200, str(status))
    check("healthz reports the expected variant",
          health.get("variant") == a.variant,
          "got %r" % health.get("variant"))

    status, listing = call(base, "/v1/tickets")
    ids = listing.get("ticket_ids", [])
    check("ticket listing is non-empty", status == 200 and len(ids) > 0, "got %d" % len(ids))

    golden_path = os.path.join(ROOT, "data", a.variant, "golden.json")
    golden = json.load(open(golden_path, encoding="utf-8"))
    check("golden dataset loads", len(golden) > 0, golden_path)
    check("every golden case maps to a real ticket",
          all(g["ticket_id"] in ids for g in golden),
          "missing: %s" % [g["ticket_id"] for g in golden if g["ticket_id"] not in ids][:5])
    check("every ticket has a golden case",
          set(ids) == {g["ticket_id"] for g in golden},
          "unlabeled: %s" % sorted(set(ids) - {g["ticket_id"] for g in golden})[:5])

    first = ids[0]
    status, ticket = call(base, "/v1/tickets/%s" % first)
    check("ticket fetch returns a body", status == 200 and bool(ticket.get("body")), str(status))

    status, _ = call(base, "/v1/tickets/TCK-000000")
    check("unknown ticket returns 404", status == 404, str(status))

    status, account = call(base, "/v1/accounts/%s" % ticket["account_id"])
    check("account fetch returns 200", status == 200, str(status))

    # Entitlements: at least one account must serve cleanly.
    served = errored = 0
    accounts = json.load(open(os.path.join(ROOT, "data", a.variant, "accounts.json"), encoding="utf-8"))
    for acc in accounts:
        st, _ = call(base, "/v1/accounts/%s/entitlements" % acc["account_id"])
        if st == 200:
            served += 1
        elif st == 500:
            errored += 1
    check("entitlements serve for most accounts", served >= len(accounts) - 2,
          "%d/%d served" % (served, len(accounts)))
    check("entitlement errors surface as 500, not a crash", errored + served == len(accounts),
          "%d served + %d errored != %d" % (served, errored, len(accounts)))

    kb = json.load(open(os.path.join(ROOT, "data", a.variant, "kb.json"), encoding="utf-8"))
    probe = " ".join(kb[0]["tags"][:2]) if kb and kb[0].get("tags") else kb[0]["title"]
    status, hits = call(base, "/v1/kb/search", params={"q": probe, "limit": 3})
    check("kb search returns results", status == 200 and len(hits.get("results", [])) > 0,
          "q=%r status=%s" % (probe, status))
    status, _ = call(base, "/v1/kb/search", params={"q": "  "})
    check("blank kb query returns 400", status == 400, str(status))

    scores = [h["score"] for h in hits.get("results", [])]
    check("kb results are ranked descending", scores == sorted(scores, reverse=True), str(scores))

    status, _ = call(base, "/v1/escalations",
                     payload={"ticket_id": first, "reason": "smoke", "summary": "smoke test"})
    check("write without confirm returns 409", status == 409, str(status))

    status, _ = call(base, "/v1/escalations", payload={"ticket_id": first, "confirm": True})
    check("write missing fields returns 422", status == 422, str(status))

    status, _ = call(base, "/v1/escalations",
                     payload={"ticket_id": "TCK-000000", "reason": "x", "summary": "y", "confirm": True})
    check("write with unknown ticket returns 422", status == 422, str(status))

    before = len(call(base, "/v1/escalations")[1]["escalations"])
    status, filed = call(base, "/v1/escalations",
                         payload={"ticket_id": first, "reason": "smoke",
                                  "summary": "smoke test", "confirm": True})
    check("confirmed write returns 201", status == 201, str(status))
    check("confirmed write is assigned an id", bool(filed.get("escalation_id")), str(filed))
    after = len(call(base, "/v1/escalations")[1]["escalations"])
    check("filed escalation is readable back", after == before + 1, "%d -> %d" % (before, after))

    kb_ids = {k["id"] for k in kb}
    bad = [(g["case_id"], k) for g in golden for k in g.get("expected_kb", []) if k not in kb_ids]
    check("golden expected_kb references exist", not bad, str(bad[:5]))

    print()
    if FAILURES:
        print("%d check(s) failed: %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
