"""Every endpoint, called once, with no dependencies.

    python3 -m server &
    python3 examples/minimal_client.py
"""
import json
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8080"


def get(path, **params):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def main():
    print("health          ", get("/healthz"))

    status, listing = get("/v1/tickets")
    ticket_id = listing["ticket_ids"][0]
    print("ticket count    ", len(listing["ticket_ids"]))

    status, ticket = get("/v1/tickets/%s" % ticket_id)
    print("ticket          ", ticket_id, "->", ticket["subject"])

    account_id = ticket["account_id"]
    status, account = get("/v1/accounts/%s" % account_id)
    print("account         ", account_id, "->", account["name"])

    status, ent = get("/v1/accounts/%s/entitlements" % account_id)
    print("entitlements    ", status, ent if status != 200 else ent["features"])

    status, hits = get("/v1/kb/search", q=ticket["subject"], limit=3)
    print("kb search       ", [(h["id"], h["score"]) for h in hits["results"]])

    # The write refuses to commit without an explicit confirmation.
    print("write, unconfirmed", post("/v1/escalations", {
        "ticket_id": ticket_id, "reason": "example", "summary": "demonstration only",
    })[0])

    status, filed = post("/v1/escalations", {
        "ticket_id": ticket_id, "reason": "example",
        "summary": "demonstration only", "confirm": True,
    })
    print("write, confirmed  ", status, filed.get("escalation_id"))
    print("escalations       ", len(get("/v1/escalations")[1]["escalations"]))


if __name__ == "__main__":
    main()
