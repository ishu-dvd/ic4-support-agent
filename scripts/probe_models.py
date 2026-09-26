#!/usr/bin/env python3
"""Probe which DigitalOcean serverless-inference models this team's key can actually call, and write
dashboard/models.json (the catalog behind the dashboard's Models tab and GET /api/models).

Stdlib only. Reads DIGITALOCEAN_ACCESS_TOKEN (catalog + prices via GET /v2/gen-ai/models) and
OPENAI_API_KEY (falls back to the DO token) for a 4-token chat completion per model. Neither value is
printed. 403 = "not available for your subscription tier"; 404 = id not served by inference.do-ai.run.

    . scripts/env.sh && python3 scripts/probe_models.py [--out dashboard/models.json] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CATALOG_URL = "https://api.digitalocean.com/v2/gen-ai/models?per_page=200"
INFERENCE_URL = "https://inference.do-ai.run/v1/chat/completions"


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _probe(model_id: str, key: str) -> int:
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                       "max_tokens": 4}).encode()
    req = urllib.request.Request(INFERENCE_URL, data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "dashboard", "models.json"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    token = os.environ.get("DIGITALOCEAN_ACCESS_TOKEN", "")
    key = os.environ.get("OPENAI_API_KEY") or token
    if not token or not key:
        print("DIGITALOCEAN_ACCESS_TOKEN (and optionally OPENAI_API_KEY) must be set; see scripts/env.sh", file=sys.stderr)
        return 2
    models = _get_json(CATALOG_URL, token).get("models", [])
    text_models = [m for m in models if m.get("id") and m.get("is_foundational")
                   and "text" in ((m.get("modalities") or {}).get("output") or ["text"])]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        codes = dict(zip((m["id"] for m in text_models), ex.map(lambda m: _probe(m["id"], key), text_models)))
    accessible, inaccessible = [], []
    for m in sorted(text_models, key=lambda m: m["id"]):
        code = codes[m["id"]]
        if code == 200:
            pricing = m.get("pricing") or {}
            accessible.append({
                "model_id": m["id"], "name": m.get("name"), "type": m.get("type"),
                "params_b": m.get("parameter_count"),
                "context": int(m["context_window"]) if str(m.get("context_window") or "").isdigit() else None,
                "price_per_1k": {
                    "input": round(float(pricing["input_price_per_million"]) * 1000, 6) if pricing.get("input_price_per_million") is not None else None,
                    "output": round(float(pricing["output_price_per_million"]) * 1000, 6) if pricing.get("output_price_per_million") is not None else None,
                },
            })
        else:
            inaccessible.append({"model_id": m["id"], "status": code})
    out = {
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": INFERENCE_URL.rsplit("/chat", 1)[0],
        "note": "HTTP 403 = model not in this team's subscription tier; 404 = id not served by the inference "
                "endpoint (embedding/image/audio ids or retired); 0 = network error.",
        "accessible": accessible, "inaccessible": inaccessible,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"accessible": len(accessible), "inaccessible": len(inaccessible), "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
