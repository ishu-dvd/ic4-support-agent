"""Lexical search over the knowledge base.

Deterministic scoring so results are stable across runs: title matches are
weighted more heavily than body matches, with a small bonus for tag hits.
"""
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "for", "in", "on", "at", "by", "with", "from", "how", "do",
    "does", "did", "my", "our", "we", "i", "it", "this", "that", "can", "not",
}

W_TITLE = 3.0
W_BODY = 1.0
W_TAG = 1.5


def tokenize(text):
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def score(query, article):
    q = Counter(tokenize(query))
    if not q:
        return 0.0
    title = Counter(tokenize(article.get("title", "")))
    body = Counter(tokenize(article.get("body", "")))
    tags = Counter(tokenize(" ".join(article.get("tags", []))))
    total = 0.0
    for term, qn in q.items():
        total += W_TITLE * qn * min(title.get(term, 0), 3)
        total += W_BODY * qn * min(body.get(term, 0), 4)
        total += W_TAG * qn * min(tags.get(term, 0), 2)
    return round(total, 4)


def search(query, articles, limit=5):
    scored = []
    for a in articles:
        s = score(query, a)
        if s > 0:
            scored.append((s, a))
    # Ties break on article id for determinism.
    scored.sort(key=lambda p: (-p[0], p[1]["id"]))
    out = []
    for s, a in scored[:limit]:
        item = {k: a[k] for k in ("id", "title", "body", "tags") if k in a}
        item["score"] = s
        item["applies_to"] = a.get("applies_to", {})
        out.append(item)
    return out
