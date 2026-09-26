"""Score-inflation guard: the agent must never know about the golden labels.

The eval reads data/<variant>/golden.json; the agent package and the single-ticket CLI must not
reference it in any form (not even in a comment - a reviewer greps for the word, so do we).
"""
from __future__ import annotations

import os

import agent

IC4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sources():
    pkg_dir = os.path.dirname(agent.__file__)
    for fname in sorted(os.listdir(pkg_dir)):
        if fname.endswith(".py"):
            yield os.path.join(pkg_dir, fname)
    yield os.path.join(IC4_ROOT, "scripts", "run_agent.py")


def test_agent_and_cli_never_mention_golden():
    offenders = []
    for path in _sources():
        with open(path, encoding="utf-8") as fh:
            if "golden" in fh.read().lower():
                offenders.append(os.path.relpath(path, IC4_ROOT))
    assert offenders == [], offenders


def test_agent_never_opens_data_files_directly():
    """Belt and braces: no direct path into data/ from the agent package (everything goes over HTTP)."""
    offenders = []
    for path in _sources():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if "data/support" in src or "data/access" in src or 'os.path.join("data"' in src:
            offenders.append(os.path.relpath(path, IC4_ROOT))
    assert offenders == [], offenders
