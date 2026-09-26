"""Per-run budget: a wall-clock deadline plus hard caps on LLM and tool calls.

`now` is injectable so tests can move time forward without sleeping. The counters are
guarded by a lock because `read_all_parallel` charges from three threads at once.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class BudgetExceeded(Exception):
    """A hard cap (tool or LLM calls) was hit. `args[0]` names which one."""


class Budget:
    def __init__(
        self,
        deadline_ms: int,
        max_llm_calls: int,
        max_tool_calls: int,
        now: Callable[[], float] = time.monotonic,
    ):
        self.deadline_ms = int(deadline_ms)
        self.max_llm_calls = int(max_llm_calls)
        self.max_tool_calls = int(max_tool_calls)
        self._now = now
        self._start = now()
        self.tool_calls = 0
        self.llm_calls = 0
        self._lock = threading.Lock()

    def elapsed_ms(self) -> int:
        return int((self._now() - self._start) * 1000)

    def remaining_ms(self) -> int:
        return max(0, self.deadline_ms - self.elapsed_ms())

    def fraction_remaining(self) -> float:
        if self.deadline_ms <= 0:
            return 0.0
        return min(1.0, max(0.0, self.remaining_ms() / float(self.deadline_ms)))

    def past_deadline(self) -> bool:
        return self.remaining_ms() <= 0

    def charge_tool(self) -> None:
        """Count one upstream call. Only the call cap raises here; the deadline is a
        separate, softer signal that callers check with `past_deadline()` so a slow
        read can still finish and be recorded rather than blow up mid-flight."""
        with self._lock:
            if self.tool_calls >= self.max_tool_calls:
                raise BudgetExceeded("tool_calls")
            self.tool_calls += 1

    def charge_llm(self) -> None:
        with self._lock:
            if self.llm_calls >= self.max_llm_calls:
                raise BudgetExceeded("llm_calls")
            self.llm_calls += 1
