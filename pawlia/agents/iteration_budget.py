"""Per-agent iteration budget — consume/refund counter with grace call."""

from __future__ import annotations


class IterationBudget:
    """Iteration counter for an agent's tool-calling loop.

    Each agent gets its own budget, capped at *max_total*.  After the
    budget is exhausted the caller may still make one additional call
    (a "grace call") so the model can finish summarising the work it
    already did.

    Thread-safe via an internal lock so concurrent subagent usage
    doesn't race.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._grace_used = False
        from threading import Lock
        self._lock = Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                if not self._grace_used:
                    self._grace_used = True
                    return True
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for non-tool-call turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)
