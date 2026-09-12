"""Lightweight in-memory rate limiting.

Appropriate for a single-process portfolio deployment. Does not require
Redis or any external infrastructure. A multi-instance production
deployment would need a shared store (e.g. Redis) instead -- documented as
a limitation, not implemented here to avoid unnecessary infrastructure.
"""

import time
from collections import defaultdict, deque


class RateLimitExceededError(RuntimeError):
    """Raised when a client exceeds the configured request rate."""


class InMemoryRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be greater than zero.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Returns True and records a hit if `key` is within its rate limit."""
        current_time = now if now is not None else time.monotonic()
        hits = self._hits[key]
        cutoff = current_time - self._window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self._max_requests:
            return False
        hits.append(current_time)
        return True

    def reset(self, key: str) -> None:
        self._hits.pop(key, None)
