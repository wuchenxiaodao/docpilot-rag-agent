"""In-memory fixed-window rate limiter (no external dependencies).

Suited to the stock single-process uvicorn deployment. A denied request does
not increment the counter, so a client hammering the limit cannot extend the
window. For multi-worker deployments this would need a shared store (Redis);
that is intentionally left out to respect the "no new dependencies" constraint.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Fixed-window counter, keyed by user id or client IP."""

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(
        self, key: str, limit: int, window_sec: float = 60.0
    ) -> tuple[bool, int]:
        """Increment the counter for ``key``; return ``(allowed, retry_after_s)``.

        ``limit <= 0`` disables the limiter (returns ``(True, 0)``) so callers
        can route a "rate limiting off" setting straight through.
        """
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            start, count = self._buckets.get(key, (now, 0))
            if now - start >= window_sec:
                start, count = now, 0
            if count >= limit:
                retry_after = max(1, int(window_sec - (now - start)) + 1)
                return False, retry_after
            count += 1
            self._buckets[key] = (start, count)
            return True, 0

    def reset(self) -> None:
        """Clear all buckets (test helper)."""
        with self._lock:
            self._buckets.clear()


rate_limiter = RateLimiter()
