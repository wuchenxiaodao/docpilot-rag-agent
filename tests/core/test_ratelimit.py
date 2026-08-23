"""Unit tests for the in-memory RateLimiter (no FastAPI, no settings)."""

import time

from core.ratelimit import RateLimiter


def test_disabled_when_limit_zero():
    rl = RateLimiter()
    for _ in range(100):
        allowed, retry = rl.check("k", limit=0)
        assert allowed is True
        assert retry == 0


def test_allows_up_to_limit_then_denies():
    rl = RateLimiter()
    assert rl.check("k", limit=3) == (True, 0)
    assert rl.check("k", limit=3) == (True, 0)
    assert rl.check("k", limit=3) == (True, 0)
    allowed, retry = rl.check("k", limit=3)
    assert allowed is False
    assert retry >= 1


def test_denied_request_does_not_increment():
    rl = RateLimiter()
    limit = 2
    rl.check("k", limit)
    rl.check("k", limit)
    # Hammer past the limit; none of these should consume a slot.
    for _ in range(5):
        allowed, _ = rl.check("k", limit)
        assert allowed is False
    # Still only 2 used; after reset+window we get a fresh budget of `limit`.
    rl.reset()
    assert rl.check("k", limit) == (True, 0)


def test_keys_are_independent():
    rl = RateLimiter()
    rl.check("alice", limit=1)
    assert rl.check("alice", limit=1)[0] is False  # alice exhausted
    assert rl.check("bob", limit=1)[0] is True  # bob unaffected


def test_window_resets_after_expiry(monkeypatch):
    rl = RateLimiter()
    # Pin monotonic time so the window rollover is deterministic.
    t = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])

    rl.check("k", limit=1)
    assert rl.check("k", limit=1)[0] is False  # within window -> denied
    t[0] += 61  # advance past the 60s window
    assert rl.check("k", limit=1)[0] is True  # new window -> allowed


def test_reset_clears_all():
    rl = RateLimiter()
    rl.check("a", limit=1)
    rl.check("b", limit=1)
    rl.reset()
    assert rl.check("a", limit=1)[0] is True
    assert rl.check("b", limit=1)[0] is True
