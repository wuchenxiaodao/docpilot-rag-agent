"""Redis 令牌桶限流（原子 Lua），替代单进程内存版。

动机：原 in-memory 固定窗口只对单 worker 有效，uvicorn 起多进程后每个进程
各数各的，真实限流值会被放大 N 倍。Redis 版全进程共享同一桶。

令牌桶 vs 固定窗口：固定窗口有边界突刺（窗口切换瞬间可打入 2x 流量），
令牌桶以恒定速率补充，平滑得多。

Lua 原子性：读桶-算补充-扣减-写回必须一步完成，否则并发下会超发。
时间由 Python 侧传入（而非 Lua 内调 TIME），便于测试注入。
"""

import logging
import time

from cache.client import get_redis

logger = logging.getLogger(__name__)

_BUCKET_LUA = """
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
local ts = tonumber(redis.call('HGET', KEYS[1], 'ts'))
if tokens == nil or ts == nil then
    tokens = tonumber(ARGV[1])
    ts = tonumber(ARGV[3])
end
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

tokens = math.min(capacity, tokens + (now - ts) * rate)

local allowed = 0
local retry_after = 1
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
    retry_after = 0
else
    retry_after = math.max(1, math.ceil((requested - tokens) / rate))
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], 7200)
return {allowed, retry_after}
"""

_KEY_PREFIX = "docpilot:rl:"


class RedisTokenBucket:
    """check() 接口与 core.ratelimit.RateLimiter 兼容，可无缝替换。"""

    def __init__(self) -> None:
        self._script = None

    def _get_script(self):
        if self._script is None:
            self._script = get_redis().register_script(_BUCKET_LUA)
        return self._script

    def check(
        self, key: str, limit: int, window_sec: float = 60.0, *, now: float | None = None
    ) -> tuple[bool, int]:
        """扣一个令牌；返回 (allowed, retry_after_s)。limit<=0 直接放行。"""
        if limit <= 0:
            return True, 0
        if now is None:
            now = time.time()
        rate = limit / window_sec
        script = self._get_script()
        result = script(keys=[_KEY_PREFIX + key], args=[limit, rate, now, 1])
        allowed, retry_after = int(result[0]), int(result[1])
        return bool(allowed), retry_after

    def reset(self, key: str | None = None) -> None:
        """测试辅助：清桶。"""
        r = get_redis()
        if key is None:
            for k in r.scan_iter(match=_KEY_PREFIX + "*", count=100):
                r.delete(k)
        else:
            r.delete(_KEY_PREFIX + key)


redis_token_bucket = RedisTokenBucket()
