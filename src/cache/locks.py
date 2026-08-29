"""分布式锁（Redis SET NX EX + Lua 释放）。

为什么释放要用 Lua：GET 判断 + DEL 两步不是原子的——锁可能恰好在这中间
过期并被别人抢到，然后被误删。Lua 里 compare-and-del 是单命令原子执行。

token 是随机值：只有持锁者能删自己的锁，防止 A 的锁过期后 B 持锁、
A 迟到回来把 B 的锁删掉。
"""

import logging
import uuid

from cache.client import get_redis

logger = logging.getLogger(__name__)

_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def acquire_lock(name: str, ttl_seconds: int, *, r=None) -> str | None:
    """尝试取锁。成功返回 token（释放时用），失败返回 None（别人持有）。"""
    redis_cli = r if r is not None else get_redis()
    token = uuid.uuid4().hex
    ok = redis_cli.set(name, token, nx=True, ex=ttl_seconds)
    return token if ok else None


def release_lock(name: str, token: str, *, r=None) -> bool:
    """原子释放（Lua）。token 不匹配（锁已易主/已过期）返回 False。"""
    redis_cli = r if r is not None else get_redis()
    script = redis_cli.register_script(_RELEASE_LUA)
    return bool(script(keys=[name], args=[token]))
