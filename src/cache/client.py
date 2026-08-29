"""Redis 客户端管理（单例连接池）。

设计要点：
- decode_responses=True：业务键值都是字符串/JSON；嵌入向量用 base64 存，
  避免二进制与文本模式混用的坑
- 惰性建连：Redis 不在线时不阻断导入，调用方捕获异常走降级路径
"""

import logging

import redis

from core.settings import settings

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """返回进程级共享的 Redis 客户端。不可用时抛异常，由调用方降级。"""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def reset_client() -> None:
    """测试辅助：重置单例，让下一次 get_redis 重新建连。"""
    global _client
    _client = None


def redis_available() -> bool:
    try:
        return bool(get_redis().ping())
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return False
