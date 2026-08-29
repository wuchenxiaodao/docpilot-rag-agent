"""QA 缓存（阶段 2 核心）：问过的问题直接回缓存，跳过整条 RAG 链路。

两级查找：
1. 精确命中：规范化后 md5 做 key，O(1)
2. 语义命中：嵌入向量化后与缓存集做余弦相似度（>= 阈值视为同一问题），
   当前规模 O(n) 扫描可接受，规模化需换向量索引

存储结构：
- docpilot:qa:{hash}   -> JSON {question, answer, citations}
- docpilot:qa:emb      -> hash，field={hash}，value=base64(float32 向量)

一致性（Cache-Aside）：知识库变更（/ingest 成功）时整体失效 qa:*，
宁可重算不给脏答案——RAG 场景答案依赖库内容，这是正确取舍。
"""

import base64
import hashlib
import json
import logging
import re
import time

import numpy as np

from cache.client import get_redis
from core.settings import settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "docpilot:qa:"
EMB_KEY = "docpilot:qa:emb"

_WS_RE = re.compile(r"\s+")


def _normalize(question: str) -> str:
    return _WS_RE.sub(" ", question.strip())


def question_hash(question: str) -> str:
    return hashlib.md5(_normalize(question).encode("utf-8")).hexdigest()


def _encode_embedding(vector: list[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode_embedding(blob: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(blob), dtype=np.float32)


def lookup(
    question: str,
    embed_fn=None,
    *,
    r=None,
) -> dict | None:
    """查缓存。返回 {question, answer, citations, hit: 'exact'|'semantic'} 或 None。

    embed_fn=None 时跳过语义级（只做精确命中）。Redis 异常向上抛，由服务层降级。
    """
    redis_cli = r if r is not None else get_redis()
    key = KEY_PREFIX + question_hash(question)
    raw = redis_cli.get(key)
    if raw is not None:
        data = json.loads(raw)
        data["hit"] = "exact"
        return data

    if embed_fn is None:
        return None

    emb_map = redis_cli.hgetall(EMB_KEY)
    if not emb_map:
        return None
    if len(emb_map) > settings.QA_CACHE_MAX_SCAN:
        logger.warning("QA cache embedding set exceeded scan bound: %d", len(emb_map))

    query = np.asarray(embed_fn(question), dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return None
    query = query / query_norm

    best_hash, best_score = None, -1.0
    for field, blob in list(emb_map.items())[: settings.QA_CACHE_MAX_SCAN]:
        vec = _decode_embedding(blob)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        score = float(np.dot(query, vec / norm))
        if score > best_score:
            best_hash, best_score = field, score

    if best_hash is None or best_score < settings.QA_CACHE_SIMILARITY_THRESHOLD:
        return None

    raw = redis_cli.get(KEY_PREFIX + best_hash)
    if raw is None:
        # 答案键已过期但嵌入残留：清理悬挂 field，防下次空扫
        redis_cli.hdel(EMB_KEY, best_hash)
        return None
    data = json.loads(raw)
    data["hit"] = "semantic"
    data["similarity"] = round(best_score, 4)
    return data


def store(
    question: str,
    answer: str,
    citations: list[dict] | None = None,
    *,
    r=None,
) -> None:
    """写入一条 QA 缓存（带 TTL）。embed_fn 由调用方注入以复用已加载的模型。"""
    redis_cli = r if r is not None else get_redis()
    qhash = question_hash(question)
    payload = json.dumps(
        {"question": _normalize(question), "answer": answer, "citations": citations},
        ensure_ascii=False,
    )
    redis_cli.set(KEY_PREFIX + qhash, payload, ex=settings.QA_CACHE_TTL_SECONDS)


def store_embedding(question: str, vector: list[float], *, r=None) -> None:
    """配套写入问题的嵌入向量（语义级查找用）。哈希键与嵌入哈希同 TTL。"""
    redis_cli = r if r is not None else get_redis()
    redis_cli.hset(EMB_KEY, question_hash(question), _encode_embedding(vector))
    redis_cli.expire(EMB_KEY, settings.QA_CACHE_TTL_SECONDS)


def invalidate_all(*, r=None) -> int:
    """知识库变更后的整体失效（Cache-Aside 的删缓存侧）。

    用 SCAN 游标分批删除而非 KEYS *：KEYS 会一次性遍历阻塞单线程的 Redis，
    键多时是生产事故级操作。
    """
    redis_cli = r if r is not None else get_redis()
    deleted = 0
    for key in redis_cli.scan_iter(match=KEY_PREFIX + "*", count=100):
        redis_cli.delete(key)
        deleted += 1
    if redis_cli.exists(EMB_KEY):
        redis_cli.delete(EMB_KEY)
        deleted += 1
    return deleted
