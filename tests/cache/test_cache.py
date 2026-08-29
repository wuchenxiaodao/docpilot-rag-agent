"""阶段 2 Redis 层测试。

- 纯逻辑（规范化/哈希/阈值判断）不依赖 Redis，常跑
- 缓存回写/语义命中/锁/令牌桶依赖 Redis 容器，挂 docker 标记
  （跑法：pytest tests/cache --run-docker）
"""

import uuid

import numpy as np
import pytest

from cache.client import get_redis
from cache.locks import acquire_lock, release_lock
from cache.semantic_cache import (
    invalidate_all,
    lookup,
    question_hash,
    store,
    store_embedding,
)
from cache.token_bucket import RedisTokenBucket


# ---------- 纯逻辑 ----------


def test_question_hash_normalizes_whitespace():
    assert question_hash("病假 几天") == question_hash("  病假   几天  ")


def test_question_hash_case_and_punctuation_differ():
    # 规范化只处理空白：不同问题必须不同 key
    assert question_hash("a b") != question_hash("ab")


def _fake_embed_close():
    # 与已存向量近似同向（余弦 ~0.995）
    return [0.995, 0.1]


def _fake_embed_far():
    return [0.0, 1.0]


# ---------- Redis 集成 ----------


@pytest.fixture(autouse=True)
def _clean_qa_keys():
    """每个用例前后清掉 docpilot:qa:* 与 docpilot:rl:*，避免用例间串扰。"""
    def _purge():
        try:
            r = get_redis()
            r.delete(*[k for k in r.scan_iter(match="docpilot:qa:*")])
            r.delete(*[k for k in r.scan_iter(match="docpilot:rl:*")])
            r.delete(*[k for k in r.scan_iter(match="docpilot:lock:*")])
        except Exception:
            pass

    _purge()
    yield
    _purge()


@pytest.mark.docker
class TestSemanticCache:
    def test_exact_roundtrip(self):
        q = f"q-{uuid.uuid4().hex[:8]}"
        store(q, "答案 A", [{"source": "doc.pdf", "page": 3}])
        hit = lookup(q)
        assert hit is not None
        assert hit["answer"] == "答案 A"
        assert hit["citations"][0]["source"] == "doc.pdf"
        assert hit["hit"] == "exact"

    def test_miss_returns_none(self):
        assert lookup(f"nope-{uuid.uuid4().hex[:8]}") is None

    def test_semantic_hit_near_duplicate(self):
        q1 = f"q-{uuid.uuid4().hex[:8]}"
        store(q1, "语义答案")
        store_embedding(q1, [1.0, 0.0])
        # 近似问句（不同 hash）但嵌入同向 -> 语义级命中
        q2 = f"q2-{uuid.uuid4().hex[:8]}"
        hit = lookup(q2, embed_fn=lambda s: [0.995, 0.1])
        assert hit is not None
        assert hit["hit"] == "semantic"
        assert hit["answer"] == "语义答案"
        assert hit["similarity"] >= 0.95

    def test_semantic_miss_different_topic(self):
        q1 = f"q-{uuid.uuid4().hex[:8]}"
        store(q1, "正交问题的答案")
        store_embedding(q1, [1.0, 0.0])
        q2 = f"q2-{uuid.uuid4().hex[:8]}"
        assert lookup(q2, embed_fn=lambda s: [0.0, 1.0]) is None

    def test_invalidate_all_clears_everything(self):
        q = f"q-{uuid.uuid4().hex[:8]}"
        store(q, "待失效")
        store_embedding(q, [1.0, 0.0])
        deleted = invalidate_all()
        assert deleted >= 2  # 答案键 + 嵌入哈希
        assert lookup(q) is None

    def test_ttl_is_set(self):
        q = f"q-{uuid.uuid4().hex[:8]}"
        store(q, "带过期")
        ttl = get_redis().ttl(f"docpilot:qa:{question_hash(q)}")
        assert 0 < ttl <= 3600


@pytest.mark.docker
class TestDistributedLock:
    def test_acquire_release_cycle(self):
        name = f"lock:test:{uuid.uuid4().hex[:8]}"
        token = acquire_lock(name, 10)
        assert token is not None
        assert release_lock(name, token) is True
        # 释放后可再取
        token2 = acquire_lock(name, 10)
        assert token2 is not None
        release_lock(name, token2)

    def test_second_acquire_blocked(self):
        name = f"lock:test:{uuid.uuid4().hex[:8]}"
        t1 = acquire_lock(name, 10)
        assert acquire_lock(name, 10) is None
        release_lock(name, t1)

    def test_release_with_wrong_token_fails(self):
        name = f"lock:test:{uuid.uuid4().hex[:8]}"
        t1 = acquire_lock(name, 10)
        # 伪造别人的 token：删除必须失败（锁还在）
        assert release_lock(name, "not-the-token") is False
        # 真正的持锁者仍能释放
        assert release_lock(name, t1) is True


@pytest.mark.docker
class TestTokenBucket:
    def test_allows_capacity_then_denies(self):
        bucket = RedisTokenBucket()
        key = f"tb-{uuid.uuid4().hex[:8]}"
        try:
            assert bucket.check(key, limit=3, window_sec=60) == (True, 0)
            assert bucket.check(key, limit=3, window_sec=60) == (True, 0)
            assert bucket.check(key, limit=3, window_sec=60) == (True, 0)
            allowed, retry = bucket.check(key, limit=3, window_sec=60)
            assert allowed is False
            assert retry >= 1
        finally:
            bucket.reset(key)

    def test_refill_over_time(self):
        # 时间由 Python 注入：确定性验证"令牌随时间恢复"
        bucket = RedisTokenBucket()
        key = f"tb-{uuid.uuid4().hex[:8]}"
        try:
            now = 1000.0
            bucket.check(key, limit=1, window_sec=60, now=now)
            assert bucket.check(key, limit=1, window_sec=60, now=now)[0] is False
            # 60 秒后桶按速率补满 1 个
            allowed, _ = bucket.check(key, limit=1, window_sec=60, now=now + 61)
            assert allowed is True
        finally:
            bucket.reset(key)

    def test_keys_independent(self):
        bucket = RedisTokenBucket()
        k1, k2 = f"tb-{uuid.uuid4().hex[:6]}", f"tb-{uuid.uuid4().hex[:6]}"
        try:
            bucket.check(k1, limit=1, window_sec=60)
            assert bucket.check(k1, limit=1, window_sec=60)[0] is False
            assert bucket.check(k2, limit=1, window_sec=60)[0] is True
        finally:
            bucket.reset(k1)
            bucket.reset(k2)
