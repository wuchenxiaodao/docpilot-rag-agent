"""阶段 1 持久层测试。

- 纯逻辑（哈希、状态机迁移表）不依赖 MySQL，常跑
- 秒传/游标分页/状态机全链路依赖 MySQL 容器，挂 docker 标记
  （跑法：pytest tests/db --run-docker）
"""

import uuid

import pytest

from db.models import DocumentStatus
from db.repository import (
    _TRANSITIONS,
    compute_file_hash,
    create_session,
    append_message,
    count_messages,
    find_document_by_hash,
    list_messages_page,
    register_document,
    transition_status,
)
from db.session import session_scope


# ---------- 纯逻辑：不需要数据库 ----------


def test_compute_file_hash_stable():
    content = b"hello docpilot"
    assert compute_file_hash(content) == compute_file_hash(content)
    assert compute_file_hash(content) != compute_file_hash(b"hello docpilot!")


def test_state_machine_legal_paths():
    assert DocumentStatus.PROCESSING in _TRANSITIONS[DocumentStatus.PENDING]
    assert DocumentStatus.READY in _TRANSITIONS[DocumentStatus.PROCESSING]
    assert DocumentStatus.FAILED in _TRANSITIONS[DocumentStatus.PROCESSING]
    assert DocumentStatus.PENDING in _TRANSITIONS[DocumentStatus.FAILED]


def test_state_machine_no_skip_ready_to_processing():
    """READY 是终态：不允许 ready -> processing 的回流。"""
    assert DocumentStatus.PROCESSING not in _TRANSITIONS.get(DocumentStatus.READY, set())


# ---------- MySQL 集成：需要容器 ----------


@pytest.mark.docker
class TestRegisterAndStateMachine:
    def test_register_then_ready_and_dedup(self):
        filename = f"t-{uuid.uuid4().hex[:8]}.pdf"
        content = b"fake pdf bytes for hash test"
        with session_scope() as db:
            doc, created = register_document(db, filename, content)
            assert created is True
            assert doc.status == DocumentStatus.PENDING
            did = doc.id

            transition_status(db, did, DocumentStatus.PROCESSING)
            transition_status(
                db, did, DocumentStatus.READY, chunks_added=7, chunks_deleted=0
            )

        # 同 hash 同文件名再传：秒传，created=False，不新增行
        with session_scope() as db:
            again, created2 = register_document(db, filename, content)
            assert created2 is False
            assert again.id == did
            assert again.chunks_added == 7

    def test_failed_doc_can_retry(self):
        filename = f"t-{uuid.uuid4().hex[:8]}.pdf"
        content = b"another fake payload"
        with session_scope() as db:
            doc, _ = register_document(db, filename, content)
            did = doc.id
            transition_status(db, did, DocumentStatus.PROCESSING)
            transition_status(db, did, DocumentStatus.FAILED, error_message="boom")

            # 失败后重传：复用行、状态回 pending
            doc2, created = register_document(db, filename, content)
            assert doc2.id == did
            assert created is True
            assert doc2.status == DocumentStatus.PENDING

    def test_illegal_transition_raises(self):
        filename = f"t-{uuid.uuid4().hex[:8]}.pdf"
        with session_scope() as db:
            doc, _ = register_document(db, filename, b"payload x")
            with pytest.raises(ValueError):
                # pending 直接跳 ready 是非法的
                transition_status(db, doc.id, DocumentStatus.READY)


@pytest.mark.docker
class TestCursorPagination:
    def test_page_through_messages(self):
        with session_scope() as db:
            chat = create_session(db, title="分页测试")
            sid = chat.id
            for i in range(5):
                append_message(db, sid, "user" if i % 2 == 0 else "assistant", f"msg-{i}")

        # limit=2 翻三页
        with session_scope() as db:
            page1, cursor1 = list_messages_page(db, sid, limit=2)
            assert [m.content for m in page1] == ["msg-0", "msg-1"]
            assert cursor1 is not None

            page2, cursor2 = list_messages_page(db, sid, cursor=cursor1, limit=2)
            assert [m.content for m in page2] == ["msg-2", "msg-3"]

            page3, cursor3 = list_messages_page(db, sid, cursor=cursor2, limit=2)
            assert [m.content for m in page3] == ["msg-4"]
            assert cursor3 is None  # 没有更多了
            assert count_messages(db, sid) == 5

    def test_empty_session(self):
        with session_scope() as db:
            chat = create_session(db, title="空会话")
            page, cursor = list_messages_page(db, chat.id)
            assert page == []
            assert cursor is None
