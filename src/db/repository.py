"""业务数据访问层：秒传、状态机、游标分页、对话记录。

对应改造清单阶段 1 的核心任务。所有函数接收调用方管理的 Session，
事务边界由 service 层的 session_scope 控制。
"""

import hashlib
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import ChatMessage, ChatSession, Document, DocumentStatus, MessageRole


def compute_file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def find_document_by_hash(session: Session, file_hash: str, filename: str) -> Document | None:
    stmt = select(Document).where(
        Document.file_hash == file_hash, Document.filename == filename
    )
    return session.scalar(stmt)


def register_document(session: Session, filename: str, content: bytes) -> tuple[Document, bool]:
    """登记一次上传。返回 (文档, 是否秒传)。

    秒传逻辑：同 hash + 同文件名的 READY 文档已存在时，不再重复解析嵌入，
    直接返回既有记录（created=False 表示库未变）。非 READY（processing/failed）
    依然走全量流程，给重试留活口。
    """
    file_hash = compute_file_hash(content)
    existing = find_document_by_hash(session, file_hash, filename)
    if existing is not None and existing.status == DocumentStatus.READY:
        return existing, False

    if existing is not None:
        # 上次失败/处理中断：复用行，状态推回 pending
        existing.status = DocumentStatus.PENDING
        existing.size_bytes = len(content)
        existing.error_message = None
        return existing, True

    doc = Document(
        file_hash=file_hash,
        filename=filename,
        size_bytes=len(content),
        status=DocumentStatus.PENDING,
    )
    session.add(doc)
    session.flush()  # 拿自增 id
    return doc, True


# 合法状态机迁移：pending -> processing -> ready/failed；failed -> pending(重试)
_TRANSITIONS: dict[DocumentStatus, set[DocumentStatus]] = {
    DocumentStatus.PENDING: {DocumentStatus.PROCESSING},
    DocumentStatus.PROCESSING: {DocumentStatus.READY, DocumentStatus.FAILED},
    DocumentStatus.FAILED: {DocumentStatus.PENDING},
}


def transition_status(
    session: Session,
    document_id: int,
    to_status: DocumentStatus,
    *,
    error_message: str | None = None,
    chunks_added: int | None = None,
    chunks_deleted: int | None = None,
) -> Document:
    """带守卫的状态迁移：非法跳转直接抛 ValueError，防止把 failed 改成 ready 之类的脏状态。"""
    doc = session.get(Document, document_id)
    if doc is None:
        raise ValueError(f"Document {document_id} not found")
    allowed = _TRANSITIONS.get(doc.status, set())
    if to_status not in allowed:
        raise ValueError(f"Illegal transition {doc.status} -> {to_status}")
    doc.status = to_status
    if error_message is not None:
        doc.error_message = error_message
    if chunks_added is not None:
        doc.chunks_added = chunks_added
    if chunks_deleted is not None:
        doc.chunks_deleted = chunks_deleted
    return doc


def list_documents(
    session: Session, *, status: DocumentStatus | None = None, limit: int = 50
) -> list[Document]:
    stmt = select(Document).order_by(Document.id.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(Document.status == status)
    return list(session.scalars(stmt))


def create_session(
    session: Session, title: str = "新会话", session_id: str | None = None
) -> ChatSession:
    chat = ChatSession(id=session_id or uuid.uuid4().hex, title=title)
    session.add(chat)
    session.flush()
    return chat


def get_session_by_id(session: Session, session_id: str) -> ChatSession | None:
    return session.get(ChatSession, session_id)


def append_message(
    session: Session,
    session_id: str,
    role: MessageRole,
    content: str,
    *,
    citations: dict | None = None,
    latency_ms: float | None = None,
) -> ChatMessage:
    # mysql.JSON 会把 Python None 序列化成字符串 'null'（不是 SQL NULL），
    # 所以空值列干脆不进 INSERT，让数据库默认值生效
    optional: dict = {}
    if citations is not None:
        optional["citations"] = citations
    if latency_ms is not None:
        optional["latency_ms"] = latency_ms
    msg = ChatMessage(session_id=session_id, role=role, content=content, **optional)
    session.add(msg)
    session.flush()
    return msg


def list_messages_page(
    session: Session, session_id: str, *, cursor: int | None = None, limit: int = 20
) -> tuple[list[ChatMessage], int | None]:
    """游标分页：WHERE session_id=? AND id>? ORDER BY id LIMIT n。

    返回 (本页消息, 下一页游标)。next_cursor 为 None 表示没有更多。
    相比 OFFSET 深分页，扫描量恒定为 limit 行。
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .limit(limit + 1)
    )
    if cursor is not None:
        stmt = stmt.where(ChatMessage.id > cursor)
    rows = list(session.scalars(stmt))
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = page[-1].id if has_more and page else None
    return page, next_cursor


def count_messages(session: Session, session_id: str) -> int:
    return session.scalar(
        select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
    ) or 0
