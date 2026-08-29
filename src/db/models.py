"""DocPilot 业务表的 SQLAlchemy 模型（MySQL）。

与 memory/ 目录的 LangGraph checkpointer 分离：那套管 agent 会话状态，
这套管业务数据（文档元数据、对话记录、任务、反馈），全部走 MySQL。

表设计对应的面试考点：
- documents.file_hash 唯一索引 -> 秒传/防重复上传
- documents.status 状态机 -> pending/processing/ready/failed
- chat_messages (session_id, id) 复合游标分页 -> 深分页优化
- IngestType JSON 列 -> MySQL 8 JSON 类型
"""

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CHAR,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"        # 已登记，等待解析
    PROCESSING = "processing"  # 解析+嵌入中
    READY = "ready"            # 已入向量库，可检索
    FAILED = "failed"          # 解析或嵌入失败


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Document(Base):
    """上传文档的元数据。content 本体在 Chroma 向量库里，这里只管登记与状态。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_hash: Mapped[str] = mapped_column(CHAR(32), nullable=False)  # MD5 十六进制
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False, length=16),
        default=DocumentStatus.PENDING,
        nullable=False,
    )
    chunks_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        # 秒传：同哈希同文件名视为同一文档，重复上传直接命中
        UniqueConstraint("file_hash", "filename", name="uq_documents_hash_filename"),
        Index("idx_documents_status_created", "status", "created_at"),
        CheckConstraint("size_bytes >= 0", name="ck_documents_size_positive"),
    )


class ChatSession(Base):
    """一次问答会话（对应前端一个 thread）。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    title: Mapped[str] = mapped_column(String(255), default="新会话", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ChatMessage(Base):
    """会话内的单条消息。游标分页按 (session_id, id) 走复合索引。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, native_enum=False, length=16), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # 引用的 chunk 元数据
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)  # 生成耗时
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        # 游标分页索引：WHERE session_id=? AND id>? ORDER BY id LIMIT n
        Index("idx_messages_session_id", "session_id", "id"),
    )


class IngestJob(Base):
    """阶段 3 Celery 异步任务的记录表（先建表占位，接入在阶段 3）。"""

    __tablename__ = "ingest_jobs"

    id: Mapped[str] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    state: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)  # queued/running/done/dead
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_jobs_state_created", "state", "created_at"),)


class FeedbackEvent(Base):
    """用户对回答的点赞/点踩，替代只进 LangSmith 的单向反馈。"""

    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # like/dislike
    reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("session_id", "message_id", "kind", name="uq_feedback_once"),
        Index("idx_feedback_session", "session_id", "id"),
    )
