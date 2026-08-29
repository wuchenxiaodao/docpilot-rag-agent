"""MySQL 引擎与会话管理。

连接池参数（面试点：为什么不用 Executors 固定线程池的思路类似——
池化复用昂贵的建连开销，上限防打爆数据库连接数）：
- pool_size=5 常驻连接
- max_overflow=10 峰值借出，用完归还即销毁
- pool_pre_ping 用前探活，避免拿到被 MySQL wait_timeout 掐死的连接
- pool_recycle=3600 早于 wait_timeout(默认 8h) 主动换血
"""

import logging
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.settings import settings
from db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def mysql_url() -> str:
    password = settings.MYSQL_PASSWORD.get_secret_value() if settings.MYSQL_PASSWORD else ""
    return (
        f"mysql+pymysql://{settings.MYSQL_USER}:{quote_plus(password)}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
        "?charset=utf8mb4"
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            mysql_url(),
            pool_size=settings.MYSQL_POOL_SIZE,
            max_overflow=settings.MYSQL_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.MYSQL_ECHO,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope():
    """事务作用域：正常提交、异常回滚、总是关闭。"""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_tables() -> None:
    """建表。开发期用 create_all；结构演进后切 Alembic（阶段 1 收尾做）。"""
    Base.metadata.create_all(get_engine())
    logger.info("MySQL tables ensured at %s", settings.MYSQL_DB)
