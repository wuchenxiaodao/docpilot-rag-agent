"""Alembic 迁移环境：连接串从 core.settings 取（与业务代码同源），
target_metadata 挂 db.Base，支持 autogenerate。"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 把 src 加进 sys.path，使 alembic 既能从仓库根目录跑
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.settings import settings  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 用应用配置覆盖 ini 里的连接串（密码不出现在 alembic.ini）
password = settings.MYSQL_PASSWORD.get_secret_value() if settings.MYSQL_PASSWORD else ""
from urllib.parse import quote_plus  # noqa: E402

config.set_main_option(
    "sqlalchemy.url",
    f"mysql+pymysql://{settings.MYSQL_USER}:{quote_plus(password)}"
    f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}?charset=utf8mb4",
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
