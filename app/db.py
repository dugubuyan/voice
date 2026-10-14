"""数据库连接与会话管理（SQLite + SQLAlchemy 2.0）。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import config


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


_engine = create_engine(
    config.database_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False}
    if config.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def get_engine():
    return _engine


def init_db() -> None:
    """建表 + 轻量列迁移（SQLite）。"""
    # 确保模型已注册到 Base.metadata
    from app import models  # noqa: F401

    Base.metadata.create_all(_engine)
    _ensure_sqlite_columns()


# 既有库上补齐后加的列（SQLite 支持带默认值的 ADD COLUMN；已存在则跳过）。
# 保数据、幂等：每次启动检查一次，缺列才加。
_SQLITE_ADD_COLUMNS = {
    "elders": {
        "address": "ALTER TABLE elders ADD COLUMN address VARCHAR(256) DEFAULT ''",
        "allergies": "ALTER TABLE elders ADD COLUMN allergies JSON DEFAULT '[]'",
    },
}


def _ensure_sqlite_columns() -> None:
    if not config.database_url.startswith("sqlite"):
        return
    from sqlalchemy import text

    with _engine.begin() as conn:
        for table, columns in _SQLITE_ADD_COLUMNS.items():
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            if not existing:
                continue  # 表还不存在（create_all 之后一般已建）；跳过
            for col, ddl in columns.items():
                if col not in existing:
                    conn.execute(text(ddl))


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务性会话上下文。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
