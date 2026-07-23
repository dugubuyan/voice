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
    """建表。"""
    # 确保模型已注册到 Base.metadata
    from app import models  # noqa: F401

    Base.metadata.create_all(_engine)


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
