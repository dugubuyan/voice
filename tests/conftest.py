"""测试夹具：内存数据库 + 种子数据 + FakeLLM 大脑。"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db as db_module
from app.db import Base


@pytest.fixture()
def session():
    """每个测试一个独立的内存 SQLite 会话。"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )
    # 让所有模块共享该引擎/会话工厂
    import app.models  # noqa: F401  确保模型注册

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    # 覆盖 db 模块的会话工厂，供调度器等内部 session_scope 使用
    db_module.SessionLocal = TestSession
    db_module._engine = engine

    s = TestSession()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def elders(session):
    """写入种子数据，返回 {name: Elder}。"""
    from app.models import Elder, Medication
    from seed.seed_data import SEED

    mapping = {}
    for item in SEED:
        elder = Elder(**item["elder"])
        session.add(elder)
        session.flush()
        for m in item["meds"]:
            session.add(Medication(elder_id=elder.id, **m))
        mapping[elder.name] = elder
    session.commit()
    return mapping


@pytest.fixture()
def notifier(session):
    from app.notify.notifier import ConsoleNotifier

    return ConsoleNotifier(session, echo=False)


@pytest.fixture()
def brain(session, notifier):
    from app.agent.brain import Brain
    from app.agent.llm import FakeLLMClient

    return Brain(session, FakeLLMClient(), notifier)


@pytest.fixture()
def started(brain, elders):
    """返回一个便捷函数：为指定老人初始化运行态并返回 (elder_id, elder_id)。"""

    def _start(name: str = "李桂芳") -> tuple[int, int]:
        elder_id = elders[name].id
        brain.get_runtime(elder_id)  # 惰性初始化
        return elder_id, elder_id    # 返回 (elder_id, elder_id) 兼容旧的 (elder_id, sid) 用法

    return _start
