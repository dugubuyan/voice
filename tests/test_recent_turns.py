"""RecentTurnLog：落盘/恢复/巩固标记/清理。对应需求 8。"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from app.agent.runtime import Turn
from app.memory.recent import RecentTurnLog
from app.models import RecentTurn


def _turn(role: str, content: str, at: datetime) -> Turn:
    return Turn(role=role, content=content, at=at)


def test_append_then_load_recent_order_and_fields(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)

    base = datetime(2024, 1, 1, 9, 0, 0)
    log.append(elder_id, _turn("user", "早上好", base))
    log.append(elder_id, _turn("assistant", "早呀，睡得好吗", base + timedelta(seconds=1)))
    log.append(elder_id, _turn("user", "还行", base + timedelta(seconds=2)))

    loaded = log.load_recent(elder_id, limit=10)
    assert [t.role for t in loaded] == ["user", "assistant", "user"]
    assert [t.content for t in loaded] == ["早上好", "早呀，睡得好吗", "还行"]
    # 时间升序
    assert loaded[0].at <= loaded[1].at <= loaded[2].at


def test_load_recent_respects_limit_keeps_latest(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)

    base = datetime(2024, 1, 1, 9, 0, 0)
    for i in range(5):
        log.append(elder_id, _turn("user", f"第{i}句", base + timedelta(seconds=i)))

    loaded = log.load_recent(elder_id, limit=2)
    # 只取最近 2 条，且时间升序
    assert [t.content for t in loaded] == ["第3句", "第4句"]


def test_load_recent_isolated_by_elder(session, elders):
    a = elders["李桂芳"].id
    names = list(elders.keys())
    b = elders[names[1]].id if len(names) > 1 else a
    log = RecentTurnLog(session)
    base = datetime(2024, 1, 1, 9, 0, 0)
    log.append(a, _turn("user", "A 的话", base))
    if b != a:
        log.append(b, _turn("user", "B 的话", base))
        assert [t.content for t in log.load_recent(a, 10)] == ["A 的话"]
        assert [t.content for t in log.load_recent(b, 10)] == ["B 的话"]


def test_append_swallows_exception(session, elders, monkeypatch):
    """落盘内部异常不得冒泡（需求 8.3）。"""
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)

    def boom(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(session, "add", boom)

    # 不应抛出
    log.append(elder_id, _turn("user", "触发异常", datetime.now()))


def test_mark_consolidated_sets_flags(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)
    base = datetime(2024, 1, 1, 9, 0, 0)
    for i in range(3):
        log.append(elder_id, _turn("user", f"句{i}", base + timedelta(seconds=i)))

    rows = list(
        session.execute(
            select(RecentTurn).where(RecentTurn.elder_id == elder_id).order_by(RecentTurn.id)
        ).scalars().all()
    )
    ids = [r.id for r in rows]

    log.mark_consolidated(ids[:2], episode_id=42)

    refreshed = list(
        session.execute(
            select(RecentTurn).where(RecentTurn.elder_id == elder_id).order_by(RecentTurn.id)
        ).scalars().all()
    )
    assert refreshed[0].consolidated is True and refreshed[0].episode_id == 42
    assert refreshed[1].consolidated is True and refreshed[1].episode_id == 42
    # 未标记的保持原样
    assert refreshed[2].consolidated is False and refreshed[2].episode_id is None


def test_mark_consolidated_empty_is_noop(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)
    log.append(elder_id, _turn("user", "只有一句", datetime.now()))
    # 空列表不报错、不改动
    log.mark_consolidated([], episode_id=1)
    row = session.execute(select(RecentTurn)).scalars().first()
    assert row.consolidated is False


def test_purge_only_consolidated_and_expired(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)
    now = datetime.now()

    # 四类行：
    #  1) 已巩固 + 超期 -> 应删
    #  2) 已巩固 + 未超期 -> 保留
    #  3) 未巩固 + 超期 -> 保留（未巩固绝不删）
    #  4) 未巩固 + 未超期 -> 保留
    old = now - timedelta(days=10)
    fresh = now - timedelta(hours=1)

    r1 = RecentTurn(elder_id=elder_id, role="user", content="巩固且超期", at=old, consolidated=True)
    r2 = RecentTurn(elder_id=elder_id, role="user", content="巩固未超期", at=fresh, consolidated=True)
    r3 = RecentTurn(elder_id=elder_id, role="user", content="未巩固超期", at=old, consolidated=False)
    r4 = RecentTurn(elder_id=elder_id, role="user", content="未巩固未超期", at=fresh, consolidated=False)
    session.add_all([r1, r2, r3, r4])
    session.flush()

    deleted = log.purge_consolidated(elder_id, keep_days=3)
    assert deleted == 1

    remaining = {
        r.content
        for r in session.execute(
            select(RecentTurn).where(RecentTurn.elder_id == elder_id)
        ).scalars().all()
    }
    assert remaining == {"巩固未超期", "未巩固超期", "未巩固未超期"}


def test_purge_returns_zero_when_nothing_matches(session, elders):
    elder_id = elders["李桂芳"].id
    log = RecentTurnLog(session)
    log.append(elder_id, _turn("user", "新鲜且未巩固", datetime.now()))
    assert log.purge_consolidated(elder_id, keep_days=3) == 0
