"""ForesightStore 测试：到期边界 / expire_due 幂等 / deactivate / due_date 不改写。

对应需求 3、Property 4（单调失活）、Property 8（时间二分：不改事件时间）。
"""
from __future__ import annotations

from datetime import date, timedelta

from app.memory.foresight import ForesightDraft, ForesightStore


def _add(store, elder_id, content="复查血糖", kind="appointment", due=""):
    return store.add(elder_id, ForesightDraft(content=content, kind=kind, due_date=due))


def test_due_today_still_valid(session):
    """到期日=今天 → 未过期，仍在 active_valid，且 expire_due 不处理它。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    fs = _add(store, 1, due=today.isoformat())

    assert store.expire_due(1, today) == 0
    valid = store.active_valid(1, today)
    assert fs.id in [f.id for f in valid]
    assert fs.active is True


def test_due_yesterday_expires(session):
    """到期日=昨天 → expire_due 失活，且不再出现在 active_valid。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    yesterday = today - timedelta(days=1)
    fs = _add(store, 1, due=yesterday.isoformat())

    assert store.expire_due(1, today) == 1
    assert fs.active is False
    assert fs.reason == "expired"
    assert fs.deactivated_at is not None
    assert fs.id not in [f.id for f in store.active_valid(1, today)]


def test_future_due_not_expired(session):
    """未来到期日不受影响。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    fs = _add(store, 1, due=(today + timedelta(days=5)).isoformat())

    assert store.expire_due(1, today) == 0
    assert fs.id in [f.id for f in store.active_valid(1, today)]


def test_no_due_date_always_valid(session):
    """空到期日：视为无期限，始终有效、永不过期。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    fs = _add(store, 1, due="")

    assert store.expire_due(1, today) == 0
    assert fs.id in [f.id for f in store.active_valid(1, today)]


def test_expire_due_idempotent(session):
    """连续调用两次：第二次返回 0，不改动已失活记录。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    fs = _add(store, 1, due=(today - timedelta(days=2)).isoformat())

    assert store.expire_due(1, today) == 1
    first_at = fs.deactivated_at
    # 第二次不再处理
    assert store.expire_due(1, today) == 0
    assert fs.deactivated_at == first_at
    assert fs.reason == "expired"


def test_deactivate_done_and_cancelled(session):
    """deactivate 完成/取消生效，从 active_valid 移除。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    due = (today + timedelta(days=3)).isoformat()
    done = _add(store, 1, content="买菜", due=due)
    cancelled = _add(store, 1, content="约理发", due=due)

    store.deactivate(done.id, "done")
    store.deactivate(cancelled.id, "cancelled")

    assert done.active is False and done.reason == "done"
    assert cancelled.active is False and cancelled.reason == "cancelled"
    assert done.deactivated_at is not None
    valid_ids = [f.id for f in store.active_valid(1, today)]
    assert done.id not in valid_ids
    assert cancelled.id not in valid_ids


def test_deactivate_missing_id_is_noop(session):
    """对不存在的 id 调 deactivate 不报错。"""
    store = ForesightStore(session)
    store.deactivate(9999, "done")  # 不抛异常即可


def test_active_valid_excludes_expired_and_deactivated(session):
    """active_valid 只返回有效项，排除过期与失活；结果按 due_date 升序。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    expired = _add(store, 1, content="过期项", due=(today - timedelta(days=1)).isoformat())
    far = _add(store, 1, content="远期", due=(today + timedelta(days=10)).isoformat())
    near = _add(store, 1, content="近期", due=(today + timedelta(days=1)).isoformat())
    done = _add(store, 1, content="已完成", due=(today + timedelta(days=2)).isoformat())

    store.expire_due(1, today)
    store.deactivate(done.id, "done")

    valid = store.active_valid(1, today)
    ids = [f.id for f in valid]
    assert expired.id not in ids
    assert done.id not in ids
    # 升序：近期在远期之前
    assert ids == [near.id, far.id]


def test_due_date_never_rewritten_by_expire(session):
    """expire_due 后 due_date 原值不变（C4：绝不改事件时间）。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    original_due = (today - timedelta(days=1)).isoformat()
    fs = _add(store, 1, due=original_due)

    store.expire_due(1, today)
    assert fs.due_date == original_due


def test_due_date_never_rewritten_by_deactivate(session):
    """deactivate 后 due_date 原值不变（C4）。"""
    store = ForesightStore(session)
    original_due = "2025-04-01"
    fs = _add(store, 1, due=original_due)

    store.deactivate(fs.id, "cancelled")
    assert fs.due_date == original_due


def test_add_preserves_due_date_verbatim(session):
    """add 原样保存 due_date，不做归一化改写。"""
    store = ForesightStore(session)
    fs = _add(store, 1, due="2025-12-31")
    assert fs.due_date == "2025-12-31"


def test_invalid_due_date_treated_as_no_deadline(session):
    """非法 due_date 字符串：健壮处理，视为无期限，不过期。"""
    store = ForesightStore(session)
    today = date(2025, 3, 10)
    fs = _add(store, 1, due="not-a-date")

    assert store.expire_due(1, today) == 0
    assert fs.id in [f.id for f in store.active_valid(1, today)]
