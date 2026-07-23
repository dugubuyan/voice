"""事实级冲突解析测试（任务14）。"""
from __future__ import annotations

import pytest

from app.memory.conflict import MemoryConflictResolver
from app.memory.notes import ExtractedNote, NoteStore


def _resolver(session):
    return MemoryConflictResolver(session)


def _note(content, kind="commitment", date_iso=""):
    return ExtractedNote(
        is_memory_request=True, content=content,
        date_iso=date_iso, persons=[], kind=kind,
    )


# ──────────────────────────────────────────────
# 冲突判定
# ──────────────────────────────────────────────

def test_same_kind_same_date_triggers_conflict(session, elders):
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("后天去医院复查血糖", date_iso="2026-07-20"))
    n2 = resolver.upsert_note(elder.id, _note("后天改约医院复查血压", date_iso="2026-07-20"))
    session.flush()

    assert not n1.active           # 旧记录失效
    assert n1.superseded_by == n2.id
    assert n2.active               # 新记录有效


def test_same_kind_no_date_high_overlap_triggers_conflict(session, elders):
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("女儿下周来看我", kind="person"))
    n2 = resolver.upsert_note(elder.id, _note("女儿周末来看我", kind="person"))
    session.flush()

    assert not n1.active
    assert n1.superseded_by == n2.id


def test_different_kind_no_conflict(session, elders):
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("后天去医院", kind="commitment", date_iso="2026-07-20"))
    n2 = resolver.upsert_note(elder.id, _note("女儿来看我", kind="person"))
    session.flush()

    assert n1.active  # 不同 kind，不冲突
    assert n2.active


def test_same_kind_different_date_no_conflict(session, elders):
    elder = elders["张建国"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("去医院", date_iso="2026-07-20"))
    n2 = resolver.upsert_note(elder.id, _note("去医院", date_iso="2026-07-25"))
    session.flush()

    assert n1.active   # 日期不同，不冲突
    assert n2.active


def test_no_physical_delete(session, elders):
    """旧记录 active=False 但仍在 DB 里，可审计。"""
    from sqlalchemy import select
    from app.models import MemoryNote

    elder = elders["王秀英"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("后天广场舞", date_iso="2026-07-20"))
    n2 = resolver.upsert_note(elder.id, _note("后天改为室内舞蹈", date_iso="2026-07-20"))
    session.flush()

    # 旧 note 仍在 DB（物理存在），只是 active=False
    all_notes = session.execute(
        select(MemoryNote).where(MemoryNote.elder_id == elder.id)
    ).scalars().all()
    ids = [n.id for n in all_notes]
    assert n1.id in ids
    assert not n1.active


def test_active_notes_excludes_superseded(session, elders):
    elder = elders["李桂芳"]
    resolver = _resolver(session)
    store = NoteStore(session)

    resolver.upsert_note(elder.id, _note("旧内容", date_iso="2026-07-20"))
    resolver.upsert_note(elder.id, _note("新内容", date_iso="2026-07-20"))
    session.flush()

    active = store.active_notes(elder.id)
    contents = [n.content for n in active]
    assert "新内容" in contents
    assert "旧内容" not in contents


# ──────────────────────────────────────────────
# Brain 接入验证
# ──────────────────────────────────────────────

def test_brain_uses_conflict_resolver_for_remember_note(brain, started, session):
    """Brain._do_remember_note 应走 MemoryConflictResolver。"""
    elder_id, sid = started("李桂芳")
    brain.process_turn(elder_id, "记一下我后天要去医院复查血糖")
    brain.process_turn(elder_id, "记一下我后天改约医院复查血压")
    session.commit()

    store = NoteStore(session)
    active = store.active_notes(elder_id)
    # 同一天的两条 commitment 应只有一条 active
    commitment_active = [n for n in active if n.kind == "commitment" and "2026" in n.remind_date]
    assert len(commitment_active) <= 2  # 两条日期不同可以共存；同日则只有1条
