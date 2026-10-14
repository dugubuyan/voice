"""冲突/不确定扩展测试（任务 8）：confidence 标注 + source_episode_id 追溯 + 向后兼容。

覆盖点：
- 写入 confidence="uncertain" 的 note，读回标注正确。
- 传 source_episode_id 时正确落库；不传时为 None（向后兼容）。
- recency-wins：同 kind+date 的两条 → 旧 active=False、superseded_by 指向新 id（P12 不变）。
- 现有行为回归：不传新参数时 upsert_note 结果与之前一致。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.memory.conflict import MemoryConflictResolver
from app.memory.notes import ExtractedNote, NoteStore
from app.models import Episode, MemoryNote


def _resolver(session):
    return MemoryConflictResolver(session)


def _note(content, kind="commitment", date_iso=""):
    return ExtractedNote(
        is_memory_request=True, content=content,
        date_iso=date_iso, persons=[], kind=kind,
    )


def _make_episode(session, elder_id) -> Episode:
    now = datetime(2026, 7, 18, 10, 0, 0)
    ep = Episode(
        elder_id=elder_id, summary="聊了聊最近的身体情况",
        mood="平和", topics=["健康"], started_at=now, ended_at=now,
    )
    session.add(ep)
    session.flush()
    return ep


# ──────────────────────────────────────────────
# confidence 标注
# ──────────────────────────────────────────────

def test_uncertain_confidence_persisted(session, elders):
    """写入 uncertain 的 note，读回标注正确。"""
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    row = resolver.upsert_note(
        elder.id, _note("好像说下周要去体检，不太确定"),
        confidence="uncertain",
    )
    session.flush()

    reloaded = session.get(MemoryNote, row.id)
    assert reloaded.confidence == "uncertain"


def test_certain_is_default_confidence(session, elders):
    """不传 confidence 时默认 certain。"""
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    row = resolver.upsert_note(elder.id, _note("后天去医院复查血糖", date_iso="2026-07-20"))
    session.flush()

    assert session.get(MemoryNote, row.id).confidence == "certain"


# ──────────────────────────────────────────────
# source_episode_id 追溯
# ──────────────────────────────────────────────

def test_source_episode_id_persisted(session, elders):
    """传 source_episode_id 时正确落库。"""
    elder = elders["李桂芳"]
    ep = _make_episode(session, elder.id)
    resolver = _resolver(session)

    row = resolver.upsert_note(
        elder.id, _note("孙子考上大学了", kind="event"),
        source_episode_id=ep.id,
    )
    session.flush()

    assert session.get(MemoryNote, row.id).source_episode_id == ep.id


def test_source_episode_id_defaults_none(session, elders):
    """不传 source_episode_id 时为 None（向后兼容）。"""
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    row = resolver.upsert_note(elder.id, _note("女儿周末来看我", kind="person"))
    session.flush()

    assert session.get(MemoryNote, row.id).source_episode_id is None


# ──────────────────────────────────────────────
# recency-wins 审计（P12 不变）
# ──────────────────────────────────────────────

def test_recency_wins_audit_unchanged_with_new_params(session, elders):
    """同 kind+date 的两条 → 旧 active=False、superseded_by 指向新 id。"""
    elder = elders["王秀英"]
    resolver = _resolver(session)
    ep = _make_episode(session, elder.id)

    n1 = resolver.upsert_note(
        elder.id, _note("后天广场舞", date_iso="2026-07-20"),
        source_episode_id=ep.id, confidence="uncertain",
    )
    n2 = resolver.upsert_note(
        elder.id, _note("后天改为室内舞蹈", date_iso="2026-07-20"),
        source_episode_id=ep.id,
    )
    session.flush()

    assert not n1.active                 # 旧记录失效
    assert n1.superseded_by == n2.id     # 取代指针
    assert n2.active                     # 新记录有效

    # 物理未删除，可审计
    all_ids = [
        n.id for n in session.execute(
            select(MemoryNote).where(MemoryNote.elder_id == elder.id)
        ).scalars().all()
    ]
    assert n1.id in all_ids


# ──────────────────────────────────────────────
# 现有行为回归：不传新参数时结果与之前一致
# ──────────────────────────────────────────────

def test_backward_compatible_no_new_params(session, elders):
    """不传新参数时 upsert_note 结果与之前一致。"""
    elder = elders["李桂芳"]
    resolver = _resolver(session)

    n1 = resolver.upsert_note(elder.id, _note("后天去医院复查血糖", date_iso="2026-07-20"))
    n2 = resolver.upsert_note(elder.id, _note("后天改约医院复查血压", date_iso="2026-07-20"))
    session.flush()

    # recency-wins 语义不变
    assert not n1.active
    assert n1.superseded_by == n2.id
    assert n2.active
    # 新列取默认值，不影响旧行为
    assert n2.confidence == "certain"
    assert n2.source_episode_id is None

    # active_notes 只保留新记录
    active = NoteStore(session).active_notes(elder.id)
    contents = [n.content for n in active]
    assert "后天改约医院复查血压" in contents
    assert "后天去医院复查血糖" not in contents
