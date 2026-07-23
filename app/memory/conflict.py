"""事实级冲突解析：recency-wins + 显式失效（不物理删除，保留审计痕迹）。

冲突判定规则（MVP，规则驱动）：
  同 kind + 同 remind_date（均非空时）→ 视为同一约定的更新
  同 kind + 均无 remind_date + content 关键词重叠度高 → 视为同一事实更新
  其余 → 不冲突，直接新增

旧记录：active=False + superseded_by 指向新记录 id，不物理删除。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.notes import ExtractedNote, NoteStore
from app.models import MemoryNote


def _keyword_overlap(a: str, b: str, threshold: float = 0.4) -> bool:
    """粗略判断两段文本关键词重叠度是否超过阈值（Jaccard 字符集）。"""
    if not a or not b:
        return False
    sa = set(a.replace(" ", ""))
    sb = set(b.replace(" ", ""))
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= threshold


class MemoryConflictResolver:
    """事实级冲突解析器。通过 upsert_note 代替直接调 NoteStore.add_note。"""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._store = NoteStore(session)

    def upsert_note(self, elder_id: int, candidate: ExtractedNote) -> MemoryNote:
        """写入新事实，并把语义冲突的旧记录显式失效。

        冲突判定优先级：
          1. 同 kind + 同 remind_date（两者均非空）→ 冲突
          2. 同 kind + 均无 remind_date + content 关键词高度重叠 → 冲突
          否则不冲突，直接新增。
        """
        existing = self._find_conflicting(elder_id, candidate)
        new_row = self._store.add_note(elder_id, candidate)

        if existing is not None:
            existing.active = False              # 显式失效
            existing.superseded_by = new_row.id  # 取代指针
            self._session.flush()

        return new_row

    def _find_conflicting(
        self, elder_id: int, candidate: ExtractedNote
    ) -> MemoryNote | None:
        """找到与 candidate 语义冲突的最近一条 active 记录；找不到返回 None。"""
        stmt = (
            select(MemoryNote)
            .where(
                MemoryNote.elder_id == elder_id,
                MemoryNote.active == True,  # noqa: E712
                MemoryNote.kind == candidate.kind,
            )
            .order_by(MemoryNote.created_at.desc())
            .limit(10)
        )
        candidates = list(self._session.execute(stmt).scalars().all())

        for row in candidates:
            # 规则 1：同 remind_date（均非空）
            if candidate.date_iso and row.remind_date == candidate.date_iso:
                return row
            # 规则 2：均无日期 + 内容关键词高度重叠
            if not candidate.date_iso and not row.remind_date:
                if _keyword_overlap(candidate.content, row.content):
                    return row
        return None
