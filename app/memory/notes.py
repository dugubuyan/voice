"""事实型记忆：数据结构 + 日期状态动态标注 + 存储。

意图识别与结构化提取已统一到 NLU 层（app/agent/nlu.py），本模块只负责：
- ExtractedNote：事实记忆的数据结构
- date_status_label：注入 prompt 时按当天实时计算日期状态（今天/明天/已过期…）
- NoteStore：读写与上下文构建
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import MemoryNote


# ──────────────────────────────────────────────
# 事实记忆数据结构
# ──────────────────────────────────────────────

@dataclass
class ExtractedNote:
    is_memory_request: bool
    content: str           # 要记住的核心内容
    date_iso: str          # YYYY-MM-DD，无则空串
    persons: list[str]     # 明确提到的人名
    kind: str              # commitment / person / event / health / other


# ──────────────────────────────────────────────
# 日期状态动态标注
# ──────────────────────────────────────────────

def date_status_label(date_iso: str, today: date | None = None) -> str:
    """把绝对日期转成对 LLM 友好的状态描述，随当天日期实时变化。"""
    if not date_iso:
        return ""
    today = today or date.today()
    try:
        d = date.fromisoformat(date_iso)
    except ValueError:
        return f"[{date_iso}]"
    diff = (d - today).days
    if diff < -1:
        return f"【已过期 {d.month}月{d.day}日，{-diff}天前】"
    if diff == -1:
        return f"【昨天 {d.month}月{d.day}日，已过期】"
    if diff == 0:
        return f"【今天 {d.month}月{d.day}日】"
    if diff == 1:
        return f"【明天 {d.month}月{d.day}日】"
    if diff == 2:
        return f"【后天 {d.month}月{d.day}日】"
    return f"【{d.month}月{d.day}日，还有{diff}天】"


# ──────────────────────────────────────────────
# 存储
# ──────────────────────────────────────────────

class NoteStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_note(self, elder_id: int, note: ExtractedNote) -> MemoryNote:
        row = MemoryNote(
            elder_id=elder_id,
            content=note.content,
            kind=note.kind,
            remind_date=note.date_iso,
            active=True,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def active_notes(self, elder_id: int, limit: int = 20) -> list[MemoryNote]:
        stmt = (
            select(MemoryNote)
            .where(MemoryNote.elder_id == elder_id, MemoryNote.active == True)  # noqa: E712
            .order_by(MemoryNote.created_at.asc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def build_context(self, elder_id: int, today: date | None = None) -> str:
        """整理成可注入 prompt 的事实清单，日期状态实时计算。"""
        today = today or date.today()
        notes = self.active_notes(elder_id)
        if not notes:
            return ""
        lines = []
        for n in notes:
            label = date_status_label(n.remind_date, today)
            lines.append(f"- {label} {n.content}".strip())
        return "\n".join(lines)
