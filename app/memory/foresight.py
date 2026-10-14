"""带时限前瞻（Foresight）：会自动过期的未来关注点。需求 3。

生命周期：
- Active：抽取识别出的未来关注点（带 due_date）。有效期内可被召回注入。
- Inactive：now 越过 due_date（过期）/ 老人说已完成或取消。保留审计，不物理删除。

约束 C4：只写「对话时间」（deactivated_at 等 wall-clock），**绝不改写 due_date**（事件时间）。

风格参考 app/memory/notes.py：日期用 date.fromisoformat 解析，对空串/非法值健壮。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Foresight


# ──────────────────────────────────────────────
# 前瞻草稿（抽取产出的轻量数据结构，见 design.md Core Types）
# ──────────────────────────────────────────────

@dataclass
class ForesightDraft:
    content: str
    kind: str               # appointment / commitment / todo
    due_date: str           # YYYY-MM-DD，可空（无期限则不作为 foresight）


# ──────────────────────────────────────────────
# 内部工具：健壮的到期日解析
# ──────────────────────────────────────────────

def _parse_due(due_date: str | None) -> date | None:
    """把 due_date 字符串解析为 date；空串/None/非法值返回 None（视为无期限）。"""
    if not due_date:
        return None
    try:
        return date.fromisoformat(due_date)
    except ValueError:
        return None


# ──────────────────────────────────────────────
# 存储
# ──────────────────────────────────────────────

class ForesightStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        elder_id: int,
        draft: ForesightDraft,
        source_episode_id: int | None = None,
    ) -> Foresight:
        """写入一条前瞻。due_date 原样落库，不做任何改写（C4）。"""
        row = Foresight(
            elder_id=elder_id,
            content=draft.content,
            kind=draft.kind or "commitment",
            due_date=draft.due_date or "",
            active=True,
            source_episode_id=source_episode_id,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def active_valid(self, elder_id: int, today: date) -> list[Foresight]:
        """当前有效的前瞻：active=True 且（无到期日 或 到期日 >= today）。

        按 due_date 升序返回（空到期日排在最后）。
        """
        stmt = (
            select(Foresight)
            .where(Foresight.elder_id == elder_id, Foresight.active == True)  # noqa: E712
        )
        rows = list(self._session.execute(stmt).scalars().all())
        valid: list[Foresight] = []
        for r in rows:
            due = _parse_due(r.due_date)
            if due is None or due >= today:
                valid.append(r)
        # 升序；无到期日（None）排最后
        valid.sort(key=lambda r: (_parse_due(r.due_date) is None, _parse_due(r.due_date) or date.max))
        return valid

    def expire_due(self, elder_id: int, today: date) -> int:
        """把已越过到期日的 active 前瞻失活，返回处理条数。

        条件：active=True 且 due_date 非空可解析 且 due_date < today。
        置 active=False、reason="expired"、deactivated_at=now。
        幂等：已 active=False 的不再处理（第二次调用返回 0）。
        不改写 due_date（C4）。
        """
        stmt = (
            select(Foresight)
            .where(Foresight.elder_id == elder_id, Foresight.active == True)  # noqa: E712
        )
        rows = list(self._session.execute(stmt).scalars().all())
        now = datetime.now()
        count = 0
        for r in rows:
            due = _parse_due(r.due_date)
            if due is not None and due < today:
                r.active = False
                r.reason = "expired"
                r.deactivated_at = now
                count += 1
        if count:
            self._session.flush()
        return count

    def deactivate(self, foresight_id: int, reason: str) -> None:
        """老人明确说已完成/取消时失活。reason 取 done/cancelled。

        置 active=False、写 reason、deactivated_at=now。不改写 due_date（C4）。
        """
        row = self._session.get(Foresight, foresight_id)
        if row is None:
            return
        row.active = False
        row.reason = reason
        row.deactivated_at = datetime.now()
        self._session.flush()
