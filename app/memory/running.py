"""RunningSummary 读写：每位老人维护一条持续演进的 active 摘要行。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import RunningSummary


class RunningSummaryRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_active(self, elder_id: int) -> RunningSummary | None:
        stmt = (
            select(RunningSummary)
            .where(RunningSummary.elder_id == elder_id, RunningSummary.active == True)  # noqa: E712
            .order_by(RunningSummary.updated_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def upsert(
        self,
        elder_id: int,
        *,
        summary: str,
        mood: str,
        topics: list[str],
        covers_from: datetime,
        covers_to: datetime,
    ) -> RunningSummary:
        """原地更新 active 行；无则创建。

        covers_from 仅首次创建时写入（最早内容时间戳不变）。
        covers_to 每次前移（反映最近并入内容的时间）。
        """
        row = self.get_active(elder_id)
        if row is None:
            row = RunningSummary(
                elder_id=elder_id,
                covers_from=covers_from,
                active=True,
            )
            self._session.add(row)
        row.summary = summary
        row.mood = mood
        row.topics = list(topics)
        row.covers_to = covers_to
        self._session.flush()
        return row

    def build_context(self, elder_id: int) -> str:
        """返回可注入 system prompt 的摘要文本，无则返回空串。"""
        row = self.get_active(elder_id)
        return row.summary if row else ""
