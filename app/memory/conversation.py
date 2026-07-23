"""对话记忆（跨会话）：保存结构化摘要，而非全量历史。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ConversationSummary


class ConversationMemory:
    def __init__(self, session: Session) -> None:
        self._session = session

    def last_summary(self, elder_id: int) -> ConversationSummary | None:
        stmt = (
            select(ConversationSummary)
            .where(ConversationSummary.elder_id == elder_id)
            .order_by(ConversationSummary.started_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()

    def save_summary(
        self,
        elder_id: int,
        session_id: str,
        summary: str,
        mood: str = "",
        topics: list[str] | None = None,
        started_at: datetime | None = None,
    ) -> ConversationSummary:
        row = ConversationSummary(
            elder_id=elder_id,
            session_id=session_id,
            summary=summary,
            mood=mood,
            topics=topics or [],
            started_at=started_at or datetime.now(),
            ended_at=datetime.now(),
        )
        self._session.add(row)
        self._session.flush()
        return row
