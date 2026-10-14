"""近期对话原文落盘（RecentTurn）：append-only 落盘 + 重启恢复 + 巩固标记 + 清理。

设计对应需求 8：
- 每轮把 user/assistant 原文 append 落盘（best-effort，落盘失败绝不影响对话返回）。
- get_runtime 时按时间恢复最近窗口。
- 已巩固为 Episode 的旧轮次可被清理 Trigger 清除。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.agent.runtime import Turn
from app.models import RecentTurn


class RecentTurnLog:
    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, elder_id: int, turn) -> None:
        """把一个 Turn(role/content/at) 落盘。

        best-effort：任何异常都吞掉不外抛——对话链路绝不能被落盘失败影响（需求 8.3）。
        """
        try:
            row = RecentTurn(
                elder_id=elder_id,
                role=turn.role,
                content=turn.content,
                at=turn.at,
            )
            self._session.add(row)
            self._session.flush()
        except Exception:
            # 落盘失败静默降级；回滚以免污染后续同一 session 的写入。
            try:
                self._session.rollback()
            except Exception:
                pass

    def load_recent(self, elder_id: int, limit: int) -> list[Turn]:
        """按时间升序返回最近 limit 条，转回 Turn 对象（用于重启恢复窗口）。"""
        stmt = (
            select(RecentTurn)
            .where(RecentTurn.elder_id == elder_id)
            .order_by(RecentTurn.at.desc(), RecentTurn.id.desc())
            .limit(limit)
        )
        rows = list(self._session.execute(stmt).scalars().all())
        rows.reverse()  # desc 取最近 limit 条后翻转为时间升序
        return [Turn(role=r.role, content=r.content, at=r.at) for r in rows]

    def mark_consolidated(self, turn_ids: list[int], episode_id: int) -> None:
        """把这些行标记为已巩固，并关联到给定 Episode。"""
        if not turn_ids:
            return
        stmt = select(RecentTurn).where(RecentTurn.id.in_(turn_ids))
        for row in self._session.execute(stmt).scalars().all():
            row.consolidated = True
            row.episode_id = episode_id
        self._session.flush()

    def purge_consolidated(self, elder_id: int, keep_days: int) -> int:
        """删除已巩固且超期（at 早于 now - keep_days 天）的行，返回删除条数。

        不删 consolidated=False 的行（未巩固原文必须保留，重启才能恢复）。
        """
        cutoff = datetime.now() - timedelta(days=keep_days)
        stmt = delete(RecentTurn).where(
            RecentTurn.elder_id == elder_id,
            RecentTurn.consolidated == True,  # noqa: E712
            RecentTurn.at < cutoff,
        )
        result = self._session.execute(stmt)
        self._session.flush()
        return result.rowcount or 0
