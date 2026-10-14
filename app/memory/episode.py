"""情景记忆持久化 + 检索（Phase 3，需求 2/5/11）。

Episode 是长期记忆的主载体：一段语义连贯对话被抽取为摘要 + 情绪 + 主题，
永久落库、可检索，取代 Phase 1「fold 后销毁」的单条漂移摘要。

检索实现选择——起步用 SQLite LIKE：
    对 summary / topics 做包含匹配（按 ended_at 倒序）。选择 LIKE 而非 FTS5/向量的原因：
    - 实现简单稳健、零额外依赖，跨平台且测试环境必然可用；
    - FTS5 需要 SQLite 编译期开启扩展，部分环境（含 CI/测试）不保证可用，
      引入即有「本地跑得动、别处跑不动」的风险，违背本项目「本地单机、离线可跑」原则；
    - 中文场景下 LIKE 的包含匹配对关键词召回已「必要且充分」，满足起步需要。
    预留：后续可替换为 FTS5 / 向量检索，`search()` 接口签名与返回类型保持不变，
    调用方（Recaller）无需改动。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Episode


class EpisodeStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        elder_id: int,
        *,
        summary: str,
        mood: str,
        topics: list[str],
        started_at,
        ended_at,
        visibility: str = "family",
    ) -> Episode:
        """写入一条情景记忆并返回（已 flush，可读回自增 id）。"""
        episode = Episode(
            elder_id=elder_id,
            summary=summary,
            mood=mood,
            topics=list(topics or []),
            started_at=started_at,
            ended_at=ended_at,
            visibility=visibility,
        )
        self._session.add(episode)
        self._session.flush()
        return episode

    def search(self, elder_id: int, query: str, limit: int = 6) -> list[Episode]:
        """按 query 检索该老人的 Episode（summary/topics 包含匹配，ended_at 倒序）。

        起步用 SQLite LIKE 对 summary 做包含匹配（见模块顶部说明）。topics 是 JSON 列，
        SQLite 默认以 \\uXXXX 转义存储中文，直接 LIKE 匹配不到中文关键词，故 topics 的
        包含匹配在 Python 侧完成——语义仍是「包含匹配」，且更跨平台。空 query 返回空列表。
        """
        q = (query or "").strip()
        if not q:
            return []
        stmt = (
            select(Episode)
            .where(Episode.elder_id == elder_id)
            .order_by(Episode.ended_at.desc())
        )
        rows = self._session.execute(stmt).scalars().all()
        hits = [
            ep
            for ep in rows
            if q in (ep.summary or "")
            or any(q in str(t) for t in (ep.topics or []))
        ]
        return hits[:limit]

    def unconsolidated(self, elder_id: int) -> list[Episode]:
        """返回未巩固的 Episode，按 started_at 升序（供巩固触发按序归并）。"""
        stmt = (
            select(Episode)
            .where(
                Episode.elder_id == elder_id,
                Episode.consolidated == False,  # noqa: E712
            )
            .order_by(Episode.started_at.asc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def mark_consolidated(
        self, episode_ids: list[int], scene_id: int | None
    ) -> None:
        """把给定 Episode 置为已巩固并挂上所属场景（幂等，重复调用不产生副作用）。"""
        if not episode_ids:
            return
        stmt = select(Episode).where(Episode.id.in_(episode_ids))
        for ep in self._session.execute(stmt).scalars().all():
            ep.consolidated = True
            ep.scene_id = scene_id
        self._session.flush()
