"""语义巩固器（Consolidator，Phase 3，需求 4/7）。

把未巩固的 Episode 归并为主题 MemScene，重写老人画像 ElderProfile，
并归纳健康/情绪趋势 TrendMemory。一次 LLM 调用产出稳定结构。

设计要点（对应 design.md 组件 6）：
  - 归并：为每个返回 scene 按 theme 找该 elder 的 active MemScene；存在则原地更新
    （summary + 并入 member_episode_ids），不存在则新建。**同 theme 至多一个 active**
    （需求 4.3 / P5）——本实现「同 theme 原地更新那一条 active」，天然保证唯一 active。
  - 画像：ElderProfile（该 elder 唯一一行）用 LLM 返回的 profile **整体重写**，
    有界（截断到 _PROFILE_MAX_CHARS），不无限追加（需求 4.2）。
  - 趋势：TrendMemory 按 kind 更新或新增。**数值只应来自结构化源**——
    prompt 明确要求「趋势描述基于给定健康读数，不要编造数值」，代码侧
    based_on 只记录 HealthReading 的 id（精确源），TrendMemory 不由 LLM 凭空
    生成数值（需求 7.3）。
  - 幂等（需求 9.4）：只处理 unconsolidated 的 Episode，处理后 mark_consolidated；
    重复调用不再处理同一批、不产生重复 scene。
  - 降级（需求 4.5）：LLM 调用/解析失败 → 保持既有 MemScene/画像/趋势不变、
    **不 mark_consolidated**（下次可重试）、静默返回空结果，不抛异常、不损坏既有数据。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.episode import EpisodeStore
from app.memory.events import EventStore
from app.models import ElderProfile, MemScene, TrendMemory

CONSOLIDATE_PROMPT_MARKER = "【记忆巩固】"

# 画像重写有界上限，防止单文档无限增长（需求 4.2）
_PROFILE_MAX_CHARS = 1500
# 用于喂给 prompt 作为趋势依据的健康读数条数
_HEALTH_READING_LIMIT = 10

CONSOLIDATE_PROMPT = """\
【记忆巩固】你在维护一位老人的长期陪护记忆。下面给你【待巩固情景】（按时间从早到晚，
每条前有下标 [i]）、【现有画像】、【现有场景主题】与【健康读数】。请：
1) 把相关情景归并成主题场景（scenes），每个场景给出 theme（主题名）、summary（该主题的
   综合摘要）、member_indices（属于该主题的情景下标数组，指向【待巩固情景】的下标）。
   同一主题请沿用【现有场景主题】里已有的名字，便于归并。
2) 重写一份完整、稳定、有界的老人画像（profile），综合现有画像与新情景，避免流水账。
3) 归纳健康/情绪趋势（trends），每条含 kind（health 或 mood）与 content。
   **趋势描述必须基于给定的【健康读数】，绝不要编造任何数值**；没有可靠依据就不要给该趋势。

只输出 JSON，不要额外文字：
{{"scenes":[{{"theme":"主题名","summary":"综合摘要","member_indices":[0,1]}}],
  "profile":"重写后的完整画像（有界）",
  "trends":[{{"kind":"health/mood","content":"趋势描述，不含编造数值"}}]}}

【待巩固情景】
{episodes}

【现有画像】
{profile}

【现有场景主题】
{themes}

【健康读数】
{health}
"""


@dataclass
class ConsolidationResult:
    """一次巩固产出的可断言摘要（便于测试）。"""

    processed_episode_ids: list[int] = field(default_factory=list)
    scene_ids: list[int] = field(default_factory=list)
    updated_profile: bool = False
    trend_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.processed_episode_ids


class Consolidator:
    """语义巩固器（依赖注入 LLM + DB session）。"""

    def __init__(self, llm, session: Session) -> None:
        self._llm = llm
        self._session = session
        self._episodes = EpisodeStore(session)
        self._events = EventStore(session)

    # ──────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────

    def consolidate(self, elder_id: int) -> ConsolidationResult:
        """把该老人未巩固的 Episode 归并进 MemScene，重写画像，归纳趋势。

        Returns:
            ConsolidationResult。无未巩固 Episode 时为空结果（no-op）；
            LLM 调用/解析失败时同样返回空结果且不改动任何既有数据（降级）。

        Raises:
            不抛出异常。
        """
        episodes = self._episodes.unconsolidated(elder_id)
        if not episodes:
            return ConsolidationResult()  # no-op

        health = self._events.recent_health(elder_id, limit=_HEALTH_READING_LIMIT)

        # 先做「读 LLM + 解析」；失败即降级返回，不触碰任何既有数据（需求 4.5）。
        parsed = self._call_llm(elder_id, episodes, health)
        if parsed is None:
            return ConsolidationResult()

        scenes_data, profile_text, trends_data = parsed

        # ===== 归并 MemScene（同 theme 至多一个 active，P5）=====
        scene_ids: list[int] = []
        assigned: set[int] = set()
        for scene_data in scenes_data:
            theme = str(scene_data.get("theme", "")).strip()
            if not theme:
                continue
            summary = str(scene_data.get("summary", "")).strip()
            member_ep_ids = self._map_indices(
                scene_data.get("member_indices", []), episodes
            )
            scene = self._merge_scene(elder_id, theme, summary, member_ep_ids)
            scene_ids.append(scene.id)
            # 该批 Episode 挂到所属 scene 并标记已巩固
            self._episodes.mark_consolidated(member_ep_ids, scene.id)
            assigned.update(member_ep_ids)

        # 未被任何 scene 引用的 Episode 也要标记已巩固，保证幂等（下次不再处理同一批）
        leftover = [ep.id for ep in episodes if ep.id not in assigned]
        self._episodes.mark_consolidated(leftover, None)

        # ===== 画像整体重写（单文档、有界，需求 4.2）=====
        updated_profile = self._rewrite_profile(elder_id, profile_text)

        # ===== 趋势归纳（数值仅取自结构化源，需求 7.3）=====
        window_from = episodes[0].started_at
        window_to = episodes[-1].ended_at
        trend_count = self._upsert_trends(
            elder_id, trends_data, health, window_from, window_to
        )

        return ConsolidationResult(
            processed_episode_ids=[ep.id for ep in episodes],
            scene_ids=scene_ids,
            updated_profile=updated_profile,
            trend_count=trend_count,
        )

    # ──────────────────────────────────────────
    # LLM 调用 + 解析（失败返回 None → 触发降级）
    # ──────────────────────────────────────────

    def _call_llm(self, elder_id, episodes, health):
        """返回 (scenes, profile, trends) 或 None（失败降级）。"""
        prompt = self._build_prompt(elder_id, episodes, health)
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            scenes = data.get("scenes", [])
            trends = data.get("trends", [])
            profile = str(data.get("profile", "")).strip()
            if not isinstance(scenes, list):
                scenes = []
            if not isinstance(trends, list):
                trends = []
            # 至少要有一个可归并的 scene，否则视为无效产出、降级
            valid_scenes = [s for s in scenes if isinstance(s, dict) and str(s.get("theme", "")).strip()]
            if not valid_scenes:
                raise ValueError("no valid scene")
            return valid_scenes, profile, trends
        except Exception:
            return None

    def _build_prompt(self, elder_id, episodes, health) -> str:
        episode_lines = "\n".join(
            f"[{i}] {(ep.summary or '').strip()}" for i, ep in enumerate(episodes)
        )
        existing_profile = self._existing_profile_text(elder_id) or "（暂无）"
        themes = self._existing_themes(elder_id)
        theme_text = "、".join(themes) if themes else "（暂无）"
        if health:
            health_text = "\n".join(
                f"- {r.kind}: {r.value}{(' ' + r.unit) if r.unit else ''}" for r in health
            )
        else:
            health_text = "（暂无结构化健康读数）"
        return CONSOLIDATE_PROMPT.format(
            episodes=episode_lines or "（无）",
            profile=existing_profile,
            themes=theme_text,
            health=health_text,
        )

    # ──────────────────────────────────────────
    # MemScene 归并
    # ──────────────────────────────────────────

    def _merge_scene(
        self, elder_id: int, theme: str, summary: str, member_ep_ids: list[int]
    ) -> MemScene:
        """按 theme 找该 elder 的 active MemScene：存在则原地更新，否则新建。

        原地更新保证同 elder 同 theme 任意时刻至多一条 active（P5）。
        """
        scene = self._find_active_scene(elder_id, theme)
        if scene is not None:
            if summary:
                scene.summary = summary
            merged = list(scene.member_episode_ids or [])
            for eid in member_ep_ids:
                if eid not in merged:
                    merged.append(eid)
            scene.member_episode_ids = merged  # 整体重赋触发 JSON 变更追踪
        else:
            scene = MemScene(
                elder_id=elder_id,
                theme=theme,
                summary=summary,
                member_episode_ids=list(member_ep_ids),
                active=True,
            )
            self._session.add(scene)
        self._session.flush()
        return scene

    def _find_active_scene(self, elder_id: int, theme: str) -> MemScene | None:
        stmt = (
            select(MemScene)
            .where(
                MemScene.elder_id == elder_id,
                MemScene.theme == theme,
                MemScene.active == True,  # noqa: E712
            )
            .order_by(MemScene.id.desc())
        )
        return self._session.execute(stmt).scalars().first()

    def _existing_themes(self, elder_id: int) -> list[str]:
        stmt = (
            select(MemScene.theme)
            .where(
                MemScene.elder_id == elder_id,
                MemScene.active == True,  # noqa: E712
            )
            .distinct()
        )
        return [t for t in self._session.execute(stmt).scalars().all() if t]

    # ──────────────────────────────────────────
    # 画像重写
    # ──────────────────────────────────────────

    def _existing_profile_text(self, elder_id: int) -> str:
        prof = self._get_profile(elder_id)
        return prof.content if prof else ""

    def _get_profile(self, elder_id: int) -> ElderProfile | None:
        stmt = select(ElderProfile).where(ElderProfile.elder_id == elder_id)
        return self._session.execute(stmt).scalars().first()

    def _rewrite_profile(self, elder_id: int, profile_text: str) -> bool:
        """用 LLM 返回的 profile 整体重写单文档画像（有界）。空则跳过。"""
        text = (profile_text or "").strip()[:_PROFILE_MAX_CHARS]
        if not text:
            return False
        prof = self._get_profile(elder_id)
        if prof is None:
            prof = ElderProfile(elder_id=elder_id, content=text)
            self._session.add(prof)
        else:
            prof.content = text  # 整体重写，非追加（有界稳定）
        self._session.flush()
        return True

    # ──────────────────────────────────────────
    # 趋势归纳
    # ──────────────────────────────────────────

    def _upsert_trends(
        self, elder_id, trends_data, health, window_from, window_to
    ) -> int:
        """按 kind 更新或新增 TrendMemory。

        based_on 只记录结构化 HealthReading 的 id（精确源）——TrendMemory 不承载
        由 LLM 凭空生成的数值，数值溯源一律回到结构化读数（需求 7.3）。
        """
        health_ids = [r.id for r in health]
        count = 0
        for item in trends_data:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).strip()
            content = str(item.get("content", "")).strip()
            if kind not in {"health", "mood"} or not content:
                continue
            # health 趋势的数值依据来自结构化读数 id；mood 趋势无数值依据
            based_on = list(health_ids) if kind == "health" else []
            existing = self._find_trend(elder_id, kind)
            if existing is None:
                self._session.add(
                    TrendMemory(
                        elder_id=elder_id,
                        kind=kind,
                        content=content,
                        window_from=window_from,
                        window_to=window_to,
                        based_on=based_on,
                    )
                )
            else:
                existing.content = content
                existing.window_from = window_from
                existing.window_to = window_to
                existing.based_on = based_on
            self._session.flush()
            count += 1
        return count

    def _find_trend(self, elder_id: int, kind: str) -> TrendMemory | None:
        stmt = (
            select(TrendMemory)
            .where(TrendMemory.elder_id == elder_id, TrendMemory.kind == kind)
            .order_by(TrendMemory.id.desc())
        )
        return self._session.execute(stmt).scalars().first()

    # ──────────────────────────────────────────
    # 工具
    # ──────────────────────────────────────────

    @staticmethod
    def _map_indices(raw_indices, episodes) -> list[int]:
        """把 member_indices（指向输入 Episode 列表的下标）映射回 episode id。"""
        if not isinstance(raw_indices, list):
            return []
        ids: list[int] = []
        n = len(episodes)
        for idx in raw_indices:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                eid = episodes[i].id
                if eid not in ids:
                    ids.append(eid)
        return ids
