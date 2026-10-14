"""Consolidator 测试：归并 / 画像重写 / 趋势 / 幂等 / 降级 / FakeLLM 分支。

用 conftest 的内存 DB（session 夹具）。覆盖 design.md 的正确性属性：
- 幂等（需求 9.4）：N 条未巩固 Episode，consolidate 一次后全部 consolidated=True；
  再调一次 → no-op（不重复建 scene、不重复处理）。
- 同 theme 唯一 active（P5，需求 4.3）：两次巩固产生同 theme，断言至多一条 active。
- 画像重写（需求 4.2）：consolidate 后 ElderProfile 存在且非空；再次巩固是「重写」
  而非无限追加（长度有界）。
- 趋势（需求 7.3）：TrendMemory 写入；health 趋势的 based_on 溯源到结构化 HealthReading
  的 id（数值来自结构化源，不凭空生成）。
- 降级（需求 4.5）：LLM 抛错 → 既有数据不变、Episode 仍未巩固（可下次重试）、不抛异常。
- FakeLLM 对【记忆巩固】返回可解析合法 JSON。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.agent.llm import FakeLLMClient
from app.memory.consolidate import (
    CONSOLIDATE_PROMPT_MARKER,
    Consolidator,
    ConsolidationResult,
)
from app.memory.episode import EpisodeStore
from app.memory.events import EventStore
from app.models import Elder, ElderProfile, Episode, MemScene, TrendMemory


# ──────────────────────────────────────────
# 夹具 / 辅助
# ──────────────────────────────────────────

def _elder_id(session) -> int:
    elder = Elder(name="赵德海", age=80)
    session.add(elder)
    session.flush()
    return elder.id


def _add_episode(session, elder_id, summary, *, day=1, consolidated=False) -> Episode:
    ep = Episode(
        elder_id=elder_id,
        summary=summary,
        mood="平静",
        topics=[],
        started_at=datetime(2024, 1, day, 9, 0, 0),
        ended_at=datetime(2024, 1, day, 9, 30, 0),
        consolidated=consolidated,
    )
    session.add(ep)
    session.flush()
    return ep


class _BoomLLM:
    """恒抛错的 LLM 桩，用于降级路径测试。"""

    def chat(self, messages):
        raise RuntimeError("llm down")


class _StubLLM:
    """可控返回的 LLM 桩：按固定 JSON 返回，便于精确断言。"""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def chat(self, messages):
        self.calls.append(messages[-1]["content"])
        return json.dumps(self._payload, ensure_ascii=False)


def _active_scenes(session, elder_id, theme=None):
    stmt = select(MemScene).where(
        MemScene.elder_id == elder_id,
        MemScene.active == True,  # noqa: E712
    )
    if theme is not None:
        stmt = stmt.where(MemScene.theme == theme)
    return list(session.execute(stmt).scalars().all())


# ──────────────────────────────────────────
# 幂等
# ──────────────────────────────────────────

def test_consolidate_marks_all_and_is_idempotent(session):
    elder_id = _elder_id(session)
    for i in range(3):
        _add_episode(session, elder_id, f"第{i}段对话", day=i + 1)

    cons = Consolidator(FakeLLMClient(), session)
    result = cons.consolidate(elder_id)

    # 全部被处理并标记
    assert len(result.processed_episode_ids) == 3
    assert EpisodeStore(session).unconsolidated(elder_id) == []
    session.expire_all()
    for ep in session.execute(select(Episode)).scalars().all():
        assert ep.consolidated is True

    scenes_after_first = _active_scenes(session, elder_id)
    assert len(scenes_after_first) >= 1

    # 再调一次：无未巩固 → no-op，不重复建 scene
    result2 = cons.consolidate(elder_id)
    assert result2.is_empty
    assert result2.processed_episode_ids == []
    scenes_after_second = _active_scenes(session, elder_id)
    assert len(scenes_after_second) == len(scenes_after_first)


def test_consolidate_empty_is_noop(session):
    elder_id = _elder_id(session)
    cons = Consolidator(FakeLLMClient(), session)
    result = cons.consolidate(elder_id)
    assert result.is_empty
    assert _active_scenes(session, elder_id) == []
    assert session.execute(select(ElderProfile)).scalars().first() is None


# ──────────────────────────────────────────
# 同 theme 唯一 active（P5）
# ──────────────────────────────────────────

def test_same_theme_single_active_scene(session):
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "第一批对话", day=1)

    # FakeLLM 固定返回 theme="日常点滴"
    cons = Consolidator(FakeLLMClient(), session)
    cons.consolidate(elder_id)

    # 再来一批 Episode，第二次巩固产生相同 theme
    _add_episode(session, elder_id, "第二批对话", day=2)
    cons.consolidate(elder_id)

    active = _active_scenes(session, elder_id, theme="日常点滴")
    assert len(active) == 1
    # 第二次的 member 被并入同一条 active scene
    assert len(active[0].member_episode_ids) == 2


# ──────────────────────────────────────────
# 画像重写：存在、非空、有界（非无限追加）
# ──────────────────────────────────────────

def test_profile_rewritten_not_appended(session):
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "聊了聊天气", day=1)

    cons = Consolidator(FakeLLMClient(), session)
    cons.consolidate(elder_id)

    prof = session.execute(select(ElderProfile)).scalars().first()
    assert prof is not None
    assert prof.content.strip() != ""
    len_after_first = len(prof.content)

    # 第二次巩固：整体重写，不是无限追加
    _add_episode(session, elder_id, "又聊了聊", day=2)
    cons.consolidate(elder_id)
    session.refresh(prof)
    # 单文档仍唯一
    assert len(session.execute(select(ElderProfile)).scalars().all()) == 1
    # 长度没有线性增长（重写而非追加）
    assert len(prof.content) == len_after_first


def test_profile_bounded_length(session):
    """LLM 返回超长 profile 时被截断到有界长度。"""
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "一段对话", day=1)

    huge = "养" * 5000
    stub = _StubLLM(
        {
            "scenes": [{"theme": "日常", "summary": "综合", "member_indices": [0]}],
            "profile": huge,
            "trends": [],
        }
    )
    cons = Consolidator(stub, session)
    cons.consolidate(elder_id)

    prof = session.execute(select(ElderProfile)).scalars().first()
    assert prof is not None
    assert len(prof.content) <= 1500


# ──────────────────────────────────────────
# 趋势：写入 + 数值来自结构化源
# ──────────────────────────────────────────

def test_trend_health_based_on_structured_source(session):
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "聊到最近血压", day=1)

    # 结构化健康读数（精确源）
    events = EventStore(session)
    r1 = events.add_health_reading(elder_id, "blood_pressure", "130/85", "mmHg")
    r2 = events.add_health_reading(elder_id, "blood_pressure", "128/84", "mmHg")

    # 桩 LLM 返回 health 趋势（描述不含编造数值）
    stub = _StubLLM(
        {
            "scenes": [{"theme": "健康", "summary": "关注血压", "member_indices": [0]}],
            "profile": "老人关注血压，作息规律。",
            "trends": [
                {"kind": "health", "content": "近期血压总体平稳，数值以结构化记录为准。"},
                {"kind": "mood", "content": "情绪平稳。"},
            ],
        }
    )
    cons = Consolidator(stub, session)
    result = cons.consolidate(elder_id)

    assert result.trend_count == 2
    health_trend = session.execute(
        select(TrendMemory).where(TrendMemory.kind == "health")
    ).scalars().first()
    assert health_trend is not None
    # based_on 溯源到结构化 HealthReading 的 id（数值不凭空生成）
    assert set(health_trend.based_on) == {r1.id, r2.id}

    # mood 趋势无数值依据
    mood_trend = session.execute(
        select(TrendMemory).where(TrendMemory.kind == "mood")
    ).scalars().first()
    assert mood_trend is not None
    assert mood_trend.based_on == []


def test_trend_upsert_by_kind(session):
    """同 kind 趋势更新而非重复新增。"""
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "第一段", day=1)

    stub = _StubLLM(
        {
            "scenes": [{"theme": "日常", "summary": "s", "member_indices": [0]}],
            "profile": "p",
            "trends": [{"kind": "mood", "content": "情绪平稳"}],
        }
    )
    cons = Consolidator(stub, session)
    cons.consolidate(elder_id)

    _add_episode(session, elder_id, "第二段", day=2)
    stub._payload["trends"] = [{"kind": "mood", "content": "情绪略有好转"}]
    cons.consolidate(elder_id)

    mood_trends = session.execute(
        select(TrendMemory).where(TrendMemory.kind == "mood")
    ).scalars().all()
    assert len(mood_trends) == 1
    assert mood_trends[0].content == "情绪略有好转"


# ──────────────────────────────────────────
# 降级：LLM 抛错 → 既有数据不变、Episode 仍未巩固、不抛异常
# ──────────────────────────────────────────

def test_llm_failure_degrades_without_side_effects(session):
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "一段对话", day=1)
    _add_episode(session, elder_id, "又一段", day=2)

    cons = Consolidator(_BoomLLM(), session)
    result = cons.consolidate(elder_id)  # 不抛异常

    assert result.is_empty
    # Episode 仍全部未巩固（下次可重试）
    assert len(EpisodeStore(session).unconsolidated(elder_id)) == 2
    # 没有产生任何 scene / 画像 / 趋势
    assert _active_scenes(session, elder_id) == []
    assert session.execute(select(ElderProfile)).scalars().first() is None
    assert session.execute(select(TrendMemory)).scalars().all() == []


def test_invalid_json_degrades(session):
    """LLM 返回不合法 JSON → 降级，不损坏既有数据。"""
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "一段", day=1)

    class _BadJSON:
        def chat(self, messages):
            return "这不是 JSON"

    cons = Consolidator(_BadJSON(), session)
    result = cons.consolidate(elder_id)
    assert result.is_empty
    assert len(EpisodeStore(session).unconsolidated(elder_id)) == 1


def test_llm_failure_preserves_existing_memory(session):
    """已有 MemScene/画像在一次失败巩固后保持不变。"""
    elder_id = _elder_id(session)
    _add_episode(session, elder_id, "第一段", day=1)

    # 先用 FakeLLM 成功巩固一次，建立既有记忆
    Consolidator(FakeLLMClient(), session).consolidate(elder_id)
    prof_before = session.execute(select(ElderProfile)).scalars().first().content
    scenes_before = len(_active_scenes(session, elder_id))

    # 新增未巩固 Episode，但这次 LLM 失败
    _add_episode(session, elder_id, "第二段", day=2)
    Consolidator(_BoomLLM(), session).consolidate(elder_id)

    # 既有画像/场景不变；新 Episode 仍未巩固
    prof_after = session.execute(select(ElderProfile)).scalars().first().content
    assert prof_after == prof_before
    assert len(_active_scenes(session, elder_id)) == scenes_before
    assert len(EpisodeStore(session).unconsolidated(elder_id)) == 1


# ──────────────────────────────────────────
# FakeLLM 对【记忆巩固】返回可解析合法 JSON
# ──────────────────────────────────────────

def test_fake_llm_consolidate_branch_returns_valid_json():
    fake = FakeLLMClient()
    prompt = (
        f"{CONSOLIDATE_PROMPT_MARKER} ...\n"
        "【待巩固情景】\n[0] 聊天\n[1] 散步\n\n"
        "【现有画像】\n（暂无）\n\n【现有场景主题】\n（暂无）\n\n"
        "【健康读数】\n- blood_pressure: 130/85 mmHg\n"
    )
    raw = fake.chat([{"role": "user", "content": prompt}])
    data = json.loads(raw)

    assert isinstance(data["scenes"], list) and len(data["scenes"]) >= 1
    scene = data["scenes"][0]
    assert scene["theme"]
    assert scene["member_indices"] == [0, 1]
    assert isinstance(data["profile"], str) and data["profile"].strip()
    # 含结构化健康读数 → 应给出 health 趋势
    kinds = {t["kind"] for t in data["trends"]}
    assert "health" in kinds and "mood" in kinds


def test_fake_llm_consolidate_no_health_omits_health_trend():
    fake = FakeLLMClient()
    prompt = (
        f"{CONSOLIDATE_PROMPT_MARKER}\n"
        "【待巩固情景】\n[0] 聊天\n\n【现有画像】\n（暂无）\n\n"
        "【现有场景主题】\n（暂无）\n\n【健康读数】\n（暂无结构化健康读数）\n"
    )
    data = json.loads(fake.chat([{"role": "user", "content": prompt}]))
    kinds = {t["kind"] for t in data["trends"]}
    assert "mood" in kinds
    assert "health" not in kinds
