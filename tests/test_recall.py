"""FtsRecaller 测试：多源融合 / 有界截断 / 空结果不编造 / 过期不注入 / 防编造措辞。

对应需求 5（检索式回忆）、Property 6（召回有界）。
使用 conftest 的内存 DB session。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import app.config as config_module
from app.memory.episode import EpisodeStore
from app.memory.foresight import ForesightDraft, ForesightStore
from app.memory.notes import ExtractedNote, NoteStore
from app.memory.recall import (
    INJECTION_HEADER,
    FtsRecaller,
    RecallResult,
)
from app.models import MemScene

ELDER = 1
NOW = datetime(2025, 3, 10, 9, 0, 0)
TODAY = NOW.date()


def _add_episode(session, summary, topics=None, ended=None):
    return EpisodeStore(session).add(
        ELDER,
        summary=summary,
        mood="平静",
        topics=topics or [],
        started_at=ended or NOW,
        ended_at=ended or NOW,
    )


def _add_fact(session, content, kind="other"):
    return NoteStore(session).add_note(
        ELDER, ExtractedNote(True, content, "", [], kind)
    )


def _add_foresight(session, content, due=""):
    return ForesightStore(session).add(
        ELDER, ForesightDraft(content=content, kind="commitment", due_date=due)
    )


# ──────────────────────────────────────────────
# 有界性（Property 6）
# ──────────────────────────────────────────────

def test_bounded_by_max_items(session, monkeypatch):
    """造出远超上限的可命中记忆，断言 items 条数 <= recall_max_items。"""
    monkeypatch.setattr(config_module.config, "recall_max_items", 3)
    monkeypatch.setattr(config_module.config, "recall_token_budget", 100000)

    for i in range(20):
        _add_episode(session, f"关于遛弯的对话第{i}次")
        _add_fact(session, f"遛弯相关事实{i}")
    session.commit()

    result = FtsRecaller(session).recall(ELDER, "遛弯", NOW)
    assert len(result.items) <= 3


def test_bounded_by_token_budget(session, monkeypatch):
    """累计文本长度受 recall_token_budget 限制（字符数近似）。"""
    monkeypatch.setattr(config_module.config, "recall_max_items", 50)
    monkeypatch.setattr(config_module.config, "recall_token_budget", 40)

    for i in range(20):
        _add_episode(session, f"遛弯的一段较长对话摘要内容编号{i}，聊了很久很久")
    session.commit()

    result = FtsRecaller(session).recall(ELDER, "遛弯", NOW)
    total = sum(len(it.text) for it in result.items)
    assert total <= 40


# ──────────────────────────────────────────────
# 空 query / 无命中 → 不编造
# ──────────────────────────────────────────────

def test_empty_query_returns_empty(session):
    _add_episode(session, "聊到遛弯")
    session.commit()
    result = FtsRecaller(session).recall(ELDER, "", NOW)
    assert result.items == []
    assert result.to_injection() == ""


def test_no_match_returns_empty(session):
    _add_episode(session, "聊到遛弯")
    _add_fact(session, "喜欢喝茶")
    session.commit()
    result = FtsRecaller(session).recall(ELDER, "完全不相关的关键词xyz", NOW)
    assert result.items == []
    assert result.to_injection() == ""


# ──────────────────────────────────────────────
# 过期 foresight 不出现在结果里（需求 5.4）
# ──────────────────────────────────────────────

def test_expired_foresight_excluded(session):
    """一个已过期 + 一个有效，只召回有效的那条。"""
    expired = _add_foresight(
        session, "复查血压提醒", due=(TODAY - timedelta(days=2)).isoformat()
    )
    valid = _add_foresight(
        session, "复查血糖提醒", due=(TODAY + timedelta(days=3)).isoformat()
    )
    session.commit()

    result = FtsRecaller(session).recall(ELDER, "复查", NOW)
    ref_ids = {(it.source, it.ref_id) for it in result.items}
    assert ("foresight", valid.id) in ref_ids
    assert ("foresight", expired.id) not in ref_ids


# ──────────────────────────────────────────────
# 跨源融合：同时命中 episode 与 fact
# ──────────────────────────────────────────────

def test_cross_source_fusion(session):
    ep = _add_episode(session, "上周一起聊了养花的事")
    note = _add_fact(session, "喜欢养花，尤其是月季")
    session.commit()

    result = FtsRecaller(session).recall(ELDER, "养花", NOW)
    by_source = {it.source: it for it in result.items}
    assert "episode" in by_source
    assert "fact" in by_source
    assert by_source["episode"].ref_id == ep.id
    assert by_source["fact"].ref_id == note.id
    # fact 基础分高于 episode，应排在前面
    assert result.items[0].source == "fact"


def test_scene_and_person_sources(session):
    """MemScene（active）与 Person 也能被召回并带正确 source。"""
    scene = MemScene(
        elder_id=ELDER,
        theme="健康",
        summary="最近常聊到散步锻炼身体",
        member_episode_ids=[],
        active=True,
    )
    session.add(scene)
    from app.memory.person import PersonStore

    PersonStore(session).upsert_mention(
        ELDER, name="散步搭子老王", relation="邻居", snippet="每天一起散步", at=NOW
    )
    session.commit()

    result = FtsRecaller(session).recall(ELDER, "散步", NOW)
    sources = {it.source for it in result.items}
    assert "scene" in sources
    assert "person" in sources


# ──────────────────────────────────────────────
# to_injection 含防编造措辞
# ──────────────────────────────────────────────

def test_injection_has_anti_fabrication_header(session):
    _add_fact(session, "孙子下周来看望")
    session.commit()
    result = FtsRecaller(session).recall(ELDER, "孙子", NOW)
    text = result.to_injection()
    assert text != ""
    assert INJECTION_HEADER in text
    assert "不要编造" in text
    assert "孙子下周来看望" in text


def test_recall_result_default_empty():
    """RecallResult 默认空 items，to_injection 为空串。"""
    assert RecallResult().to_injection() == ""
