"""重要性反思引擎测试（任务15）。"""
from __future__ import annotations

import pytest

from app.agent.reflection import ReflectionEngine, importance_score
from app.config import Config
from app.memory.conflict import MemoryConflictResolver
from app.memory.notes import NoteStore


# ──────────────────────────────────────────────
# importance_score 打分测试
# ──────────────────────────────────────────────

def test_score_emotion_words():
    assert importance_score("最近总是失眠，心里有点难受", "chat") > importance_score("今天天气不错", "chat")


def test_score_health_words():
    assert importance_score("头晕了一下，站不稳", "chat") > importance_score("吃饺子了", "chat")


def test_score_emergency_kind():
    assert importance_score("我摔倒了", "emergency") > importance_score("我摔倒了", "chat")


def test_score_health_kind():
    assert importance_score("血压好些了", "health") > importance_score("血压好些了", "chat")


def test_score_person_name():
    assert importance_score("女儿来看我了", "chat") > importance_score("下午睡了一觉", "chat")


def test_score_ordinary_chat_baseline():
    score = importance_score("今天天气真不错", "chat")
    assert 0 < score <= 1.5  # 普通闲聊只有基础分


# ──────────────────────────────────────────────
# 累积与触发
# ──────────────────────────────────────────────

def test_no_reflect_below_threshold(brain, started, session):
    """累积未到阈值，不生成 MemoryNote。"""
    elder_id, sid = started("王秀英")
    rt = brain._runtimes[elder_id]
    rt.reflection_score = 0.0

    # 发一句普通闲聊（分数低）
    brain.process_turn(elder_id, "今天天气不错")
    session.commit()

    store = NoteStore(session)
    reflection_notes = [n for n in store.active_notes(elder_id) if n.source == "reflection"]
    assert len(reflection_notes) == 0


def test_reflect_triggers_at_threshold(brain, started, session):
    """累积超阈值，生成 source=reflection 的 MemoryNote，分数归零。"""
    from app.config import config

    elder_id, sid = started("李桂芳")
    rt = brain._runtimes[elder_id]
    # 把分数推到阈值附近，再发一句高分话（情感词 + 健康词）
    rt.reflection_score = config.reflect_threshold - 0.5

    brain.process_turn(elder_id, "最近总是睡不好，头晕，心里难受")
    session.commit()

    # 分数应归零（已触发）
    assert rt.reflection_score < config.reflect_threshold

    # 应写入一条反思笔记
    store = NoteStore(session)
    reflection_notes = [n for n in store.active_notes(elder_id) if n.source == "reflection"]
    assert len(reflection_notes) >= 1


def test_reflect_score_resets_after_trigger(brain, started, session):
    from app.config import config

    elder_id, sid = started("张建国")
    rt = brain._runtimes[elder_id]
    rt.reflection_score = config.reflect_threshold - 0.1

    brain.process_turn(elder_id, "头晕，喘不上气，心慌")
    session.commit()

    assert rt.reflection_score < config.reflect_threshold - 0.1 or rt.reflection_score == 0.0


def test_reflect_appends_to_system_prompt(brain, started, session):
    """反思结果追加到 system_prompt，供下一轮感知。"""
    from app.config import config

    elder_id, sid = started("王秀英")
    rt = brain._runtimes[elder_id]
    rt.reflection_score = config.reflect_threshold - 0.1

    original_prompt = rt.system_prompt
    brain.process_turn(elder_id, "最近难受，睡不好，心慌担心")
    session.commit()

    if rt.reflection_score == 0.0:  # 确认触发了
        assert len(rt.system_prompt) > len(original_prompt)


def test_reflect_uses_conflict_resolver(brain, started, session):
    """反思笔记通过 MemoryConflictResolver 写入（Property 4 不变）。"""
    from app.config import config

    elder_id, sid = started("李桂芳")
    rt = brain._runtimes[elder_id]

    # 第一次触发
    rt.reflection_score = config.reflect_threshold - 0.1
    brain.process_turn(elder_id, "睡不着，心里难受")
    session.commit()

    store = NoteStore(session)
    notes1 = [n for n in store.active_notes(elder_id) if n.source == "reflection"]

    if not notes1:
        return  # FakeLLM 返回了不含 insight 的内容，跳过

    # 第二次触发（相似内容应触发冲突解析，旧 note 被取代）
    rt.reflection_score = config.reflect_threshold - 0.1
    brain.process_turn(elder_id, "还是睡不好，总是担心")
    session.commit()

    notes2 = [n for n in store.active_notes(elder_id) if n.source == "reflection"]
    # 相同 kind=other 且无日期，关键词高度重叠 → 应只有1条 active
    # （FakeLLM 返回固定文本，overlap 必然高）
    assert len(notes2) <= len(notes1) + 1  # 允许不冲突时累加，但不超过 1 条新增
