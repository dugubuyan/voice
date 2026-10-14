"""记忆三层测试：档案注入 / 事件记录 / 跨会话摘要。"""
from __future__ import annotations

import uuid

import pytest

from app.memory.events import parse_health
from app.memory.profile import ProfileRepository
from tests.scenarios import HEALTH_CASES


def test_profile_context_contains_key_info(session, elders):
    repo = ProfileRepository(session)
    elder = elders["李桂芳"]
    ctx = repo.build_context(elder)
    assert "李桂芳" in ctx
    assert "高血压" in ctx
    assert "李梅" in ctx


def test_system_prompt_injects_profile(brain, started):
    elder_id, sid = started("张建国")
    runtime = brain._runtimes[elder_id]
    assert "张建国" in runtime.system_prompt
    assert "冠心病" in runtime.system_prompt


@pytest.mark.parametrize("text,kind,value", HEALTH_CASES)
def test_parse_health(text, kind, value):
    parsed = parse_health(text)
    assert parsed is not None
    assert parsed.kind == kind
    if value is not None:
        assert parsed.value == value


def test_health_reading_is_persisted(brain, started, session):
    from app.memory.events import EventStore

    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "我今天量的血压是140/90")
    assert result.kind == "health"
    readings = EventStore(session).recent_health(elder_id)
    assert any(r.value == "140/90" for r in readings)


def test_memory_note_captured_verbatim(brain, started, session):
    """老人说'记一下…'，具体姓名/日期应逐字入库，不被摘要丢失。"""
    from app.memory.notes import NoteStore

    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "你记一下我后天要去看我一个朋友，他叫李世明")
    assert result.kind == "memory"
    assert "李世明" in result.data["content"]
    # 已解析出绝对提醒日期
    assert result.data["remind_date"]

    notes = NoteStore(session).active_notes(elder_id)
    assert any("李世明" in n.content for n in notes)


def test_memory_note_injected_next_session(brain, started, session):
    """下个会话应把事实注入 system prompt，供精确回忆、防止编造。"""
    elder_id, sid = started("李桂芳")
    brain.process_turn(elder_id, "记一下我后天要去看朋友李世明")
    # fold_all 把窗口写入 running summary
    brain._wm.fold_all(brain._runtimes[elder_id], session)
    session.commit()

    # 删除旧 runtime，让下次 get_runtime 重新构建
    del brain._runtimes[elder_id]
    runtime2 = brain.get_runtime(elder_id)
    assert "李世明" in runtime2.system_prompt
    assert "需要记住的事" in runtime2.system_prompt


def test_cross_session_summary_roundtrip(brain, started, session):
    from app.memory.running import RunningSummaryRepo

    elder_id, sid = started("王秀英")
    brain.process_turn(elder_id, "今天去跳广场舞了，可开心了")
    # fold_all 把窗口折叠写入 RunningSummary
    brain._wm.fold_all(brain._runtimes[elder_id], session)
    session.commit()

    # RunningSummaryRepo 里应有记录
    repo = RunningSummaryRepo(session)
    row = repo.get_active(elder_id)
    assert row is not None
    assert row.summary  # 生成了摘要

    # 删除旧 runtime，重新构建：running summary 从 DB 恢复到内存镜像
    del brain._runtimes[elder_id]
    runtime2 = brain.get_runtime(elder_id)
    # running summary 不再烘焙进 system_prompt（改为每轮动态注入），
    # 但应恢复到 runtime.running_summary 内存镜像，供 _handle_chat 动态注入
    assert runtime2.running_summary  # 内存镜像有值
    assert runtime2.running_summary == row.summary
    # 且 _handle_chat 会把它注入到发给 LLM 的 messages 中
    import app.agent.brain as brain_mod
    captured = {}
    orig = runtime2  # noqa
    # 直接验证注入逻辑：构造一次 chat，拦截 llm.chat 的 messages
    original_chat = brain._llm.chat
    def _spy(messages):
        captured["messages"] = messages
        return original_chat(messages)
    brain._llm.chat = _spy
    try:
        brain._handle_chat(runtime2)
    finally:
        brain._llm.chat = original_chat
    injected = "".join(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert runtime2.running_summary in injected


# ===== 任务5：档案上下文增强（用药时间兜底）=====

def test_profile_context_medication_with_time_and_timing(session, elders):
    """李桂芳用药行应含时间点（08:00）和饭前后（饭后）。"""
    repo = ProfileRepository(session)
    elder = elders["李桂芳"]
    ctx = repo.build_context(elder)
    assert "08:00" in ctx, f"应含时间点 08:00，实际：{ctx}"
    assert "饭后" in ctx, f"应含饭后，实际：{ctx}"


def test_profile_context_no_medication_shows_wu(session):
    """无药老人用药行应显示"无"。"""
    from app.models import Elder
    elder = Elder(name="测试无药老人", age=70, conditions=[], emergency_contacts=[])
    session.add(elder)
    session.flush()
    repo = ProfileRepository(session)
    ctx = repo.build_context(elder)
    assert "常用药物：无" in ctx, f"应含'常用药物：无'，实际：{ctx}"
