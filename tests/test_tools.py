"""工具调用测试：验证层2事件记忆能被正确查询并返回结构化结果。

测试使用 FakeLLMClient（不触发真实 tool calling），
直接测试工具执行层（execute_tool）的正确性，保证离线可跑。
"""
from __future__ import annotations

from datetime import date

import pytest

from app.agent.tools import execute_tool
from app.memory.events import EventStore
from app.models import Medication, MedicationLog


# ──────────────────────────────────────────────
# 工具执行层直接测试
# ──────────────────────────────────────────────

def test_query_health_records_blood_pressure(session, elders):
    elder = elders["李桂芳"]
    store = EventStore(session)
    store.add_health_reading(elder.id, "blood_pressure", "140/90", "mmHg")
    store.add_health_reading(elder.id, "blood_pressure", "138/88", "mmHg")
    session.commit()

    result = execute_tool(
        "query_health_records",
        '{"kind": "blood_pressure", "limit": 5}',
        session, elder.id,
    )
    assert "140/90" in result
    assert "血压" in result


def test_query_health_records_all(session, elders):
    elder = elders["李桂芳"]
    store = EventStore(session)
    store.add_health_reading(elder.id, "blood_pressure", "140/90", "mmHg")
    store.add_health_reading(elder.id, "blood_sugar", "7.2", "mmol/L")
    session.commit()

    result = execute_tool(
        "query_health_records",
        '{"kind": "all", "limit": 5}',
        session, elder.id,
    )
    assert "血压" in result
    assert "血糖" in result


def test_query_health_no_records(session, elders):
    elder = elders["王秀英"]
    result = execute_tool(
        "query_health_records",
        '{"kind": "blood_pressure"}',
        session, elder.id,
    )
    assert "暂无" in result


def test_query_medication_history(session, elders):
    elder = elders["李桂芳"]
    med = elder.medications[0]
    today = date.today().isoformat()
    log = MedicationLog(
        elder_id=elder.id,
        medication_id=med.id,
        scheduled_date=today,
        scheduled_time="08:00",
        status="taken",
        reminder_count=1,
    )
    session.add(log)
    session.commit()

    result = execute_tool(
        "query_medication_history",
        '{"days": 1}',
        session, elder.id,
    )
    assert med.name in result
    assert "已服用" in result


def test_query_medication_missed(session, elders):
    elder = elders["李桂芳"]
    med = elder.medications[0]
    today = date.today().isoformat()
    log = MedicationLog(
        elder_id=elder.id,
        medication_id=med.id,
        scheduled_date=today,
        scheduled_time="08:00",
        status="missed",
        reminder_count=3,
    )
    session.add(log)
    session.commit()

    result = execute_tool(
        "query_medication_history",
        '{"days": 1}',
        session, elder.id,
    )
    assert "漏服" in result


def test_query_emergency_history(session, elders):
    from app.models import EmergencyEvent

    elder = elders["陈明远"]
    session.add(EmergencyEvent(
        elder_id=elder.id,
        level="alert",
        trigger_text="我这胸口有点闷",
        status="cancelled",
        actions=["已告警家属"],
    ))
    session.commit()

    result = execute_tool(
        "query_emergency_history",
        '{"limit": 3}',
        session, elder.id,
    )
    assert "胸口" in result
    assert "告警" in result


def test_query_emergency_none(session, elders):
    elder = elders["王秀英"]
    result = execute_tool(
        "query_emergency_history",
        "{}",
        session, elder.id,
    )
    assert "没有" in result


def test_unknown_tool_returns_error(session, elders):
    elder = elders["李桂芳"]
    result = execute_tool("nonexistent_tool", "{}", session, elder.id)
    assert "未知工具" in result


# ──────────────────────────────────────────────
# tool schema 完整性检查
# ──────────────────────────────────────────────

def test_tool_schemas_valid():
    from app.agent.tools import TOOL_SCHEMAS

    assert len(TOOL_SCHEMAS) == 3
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "query_health_records" in names
    assert "query_medication_history" in names
    assert "query_emergency_history" in names
    for t in TOOL_SCHEMAS:
        assert "description" in t["function"]
        assert "parameters" in t["function"]


# ──────────────────────────────────────────────
# FakeLLM 不触发工具调用（测试隔离）
# ──────────────────────────────────────────────

def test_fake_llm_no_tool_calls():
    from app.agent.llm import FakeLLMClient
    from app.agent.tools import TOOL_SCHEMAS

    llm = FakeLLMClient()
    resp = llm.chat_with_tools(
        [{"role": "user", "content": "我最近血压怎么样？"}],
        TOOL_SCHEMAS,
    )
    assert not resp.has_tool_calls
    assert resp.content
