"""用药时间表 executor 测试（query_medication_schedule）。

覆盖：
- 有多种药的老人 → 渲染含时间点和 timing 中文
- 无用药计划的老人 → 返回"没有在管的用药安排"
- 只读断言：调用前后 Medication 数量不变（Property 3）
"""
from __future__ import annotations

import pytest

from app.agent.tools import execute_tool
from app.models import Elder, Medication


# ──────────────────────────────────────────────
# 测试 1：李桂芳（两种药）→ 渲染含时间点和饭后
# ──────────────────────────────────────────────

def test_schedule_renders_two_meds(session, elders):
    """氨氯地平 08:00 饭后 / 二甲双胍 08:00+18:00 饭后 均出现在输出中。"""
    elder = elders["李桂芳"]
    result = execute_tool("query_medication_schedule", "{}", session, elder.id)

    assert "氨氯地平" in result
    assert "二甲双胍" in result
    assert "08:00" in result
    assert "18:00" in result
    assert "饭后" in result
    assert result.startswith("您的用药安排是：")


# ──────────────────────────────────────────────
# 测试 2：无用药计划的老人 → 明确告知没有安排
# ──────────────────────────────────────────────

def test_schedule_no_meds(session):
    """临时创建一个没有任何 Medication 的老人，应返回'没有在管的用药安排'。"""
    elder = Elder(
        name="测试无药老人",
        age=70,
        gender="男",
        conditions=[],
        emergency_contacts=[],
    )
    session.add(elder)
    session.flush()

    result = execute_tool("query_medication_schedule", "{}", session, elder.id)
    assert "没有在管的用药安排" in result


# ──────────────────────────────────────────────
# 测试 3：只读断言（Property 3）
# ──────────────────────────────────────────────

def test_schedule_is_read_only(session, elders):
    """execute_tool 调用前后 Medication 总数不变（无写操作副作用）。"""
    elder = elders["李桂芳"]
    count_before = session.query(Medication).count()
    execute_tool("query_medication_schedule", "{}", session, elder.id)
    count_after = session.query(Medication).count()
    assert count_before == count_after


# ──────────────────────────────────────────────
# 端到端 Brain 测试（任务 4）
# ──────────────────────────────────────────────

def test_brain_schedule_ji_dian_chi_yao(brain, elders):
    """'我几点吃药' → FakeLLM 识别为 scope=schedule → kind=medication, answer=schedule, reply 含 08:00。"""
    elder = elders["李桂芳"]
    brain.get_runtime(elder.id)
    result = brain.process_turn(elder.id, "我几点吃药")
    assert result.kind == "medication"
    assert result.data.get("answer") == "schedule"
    assert "08:00" in result.reply


def test_brain_schedule_yi_ban_ji_dian(brain, elders):
    """'我一般几点吃药' → 同样走 schedule 分支。"""
    elder = elders["李桂芳"]
    brain.get_runtime(elder.id)
    result = brain.process_turn(elder.id, "我一般几点吃药")
    assert result.kind == "medication"
    assert result.data.get("answer") == "schedule"
    assert "08:00" in result.reply


def test_brain_today_not_schedule_regression(brain, elders):
    """'我吃药了吗' → 走 today 分支，answer 不是 schedule（回归 Property 4）。"""
    elder = elders["李桂芳"]
    brain.get_runtime(elder.id)
    result = brain.process_turn(elder.id, "我吃药了吗")
    assert result.kind == "medication"
    assert result.data.get("answer") != "schedule"
