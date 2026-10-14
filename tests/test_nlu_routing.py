"""意图识别层 + 代码路由测试（使用离线 FakeLLM 的 NLU 存根）。"""
from __future__ import annotations

from datetime import date, timedelta

from app.agent.nlu import parse_nlu_response


# ──────────────────────────────────────────────
# NLU 解析
# ──────────────────────────────────────────────

def test_parse_nlu_valid():
    raw = '{"intents": [{"name": "confirm_medication", "slots": {}}]}'
    intents = parse_nlu_response(raw)
    assert len(intents) == 1
    assert intents[0].name == "confirm_medication"


def test_parse_nlu_code_fence():
    raw = '```json\n{"intents": [{"name": "chitchat", "slots": {}}]}\n```'
    intents = parse_nlu_response(raw)
    assert intents[0].name == "chitchat"


def test_parse_nlu_garbage_falls_back_to_chitchat():
    intents = parse_nlu_response("这不是JSON")
    assert intents[0].name == "chitchat"


def test_parse_nlu_unknown_intent_dropped():
    raw = '{"intents": [{"name": "launch_missiles", "slots": {}}]}'
    intents = parse_nlu_response(raw)
    assert intents[0].name == "chitchat"  # 未知意图被丢弃后兜底


# ──────────────────────────────────────────────
# 路由到处理函数（端到端，走 Brain + FakeLLM NLU）
# ──────────────────────────────────────────────

def test_route_confirm_medication(brain, started, session):
    from app.medication.service import MedicationService
    from app.notify.notifier import ConsoleNotifier

    elder_id, sid = started("李桂芳")
    svc = MedicationService(session, ConsoleNotifier(session, echo=False))
    elder = brain._profiles.get(elder_id)
    svc.trigger_reminder(elder, elder.medications[0].id, "08:00")

    result = brain.process_turn(elder_id, "药我都吃过了")
    assert result.kind == "medication"
    assert result.data.get("answer") == "confirmed"


def test_route_record_health(brain, started, session):
    from app.memory.events import EventStore

    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "我今天血糖是7.2")
    assert result.kind == "health"
    assert result.data.get("value") == "7.2"
    readings = EventStore(session).recent_health(elder_id)
    assert any(r.value == "7.2" for r in readings)


def test_route_chitchat(brain, started):
    elder_id, sid = started("王秀英")
    result = brain.process_turn(elder_id, "今天天气真不错")
    assert result.kind == "chat"
    assert result.reply


def test_route_record_appointment(brain, started, session):
    from app.appointment.service import AppointmentService

    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "帮我记一下后天要去医院复查血糖")
    assert result.kind == "appointment"
    assert result.data.get("date_iso")  # 后天已解析为绝对日期

    appts = AppointmentService(session).all_active(elder_id)
    assert len(appts) == 1
    assert "复查" in appts[0].title


def test_route_query_appointment(brain, started, session):
    elder_id, sid = started("李桂芳")
    brain.process_turn(elder_id, "帮我记一下后天要去医院复查")
    result = brain.process_turn(elder_id, "我最近有什么安排")
    assert result.kind == "appointment"
    assert result.data.get("count", 0) >= 1


def test_appointment_expired_not_in_upcoming(session, elders):
    from app.appointment.service import AppointmentService

    elder = elders["李桂芳"]
    svc = AppointmentService(session)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    svc.create(elder.id, "过期的事", date_iso=yesterday)
    svc.create(elder.id, "将来的事", date_iso=tomorrow)
    session.commit()

    upcoming = svc.upcoming(elder.id, date.today())
    titles = [a.title for a in upcoming]
    assert "将来的事" in titles
    assert "过期的事" not in titles


# ──────────────────────────────────────────────
# schedule facet 识别（任务 3）
# ──────────────────────────────────────────────

def _nlu(brain, elder, text):
    """调用 NLU 层，返回第一个意图。"""
    intents = brain._run_nlu(elder, text)
    return intents[0]


def test_nlu_schedule_jidie(brain, started, elders):
    """'我几点吃药' → query_medication + scope=schedule"""
    elder_id, _ = started("李桂芳")
    elder = elders["李桂芳"]
    intent = _nlu(brain, elder, "我几点吃药")
    assert intent.name == "query_medication"
    assert intent.slots.get("scope") == "schedule"


def test_nlu_schedule_yiban_jidie(brain, started, elders):
    """'我一般几点吃药' → query_medication + scope=schedule"""
    elder_id, _ = started("李桂芳")
    elder = elders["李桂芳"]
    intent = _nlu(brain, elder, "我一般几点吃药")
    assert intent.name == "query_medication"
    assert intent.slots.get("scope") == "schedule"


def test_nlu_schedule_shenmeshihou(brain, started, elders):
    """'我的吃药时间是什么时候' → query_medication + scope=schedule"""
    elder_id, _ = started("李桂芳")
    elder = elders["李桂芳"]
    intent = _nlu(brain, elder, "我的吃药时间是什么时候")
    assert intent.name == "query_medication"
    assert intent.slots.get("scope") == "schedule"


def test_nlu_regression_today(brain, started, elders):
    """回归：'我吃药了吗' 仍为 scope=today"""
    elder_id, _ = started("李桂芳")
    elder = elders["李桂芳"]
    intent = _nlu(brain, elder, "我吃药了吗")
    assert intent.name == "query_medication"
    assert intent.slots.get("scope") == "today"


def test_nlu_regression_confirm(brain, started, elders):
    """回归：'药吃过了' 仍为 confirm_medication"""
    elder_id, _ = started("李桂芳")
    elder = elders["李桂芳"]
    intent = _nlu(brain, elder, "药吃过了")
    assert intent.name == "confirm_medication"
