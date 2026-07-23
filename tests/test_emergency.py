"""紧急检测与分级响应测试。"""
from __future__ import annotations

import pytest

from app.emergency import detector
from app.emergency.detector import Level
from tests.scenarios import CONFIRM_CASES, EMERGENCY_CASES


@pytest.mark.parametrize("text,level,need_confirm", EMERGENCY_CASES)
def test_detect_levels(text, level, need_confirm):
    d = detector.detect(text)
    assert d.level == level, f"{text} -> {d.level}, 期望 {level}（{d.reason}）"
    assert d.need_confirmation == need_confirm


@pytest.mark.parametrize("text,verdict", CONFIRM_CASES)
def test_interpret_confirmation(text, verdict):
    assert detector.interpret_confirmation(text) == verdict


def test_rescue_triggers_emergency_and_notifies_all(brain, started, notifier):
    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "救命啊，我不行了")
    assert result.kind == "emergency"
    assert result.level == Level.EMERGENCY.value
    # 李桂芳有两个联系人，紧急应全部通知
    assert len(notifier.sent) == 2
    assert all(n.kind == "emergency" for n in notifier.sent)


def test_alert_then_cancel_downgrades(brain, started, notifier):
    elder_id, sid = started("李桂芳")
    r1 = brain.process_turn(elder_id, "我这胸口有点闷")
    assert r1.level == Level.ALERT.value
    # 告警先通知第一联系人
    assert len(notifier.sent) == 1
    r2 = brain.process_turn(elder_id, "没事没事，我就是喘口气")
    assert r2.data.get("resolved") is True
    # 取消后不应再新增通知
    assert len(notifier.sent) == 1


def test_alert_then_confirm_escalates(brain, started, notifier):
    elder_id, sid = started("李桂芳")
    brain.process_turn(elder_id, "我头晕得厉害")
    before = len(notifier.sent)
    r2 = brain.process_turn(elder_id, "是的，特别难受")
    assert r2.data.get("escalated") is True
    # 升级后应通知全部联系人
    assert len(notifier.sent) > before


def test_silence_escalates_pending(brain, started, notifier):
    elder_id, sid = started("张建国")
    brain.process_turn(elder_id, "我有点站不起来")
    r = brain.handle_silence(elder_id)
    assert r is not None
    assert r.data.get("reason") == "silence"
    assert any("紧急升级" in n.content for n in notifier.sent)


def test_reported_incident_does_not_alarm(brain, started, notifier):
    elder_id, sid = started("李桂芳")
    result = brain.process_turn(elder_id, "我上次差点摔倒，吓死我了")
    # 仅观察，不通知家属
    assert result.level == Level.OBSERVE.value
    assert len(notifier.sent) == 0
