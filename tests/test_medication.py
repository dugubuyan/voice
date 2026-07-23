"""用药提醒、确认闭环与防重复服用测试。"""
from __future__ import annotations

from datetime import date

import pytest

from app.medication.service import MedicationService
from tests.scenarios import MED_DIALOG_CASES


def _service(session, notifier):
    return MedicationService(session, notifier)


def test_reminder_creates_pending_log(session, elders, notifier):
    svc = _service(session, notifier)
    elder = elders["李桂芳"]
    med = elder.medications[0]
    result = svc.trigger_reminder(elder, med.id, "08:00")
    assert result.already_taken is False
    assert result.log.status == "pending"
    assert med.name in result.text


def test_confirm_taken_then_no_double_dose(session, elders, notifier):
    svc = _service(session, notifier)
    elder = elders["李桂芳"]
    med = elder.medications[0]
    r1 = svc.trigger_reminder(elder, med.id, "08:00")
    svc.confirm_taken(r1.log)
    assert svc.taken_today(elder.id, med.id) is True
    # 再次到点提醒应识别为已服用，不重复催
    r2 = svc.trigger_reminder(elder, med.id, "08:00")
    assert r2.already_taken is True
    assert "已经吃过" in r2.text


def test_escalation_after_max_reminders(session, elders, notifier):
    from app.config import config

    svc = _service(session, notifier)
    elder = elders["李桂芳"]
    med = elder.medications[0]
    log = None
    for _ in range(config.medication_max_reminders):
        log = svc.trigger_reminder(elder, med.id, "08:00").log
    assert svc.should_escalate(log) is True
    svc.mark_missed_and_escalate(elder, log)
    assert log.status == "missed"
    assert any(n.kind == "medication" for n in notifier.sent)


def test_defer_keeps_pending(session, elders, notifier):
    svc = _service(session, notifier)
    elder = elders["李桂芳"]
    med = elder.medications[0]
    log = svc.trigger_reminder(elder, med.id, "08:00").log
    svc.defer(log)
    assert log.status == "deferred"
    pending = svc.pending_today(elder.id)
    assert log.id in [p.id for p in pending]


@pytest.mark.parametrize("text,kind,data_frag", MED_DIALOG_CASES)
def test_medication_dialog(brain, started, text, kind, data_frag):
    elder_id, sid = started("李桂芳")
    # 先制造一个待服用记录（模拟到点提醒）
    from app.medication.service import MedicationService
    from app.notify.notifier import ConsoleNotifier

    svc = MedicationService(brain._session, ConsoleNotifier(brain._session, echo=False))
    elder = brain._profiles.get(elder_id)
    svc.trigger_reminder(elder, elder.medications[0].id, "08:00")

    result = brain.process_turn(elder_id, text)
    assert result.kind == kind
    for k, v in data_frag.items():
        assert result.data.get(k) == v


def test_query_when_already_taken(brain, started):
    elder_id, sid = started("李桂芳")
    from app.notify.notifier import ConsoleNotifier

    svc = MedicationService(brain._session, ConsoleNotifier(brain._session, echo=False))
    elder = brain._profiles.get(elder_id)
    r = svc.trigger_reminder(elder, elder.medications[0].id, "08:00")
    svc.confirm_taken(r.log)

    result = brain.process_turn(elder_id, "我今天吃药了吗？")
    assert result.data.get("answer") == "already_taken"
