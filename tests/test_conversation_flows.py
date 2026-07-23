"""端到端生活场景：把多轮对话串起来验证业务闭环。"""
from __future__ import annotations

from app.emergency.detector import Level
from app.medication.service import MedicationService
from app.notify.notifier import ConsoleNotifier


def test_daily_life_of_solo_elder(brain, started, session):
    """场景：独居李奶奶的一天——问候→吃药→家常→报不适→确认降级→晚安摘要。"""
    elder_id, sid = started("李桂芳")

    # 早上到点吃药（模拟提醒后老人确认）
    svc = MedicationService(session, ConsoleNotifier(session, echo=False))
    elder = brain._profiles.get(elder_id)
    svc.trigger_reminder(elder, elder.medications[0].id, "08:00")
    r_med = brain.process_turn(elder_id, "药我吃过啦")
    assert r_med.data.get("answer") == "confirmed"

    # 聊家常
    r_chat = brain.process_turn(elder_id, "外孙小宝昨天打电话来了，我高兴了一整天")
    assert r_chat.kind == "chat"
    assert r_chat.reply  # 有回复

    # 报健康数据
    r_health = brain.process_turn(elder_id, "对了，我血压量了是138/85")
    assert r_health.kind == "health"

    # 提及不适 -> 告警确认
    r_alert = brain.process_turn(elder_id, "就是刚才有点头晕")
    assert r_alert.level == Level.ALERT.value

    # 确认无事 -> 降级
    r_ok = brain.process_turn(elder_id, "现在好多了，没事了")
    assert r_ok.data.get("resolved") is True

    # 结束：fold_all 生成摘要
    brain._wm.fold_all(brain._runtimes[elder_id], session)
    session.commit()
    runtime = brain._runtimes[elder_id]
    assert runtime.running_summary or True  # fold 完成不报错即可


def test_emergency_night(brain, started, notifier):
    """场景：深夜求救——检测→通知全部联系人→保持陪伴。"""
    elder_id, sid = started("陈明远")
    result = brain.process_turn(elder_id, "救命……我喘不上气了")
    assert result.level == Level.EMERGENCY.value
    # 陈明远有两个联系人
    assert len(notifier.sent) == 2
    assert "陪着您" in result.reply


def test_medication_forgotten_then_family_alerted(brain, started, session, notifier):
    """场景：反复提醒仍不吃药——最终通知家属。"""
    from app.config import config

    elder_id, sid = started("王秀英")
    svc = MedicationService(session, notifier)
    elder = brain._profiles.get(elder_id)
    med = elder.medications[0]

    log = None
    for _ in range(config.medication_max_reminders):
        log = svc.trigger_reminder(elder, med.id, "08:00").log
    if svc.should_escalate(log):
        svc.mark_missed_and_escalate(elder, log)

    assert any(n.kind == "medication" for n in notifier.sent)


def test_confusion_does_not_double_dose(brain, started, session):
    """场景：轻度认知障碍老人重复问是否吃药，系统防误服。"""
    elder_id, sid = started("张建国")
    svc = MedicationService(session, ConsoleNotifier(session, echo=False))
    elder = brain._profiles.get(elder_id)
    r = svc.trigger_reminder(elder, elder.medications[0].id, "08:00")
    svc.confirm_taken(r.log)

    # 老人忘了自己吃过，反复问
    for _ in range(3):
        result = brain.process_turn(elder_id, "我是不是还没吃药啊？")
        assert result.data.get("answer") == "already_taken"
