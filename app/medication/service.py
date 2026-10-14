"""用药服务：主动提醒 + 确认闭环 + 防重复服用 + 未服用升级家属。

核心：这是"定时触发 + 确认闭环"系统，不是被动问答。
剂量等信息只来自结构化数据库，绝不由 LLM 生成。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import config
from app.models import Elder, Medication, MedicationLog
from app.notify.notifier import Notifier

_TIMING_TEXT = {
    "before_meal": "饭前",
    "after_meal": "饭后",
    "anytime": "",
}


@dataclass
class ReminderResult:
    log: MedicationLog
    medication: Medication
    text: str
    already_taken: bool = False


class MedicationService:
    def __init__(self, session: Session, notifier: Notifier) -> None:
        self._session = session
        self._notifier = notifier

    # ---- 用药计划录入（设置模式经此写入，不改动确认/防重复等安全判定）----
    def create_plan(
        self,
        elder_id: int,
        *,
        name: str,
        dose: str,
        times: list[str] | None = None,
        timing: str = "anytime",
        color_or_shape: str = "",
        notes: str = "",
    ) -> Medication:
        """新增一条用药计划（结构化数据录入）。

        仅创建 Medication 计划行；服药确认、防重复服用、漏服升级等既有安全逻辑
        完全不变（那些走 MedicationLog，不受本方法影响）。剂量/时间来自结构化录入，
        绝不由 LLM 生成。
        """
        med = Medication(
            elder_id=elder_id,
            name=name,
            dose=dose,
            times=list(times or []),
            timing=timing or "anytime",
            color_or_shape=color_or_shape,
            notes=notes,
        )
        self._session.add(med)
        self._session.flush()
        return med

    # ---- 查询 ----
    def _med(self, medication_id: int) -> Medication:
        return self._session.get(Medication, medication_id)

    def taken_today(self, elder_id: int, medication_id: int, on: date | None = None) -> bool:
        """今天该药是否已确认服用（防重复服用依据）。"""
        day = (on or date.today()).isoformat()
        stmt = select(MedicationLog).where(
            MedicationLog.elder_id == elder_id,
            MedicationLog.medication_id == medication_id,
            MedicationLog.scheduled_date == day,
            MedicationLog.status == "taken",
        )
        return self._session.execute(stmt).first() is not None

    def _reminder_text(self, elder: Elder, med: Medication) -> str:
        timing = _TIMING_TEXT.get(med.timing, "")
        shape = f"（{med.color_or_shape}）" if med.color_or_shape else ""
        timing_part = f"，{timing}吃" if timing else ""
        return f"{elder.name}，现在该吃{med.name}了{shape}，{med.dose}{timing_part}。吃过了记得跟我说一声哦。"

    # ---- 主动提醒 ----
    def trigger_reminder(
        self, elder: Elder, medication_id: int, scheduled_time: str, on: date | None = None
    ) -> ReminderResult:
        """到点触发一次提醒。若今日已服用则不重复提醒。"""
        med = self._med(medication_id)
        day = (on or date.today()).isoformat()

        if self.taken_today(elder.id, medication_id, on):
            log = self._get_or_create_log(elder.id, medication_id, day, scheduled_time)
            return ReminderResult(
                log=log,
                medication=med,
                text=f"{elder.name}，{med.name}今天已经吃过了，不用再吃啦。",
                already_taken=True,
            )

        log = self._get_or_create_log(elder.id, medication_id, day, scheduled_time)
        log.reminder_count += 1
        self._session.flush()
        return ReminderResult(log=log, medication=med, text=self._reminder_text(elder, med))

    def _get_or_create_log(
        self, elder_id: int, medication_id: int, day: str, scheduled_time: str
    ) -> MedicationLog:
        stmt = select(MedicationLog).where(
            MedicationLog.elder_id == elder_id,
            MedicationLog.medication_id == medication_id,
            MedicationLog.scheduled_date == day,
            MedicationLog.scheduled_time == scheduled_time,
        )
        log = self._session.execute(stmt).scalars().first()
        if log is None:
            log = MedicationLog(
                elder_id=elder_id,
                medication_id=medication_id,
                scheduled_date=day,
                scheduled_time=scheduled_time,
                status="pending",
                reminder_count=0,
            )
            self._session.add(log)
            self._session.flush()
        return log

    # ---- 确认闭环 ----
    def confirm_taken(self, log: MedicationLog) -> None:
        log.status = "taken"
        log.confirmed_at = datetime.now()
        self._session.flush()

    def defer(self, log: MedicationLog) -> None:
        """老人说'待会儿吃'，延后再提醒。"""
        log.status = "deferred"
        self._session.flush()

    def mark_missed_and_escalate(self, elder: Elder, log: MedicationLog) -> None:
        """多次提醒仍未服用，标记漏服并通知家属。"""
        log.status = "missed"
        self._session.flush()
        med = self._med(log.medication_id)
        contacts = sorted(elder.emergency_contacts or [], key=lambda c: c.get("priority", 99))
        if contacts:
            self._notifier.send(
                elder.id,
                contacts[0].get("name", "家属"),
                "medication",
                f"{elder.name} 未按时服用{med.name}（{log.scheduled_time}），已提醒"
                f"{log.reminder_count}次仍未确认，请留意。",
            )

    def pending_today(self, elder_id: int, on: date | None = None) -> list[MedicationLog]:
        day = (on or date.today()).isoformat()
        stmt = select(MedicationLog).where(
            MedicationLog.elder_id == elder_id,
            MedicationLog.scheduled_date == day,
            MedicationLog.status.in_(["pending", "deferred"]),
        )
        return list(self._session.execute(stmt).scalars().all())

    def should_escalate(self, log: MedicationLog) -> bool:
        return log.reminder_count >= config.medication_max_reminders
