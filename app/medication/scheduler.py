"""用药提醒调度器（APScheduler）。

单机演示用 BackgroundScheduler，从数据库的用药计划构建定时任务。
到点后通过 OutputChannel 主动"开口"提醒老人。
"""
from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db import session_scope
from app.medication.service import MedicationService
from app.models import Elder, Medication
from app.notify.notifier import ConsoleNotifier

# 提醒回调签名：收到 (elder_id, elder_name, medication_id, scheduled_time, text)
ReminderCallback = Callable[[int, str, int, str, str], None]


class MedicationScheduler:
    def __init__(self, on_reminder: ReminderCallback) -> None:
        self._scheduler = BackgroundScheduler()
        self._on_reminder = on_reminder

    def load_jobs(self) -> int:
        """从数据库加载所有老人的用药计划为定时任务。返回任务数。"""
        count = 0
        with session_scope() as session:
            meds = session.query(Medication).all()
            for med in meds:
                elder = session.get(Elder, med.elder_id)
                for t in med.times or []:
                    hour, minute = t.split(":")
                    self._scheduler.add_job(
                        self._fire,
                        CronTrigger(hour=int(hour), minute=int(minute)),
                        args=[med.elder_id, elder.name, med.id, t],
                        id=f"med-{med.id}-{t}",
                        replace_existing=True,
                    )
                    count += 1
        return count

    def _fire(self, elder_id: int, elder_name: str, medication_id: int, scheduled_time: str) -> None:
        with session_scope() as session:
            elder = session.get(Elder, elder_id)
            service = MedicationService(session, ConsoleNotifier(session))
            result = service.trigger_reminder(elder, medication_id, scheduled_time)
            self._on_reminder(
                elder_id, elder_name, medication_id, scheduled_time, result.text
            )

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)
