"""紧急事件服务：落库、分级响应、家属升级。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.emergency.detector import Level
from app.models import Elder, EmergencyEvent
from app.notify.notifier import Notifier


class EmergencyService:
    def __init__(self, session: Session, notifier: Notifier) -> None:
        self._session = session
        self._notifier = notifier

    def _contacts(self, elder: Elder) -> list[dict]:
        contacts = elder.emergency_contacts or []
        return sorted(contacts, key=lambda c: c.get("priority", 99))

    def open_event(self, elder: Elder, level: Level, trigger_text: str) -> EmergencyEvent:
        """创建紧急事件并按级别执行响应动作。"""
        event = EmergencyEvent(
            elder_id=elder.id,
            level=level.value,
            trigger_text=trigger_text,
            status="open",
            actions=[],
        )
        self._session.add(event)
        self._session.flush()
        self._respond(elder, event, level)
        return event

    def _respond(self, elder: Elder, event: EmergencyEvent, level: Level) -> None:
        actions = list(event.actions or [])
        contacts = self._contacts(elder)

        if level == Level.OBSERVE:
            actions.append("记录并稍后回访")
        elif level == Level.ALERT:
            if contacts:
                first = contacts[0]
                self._notifier.send(
                    elder.id,
                    first.get("name", "家属"),
                    "emergency",
                    f"{elder.name} 疑似身体不适（触发：{event.trigger_text}），请尽快确认情况。",
                )
                actions.append(f"已告警家属 {first.get('name')}")
        elif level == Level.EMERGENCY:
            for c in contacts:
                self._notifier.send(
                    elder.id,
                    c.get("name", "家属"),
                    "emergency",
                    f"【紧急】{elder.name} 触发紧急求助（{event.trigger_text}），"
                    f"请立即联系并考虑拨打120。",
                )
            actions.append(f"已通知全部联系人({len(contacts)})并建议拨打120")
            actions.append("保持通话，持续安抚")

        event.actions = actions
        self._session.flush()

    def escalate(self, event: EmergencyEvent, elder: Elder, reason: str) -> None:
        """升级：alert -> emergency（如确认或沉默超时）。"""
        event.status = "confirmed"
        event.level = Level.EMERGENCY.value
        actions = list(event.actions or [])
        actions.append(f"升级为紧急：{reason}")
        event.actions = actions
        self._session.flush()
        for c in self._contacts(elder):
            self._notifier.send(
                elder.id,
                c.get("name", "家属"),
                "emergency",
                f"【紧急升级】{elder.name}：{reason}。请立即处理，必要时拨打120。",
            )

    def cancel(self, event: EmergencyEvent, reason: str) -> None:
        """降级/取消：老人确认无事。"""
        event.status = "cancelled"
        actions = list(event.actions or [])
        actions.append(f"老人确认无事，取消：{reason}")
        event.actions = actions
        event.resolved_at = datetime.now()
        self._session.flush()
