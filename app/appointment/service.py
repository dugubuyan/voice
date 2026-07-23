"""预约/日程服务：结构化存储，支持查询与到点提醒。"""
from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Appointment


class AppointmentService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        elder_id: int,
        title: str,
        date_iso: str = "",
        time: str = "",
        persons: list[str] | None = None,
        location: str = "",
    ) -> Appointment:
        appt = Appointment(
            elder_id=elder_id,
            title=title,
            date_iso=date_iso,
            time=time,
            persons=persons or [],
            location=location,
            status="active",
        )
        self._session.add(appt)
        self._session.flush()
        return appt

    def upcoming(self, elder_id: int, on: date | None = None, limit: int = 10) -> list[Appointment]:
        """未来（含今天）的有效预约，按日期升序。无日期的排最后。"""
        today = (on or date.today()).isoformat()
        stmt = (
            select(Appointment)
            .where(
                Appointment.elder_id == elder_id,
                Appointment.status == "active",
            )
            .order_by(Appointment.date_iso.asc())
            .limit(limit)
        )
        rows = list(self._session.execute(stmt).scalars().all())
        # 过滤掉已过期的（有日期且早于今天）
        return [a for a in rows if not a.date_iso or a.date_iso >= today]

    def all_active(self, elder_id: int, limit: int = 20) -> list[Appointment]:
        stmt = (
            select(Appointment)
            .where(Appointment.elder_id == elder_id, Appointment.status == "active")
            .order_by(Appointment.date_iso.asc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())
