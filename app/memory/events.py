"""事件记忆（结构化）：健康读数与事件时间线。是三大模块共享的地基。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EmergencyEvent, HealthReading, MedicationLog


@dataclass
class ParsedReading:
    kind: str
    value: str
    unit: str


# 简单的健康数据抽取（血压/血糖/心率/体温/体重）
_PATTERNS = [
    ("blood_pressure", re.compile(r"血压.*?(\d{2,3})\D{0,3}(\d{2,3})"), "mmHg"),
    ("blood_sugar", re.compile(r"血糖.*?(\d{1,2}(?:\.\d)?)"), "mmol/L"),
    ("heart_rate", re.compile(r"(?:心率|脉搏).*?(\d{2,3})"), "次/分"),
    ("temperature", re.compile(r"(?:体温|发烧|发热).*?(\d{2}(?:\.\d)?)"), "℃"),
    ("weight", re.compile(r"体重.*?(\d{2,3}(?:\.\d)?)"), "kg"),
]


def parse_health(text: str) -> ParsedReading | None:
    """从口语中抽取健康读数，抽不到返回 None。"""
    for kind, pat, unit in _PATTERNS:
        m = pat.search(text or "")
        if m:
            if kind == "blood_pressure":
                value = f"{m.group(1)}/{m.group(2)}"
            else:
                value = m.group(1)
            return ParsedReading(kind=kind, value=value, unit=unit)
    return None


class EventStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_health_reading(
        self, elder_id: int, kind: str, value: str, unit: str = "", note: str = ""
    ) -> HealthReading:
        reading = HealthReading(
            elder_id=elder_id, kind=kind, value=value, unit=unit, note=note
        )
        self._session.add(reading)
        self._session.flush()
        return reading

    def recent_health(self, elder_id: int, limit: int = 10) -> list[HealthReading]:
        stmt = (
            select(HealthReading)
            .where(HealthReading.elder_id == elder_id)
            .order_by(HealthReading.recorded_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def recent_emergencies(self, elder_id: int, limit: int = 5) -> list[EmergencyEvent]:
        stmt = (
            select(EmergencyEvent)
            .where(EmergencyEvent.elder_id == elder_id)
            .order_by(EmergencyEvent.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())

    def medication_history(self, elder_id: int, limit: int = 20) -> list[MedicationLog]:
        stmt = (
            select(MedicationLog)
            .where(MedicationLog.elder_id == elder_id)
            .order_by(MedicationLog.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())
