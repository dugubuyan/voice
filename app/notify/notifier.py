"""家属/社区通知。当前为控制台实现，接口可替换为短信/微信/电话。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from sqlalchemy.orm import Session

from app.models import Notification


@dataclass
class NotifyRecord:
    elder_id: int
    recipient: str
    kind: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)


class Notifier(Protocol):
    """通知接口。将来替换为真实短信/微信/电话渠道只需实现此协议。"""

    def send(self, elder_id: int, recipient: str, kind: str, content: str) -> None: ...


class ConsoleNotifier:
    """把通知打到控制台，并写入 Notification 表；同时在内存留存便于测试断言。"""

    def __init__(self, session: Session, echo: bool = True) -> None:
        self._session = session
        self._echo = echo
        self.sent: list[NotifyRecord] = []

    def send(self, elder_id: int, recipient: str, kind: str, content: str) -> None:
        record = NotifyRecord(elder_id=elder_id, recipient=recipient, kind=kind, content=content)
        self.sent.append(record)
        self._session.add(
            Notification(
                elder_id=elder_id,
                recipient=recipient,
                channel="console",
                kind=kind,
                content=content,
            )
        )
        self._session.flush()
        if self._echo:
            icon = {"emergency": "🚨", "medication": "💊", "health": "🩺"}.get(kind, "🔔")
            print(f"\n{icon} [通知家属 -> {recipient}] {content}\n")
