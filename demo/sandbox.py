"""访客沙箱：每位访客一份独立的内存库 + 种子老人 + Brain。

为什么需要：app.cli 是单用户进程，直接对着 elder_care.db 读写；
放到网站上会让所有访客共享同一份记忆、互相污染，也会写脏主库。
这里不碰 app.db 的全局引擎，自建内存 SQLite 引擎，访客之间、访客与主库之间完全隔离。
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.brain import Brain
from app.agent.llm import LLMClient
from app.db import Base
from app.medication.service import MedicationService
from app.memory.episode import EpisodeStore
from app.memory.events import EventStore
from app.memory.foresight import ForesightStore
from app.memory.profile import ProfileRepository
from app.models import (
    Elder,
    ElderProfile,
    Episode,
    MemoryNote,
    MemScene,
    Medication,
    Notification,
)
from app.notify.notifier import ConsoleNotifier
from seed.seed_data import SEED


class SandboxError(Exception):
    """可直接展示给访客的业务错误。"""


class SandboxFull(SandboxError):
    pass


class TurnLimitReached(SandboxError):
    pass


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%m-%d %H:%M") if dt else ""


def _fmt_db(dt: datetime | None) -> str:
    """库里 server_default=now() 写入的是 UTC 朴素时间，换成服务器本地时间，
    与 Python 侧 datetime.now() 写入的时间（情景等）保持一致。"""
    if dt is None:
        return ""
    return _fmt(dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None))


class Sandbox:
    def __init__(self, llm: LLMClient, max_turns: int) -> None:
        self.created_at = time.monotonic()
        self.last_used = self.created_at
        self.turns = 0
        self._max_turns = max_turns
        # Brain 与 Session 都不是线程安全的，一个沙箱内所有操作串行
        self.lock = threading.RLock()

        # 内存库 + StaticPool：整个沙箱共享一条连接，库随沙箱一起销毁
        self._engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        from app import models  # noqa: F401  确保模型已注册到 Base.metadata

        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(
            bind=self._engine, expire_on_commit=False, future=True
        )()
        self._seed()

        self._notifier = ConsoleNotifier(self._session, echo=False)
        self._brain = Brain(self._session, llm, self._notifier)
        self._profiles = ProfileRepository(self._session)
        self._med = MedicationService(self._session, self._notifier)

    # ---- 生命周期 ----
    def _seed(self) -> None:
        for item in SEED:
            elder = Elder(**item["elder"])
            self._session.add(elder)
            self._session.flush()
            for m in item["meds"]:
                self._session.add(Medication(elder_id=elder.id, **m))
        self._session.commit()

    def close(self) -> None:
        with self.lock:
            try:
                self._session.close()
            finally:
                self._engine.dispose()

    def touch(self) -> None:
        self.last_used = time.monotonic()

    # ---- 查询 ----
    def elders(self) -> list[dict]:
        with self.lock:
            out = []
            for e in self._profiles.list_all():
                times = sorted(
                    {t for m in e.medications for t in (m.times or [])}
                )
                out.append(
                    {
                        "id": e.id,
                        "name": e.name,
                        "age": e.age,
                        "conditions": list(e.conditions or []),
                        "med_times": times,
                    }
                )
            return out

    def _elder(self, elder_id: int) -> Elder:
        elder = self._profiles.get(elder_id)
        if elder is None:
            raise SandboxError("没有这位老人")
        self._brain.get_runtime(elder_id)
        return elder

    # ---- 动作 ----
    def chat(self, elder_id: int, text: str) -> dict:
        with self.lock:
            if self.turns >= self._max_turns:
                raise TurnLimitReached(
                    f"这次演示已达到 {self._max_turns} 轮上限，请点「重新开始」。"
                )
            self._elder(elder_id)
            result = self._brain.process_turn(elder_id, text)
            self._session.commit()
            self.turns += 1
            return {
                "reply": result.reply,
                "kind": result.kind,
                "level": result.level,
                "actions": list(result.actions or []),
            }

    def remind(self, elder_id: int, hhmm: str) -> list[str]:
        """模拟到点：触发该时间点的用药提醒（等价 CLI /time）。"""
        with self.lock:
            elder = self._elder(elder_id)
            texts = []
            for med in self._session.query(Medication).filter(
                Medication.elder_id == elder_id
            ):
                if hhmm in (med.times or []):
                    texts.append(self._med.trigger_reminder(elder, med.id, hhmm).text)
            self._session.commit()
            return texts

    def silence(self, elder_id: int) -> str | None:
        """模拟老人沉默（等价 CLI /silence）。"""
        with self.lock:
            self._elder(elder_id)
            result = self._brain.handle_silence(elder_id)
            self._session.commit()
            return result.reply if result else None

    def pulse(self, minutes: int) -> None:
        """驱动一次记忆脉冲，并把「现在」往后拨 minutes 分钟模拟空闲（等价 CLI /tick）。"""
        with self.lock:
            self._brain.on_pulse(datetime.now() + timedelta(minutes=minutes))
            self._session.commit()

    # ---- 状态面板（等价 CLI /who /mem /events 的结构化版本）----
    def state(self, elder_id: int) -> dict:
        with self.lock:
            s = self._session
            elder = self._elder(elder_id)
            rt = self._brain.get_runtime(elder_id)
            today = date.today()

            episodes = list(
                s.execute(
                    select(Episode)
                    .where(Episode.elder_id == elder_id)
                    .order_by(Episode.ended_at.desc())
                ).scalars()
            )
            scenes = list(
                s.execute(
                    select(MemScene).where(
                        MemScene.elder_id == elder_id, MemScene.active == True  # noqa: E712
                    )
                ).scalars()
            )
            foresights = ForesightStore(s).active_valid(elder_id, today)
            profile = s.execute(
                select(ElderProfile).where(ElderProfile.elder_id == elder_id)
            ).scalars().first()
            notes = list(
                s.execute(
                    select(MemoryNote)
                    .where(
                        MemoryNote.elder_id == elder_id,
                        MemoryNote.active == True,  # noqa: E712
                    )
                    .order_by(MemoryNote.created_at.desc())
                    .limit(8)
                ).scalars()
            )
            events = EventStore(s)
            notifications = list(
                s.execute(
                    select(Notification)
                    .where(Notification.elder_id == elder_id)
                    .order_by(Notification.id.desc())
                    .limit(10)
                ).scalars()
            )
            meds = list(
                s.query(Medication).filter(Medication.elder_id == elder_id)
            )
            unconsolidated = len(EpisodeStore(s).unconsolidated(elder_id))
            episode_total = s.execute(
                select(func.count()).select_from(Episode).where(Episode.elder_id == elder_id)
            ).scalar_one()

            return {
                "turns_used": self.turns,
                "turns_max": self._max_turns,
                "mode": rt.mode,
                "pending_emergency": rt.pending_emergency_id is not None,
                "pending_profile_field": (
                    rt.pending_profile_slot.field if rt.pending_profile_slot else None
                ),
                "profile": {
                    "name": elder.name,
                    "age": elder.age,
                    "conditions": list(elder.conditions or []),
                    "address": elder.address or "",
                    "allergies": list(elder.allergies or []),
                    "contacts": [
                        f"{c.get('name', '')}（{c.get('relation', '')}）"
                        for c in (elder.emergency_contacts or [])
                    ],
                },
                "medications": [
                    {
                        "name": m.name,
                        "dose": m.dose,
                        "times": list(m.times or []),
                        "taken_today": self._med.taken_today(elder_id, m.id),
                    }
                    for m in meds
                ],
                "memory": {
                    "running_summary": rt.running_summary or "",
                    "window_turns": len(rt.window),
                    "episode_total": episode_total,
                    "episode_unconsolidated": unconsolidated,
                    "episodes": [
                        {
                            "time": _fmt(ep.ended_at),
                            "summary": ep.summary or "",
                            "consolidated": bool(ep.consolidated),
                        }
                        for ep in episodes[:5]
                    ],
                    "scenes": [sc.theme for sc in scenes],
                    "foresights": [
                        {"due": f.due_date or "", "content": f.content} for f in foresights
                    ],
                    "profile_text": (profile.content.strip() if profile and profile.content else ""),
                    "notes": [
                        {
                            "content": n.content,
                            "remind_date": n.remind_date or "",
                            "insight": n.source == "reflection",
                        }
                        for n in notes
                    ],
                },
                "events": {
                    "health": [
                        {
                            "time": _fmt_db(r.recorded_at),
                            "text": f"{r.kind} {r.value}{r.unit}",
                        }
                        for r in events.recent_health(elder_id, limit=5)
                    ],
                    "emergencies": [
                        {
                            "time": _fmt_db(e.created_at),
                            "level": e.level,
                            "status": e.status,
                            "text": e.trigger_text,
                        }
                        for e in events.recent_emergencies(elder_id, limit=5)
                    ],
                    "notifications": [
                        {
                            "time": _fmt_db(n.created_at),
                            "recipient": n.recipient,
                            "kind": n.kind,
                            "text": n.content,
                        }
                        for n in notifications
                    ],
                },
            }


class SandboxPool:
    """沙箱池：容量上限 + 空闲过期回收（惰性回收，无后台线程）。"""

    def __init__(
        self, llm: LLMClient, *, max_sessions: int, ttl_sec: int, max_turns: int
    ) -> None:
        self._llm = llm
        self._max_sessions = max_sessions
        self._ttl_sec = ttl_sec
        self._max_turns = max_turns
        self._items: dict[str, Sandbox] = {}
        self._lock = threading.Lock()

    def _purge_locked(self) -> None:
        now = time.monotonic()
        for sid in [
            sid for sid, sb in self._items.items() if now - sb.last_used > self._ttl_sec
        ]:
            self._items.pop(sid).close()

    def create(self) -> tuple[str, Sandbox]:
        with self._lock:
            self._purge_locked()
            if len(self._items) >= self._max_sessions:
                raise SandboxFull("体验人数已满，请稍后再试。")
            sid = uuid.uuid4().hex
            sb = Sandbox(self._llm, self._max_turns)
            self._items[sid] = sb
            return sid, sb

    def get(self, sid: str) -> Sandbox | None:
        with self._lock:
            self._purge_locked()
            sb = self._items.get(sid)
            if sb is not None:
                sb.touch()
            return sb

    def drop(self, sid: str) -> None:
        with self._lock:
            sb = self._items.pop(sid, None)
        if sb is not None:
            sb.close()

    def __len__(self) -> int:
        return len(self._items)
