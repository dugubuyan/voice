"""用户工作区：每个注册用户一份独立的 SQLite 记忆库 + 一个 Brain。

为什么一人一库：
- 现有记忆体系（档案 / 近期原文 / 情景 / 巩固 / 触发器）本来就是"一个库一位老人"的单用户设计，
  一人一库可以原样复用，Brain 与记忆代码零改动。
- 用户之间物理隔离，不可能串记忆；也避免多个用户共用 SQLite 写锁，
  一个人的 LLM 调用不会把其他人的写入卡住。
- 注销 / 备份 / 导出都是对单个文件操作。

对话历史（chat_messages）永久保存，用于页面回看；它与记忆管线的 recent_turns 不同，
后者几天后会被清理，只留下情景摘要。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.brain import Brain
from app.agent.llm import LLMClient
from app.config import config
from app.db import Base
from app.models import Elder
from app.notify.notifier import ConsoleNotifier
from demo import settings


class WorkspaceError(Exception):
    """可直接展示给用户的业务错误。"""


class DailyLimitReached(WorkspaceError):
    pass


class ChatMessage(Base):
    """页面对话历史（永久保存，append-only）。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="")
    level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


def _db_path(user_id: int):
    return settings.DATA_DIR / "users" / f"{int(user_id)}.db"


def _make_engine(user_id: int):
    path = _db_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}",
        # 一个工作区内所有操作由 lock 串行，单连接即可
        connect_args={"check_same_thread": False, "timeout": 15},
        poolclass=StaticPool,
        future=True,
    )
    from app import models  # noqa: F401  确保全部模型已注册

    Base.metadata.create_all(engine)
    return engine


def provision(user_id: int, name: str, age: int) -> None:
    """注册时创建该用户的记忆库与档案（此时档案只有称呼和年龄，其余在聊天/设置模式中补充）。"""
    engine = _make_engine(user_id)
    try:
        with sessionmaker(bind=engine, expire_on_commit=False, future=True)() as s:
            s.add(Elder(name=name, age=age))
            s.commit()
    finally:
        engine.dispose()


def remove_data(user_id: int) -> None:
    """注册失败回滚时删除已创建的库文件。"""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = _db_path(user_id).with_name(f"{int(user_id)}.db{suffix}")
        p.unlink(missing_ok=True)


class Workspace:
    def __init__(self, user_id: int, llm: LLMClient) -> None:
        self.user_id = user_id
        self.last_used = time.monotonic()
        # Brain 与 Session 都不是线程安全的：一个工作区内所有操作串行
        self.lock = threading.RLock()
        if not _db_path(user_id).exists():
            raise WorkspaceError("账号数据不存在，请联系管理员")
        self._engine = _make_engine(user_id)
        self._session = sessionmaker(
            bind=self._engine, expire_on_commit=False, future=True
        )()
        elder = self._session.execute(select(Elder).order_by(Elder.id)).scalars().first()
        if elder is None:
            raise WorkspaceError("账号档案缺失，请联系管理员")
        self.elder_id = elder.id
        self._notifier = ConsoleNotifier(self._session, echo=False)
        self._brain = Brain(self._session, llm, self._notifier)
        self._brain.get_runtime(self.elder_id)  # 从落盘原文恢复近期对话窗口

    # ---- 生命周期 ----
    def close(self) -> None:
        with self.lock:
            try:
                self._session.close()
            finally:
                self._engine.dispose()

    def touch(self) -> None:
        self.last_used = time.monotonic()

    # ---- 对话 ----
    def _turns_today(self) -> int:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self._session.execute(
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.role == "user", ChatMessage.at >= start)
        ).scalar_one()

    def chat(self, text: str) -> dict:
        with self.lock:
            if self._turns_today() >= settings.USER_DAILY_TURNS:
                raise DailyLimitReached(
                    f"今天已经聊了 {settings.USER_DAILY_TURNS} 句啦，明天再来找我吧。"
                )
            started = datetime.now()
            try:
                result = self._brain.process_turn(self.elder_id, text)
                self._session.add(ChatMessage(role="user", content=text, at=started))
                self._session.add(
                    ChatMessage(
                        role="assistant", content=result.reply,
                        kind=result.kind or "", level=result.level,
                    )
                )
                self._session.commit()
            except Exception:
                self._session.rollback()
                raise
            return {
                "reply": result.reply,
                "kind": result.kind,
                "level": result.level,
                "actions": list(result.actions or []),
            }

    def history(self, limit: int = 50, before_id: int | None = None) -> list[dict]:
        """按时间正序返回一页历史；before_id 用于向上翻更早的记录。"""
        with self.lock:
            stmt = select(ChatMessage).order_by(ChatMessage.id.desc()).limit(limit)
            if before_id is not None:
                stmt = stmt.where(ChatMessage.id < before_id)
            rows = list(self._session.execute(stmt).scalars())
            rows.reverse()
            return [
                {
                    "id": m.id, "role": m.role, "text": m.content,
                    "kind": m.kind, "level": m.level,
                    "at": m.at.strftime("%Y-%m-%d %H:%M"),
                }
                for m in rows
            ]

    def pulse(self) -> None:
        """服务端低频脉冲：记忆巩固 / 前瞻过期 / 清理（等价 CLI /tick，用真实时间）。"""
        with self.lock:
            self._brain.on_pulse(datetime.now())
            self._session.commit()


class WorkspacePool:
    """工作区缓存：空闲回收 + 容量上限（超限时淘汰最久未用且不忙的）。"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._items: dict[int, Workspace] = {}
        self._lock = threading.Lock()

    def _evict_locked(self, uid: int) -> bool:
        ws = self._items[uid]
        if not ws.lock.acquire(blocking=False):  # 正在处理请求，不动它
            return False
        try:
            self._items.pop(uid)
            ws.close()
        finally:
            ws.lock.release()
        return True

    def _purge_locked(self) -> None:
        now = time.monotonic()
        for uid in [u for u, w in self._items.items() if now - w.last_used > settings.WORKSPACE_TTL_SEC]:
            self._evict_locked(uid)

    def get(self, user_id: int) -> Workspace:
        with self._lock:
            self._purge_locked()
            ws = self._items.get(user_id)
            if ws is None:
                if len(self._items) >= settings.MAX_WORKSPACES:
                    for uid in sorted(self._items, key=lambda u: self._items[u].last_used):
                        if self._evict_locked(uid):
                            break
                ws = Workspace(user_id, self._llm)
                self._items[user_id] = ws
            ws.touch()
            return ws

    def pulse_all(self) -> None:
        with self._lock:
            items = list(self._items.values())
        for ws in items:
            if not ws.lock.acquire(blocking=False):  # 正在对话的跳过，下个周期再来
                continue
            try:
                ws.pulse()
            except Exception:
                pass
            finally:
                ws.lock.release()

    def close_all(self) -> None:
        with self._lock:
            for uid in list(self._items):
                self._evict_locked(uid)

    def start_pulse_thread(self) -> threading.Event:
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(config.pulse_interval_sec):
                self.pulse_all()

        threading.Thread(target=loop, name="workspace-pulse", daemon=True).start()
        return stop
