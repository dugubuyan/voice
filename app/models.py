"""数据模型：老人档案 + 事件记忆（用药/健康/紧急）+ 对话记忆 + 通知日志。

事件记忆是三大模块（紧急检测 / 用药 / 记忆）共同的结构化地基。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Elder(Base):
    """老人档案（静态记忆）。"""

    __tablename__ = "elders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(8), default="")
    # 慢性病列表，如 ["高血压", "2型糖尿病"]
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    # 紧急联系人 [{"name":"李梅","relation":"女儿","phone":"138...","priority":1}]
    emergency_contacts: Mapped[list] = mapped_column(JSON, default=list)
    dialect: Mapped[str] = mapped_column(String(32), default="普通话")
    # 偏好/习惯/爱好，用于让对话更有温度
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    # 一等结构化字段（profile-setup-mode）：安全关键/高频精确项，设置模式经复述确认写入
    address: Mapped[str] = mapped_column(String(256), default="")   # 家庭住址
    allergies: Mapped[list] = mapped_column(JSON, default=list)      # 过敏史，如 ["青霉素","海鲜"]
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    medications: Mapped[list["Medication"]] = relationship(
        back_populates="elder", cascade="all, delete-orphan"
    )


class Medication(Base):
    """用药计划（结构化，由家属/医生录入，剂量绝不由 LLM 生成）。"""

    __tablename__ = "medications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    name: Mapped[str] = mapped_column(String(64))
    dose: Mapped[str] = mapped_column(String(64))  # 如 "1片"
    # 每日提醒时间点，如 ["08:00", "20:00"]
    times: Mapped[list] = mapped_column(JSON, default=list)
    # before_meal / after_meal / anytime
    timing: Mapped[str] = mapped_column(String(16), default="anytime")
    color_or_shape: Mapped[str] = mapped_column(String(64), default="")  # 便于老人辨认
    notes: Mapped[str] = mapped_column(Text, default="")

    elder: Mapped["Elder"] = relationship(back_populates="medications")


class MedicationLog(Base):
    """服药记录（确认闭环 + 防重复服用的依据）。"""

    __tablename__ = "medication_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    medication_id: Mapped[int] = mapped_column(ForeignKey("medications.id"))
    scheduled_date: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD
    scheduled_time: Mapped[str] = mapped_column(String(5))  # HH:MM
    # pending / taken / deferred / missed / skipped
    status: Mapped[str] = mapped_column(String(16), default="pending")
    reminder_count: Mapped[int] = mapped_column(Integer, default=0)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class HealthReading(Base):
    """健康读数（血压/血糖等），结构化便于趋势分析。"""

    __tablename__ = "health_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    # blood_pressure / blood_sugar / heart_rate / weight / temperature
    kind: Mapped[str] = mapped_column(String(24))
    value: Mapped[str] = mapped_column(String(32))  # 如 "140/90"
    unit: Mapped[str] = mapped_column(String(16), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EmergencyEvent(Base):
    """紧急事件（分级 + 全程审计留痕）。"""

    __tablename__ = "emergency_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    # observe / alert / emergency
    level: Mapped[str] = mapped_column(String(16))
    trigger_text: Mapped[str] = mapped_column(Text)
    # open / confirmed / cancelled / resolved
    status: Mapped[str] = mapped_column(String(16), default="open")
    actions: Mapped[list] = mapped_column(JSON, default=list)  # 采取的动作留痕
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class ConversationSummary(Base):
    """对话摘要记忆（跨会话），而非存全量历史。"""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    session_id: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text, default="")
    mood: Mapped[str] = mapped_column(String(32), default="")  # 情绪标签
    topics: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class MemoryNote(Base):
    """事实型记忆：老人明确要求记住的事（约定、姓名、日期等），逐字保留。

    与 ConversationSummary（模糊摘要）互补：具体细节靠本表，避免被摘要丢失或 LLM 编造。
    """

    __tablename__ = "memory_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    content: Mapped[str] = mapped_column(Text)  # 逐字保留的事实
    kind: Mapped[str] = mapped_column(String(16), default="other")  # commitment/person/event/other
    remind_date: Mapped[str] = mapped_column(String(10), default="")  # YYYY-MM-DD，可空
    active: Mapped[bool] = mapped_column(default=True)
    # recency-wins 审计（二期接线，本期加列预留）
    superseded_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("memory_notes.id"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(16), default="explicit")  # explicit / reflection
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # Phase 3 扩展：不确定标注 / 可见性 / 来源情景
    confidence: Mapped[str] = mapped_column(String(12), default="certain")  # certain/uncertain
    visibility: Mapped[str] = mapped_column(String(16), default="family")  # family/internal
    source_episode_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("episodes.id"), nullable=True
    )


class RunningSummary(Base):
    """每位老人一条持续演进的对话摘要（增量折叠更新）。

    取代批量 ConversationSummary：token 超预算时把被挤出的轮次 fold 进本行，
    covers_to 随每次折叠前移，始终反映"截至最近被挤出内容"的全貌。
    """

    __tablename__ = "running_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    summary: Mapped[str] = mapped_column(Text, default="")
    mood: Mapped[str] = mapped_column(String(32), default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)
    covers_from: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # 最早并入内容的对话时间
    covers_to: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)    # 最近并入内容的对话时间
    active: Mapped[bool] = mapped_column(default=True)   # 快照留存时旧版置 False
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Appointment(Base):
    """预约/日程（结构化）：医院复查、社区体检、家人来访等，支持查询与到点提醒。"""

    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    title: Mapped[str] = mapped_column(Text)  # 事项，如"去社区医院复查血糖"
    date_iso: Mapped[str] = mapped_column(String(10), default="")  # YYYY-MM-DD
    time: Mapped[str] = mapped_column(String(5), default="")  # HH:MM，可空
    persons: Mapped[list] = mapped_column(JSON, default=list)  # 涉及的人
    location: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(16), default="active")  # active/done/cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Notification(Base):
    """家属/社区通知日志（可追溯）。"""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    recipient: Mapped[str] = mapped_column(String(64))
    channel: Mapped[str] = mapped_column(String(16), default="console")
    kind: Mapped[str] = mapped_column(String(24))  # emergency / medication / health
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ===== Phase 3：情景记忆 + 检索式回忆 + 统一条件触发 =====


class RecentTurn(Base):
    """近期对话原文落盘（append-only，可清理）。需求 8。"""

    __tablename__ = "recent_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    role: Mapped[str] = mapped_column(String(16))  # user/assistant
    content: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime)
    episode_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("episodes.id"), nullable=True
    )
    consolidated: Mapped[bool] = mapped_column(default=False)


class Episode(Base):
    """情景：一段语义连贯对话的记录，永久保留、可检索。需求 1/2/5/11。"""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    summary: Mapped[str] = mapped_column(Text, default="")
    mood: Mapped[str] = mapped_column(String(32), default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime)  # 对话时间下界
    ended_at: Mapped[datetime] = mapped_column(DateTime)  # 对话时间上界
    consolidated: Mapped[bool] = mapped_column(default=False)
    scene_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("mem_scenes.id"), nullable=True
    )
    visibility: Mapped[str] = mapped_column(String(16), default="family")  # family/internal
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Foresight(Base):
    """带时限前瞻：会自动过期的未来关注点。需求 3。"""

    __tablename__ = "foresights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    content: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(16), default="commitment")
    due_date: Mapped[str] = mapped_column(String(10), default="")  # 事件时间, 不改写(C4)
    active: Mapped[bool] = mapped_column(default=True)
    superseded_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("foresights.id"), nullable=True
    )
    source_episode_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("episodes.id"), nullable=True
    )
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reason: Mapped[str] = mapped_column(String(32), default="")  # expired/done/cancelled/superseded
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MemScene(Base):
    """主题场景：由多个相关 Episode 归并出的主题聚合。需求 4。"""

    __tablename__ = "mem_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    theme: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text, default="")
    member_episode_ids: Mapped[list] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(default=True)
    superseded_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("mem_scenes.id"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ElderProfile(Base):
    """老人画像：单文档持续重写，保持有界稳定。需求 4。"""

    __tablename__ = "elder_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"), unique=True)  # 单文档
    content: Mapped[str] = mapped_column(Text, default="")  # 重写, 有界
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class TrendMemory(Base):
    """健康/情绪趋势记忆：数值仅取自结构化源。需求 7。"""

    __tablename__ = "trend_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    kind: Mapped[str] = mapped_column(String(16))  # health/mood
    content: Mapped[str] = mapped_column(Text)
    window_from: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    window_to: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    based_on: Mapped[list] = mapped_column(JSON, default=list)  # HealthReading id 等精确源
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Person(Base):
    """人物记忆：老人生活里反复出现的人。需求 6。"""

    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[int] = mapped_column(ForeignKey("elders.id"))
    name: Mapped[str] = mapped_column(String(64))
    relation: Mapped[str] = mapped_column(String(32), default="")
    snippets: Mapped[list] = mapped_column(JSON, default=list)
    last_mentioned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    visibility: Mapped[str] = mapped_column(String(16), default="family")


class TriggerFire(Base):
    """Trigger 去重/冷却状态：跨重启幂等依据。需求 14。"""

    __tablename__ = "trigger_fires"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    elder_id: Mapped[Optional[int]] = mapped_column(ForeignKey("elders.id"), nullable=True)
    trigger_name: Mapped[str] = mapped_column(String(48))
    dedup_key: Mapped[str] = mapped_column(String(96))  # 跨重启幂等依据
    fired_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
