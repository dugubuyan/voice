"""常驻运行态数据结构：取代 per-session 的 _SessionState。

每位老人一个 ElderRuntime，设备在线期间长期驻留内存。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock


@dataclass
class Turn:
    """一轮对话（带对话时间戳）。取代裸 {role, content}。"""

    role: str        # "user" / "assistant"
    content: str
    at: datetime     # 对话时间（wall-clock），本轮发生时刻

    def to_llm_message(self) -> dict:
        """转成 LLM messages 格式。时间感由 TimeContext 单独注入，不塞进内容。"""
        return {"role": self.role, "content": self.content}


@dataclass
class PendingSlot:
    """一个正在复述确认中的一等字段候选（profile-setup-mode）。

    未确认即未提交：确认后才由 SlotSpec.apply 落库（合同即回执）。
    """

    field: str        # 目标字段名，如 "address" / "allergies" / "contacts" / ...
    value: object     # NLU 抽取的原样候选值（str / list / dict），复述念回用
    display: str      # 复述话术里念回的可读文本


@dataclass
class ElderRuntime:
    """每-elder 常驻运行态。设备在线期间长期驻留内存。"""

    elder_id: int
    system_prompt: str
    window: list[Turn] = field(default_factory=list)   # token 有界滚动窗口
    window_tokens: int = 0                              # 窗口当前估算 token 数
    running_summary: str = ""                           # 内存镜像，与 RunningSummary 表同步
    pending_emergency_id: int | None = None
    last_activity_at: datetime = field(default_factory=datetime.now)
    reflection_score: float = 0.0   # 累积重要性分数，超阈值时触发一次反思
    # ---- 设置模式（profile-setup-mode）：均为内存态，不落库、不跨重启 ----
    mode: str = "daily"                                 # daily / setup（默认最小权限）
    setup_last_at: datetime | None = None               # 最近一次设置交互时间（超时判定）
    pending_profile_slot: PendingSlot | None = None     # 复述确认挂起态
    # 仅为关机 flush / 未来扩展预留；第一阶段固化全在 process_turn 线程内
    lock: RLock = field(default_factory=RLock)
