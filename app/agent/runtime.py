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
    # 仅为关机 flush / 未来扩展预留；第一阶段固化全在 process_turn 线程内
    lock: RLock = field(default_factory=RLock)
