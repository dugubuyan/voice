"""时间上下文：为每轮对话计算时间感并注入 LLM 上下文。

只负责"对话时间"（wall-clock / 间隔），不碰"事件时间"（remind_date / date_iso），
延续项目已有的时间二分原则。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def day_period_of(hour: int) -> str:
    """把 24 小时制的小时数映射到中文时段。"""
    if hour < 5:
        return "深夜"
    if hour < 8:
        return "清晨"
    if hour < 11:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 17:
        return "下午"
    if hour < 19:
        return "傍晚"
    if hour < 23:
        return "晚上"
    return "深夜"


@dataclass
class TimeContext:
    now: datetime
    gap_since_last: float | None   # 距上一句的秒数；None 表示本轮是首句
    day_period: str                # 清晨/上午/中午/下午/傍晚/晚上/深夜

    def to_injection(self) -> str:
        """格式化为可注入 LLM 的自然语言时间感片段。"""
        now_str = self.now.strftime("%Y年%m月%d日 %H:%M")
        weekday = "一二三四五六日"[self.now.weekday()]
        parts = [f"【当前时刻】{now_str}，星期{weekday}，{self.day_period}。"]

        g = self.gap_since_last
        if g is None:
            parts.append("这是设备启动后老人对你说的第一句话。")
        elif g < 90:
            parts.append("你们正在连续对话。")
        elif g < 3600:
            parts.append(f"距离上一句过了大约{int(g // 60)}分钟。")
        elif g < 86400:
            parts.append(f"距离上一句过了大约{int(g // 3600)}小时，可以自然地关心一下。")
        else:
            parts.append(f"距离上次聊天已经过了{int(g // 86400)}天，久违了。")
        return "".join(parts)


def build_time_context(now: datetime, last_at: datetime | None) -> TimeContext:
    """计算当前时刻的时间感（纯函数，无副作用）。

    Args:
        now: 当前时刻
        last_at: 上一轮对话的时间戳；None 表示本轮是首句
    """
    gap = (now - last_at).total_seconds() if last_at is not None else None
    return TimeContext(
        now=now,
        gap_since_last=gap,
        day_period=day_period_of(now.hour),
    )
