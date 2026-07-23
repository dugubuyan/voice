"""交互通道抽象：CLI 与语音层共用同一"输出"接口。

这是为语音扩展预留的接缝：
- CLI:  speak() = 打印文本
- 语音: speak() = 送入 TTS 播放
调度器（用药提醒等）只依赖此接口，无需关心底层是文字还是语音。
"""
from __future__ import annotations

from typing import Protocol


class OutputChannel(Protocol):
    def speak(self, elder_id: int, text: str) -> None: ...


class ConsoleChannel:
    """文本演示通道：机器人开口 = 打印。"""

    def __init__(self, name: str = "助手") -> None:
        self._name = name

    def speak(self, elder_id: int, text: str) -> None:
        print(f"🤖 {self._name}：{text}")
