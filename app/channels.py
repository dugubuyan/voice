"""交互通道抽象：CLI 与语音层共用同一"输出"接口。

这是为语音扩展预留的接缝：
- CLI:  speak() = 打印文本，stream_speak() = 逐 token 打印
- 语音: speak() = 送入 TTS 播放，stream_speak() = 流式 TTS
调度器（用药提醒等）只依赖此接口，无需关心底层是文字还是语音。
"""
from __future__ import annotations

import sys
from typing import Iterator, Protocol


class OutputChannel(Protocol):
    def speak(self, elder_id: int, text: str) -> None: ...


class ConsoleChannel:
    """文本演示通道：机器人开口 = 打印。支持流式逐 token 输出。"""

    def __init__(self, name: str = "助手") -> None:
        self._name = name

    def speak(self, elder_id: int, text: str) -> None:
        print(f"🤖 {self._name}：{text}")

    def stream_speak(self, elder_id: int, stream: Iterator[str]) -> str:
        """消费 token 流，逐字打印到终端，返回完整文本。

        首个 token 到达前打印前缀，之后边收边印，结束时换行。
        语音层可覆盖此方法把 token 送入 TTS 而非 print。
        """
        print(f"🤖 {self._name}：", end="", flush=True)
        buf: list[str] = []
        for tok in stream:
            print(tok, end="", flush=True)
            buf.append(tok)
        print()  # 换行
        return "".join(buf)
