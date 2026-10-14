"""SemanticBoundaryDetector 单元测试（离线，FakeLLM / 桩 LLM）。

覆盖：正常 JSON 解析、LLM 抛错降级、cut_index 越界裁剪、空 window、
以及 FakeLLM 对【边界检测】标记返回合法 JSON。
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.agent.llm import FakeLLMClient
from app.agent.runtime import Turn
from app.memory.boundary import (
    BOUNDARY_PROMPT,
    BOUNDARY_PROMPT_MARKER,
    BoundaryDecision,
    SemanticBoundaryDetector,
)


class _StubLLM:
    """返回固定字符串的桩 LLM。"""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def chat(self, messages: list[dict]) -> str:
        return self._raw


class _RaisingLLM:
    """chat 抛异常的桩 LLM，用于测降级。"""

    def chat(self, messages: list[dict]) -> str:
        raise RuntimeError("LLM 挂了")


def _window(n: int) -> list[Turn]:
    now = datetime.now()
    return [
        Turn("user" if i % 2 == 0 else "assistant", f"第{i}句", now)
        for i in range(n)
    ]


def test_normal_json_boundary_true():
    """正常 JSON：is_boundary=True，cut_index 在范围内原样返回。"""
    llm = _StubLLM('{"is_boundary": true, "cut_index": 2}')
    detector = SemanticBoundaryDetector(llm)
    decision = detector.detect(_window(5))
    assert decision == BoundaryDecision(True, 2)


def test_normal_json_boundary_false():
    """正常 JSON：is_boundary=False，cut_index 归 0（无意义）。"""
    llm = _StubLLM('{"is_boundary": false, "cut_index": 3}')
    detector = SemanticBoundaryDetector(llm)
    decision = detector.detect(_window(5))
    assert decision.is_boundary is False
    assert decision.cut_index == 0


def test_llm_raises_degrades_to_non_boundary():
    """LLM 抛错 → 降级为 (False, 0)，不抛异常。"""
    detector = SemanticBoundaryDetector(_RaisingLLM())
    decision = detector.detect(_window(4))
    assert decision == BoundaryDecision(False, 0)


def test_invalid_json_degrades():
    """JSON 解析失败 → 降级为 (False, 0)，不抛异常。"""
    detector = SemanticBoundaryDetector(_StubLLM("这不是 JSON"))
    decision = detector.detect(_window(4))
    assert decision == BoundaryDecision(False, 0)


def test_cut_index_over_upper_bound_clamped():
    """cut_index 越上界（999）→ 裁剪到 len(window)。"""
    window = _window(5)
    llm = _StubLLM('{"is_boundary": true, "cut_index": 999}')
    detector = SemanticBoundaryDetector(llm)
    decision = detector.detect(window)
    assert decision.is_boundary is True
    assert decision.cut_index == len(window)


def test_cut_index_below_lower_bound_clamped():
    """cut_index 越下界（-1 / 0）→ 裁剪到 1。"""
    for bad in (-1, 0):
        llm = _StubLLM(json.dumps({"is_boundary": True, "cut_index": bad}))
        detector = SemanticBoundaryDetector(llm)
        decision = detector.detect(_window(5))
        assert decision.is_boundary is True
        assert decision.cut_index == 1


def test_empty_window_returns_non_boundary():
    """空 window → (False, 0)，不调 LLM。"""
    detector = SemanticBoundaryDetector(_RaisingLLM())  # 若被调到会抛错
    decision = detector.detect([])
    assert decision == BoundaryDecision(False, 0)


def test_fakellm_boundary_marker_returns_valid_json():
    """FakeLLM 对【边界检测】返回可解析的合法 JSON。"""
    fake = FakeLLMClient()
    prompt = BOUNDARY_PROMPT.format(dialogue="1. 老人：你好")
    assert BOUNDARY_PROMPT_MARKER in prompt
    raw = fake.chat([{"role": "user", "content": prompt}])
    data = json.loads(raw)
    assert "is_boundary" in data
    assert "cut_index" in data
    assert isinstance(data["is_boundary"], bool)
    assert isinstance(data["cut_index"], int)


def test_detector_with_fakellm_end_to_end():
    """detector + FakeLLM 端到端：默认非边界，不抛错。"""
    detector = SemanticBoundaryDetector(FakeLLMClient())
    decision = detector.detect(_window(5))
    assert decision == BoundaryDecision(False, 0)
