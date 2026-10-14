"""MemCellExtractor 单元测试（离线，FakeLLM / 桩 LLM）。

覆盖：正常抽取（完整 JSON → 类型正确）、离线/异常降级、字段缺失容错、
以及 FakeLLM 对【记忆抽取】返回可解析合法 JSON 并端到端得到合法 MemCell。
"""
from __future__ import annotations

import json
from datetime import datetime

from app.agent.llm import FakeLLMClient
from app.agent.runtime import Turn
from app.memory.extractor import (
    MEMCELL_PROMPT,
    MEMCELL_PROMPT_MARKER,
    MemCell,
    MemCellExtractor,
)
from app.memory.foresight import ForesightDraft
from app.memory.notes import ExtractedNote


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


def _turns() -> list[Turn]:
    now = datetime.now()
    return [
        Turn("user", "我明天要去医院复查", now),
        Turn("assistant", "好的，我记下了", now),
        Turn("user", "对了，我孙子小明下周来看我", now),
    ]


_FULL_JSON = json.dumps(
    {
        "episode_summary": "老人提到明天复查以及孙子来访",
        "mood": "开心",
        "topics": ["医疗", "家人", "探望", "复查", "孙子", "多余话题"],
        "atomic_facts": [
            {
                "content": "明天去医院复查",
                "date_iso": "2025-01-02",
                "persons": [],
                "kind": "event",
            }
        ],
        "foresights": [
            {"content": "去医院复查", "kind": "appointment", "due_date": "2025-01-02"}
        ],
        "persons": [
            {"name": "小明", "relation": "孙子", "snippet": "下周来看我"}
        ],
    },
    ensure_ascii=False,
)


def test_normal_extraction_types_correct():
    """完整 JSON → MemCell 字段完整、类型正确。"""
    extractor = MemCellExtractor(_StubLLM(_FULL_JSON))
    cell = extractor.extract(1, _turns())

    assert isinstance(cell, MemCell)
    assert cell.episode_summary == "老人提到明天复查以及孙子来访"
    assert cell.mood == "开心"
    # topics 截断到 <= 5
    assert len(cell.topics) == 5
    assert cell.topics[0] == "医疗"

    # atomic_facts 是 ExtractedNote
    assert len(cell.atomic_facts) == 1
    fact = cell.atomic_facts[0]
    assert isinstance(fact, ExtractedNote)
    assert fact.is_memory_request is True
    assert fact.content == "明天去医院复查"
    assert fact.date_iso == "2025-01-02"
    assert fact.kind == "event"

    # foresights 是 ForesightDraft
    assert len(cell.foresights) == 1
    fs = cell.foresights[0]
    assert isinstance(fs, ForesightDraft)
    assert fs.content == "去医院复查"
    assert fs.kind == "appointment"
    assert fs.due_date == "2025-01-02"

    # persons 是 dict 列表
    assert cell.persons == [
        {"name": "小明", "relation": "孙子", "snippet": "下周来看我"}
    ]


def test_llm_raises_degrades_to_valid_empty_memcell():
    """LLM 抛错 → 合法空 MemCell，episode_summary 非空，其余为空列表。"""
    extractor = MemCellExtractor(_RaisingLLM())
    cell = extractor.extract(1, _turns())

    assert isinstance(cell, MemCell)
    assert cell.episode_summary  # 非空
    assert cell.episode_summary == "我明天要去医院复查"  # 取首句
    assert cell.mood == ""
    assert cell.topics == []
    assert cell.atomic_facts == []
    assert cell.foresights == []
    assert cell.persons == []


def test_invalid_json_degrades():
    """返回非 JSON → 降级合法空 MemCell，不抛异常。"""
    extractor = MemCellExtractor(_StubLLM("这不是 JSON"))
    cell = extractor.extract(1, _turns())
    assert cell.episode_summary  # 非空
    assert cell.atomic_facts == []
    assert cell.foresights == []
    assert cell.persons == []


def test_empty_summary_degrades():
    """LLM 返回空 episode_summary → 视为失败，降级。"""
    raw = json.dumps({"episode_summary": "", "atomic_facts": []}, ensure_ascii=False)
    extractor = MemCellExtractor(_StubLLM(raw))
    cell = extractor.extract(1, _turns())
    assert cell.episode_summary == "我明天要去医院复查"


def test_degraded_summary_all_empty_turns():
    """所有轮次内容为空 → 降级摘要仍非空（兜底文案）。"""
    now = datetime.now()
    turns = [Turn("user", "", now), Turn("assistant", "   ", now)]
    extractor = MemCellExtractor(_RaisingLLM())
    cell = extractor.extract(1, turns)
    assert cell.episode_summary == "闲聊了几句"


def test_missing_fields_tolerant():
    """部分字段缺失（只有 episode_summary）不报错，缺失字段给默认。"""
    raw = json.dumps({"episode_summary": "只有摘要"}, ensure_ascii=False)
    extractor = MemCellExtractor(_StubLLM(raw))
    cell = extractor.extract(1, _turns())
    assert cell.episode_summary == "只有摘要"
    assert cell.mood == ""
    assert cell.topics == []
    assert cell.atomic_facts == []
    assert cell.foresights == []
    assert cell.persons == []


def test_malformed_list_items_skipped():
    """列表里混入非法项（非 dict / 缺 content）被跳过，不报错。"""
    raw = json.dumps(
        {
            "episode_summary": "混杂数据",
            "atomic_facts": ["不是对象", {"content": ""}, {"content": "有效事实"}],
            "foresights": [123, {"content": ""}, {"content": "有效前瞻"}],
            "persons": ["str", {"name": ""}, {"name": "老王", "relation": "邻居"}],
        },
        ensure_ascii=False,
    )
    extractor = MemCellExtractor(_StubLLM(raw))
    cell = extractor.extract(1, _turns())
    assert len(cell.atomic_facts) == 1
    assert cell.atomic_facts[0].content == "有效事实"
    assert len(cell.foresights) == 1
    assert cell.foresights[0].content == "有效前瞻"
    assert len(cell.persons) == 1
    assert cell.persons[0]["name"] == "老王"
    assert cell.persons[0]["snippet"] == ""  # 缺失字段给默认空串


def test_fakellm_memcell_marker_returns_valid_json():
    """FakeLLM 对【记忆抽取】返回可解析的合法 JSON。"""
    fake = FakeLLMClient()
    prompt = MEMCELL_PROMPT.format(dialogue="老人：你好")
    assert MEMCELL_PROMPT_MARKER in prompt
    raw = fake.chat([{"role": "user", "content": prompt}])
    data = json.loads(raw)
    assert data["episode_summary"]  # 非空
    assert isinstance(data["topics"], list)
    assert isinstance(data["atomic_facts"], list)
    assert isinstance(data["foresights"], list)
    assert isinstance(data["persons"], list)


def test_extractor_with_fakellm_end_to_end():
    """extractor + FakeLLM 端到端得到合法 MemCell。"""
    extractor = MemCellExtractor(FakeLLMClient())
    cell = extractor.extract(1, _turns())
    assert isinstance(cell, MemCell)
    assert cell.episode_summary  # 非空
    assert cell.atomic_facts == []
    assert cell.foresights == []
    assert cell.persons == []
