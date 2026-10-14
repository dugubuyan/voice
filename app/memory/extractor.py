"""记忆抽取器（MemCellExtractor）。

把一个 raw episode（连续轮次）一次 LLM 调用抽成 MemCell：
情景摘要 + 情绪 + 主题 + 原子事实 + 前瞻 + 人物。

降级（需求 2.5/2.6）：LLM 调用或 JSON 解析失败 → 返回结构合法的降级 MemCell：
episode_summary 用降级摘要（原文首句/拼接，保证非空），其余字段为空。
绝不抛出、不阻塞（参考 app/memory/folder.py 的 try/except 降级写法）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.agent.runtime import Turn
from app.memory.foresight import ForesightDraft
from app.memory.notes import ExtractedNote

MEMCELL_PROMPT_MARKER = "【记忆抽取】"

MEMCELL_PROMPT = """\
【记忆抽取】你在维护一段"老人陪护对话"的长期记忆。下面是【本段对话】（按时间从早到晚），
请把这段对话抽取成结构化记忆。只输出 JSON，不要额外文字：
{{"episode_summary":"这段对话的简要情景摘要（非空，150字内）",
  "mood":"平静/开心/低落/焦虑/... 之一，可空",
  "topics":["话题标签", "..."],
  "atomic_facts":[{{"content":"要记住的事实核心内容","date_iso":"YYYY-MM-DD或空串","persons":["人名"],"kind":"commitment/person/event/health/other"}}],
  "foresights":[{{"content":"未来关注点","kind":"appointment/commitment/todo","due_date":"YYYY-MM-DD或空串"}}],
  "persons":[{{"name":"人名","relation":"关系称谓","snippet":"相关片段"}}]}}

字段说明：
- episode_summary：必填、非空，概括这段对话发生了什么。
- atomic_facts/foresights/persons：没有则给空数组。
- topics/persons 不必穷举，抓住重点即可。

【本段对话】
{dialogue}
"""

# 字段截断上限，防止 LLM 返回过长列表
_MAX_TOPICS = 5
_MAX_PERSONS = 10
_MAX_FACTS = 20
_MAX_FORESIGHTS = 20


@dataclass
class MemCell:
    """一次抽取产出：情景 + 事实 + 前瞻。EverMemOS MemCell 的落地。"""

    episode_summary: str
    mood: str = ""
    topics: list[str] = field(default_factory=list)
    atomic_facts: list[ExtractedNote] = field(default_factory=list)
    foresights: list[ForesightDraft] = field(default_factory=list)
    persons: list[dict] = field(default_factory=list)


class MemCellExtractor:
    """记忆抽取器（stateless，依赖注入 LLM 客户端）。"""

    def __init__(self, llm) -> None:
        self._llm = llm

    def extract(self, elder_id: int, turns: list[Turn]) -> MemCell:
        """把一段连续轮次抽取成 MemCell。

        Args:
            elder_id: 老人 id（当前仅用于调用签名一致，不写 DB）。
            turns: 连续对话轮次。

        Returns:
            结构合法的 MemCell。

        Raises:
            不抛出异常：LLM 调用或 JSON 解析失败时返回降级 MemCell
            （episode_summary 非空、其余为空）。落库由调用方事务负责。
        """
        dialogue = "\n".join(
            f"{'老人' if t.role == 'user' else '助手'}：{t.content}"
            for t in turns
        )
        prompt = MEMCELL_PROMPT.format(dialogue=dialogue)

        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")

            episode_summary = str(data.get("episode_summary", "")).strip()[:200]
            if not episode_summary:
                raise ValueError("empty episode_summary")

            mood = str(data.get("mood", "")).strip()
            topics = self._clean_topics(data.get("topics", []))
            atomic_facts = self._clean_facts(data.get("atomic_facts", []))
            foresights = self._clean_foresights(data.get("foresights", []))
            persons = self._clean_persons(data.get("persons", []))

            return MemCell(
                episode_summary=episode_summary,
                mood=mood,
                topics=topics,
                atomic_facts=atomic_facts,
                foresights=foresights,
                persons=persons,
            )
        except Exception:
            # 降级：结构合法的空 MemCell，episode_summary 用降级摘要保证非空
            return MemCell(episode_summary=self._degraded_summary(turns))

    # ──────────────────────────────────────────
    # 字段容错清洗
    # ──────────────────────────────────────────

    @staticmethod
    def _clean_topics(raw) -> list[str]:
        if not isinstance(raw, list):
            return []
        topics = [str(t).strip() for t in raw if str(t).strip()]
        return topics[:_MAX_TOPICS]

    @staticmethod
    def _clean_facts(raw) -> list[ExtractedNote]:
        if not isinstance(raw, list):
            return []
        facts: list[ExtractedNote] = []
        for item in raw[:_MAX_FACTS]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            persons_raw = item.get("persons", [])
            persons = (
                [str(p).strip() for p in persons_raw if str(p).strip()]
                if isinstance(persons_raw, list)
                else []
            )
            facts.append(
                ExtractedNote(
                    is_memory_request=True,
                    content=content,
                    date_iso=str(item.get("date_iso", "")).strip(),
                    persons=persons,
                    kind=str(item.get("kind", "other")).strip() or "other",
                )
            )
        return facts

    @staticmethod
    def _clean_foresights(raw) -> list[ForesightDraft]:
        if not isinstance(raw, list):
            return []
        drafts: list[ForesightDraft] = []
        for item in raw[:_MAX_FORESIGHTS]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            drafts.append(
                ForesightDraft(
                    content=content,
                    kind=str(item.get("kind", "commitment")).strip() or "commitment",
                    due_date=str(item.get("due_date", "")).strip(),
                )
            )
        return drafts

    @staticmethod
    def _clean_persons(raw) -> list[dict]:
        if not isinstance(raw, list):
            return []
        persons: list[dict] = []
        for item in raw[:_MAX_PERSONS]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            persons.append(
                {
                    "name": name,
                    "relation": str(item.get("relation", "")).strip(),
                    "snippet": str(item.get("snippet", "")).strip(),
                }
            )
        return persons

    @staticmethod
    def _degraded_summary(turns: list[Turn]) -> str:
        """降级摘要：拼接老人首句/前若干轮内容，保证非空。"""
        for t in turns:
            text = (t.content or "").strip()
            if text:
                return text[:200]
        return "闲聊了几句"
