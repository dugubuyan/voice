"""检索式回忆（Recaller，需求 5）。

闲聊时按当前话题从多源记忆里「必要且充分」地召回 top-k，注入对话上下文，
取代 Phase 1/2 的「全量注入」。核心约束：

- **有界**（需求 5.3 / Property 6）：注入的条数受 `config.recall_max_items` 限制，
  累计文本长度受 `config.recall_token_budget` 限制（用字符数近似 token，见
  app/memory/working.py 的 estimate_tokens 思路）。两个上界都必须生效，
  与记忆总量无关。
- **不编造**（需求 5.2）：query 为空或无命中 → 返回空 RecallResult，
  to_injection() 返回空串，让对话回到正常闲聊而非硬凑记忆。
- **过期/失活不注入**（需求 5.4）：foresight 只取 active_valid（未过期且 active），
  facts 只取 active_notes，从源头保证「当前有效」。

检索实现选择——起步用各 Store 的关键词/字段包含匹配（LIKE 语义，非真 FTS）：
    类名保留 `Fts` 语义以标示「检索式回忆」的定位，并为后续替换为真正的
    SQLite FTS5 / 向量检索预留位置。`Recaller` 抽象接口与 `RecallResult` 返回类型
    保持稳定，将来新增 `VectorRecaller` 时调用方（_handle_chat）无需改动（需求 5.5）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import config
from app.memory.episode import EpisodeStore
from app.memory.foresight import ForesightStore
from app.memory.notes import NoteStore
from app.memory.person import PersonStore
from app.models import MemScene


# 防编造注入前缀：明确这是「大致印象」，细节以结构化记录为准，记不清不要编造。
INJECTION_HEADER = (
    "【与当前话题相关的记忆（大致印象，细节以结构化记录为准，记不清的不要编造）】"
)

# 各源基础评分：foresight / fact（结构化、精确、时效强）高于 episode / scene / person。
# 固定基础分保证结果稳定、可测；后续可叠加匹配度做更细的排序。
_SOURCE_SCORE = {
    "foresight": 1.0,
    "fact": 0.9,
    "scene": 0.6,
    "episode": 0.5,
    "person": 0.4,
}


def _estimate_len(text: str) -> int:
    """字符数近似 token 数（与 working.estimate_tokens 一致的思路）。"""
    return len(text or "")


@dataclass
class RecallItem:
    """一条召回结果：来源 + 已格式化文本 + 评分 + 源记录 id。"""

    source: str  # episode / fact / foresight / scene / person
    text: str
    score: float
    ref_id: int


@dataclass
class RecallResult:
    """一次召回的聚合结果。items 已按 score 降序并完成有界截断。"""

    items: list[RecallItem] = field(default_factory=list)

    def to_injection(self) -> str:
        """拼装成可注入 prompt 的中文文本块；items 为空时返回空串（不编造）。"""
        if not self.items:
            return ""
        lines = [INJECTION_HEADER]
        for it in self.items:
            lines.append(f"- {it.text}".rstrip())
        return "\n".join(lines)


class Recaller(ABC):
    """检索式回忆抽象接口。预留 VectorRecaller 等实现（需求 5.5）。"""

    @abstractmethod
    def recall(self, elder_id: int, query: str, now: datetime) -> RecallResult:
        """按当前话题召回「必要且充分」的 top-k 记忆。纯读，不写 DB。"""
        raise NotImplementedError


class FtsRecaller(Recaller):
    """起步实现：用各 Store 的关键词检索多源召回，融合去重后有界截断。

    注：这里是关键词包含匹配（LIKE 语义），并非真正的 FTS；类名保留 Fts 语义、
    接口预留向量替换位（需求 5.5）。
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._episodes = EpisodeStore(session)
        self._foresights = ForesightStore(session)
        self._notes = NoteStore(session)
        self._persons = PersonStore(session)

    def recall(self, elder_id: int, query: str, now: datetime) -> RecallResult:
        q = (query or "").strip()
        if not q:
            # 空 query → 不检索、不编造（需求 5.2）。
            return RecallResult(items=[])

        raw: list[RecallItem] = []
        raw.extend(self._recall_foresights(elder_id, q, now))
        raw.extend(self._recall_facts(elder_id, q))
        raw.extend(self._recall_scenes(elder_id, q))
        raw.extend(self._recall_episodes(elder_id, q))
        raw.extend(self._recall_persons(elder_id, q))

        # 融合去重：同一 (source, ref_id) 只保留一条。
        deduped: list[RecallItem] = []
        seen: set[tuple[str, int]] = set()
        for it in raw:
            key = (it.source, it.ref_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        # 稳定排序：按 score 降序（同分保持插入顺序，Python sort 稳定）。
        deduped.sort(key=lambda it: it.score, reverse=True)

        # 有界截断：条数 <= recall_max_items 且累计文本长度 <= recall_token_budget。
        # 两个上界都要生效（需求 5.3 / Property 6）。运行时读取 config，
        # 便于测试通过 monkeypatch 调小上界来验证有界性。
        max_items = config.recall_max_items
        token_budget = config.recall_token_budget

        selected: list[RecallItem] = []
        used = 0
        for it in deduped:
            if len(selected) >= max_items:
                break
            cost = _estimate_len(it.text)
            if used + cost > token_budget:
                # 超预算则跳过该条，继续尝试更短的后续条目，保持严格有界。
                continue
            selected.append(it)
            used += cost

        return RecallResult(items=selected)

    # ── 各源检索 ────────────────────────────────────────────────

    def _recall_foresights(
        self, elder_id: int, query: str, now: datetime
    ) -> list[RecallItem]:
        """只取未过期且 active 的前瞻（active_valid 已保证，需求 5.4）。"""
        items: list[RecallItem] = []
        for fs in self._foresights.active_valid(elder_id, now.date()):
            text = fs.content or ""
            if query not in text:
                continue
            due = f"（{fs.due_date}）" if fs.due_date else ""
            items.append(
                RecallItem(
                    source="foresight",
                    text=f"待办/前瞻：{text}{due}",
                    score=_SOURCE_SCORE["foresight"],
                    ref_id=fs.id,
                )
            )
        return items

    def _recall_facts(self, elder_id: int, query: str) -> list[RecallItem]:
        """只取 active 事实（active_notes 已保证不含失活项，需求 5.4）。"""
        items: list[RecallItem] = []
        for note in self._notes.active_notes(elder_id):
            text = note.content or ""
            if query not in text:
                continue
            items.append(
                RecallItem(
                    source="fact",
                    text=f"事实：{text}",
                    score=_SOURCE_SCORE["fact"],
                    ref_id=note.id,
                )
            )
        return items

    def _recall_scenes(self, elder_id: int, query: str) -> list[RecallItem]:
        """active MemScene：对 theme / summary 做包含匹配。"""
        stmt = select(MemScene).where(
            MemScene.elder_id == elder_id,
            MemScene.active == True,  # noqa: E712
        )
        items: list[RecallItem] = []
        for scene in self._session.execute(stmt).scalars().all():
            theme = scene.theme or ""
            summary = scene.summary or ""
            if query not in theme and query not in summary:
                continue
            body = summary or theme
            items.append(
                RecallItem(
                    source="scene",
                    text=f"主题记忆（{theme}）：{body}" if theme else f"主题记忆：{body}",
                    score=_SOURCE_SCORE["scene"],
                    ref_id=scene.id,
                )
            )
        return items

    def _recall_episodes(self, elder_id: int, query: str) -> list[RecallItem]:
        """情景摘要检索（EpisodeStore.search 已做 summary/topics 包含匹配）。"""
        items: list[RecallItem] = []
        for ep in self._episodes.search(elder_id, query):
            summary = ep.summary or ""
            if not summary:
                continue
            items.append(
                RecallItem(
                    source="episode",
                    text=f"曾聊到：{summary}",
                    score=_SOURCE_SCORE["episode"],
                    ref_id=ep.id,
                )
            )
        return items

    def _recall_persons(self, elder_id: int, query: str) -> list[RecallItem]:
        """人物记忆检索（PersonStore.search 对 name/relation/snippets 包含匹配）。"""
        items: list[RecallItem] = []
        for p in self._persons.search(elder_id, query=query):
            rel = f"（{p.relation}）" if p.relation else ""
            snippet = ""
            if p.snippets:
                snippet = "；".join(str(s) for s in p.snippets[-2:])
            body = f"人物：{p.name}{rel}"
            if snippet:
                body = f"{body} — {snippet}"
            items.append(
                RecallItem(
                    source="person",
                    text=body,
                    score=_SOURCE_SCORE["person"],
                    ref_id=p.id,
                )
            )
        return items
