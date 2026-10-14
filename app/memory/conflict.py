"""事实级冲突解析：recency-wins + 显式失效（不物理删除，保留审计痕迹）。

冲突判定规则（MVP，规则驱动）：
  同 kind + 同 remind_date（均非空时）→ 视为同一约定的更新
  同 kind + 均无 remind_date + content 关键词重叠度高 → 视为同一事实更新
  其余 → 不冲突，直接新增

旧记录：active=False + superseded_by 指向新记录 id，不物理删除。

────────────────────────────────────────────────────────────
不确定信息与安全相关事实的处理原则（Phase 3，原则说明，非安全代码）
────────────────────────────────────────────────────────────
1. 不确定信息（confidence="uncertain"）：
   老人的话里存疑、含糊或需要再确认的事实，以本表结构化保存，并带
   uncertain 标注，与确定信息（certain）区分。存疑信息**不进入不可回溯
   的长期摘要**（RunningSummary / ElderProfile 等重写型记忆），避免把
   未经证实的内容固化成"事实"；它只作为可追溯、可失效、可被后续确定
   信息取代（recency-wins + superseded_by）的结构化线索存在。

2. 安全相关事实（是否已服药、健康读数、紧急联系人等）：
   一律以结构化记录为准——服药状态看 MedicationLog、健康读数看
   HealthReading、紧急处置看 EmergencyEvent——**绝不由本模块的记忆
   覆盖或推断**。本模块只处理陪伴型事实记忆（约定/人物/事件等），
   即便抽取出与安全相关的表述，也不改写上述结构化真相来源。

以上为原则说明。本任务不改动任何安全代码（medication/emergency/health），
仅在陪伴型记忆写入路径上支持 confidence 标注与来源情景追溯。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory.notes import ExtractedNote, NoteStore
from app.models import MemoryNote


def _keyword_overlap(a: str, b: str, threshold: float = 0.4) -> bool:
    """粗略判断两段文本关键词重叠度是否超过阈值（Jaccard 字符集）。"""
    if not a or not b:
        return False
    sa = set(a.replace(" ", ""))
    sb = set(b.replace(" ", ""))
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= threshold


class MemoryConflictResolver:
    """事实级冲突解析器。通过 upsert_note 代替直接调 NoteStore.add_note。"""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._store = NoteStore(session)

    def upsert_note(
        self,
        elder_id: int,
        candidate: ExtractedNote,
        *,
        source_episode_id: int | None = None,
        confidence: str = "certain",
        source: str = "explicit",
    ) -> MemoryNote:
        """写入新事实，并把语义冲突的旧记录显式失效。

        冲突判定优先级：
          1. 同 kind + 同 remind_date（两者均非空）→ 冲突
          2. 同 kind + 均无 remind_date + content 关键词高度重叠 → 冲突
          否则不冲突，直接新增。

        Phase 3 可选参数（均有默认，保证向后兼容；现有调用无需改动）：
          source_episode_id: 该事实来源情景 id，透传给底层落库。
          confidence: "certain"/"uncertain"，透传给底层落库。
        recency-wins + superseded_by 审计逻辑保持不变（P12）。
        """
        existing = self._find_conflicting(elder_id, candidate, confidence)
        new_row = self._store.add_note(
            elder_id,
            candidate,
            source_episode_id=source_episode_id,
            confidence=confidence,
            source=source,
        )

        if existing is not None:
            existing.active = False              # 显式失效
            existing.superseded_by = new_row.id  # 取代指针
            self._session.flush()

        return new_row

    def _find_conflicting(
        self, elder_id: int, candidate: ExtractedNote, candidate_confidence: str = "certain"
    ) -> MemoryNote | None:
        """找到与 candidate 语义冲突的最近一条 active 记录；找不到返回 None。

        高可信保护（需求 6.2 / P6）：当候选为 uncertain 时，**不得取代**已存在的
        certain 或 source="setup" 记录——遇到这类行时跳过，让新的存疑记录仅新增、
        不使权威事实失效。certain 取代 certain / certain 取代 uncertain 仍按 recency-wins。
        """
        stmt = (
            select(MemoryNote)
            .where(
                MemoryNote.elder_id == elder_id,
                MemoryNote.active == True,  # noqa: E712
                MemoryNote.kind == candidate.kind,
            )
            .order_by(MemoryNote.created_at.desc())
            .limit(10)
        )
        candidates = list(self._session.execute(stmt).scalars().all())

        for row in candidates:
            # 高可信保护：存疑候选不取代权威（certain / setup）记录
            if candidate_confidence != "certain" and (
                row.confidence == "certain" or row.source == "setup"
            ):
                continue
            # 规则 1：同 remind_date（均非空）
            if candidate.date_iso and row.remind_date == candidate.date_iso:
                return row
            # 规则 2：均无日期 + 内容关键词高度重叠
            if not candidate.date_iso and not row.remind_date:
                if _keyword_overlap(candidate.content, row.content):
                    return row
        return None
