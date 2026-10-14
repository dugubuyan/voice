"""人物/关系记忆：归集老人生活里反复出现的人。需求 6。

从抽取出的人物称谓/片段累积成 Person 记忆：同一 elder 下同名人物只保留一条，
再次提及时更新最近提及时间、累加提及次数、追加片段（去重且有界），
关系为空时补上。不与档案里的 emergency_contacts 矛盾（本任务不强制合并，
upsert 到同名可自然融合）。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Person

# 每个人物最多保留最近若干条片段，避免无界增长
MAX_SNIPPETS = 5


class PersonStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def _find_by_name(self, elder_id: int, name: str) -> Person | None:
        stmt = select(Person).where(
            Person.elder_id == elder_id, Person.name == name
        )
        return self._session.execute(stmt).scalars().first()

    def upsert_mention(
        self,
        elder_id: int,
        name: str,
        relation: str = "",
        snippet: str = "",
        at: datetime | None = None,
    ) -> Person:
        """记录一次人物提及。

        已存在同名人物 → 更新最近提及时间 + 次数 +1 + 追加片段（去重、有界），
        关系原为空则补上；否则新建（mention_count=1）。不新建重复条目。
        """
        at = at or datetime.now()
        person = self._find_by_name(elder_id, name)
        if person is None:
            person = Person(
                elder_id=elder_id,
                name=name,
                relation=relation or "",
                snippets=[snippet] if snippet else [],
                last_mentioned_at=at,
                mention_count=1,
            )
            self._session.add(person)
            self._session.flush()
            return person

        # 更新既有条目
        person.last_mentioned_at = at
        person.mention_count = (person.mention_count or 0) + 1
        if relation and not person.relation:
            person.relation = relation
        if snippet:
            # 重新赋值新列表以触发 JSON 列的变更追踪
            snippets = list(person.snippets or [])
            if snippet not in snippets:
                snippets.append(snippet)
            # 只保留最近 MAX_SNIPPETS 条
            person.snippets = snippets[-MAX_SNIPPETS:]
        self._session.flush()
        return person

    def search(
        self,
        elder_id: int,
        query: str | None = None,
        name: str | None = None,
    ) -> list[Person]:
        """检索人物：name 精确匹配优先；query 对 name/relation/snippets 做包含匹配。

        结果按 last_mentioned_at 倒序（空值排最后）。
        """
        stmt = select(Person).where(Person.elder_id == elder_id)
        if name:
            stmt = stmt.where(Person.name == name)
        rows = list(self._session.execute(stmt).scalars().all())

        if not name and query:
            # snippets 存为 JSON（中文被转义），在 Python 侧做可靠的包含匹配
            def _match(p: Person) -> bool:
                if query in (p.name or "") or query in (p.relation or ""):
                    return True
                return any(query in s for s in (p.snippets or []))

            rows = [p for p in rows if _match(p)]

        rows.sort(
            key=lambda p: p.last_mentioned_at or datetime.min, reverse=True
        )
        return rows
