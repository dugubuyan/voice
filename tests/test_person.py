"""PersonStore（人物/关系记忆）测试。需求 6。"""
from __future__ import annotations

from datetime import datetime

from app.memory.person import MAX_SNIPPETS, PersonStore
from app.models import Person


def _elder_id(elders) -> int:
    return elders["李桂芳"].id


def test_first_upsert_creates_person(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)

    p = store.upsert_mention(eid, "王建国", relation="老朋友", snippet="一起下棋")
    session.commit()

    assert p.id is not None
    assert p.name == "王建国"
    assert p.relation == "老朋友"
    assert p.mention_count == 1
    assert p.snippets == ["一起下棋"]
    assert p.last_mentioned_at is not None


def test_repeated_upsert_updates_not_duplicates(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)

    t1 = datetime(2024, 1, 1, 9, 0, 0)
    t2 = datetime(2024, 1, 2, 9, 0, 0)
    store.upsert_mention(eid, "王建国", relation="", snippet="一起下棋", at=t1)
    p = store.upsert_mention(eid, "王建国", relation="老朋友", snippet="约好周末喝茶", at=t2)
    session.commit()

    # 不新增行
    rows = session.query(Person).filter(Person.elder_id == eid, Person.name == "王建国").all()
    assert len(rows) == 1

    assert p.mention_count == 2
    # last_mentioned_at 前移到更晚的时间
    assert p.last_mentioned_at == t2
    # 关系原为空 → 被补上
    assert p.relation == "老朋友"
    # snippet 追加
    assert p.snippets == ["一起下棋", "约好周末喝茶"]


def test_upsert_dedupes_snippets(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)

    store.upsert_mention(eid, "李梅", snippet="女儿来看我")
    p = store.upsert_mention(eid, "李梅", snippet="女儿来看我")  # 重复片段
    session.commit()

    assert p.snippets == ["女儿来看我"]
    assert p.mention_count == 2


def test_snippets_are_bounded(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)

    for i in range(MAX_SNIPPETS + 3):
        store.upsert_mention(eid, "张三", snippet=f"片段{i}")
    session.commit()

    p = store.search(eid, name="张三")[0]
    assert len(p.snippets) == MAX_SNIPPETS
    # 只保留最近的若干条
    assert p.snippets[0] == "片段3"
    assert p.snippets[-1] == f"片段{MAX_SNIPPETS + 2}"


def test_search_by_name(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)
    store.upsert_mention(eid, "王建国", relation="老朋友")
    store.upsert_mention(eid, "李梅", relation="女儿")
    session.commit()

    hits = store.search(eid, name="王建国")
    assert len(hits) == 1
    assert hits[0].name == "王建国"


def test_search_by_query_matches_relation_and_snippet(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)
    store.upsert_mention(eid, "王建国", relation="老朋友", snippet="一起下棋")
    store.upsert_mention(eid, "李梅", relation="女儿", snippet="常打电话")
    session.commit()

    # 命中 relation
    by_relation = store.search(eid, query="女儿")
    assert [p.name for p in by_relation] == ["李梅"]

    # 命中 snippet
    by_snippet = store.search(eid, query="下棋")
    assert [p.name for p in by_snippet] == ["王建国"]

    # 命中 name
    by_name = store.search(eid, query="李")
    assert [p.name for p in by_name] == ["李梅"]


def test_search_orders_by_last_mentioned_desc(session, elders):
    store = PersonStore(session)
    eid = _elder_id(elders)
    store.upsert_mention(eid, "老赵", at=datetime(2024, 1, 1))
    store.upsert_mention(eid, "老钱", at=datetime(2024, 3, 1))
    store.upsert_mention(eid, "老孙", at=datetime(2024, 2, 1))
    session.commit()

    names = [p.name for p in store.search(eid)]
    assert names == ["老钱", "老孙", "老赵"]


def test_search_scoped_to_elder(session, elders):
    store = PersonStore(session)
    a = elders["李桂芳"].id
    b = elders["张建国"].id
    store.upsert_mention(a, "王建国")
    store.upsert_mention(b, "王建国")
    session.commit()

    hits = store.search(a)
    assert len(hits) == 1
    assert all(p.elder_id == a for p in hits)
