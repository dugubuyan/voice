"""EpisodeStore 测试：落库 / LIKE 检索 / 未巩固查询 / 巩固置位。

用 conftest 的内存 DB（session 夹具）。覆盖：
- add 后按 summary / topics 关键词各命中一次，不命中的 query 返回空；
- unconsolidated 只返回未巩固；
- mark_consolidated 置位并挂 scene_id；
- Episode summary 落库后读取仍完整（P2 永久性的基本保证）。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.memory.episode import EpisodeStore
from app.models import Episode


def _elder_id(session) -> int:
    from app.models import Elder

    elder = Elder(name="王秀英", age=78)
    session.add(elder)
    session.flush()
    return elder.id


def _add(store, elder_id, **kw):
    base = dict(
        summary="",
        mood="平静",
        topics=[],
        started_at=datetime(2024, 1, 1, 9, 0, 0),
        ended_at=datetime(2024, 1, 1, 9, 10, 0),
    )
    base.update(kw)
    return store.add(elder_id, **base)


def test_search_hits_by_summary_keyword(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    _add(store, elder_id, summary="今天和老人聊了去公园散步和喂鸽子的事", topics=["散步"])

    hits = store.search(elder_id, "公园")
    assert len(hits) == 1
    assert "公园" in hits[0].summary


def test_search_hits_by_topics_keyword(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    _add(store, elder_id, summary="随便聊了几句", topics=["血压", "健康"])

    hits = store.search(elder_id, "血压")
    assert len(hits) == 1
    assert "血压" in hits[0].topics


def test_search_miss_returns_empty(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    _add(store, elder_id, summary="聊了聊天气", topics=["天气"])

    assert store.search(elder_id, "股票") == []
    # 空 query 也返回空
    assert store.search(elder_id, "  ") == []


def test_search_scoped_to_elder_and_ordered_desc(session):
    elder_id = _elder_id(session)
    other_id = _elder_id(session)
    store = EpisodeStore(session)
    _add(
        store,
        elder_id,
        summary="早上聊了钓鱼",
        topics=["钓鱼"],
        started_at=datetime(2024, 1, 1, 8, 0, 0),
        ended_at=datetime(2024, 1, 1, 8, 30, 0),
    )
    _add(
        store,
        elder_id,
        summary="下午又聊了钓鱼",
        topics=["钓鱼"],
        started_at=datetime(2024, 1, 2, 8, 0, 0),
        ended_at=datetime(2024, 1, 2, 8, 30, 0),
    )
    _add(store, other_id, summary="别的老人也聊钓鱼", topics=["钓鱼"])

    hits = store.search(elder_id, "钓鱼")
    assert len(hits) == 2  # 只返回本老人的
    # ended_at 倒序：下午在前
    assert hits[0].summary == "下午又聊了钓鱼"


def test_unconsolidated_returns_only_pending_sorted(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    ep_early = _add(
        store,
        elder_id,
        summary="较早的一段",
        started_at=datetime(2024, 1, 1, 8, 0, 0),
    )
    ep_late = _add(
        store,
        elder_id,
        summary="较晚的一段",
        started_at=datetime(2024, 1, 3, 8, 0, 0),
    )
    done = _add(store, elder_id, summary="已巩固的一段")
    done.consolidated = True
    session.flush()

    pending = store.unconsolidated(elder_id)
    ids = [ep.id for ep in pending]
    assert done.id not in ids
    # started_at 升序
    assert ids == [ep_early.id, ep_late.id]


def test_mark_consolidated_sets_flag_and_scene(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    ep1 = _add(store, elder_id, summary="一段一")
    ep2 = _add(store, elder_id, summary="一段二")

    store.mark_consolidated([ep1.id, ep2.id], scene_id=42)

    for ep in (ep1, ep2):
        session.refresh(ep)
        assert ep.consolidated is True
        assert ep.scene_id == 42

    # 巩固后不再出现在未巩固列表
    assert store.unconsolidated(elder_id) == []


def test_mark_consolidated_empty_is_noop(session):
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    _add(store, elder_id, summary="不受影响")
    # 空列表不应报错、不应改动数据
    store.mark_consolidated([], scene_id=None)
    assert len(store.unconsolidated(elder_id)) == 1


def test_summary_persisted_intact(session):
    """P2 永久性基本保证：summary 落库后读回仍完整。"""
    elder_id = _elder_id(session)
    store = EpisodeStore(session)
    long_summary = "老人回忆了年轻时在工厂的经历，提到师傅老赵，情绪温暖，话题涉及退休金与老同事聚会。"
    ep = _add(store, elder_id, summary=long_summary, topics=["回忆", "工厂"])
    ep_id = ep.id

    # 从 DB 重新读取，确保是持久化后的值
    session.expire_all()
    reloaded = session.get(Episode, ep_id)
    assert reloaded.summary == long_summary
    assert reloaded.topics == ["回忆", "工厂"]
