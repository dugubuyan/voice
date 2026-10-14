"""端到端集成：情景切分 → 巩固 → 检索式回忆（需求 5 的端到端回忆）。

全程离线确定性：conftest 内存 DB + FakeLLM，不依赖真实 LLM、不使用真实计时器
（on_pulse 由测试注入固定 now）。覆盖三阶段生命周期的贯通：

  Episodic Trace Formation：多轮对话经 EpisodeCutTrigger 切出 Episode 落库。
  Semantic Consolidation：brain.on_pulse(future) 触发 Consolidator，产出 MemScene/画像。
  Reconstructive Recollection：「你还记得……」经 FtsRecaller 召回此前 Episode/事实。

说明：Brain 内部用 datetime.now() 作对话时间戳，无法从外部注入过去时刻，故
「几天前」的历史 Episode 用 EpisodeStore 直接造历史记录来模拟跨天；「今天」的
对话则真实走 Brain.process_turn。二者共用同一 session，验证跨天召回连贯。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.config import config
from app.memory.episode import EpisodeStore
from app.memory.recall import FtsRecaller
from app.models import ElderProfile, Episode, MemScene, MemoryNote


def _talk(brain, elder_id, texts):
    """按顺序把若干句喂给 Brain，返回每轮 TurnResult 列表。"""
    return [brain.process_turn(elder_id, t) for t in texts]


def _episode_cut_texts(first: str, rounds: int) -> list[str]:
    """构造刚好触发轮数兜底切分的对话。

    每次 process_turn 追加 user+assistant 两轮，rounds 次 → 2*rounds 轮；
    首句放置可召回关键词（切出整窗时 FakeLLM 抽取以首句为情景摘要）。
    """
    texts = [first]
    texts += [f"随便聊聊第{i}句" for i in range(1, rounds)]
    return texts


# ══════════════════════════════════════════════════════════════════════
# 端到端：切分 → 巩固 → 召回
# ══════════════════════════════════════════════════════════════════════
def test_episode_formation_then_consolidate_then_recall(brain, started):
    """一段多轮对话切出 Episode，on_pulse 巩固出 MemScene/画像，随后可召回。"""
    elder_id, _ = started()

    # 轮数兜底：episode_max_turns 轮触发切分。每次 process_turn 产生 2 轮，
    # 故需 episode_max_turns/2 次调用；首句带可召回关键词「女儿李梅」。
    rounds = config.episode_max_turns // 2
    texts = _episode_cut_texts("我女儿李梅上周三来看我了，我们一起包了饺子", rounds)
    _talk(brain, elder_id, texts)

    # ── 阶段 1：Episode 落库（切分不丢段）──
    episodes = (
        brain._session.query(Episode).filter(Episode.elder_id == elder_id).all()
    )
    assert len(episodes) >= 1
    ep = episodes[0]
    assert "女儿" in ep.summary  # 首句作为情景摘要，含可召回关键词
    assert ep.consolidated is False  # 尚未巩固

    # ── 阶段 2：on_pulse 触发巩固（空闲超阈值）──
    future = datetime.now() + timedelta(seconds=config.consolidate_idle_sec + 100)
    brain.on_pulse(future)

    scenes = (
        brain._session.query(MemScene)
        .filter(MemScene.elder_id == elder_id, MemScene.active == True)  # noqa: E712
        .all()
    )
    assert len(scenes) >= 1  # 巩固产出至少一个主题场景
    profile = (
        brain._session.query(ElderProfile)
        .filter(ElderProfile.elder_id == elder_id)
        .first()
    )
    assert profile is not None and profile.content  # 画像被重写
    # 巩固后 Episode 被标记已巩固，但摘要仍然保留（Property 2）
    ep_after = brain._session.get(Episode, ep.id)
    assert ep_after.consolidated is True
    assert "女儿" in ep_after.summary

    # ── 阶段 3：检索式回忆（需求 5 端到端）──
    result = FtsRecaller(brain._session).recall(elder_id, "女儿", datetime.now())
    assert result.items, "「你还记得女儿…」应能召回到此前 Episode"
    assert any("女儿" in it.text for it in result.items)
    injection = result.to_injection()
    assert "不要编造" in injection  # 防编造措辞在场（约束 C5）


def test_remembered_fact_is_recallable(brain, started):
    """通过 remember_note 写入的事实，之后能被「你还记得」类 query 召回。"""
    elder_id, _ = started()

    brain.process_turn(elder_id, "帮我记住我女儿叫李梅，她是护士")

    notes = (
        brain._session.query(MemoryNote)
        .filter(MemoryNote.elder_id == elder_id, MemoryNote.active == True)  # noqa: E712
        .all()
    )
    assert notes, "remember_note 应写入一条事实记忆"

    result = FtsRecaller(brain._session).recall(elder_id, "李梅", datetime.now())
    assert result.items
    assert any(it.source == "fact" for it in result.items)
    assert any("李梅" in it.text for it in result.items)


def test_cross_day_old_episode_still_recallable(brain, started):
    """几天前的历史 Episode（跨天）在今天的对话后仍可被召回，验证连续性。"""
    elder_id, _ = started()

    # 造「几天前」的历史情景（直接落库模拟跨天，见模块顶部说明）。
    days_ago = datetime.now() - timedelta(days=5)
    EpisodeStore(brain._session).add(
        elder_id,
        summary="五天前和老伴一起去公园看了樱花，心情很好",
        mood="开心",
        topics=["公园", "樱花"],
        started_at=days_ago,
        ended_at=days_ago,
    )
    brain._session.commit()

    # 「今天」聊点别的（真实走 Brain）。
    brain.process_turn(elder_id, "今天天气不错，我在家浇了花")

    # 「你还记得那天去公园看樱花吗」→ 应召回五天前那段 Episode。
    result = FtsRecaller(brain._session).recall(elder_id, "樱花", datetime.now())
    assert result.items
    assert any("樱花" in it.text for it in result.items)


def test_recall_empty_when_no_related_memory(brain, started):
    """无相关记忆时召回为空、不编造（需求 5.2）。"""
    elder_id, _ = started()
    brain.process_turn(elder_id, "你好呀")

    result = FtsRecaller(brain._session).recall(
        elder_id, "完全不存在的关键词zzz", datetime.now()
    )
    assert result.items == []
    assert result.to_injection() == ""
