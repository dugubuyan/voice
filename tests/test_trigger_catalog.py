"""具体触发器目录测试（任务 12，离线、内存 DB、依赖注入桩替身）。

覆盖：
- EpisodeCutTrigger：超过 episode_max_turns / 语义边界 → 切出、Episode 落库、
  atomic_facts/foresights/persons 按桩 MemCell 写入，窗口缩短并重算 token。
- ConsolidateTrigger：idle 超阈值触发 consolidate；未达阈值不触发。
- ForesightExpireTrigger：造过期前瞻，触发后失活；当天二次 evaluate 幂等不报错。
- RecentTurnCleanupTrigger：只清已巩固超期原文（断言 purge 生效条数）。
- ProactiveCheckin：idle 超阈值且启用 → 触发；proactive_idle_sec=0 → 不触发。

全部用桩 detector/extractor/consolidator，保证离线确定性、不依赖真实 LLM。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.agent.runtime import ElderRuntime, Turn
from app.agent.triggers import (
    ConsolidateTrigger,
    EpisodeCutTrigger,
    ForesightExpireTrigger,
    ProactiveCheckinTrigger,
    RecentTurnCleanupTrigger,
    TriggerContext,
    TriggerEngine,
)
from app.config import config
from app.memory.boundary import BoundaryDecision
from app.memory.conflict import MemoryConflictResolver
from app.memory.episode import EpisodeStore
from app.memory.extractor import MemCell
from app.memory.foresight import ForesightDraft, ForesightStore
from app.memory.notes import ExtractedNote
from app.memory.person import PersonStore
from app.memory.recent import RecentTurnLog
from app.models import Elder, Episode, Foresight, MemoryNote, Person, RecentTurn


# ══════════════════════════════════════════════
# 桩替身
# ══════════════════════════════════════════════
class StubBrain:
    """只提供 TriggerEngine 依赖的鸭子类型接口。"""

    def __init__(self):
        self._runtimes: dict[int, ElderRuntime] = {}

    def add_runtime(self, elder_id, last_activity_at=None):
        rt = ElderRuntime(elder_id=elder_id, system_prompt="")
        if last_activity_at is not None:
            rt.last_activity_at = last_activity_at
        self._runtimes[elder_id] = rt
        return rt

    def iter_runtimes(self):
        return list(self._runtimes.values())


class StubDetector:
    """按预设 BoundaryDecision 返回；不调用任何 LLM。"""

    def __init__(self, decision: BoundaryDecision):
        self._decision = decision

    def detect(self, window):
        return self._decision


class StubExtractor:
    """返回预设 MemCell；记录被调用的 turns。"""

    def __init__(self, cell: MemCell):
        self._cell = cell
        self.calls: list[list] = []

    def extract(self, elder_id, turns):
        self.calls.append(list(turns))
        return self._cell


class StubConsolidator:
    def __init__(self):
        self.calls: list[int] = []

    def consolidate(self, elder_id):
        self.calls.append(elder_id)


class SpyRecent:
    """包裹真实 RecentTurnLog，记录 purge 调用与返回条数。"""

    def __init__(self, real: RecentTurnLog):
        self._real = real
        self.purged: list[tuple[int, int]] = []
        self.last_count = 0

    def purge_consolidated(self, elder_id, keep_days):
        n = self._real.purge_consolidated(elder_id, keep_days)
        self.purged.append((elder_id, keep_days))
        self.last_count = n
        return n


# ══════════════════════════════════════════════
# 夹具
# ══════════════════════════════════════════════
@pytest.fixture()
def elder_id(session):
    elder = Elder(name="测试老人", age=80)
    session.add(elder)
    session.flush()
    return elder.id


def _turn(role, content, at):
    return Turn(role=role, content=content, at=at)


def _window(n, base=None):
    base = base or datetime(2024, 1, 1, 9, 0, 0)
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(_turn(role, f"第{i}句话内容", base + timedelta(minutes=i)))
    return turns


def _ctx(session, elder_id, runtime, *, now=None, idle_sec=0.0):
    return TriggerContext(
        now=now or datetime.now(),
        session=session,
        elder_id=elder_id,
        runtime=runtime,
        idle_sec=idle_sec,
    )


# ══════════════════════════════════════════════
# EpisodeCutTrigger
# ══════════════════════════════════════════════
def _episode_cut_trigger(session, detector, extractor):
    return EpisodeCutTrigger(
        detector,
        extractor,
        EpisodeStore(session),
        MemoryConflictResolver(session),
        ForesightStore(session),
        PersonStore(session),
    )


def test_episode_cut_by_max_turns_persists_and_shrinks(session, elder_id):
    """窗口轮数达到 episode_max_turns → 切出整窗、落库、窗口清空并重算 token。"""
    rt = ElderRuntime(elder_id=elder_id, system_prompt="")
    rt.window = _window(config.episode_max_turns)
    rt.window_tokens = sum(len(t.content) for t in rt.window)

    cell = MemCell(
        episode_summary="聊了买菜和孙子的事",
        mood="开心",
        topics=["买菜", "孙子"],
        atomic_facts=[
            ExtractedNote(
                is_memory_request=True,
                content="周六孙子来看她",
                date_iso="",
                persons=["孙子"],
                kind="event",
            )
        ],
        foresights=[
            ForesightDraft(content="周六准备孙子爱吃的菜", kind="todo", due_date="2024-01-06"),
            ForesightDraft(content="无期限的念叨", kind="todo", due_date=""),
        ],
        persons=[{"name": "小明", "relation": "孙子", "snippet": "孙子周六来"}],
    )
    detector = StubDetector(BoundaryDecision(False, 0))  # 走轮数兜底，不依赖边界
    extractor = StubExtractor(cell)
    trig = _episode_cut_trigger(session, detector, extractor)

    ctx = _ctx(session, elder_id, rt)
    assert trig.when(ctx) is True
    trig.action(ctx)

    # Episode 落库
    eps = session.query(Episode).filter(Episode.elder_id == elder_id).all()
    assert len(eps) == 1
    assert eps[0].summary == "聊了买菜和孙子的事"
    assert eps[0].mood == "开心"

    # atomic_facts 写入
    notes = session.query(MemoryNote).filter(MemoryNote.elder_id == elder_id).all()
    assert len(notes) == 1
    assert notes[0].content == "周六孙子来看她"
    assert notes[0].source_episode_id == eps[0].id

    # 只有带 due_date 的 foresight 落库
    fss = session.query(Foresight).filter(Foresight.elder_id == elder_id).all()
    assert len(fss) == 1
    assert fss[0].due_date == "2024-01-06"
    assert fss[0].source_episode_id == eps[0].id

    # persons 写入
    persons = session.query(Person).filter(Person.elder_id == elder_id).all()
    assert len(persons) == 1
    assert persons[0].name == "小明"
    assert persons[0].relation == "孙子"

    # 窗口被清空（整窗切出）并重算 token
    assert rt.window == []
    assert rt.window_tokens == 0
    # 抽取拿到的是被切出的那批
    assert len(extractor.calls[0]) == config.episode_max_turns


def test_episode_cut_by_semantic_boundary_partial(session, elder_id):
    """未达轮数上限但检测到语义边界 → 只切出 window[:cut_index]，保留其余并重算 token。"""
    rt = ElderRuntime(elder_id=elder_id, system_prompt="")
    rt.window = _window(4)
    rt.window_tokens = sum(len(t.content) for t in rt.window)

    cell = MemCell(episode_summary="上一段话题", mood="平静", topics=[])
    detector = StubDetector(BoundaryDecision(True, 2))
    extractor = StubExtractor(cell)
    trig = _episode_cut_trigger(session, detector, extractor)

    ctx = _ctx(session, elder_id, rt)
    assert trig.when(ctx) is True
    assert trig._cut_index == 2
    trig.action(ctx)

    eps = session.query(Episode).filter(Episode.elder_id == elder_id).all()
    assert len(eps) == 1
    # 保留后半窗口
    assert len(rt.window) == 2
    assert rt.window[0].content == "第2句话内容"
    assert rt.window_tokens == sum(len(t.content) for t in rt.window)


def test_episode_cut_when_false_no_boundary_no_bound(session, elder_id):
    """未达轮数上限且无语义边界 → 不触发。"""
    rt = ElderRuntime(elder_id=elder_id, system_prompt="")
    rt.window = _window(3)
    detector = StubDetector(BoundaryDecision(False, 0))
    extractor = StubExtractor(MemCell(episode_summary="x"))
    trig = _episode_cut_trigger(session, detector, extractor)

    assert trig.when(_ctx(session, elder_id, rt)) is False


def test_episode_cut_empty_window_false(session, elder_id):
    rt = ElderRuntime(elder_id=elder_id, system_prompt="")
    detector = StubDetector(BoundaryDecision(True, 1))
    trig = _episode_cut_trigger(session, detector, StubExtractor(MemCell(episode_summary="x")))
    assert trig.when(_ctx(session, elder_id, rt)) is False


def test_episode_cut_dedup_prevents_repeat_via_engine(session, elder_id):
    """同一次切分经引擎评估只落一个 Episode（dedup_key 去重）。"""
    brain = StubBrain()
    rt = brain.add_runtime(elder_id)
    rt.window = _window(config.episode_max_turns)

    detector = StubDetector(BoundaryDecision(False, 0))
    extractor = StubExtractor(MemCell(episode_summary="一段", mood="", topics=[]))
    trig = _episode_cut_trigger(session, detector, extractor)
    engine = TriggerEngine(session, brain, [trig])

    engine.on_turn(elder_id)  # 触发切分：窗口清空
    engine.on_turn(elder_id)  # 窗口已空，when=False，不再切

    eps = session.query(Episode).filter(Episode.elder_id == elder_id).all()
    assert len(eps) == 1


# ══════════════════════════════════════════════
# ConsolidateTrigger
# ══════════════════════════════════════════════
def test_consolidate_triggers_on_idle(session, elder_id):
    rt = ElderRuntime(elder_id=elder_id, system_prompt="")
    consolidator = StubConsolidator()
    trig = ConsolidateTrigger(consolidator, EpisodeStore(session))

    ctx = _ctx(session, elder_id, rt, idle_sec=config.consolidate_idle_sec + 1)
    assert trig.when(ctx) is True
    trig.action(ctx)
    assert consolidator.calls == [elder_id]


def test_consolidate_triggers_on_episode_accumulation(session, elder_id):
    """未巩固 Episode 数达阈值 → 触发（即便 idle 未到）。"""
    store = EpisodeStore(session)
    now = datetime(2024, 1, 1, 9, 0, 0)
    for i in range(config.consolidate_min_episodes):
        store.add(
            elder_id,
            summary=f"情景{i}",
            mood="",
            topics=[],
            started_at=now,
            ended_at=now,
        )
    consolidator = StubConsolidator()
    trig = ConsolidateTrigger(consolidator, store)

    ctx = _ctx(session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""), idle_sec=0.0)
    assert trig.when(ctx) is True


def test_consolidate_not_triggered_below_thresholds(session, elder_id):
    consolidator = StubConsolidator()
    trig = ConsolidateTrigger(consolidator, EpisodeStore(session))

    ctx = _ctx(session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""), idle_sec=0.0)
    assert trig.when(ctx) is False


# ══════════════════════════════════════════════
# ForesightExpireTrigger
# ══════════════════════════════════════════════
def test_foresight_expire_deactivates_and_idempotent(session, elder_id):
    store = ForesightStore(session)
    # 造一条过期（昨天到期）+ 一条未来
    store.add(elder_id, ForesightDraft(content="过期约", kind="appointment", due_date="2024-01-01"))
    store.add(elder_id, ForesightDraft(content="未来约", kind="appointment", due_date="2099-01-01"))

    trig = ForesightExpireTrigger(store)
    now = datetime(2024, 1, 2, 9, 0, 0)  # today = 2024-01-02
    ctx = _ctx(session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""), now=now)

    assert trig.when(ctx) is True
    trig.action(ctx)

    rows = {f.content: f for f in session.query(Foresight).filter(Foresight.elder_id == elder_id).all()}
    assert rows["过期约"].active is False
    assert rows["过期约"].reason == "expired"
    assert rows["未来约"].active is True

    # 当天二次 evaluate 幂等（不报错、不改变结果）
    trig.action(ctx)
    rows2 = {f.content: f for f in session.query(Foresight).filter(Foresight.elder_id == elder_id).all()}
    assert rows2["过期约"].active is False
    assert rows2["未来约"].active is True


# ══════════════════════════════════════════════
# RecentTurnCleanupTrigger
# ══════════════════════════════════════════════
def test_recent_cleanup_purges_only_consolidated_expired(session, elder_id):
    real = RecentTurnLog(session)
    old = datetime.now() - timedelta(days=config.recent_keep_days + 1)
    recent_ts = datetime.now()

    # 已巩固 + 超期 → 应被清
    r1 = RecentTurn(elder_id=elder_id, role="user", content="旧原文", at=old, consolidated=True)
    # 已巩固但未超期 → 保留
    r2 = RecentTurn(elder_id=elder_id, role="user", content="新原文", at=recent_ts, consolidated=True)
    # 未巩固超期 → 保留
    r3 = RecentTurn(elder_id=elder_id, role="user", content="未巩固", at=old, consolidated=False)
    session.add_all([r1, r2, r3])
    session.flush()

    spy = SpyRecent(real)
    trig = RecentTurnCleanupTrigger(spy)
    ctx = _ctx(session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""))

    assert trig.when(ctx) is True
    trig.action(ctx)

    assert spy.purged == [(elder_id, config.recent_keep_days)]
    assert spy.last_count == 1
    remaining = {r.content for r in session.query(RecentTurn).filter(RecentTurn.elder_id == elder_id).all()}
    assert remaining == {"新原文", "未巩固"}


# ══════════════════════════════════════════════
# ProactiveCheckinTrigger
# ══════════════════════════════════════════════
def test_proactive_checkin_fires_when_idle_over_threshold(session, elder_id):
    seen: list[int] = []
    trig = ProactiveCheckinTrigger(callback=lambda ctx: seen.append(ctx.elder_id))

    ctx = _ctx(
        session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""),
        idle_sec=config.proactive_idle_sec + 1,
    )
    assert trig.when(ctx) is True
    trig.action(ctx)
    assert seen == [elder_id]


def test_proactive_checkin_not_fired_below_threshold(session, elder_id):
    trig = ProactiveCheckinTrigger()
    ctx = _ctx(
        session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""),
        idle_sec=config.proactive_idle_sec - 1,
    )
    assert trig.when(ctx) is False


def test_proactive_checkin_disabled_when_config_zero(session, elder_id, monkeypatch):
    monkeypatch.setattr(config, "proactive_idle_sec", 0)
    trig = ProactiveCheckinTrigger()
    ctx = _ctx(
        session, elder_id, ElderRuntime(elder_id=elder_id, system_prompt=""),
        idle_sec=999999,
    )
    assert trig.when(ctx) is False


# ══════════════════════════════════════════════
# 工厂
# ══════════════════════════════════════════════
def test_build_default_triggers_wires_catalog(session):
    from app.agent.llm import FakeLLMClient
    from app.agent.triggers import build_default_triggers

    triggers = build_default_triggers(session, FakeLLMClient(), brain=None)
    names = [t.name for t in triggers]
    assert names == [
        "episode_cut",
        "consolidate",
        "foresight_expire",
        "recent_cleanup",
        "proactive_checkin",
    ]
