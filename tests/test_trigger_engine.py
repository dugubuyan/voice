"""TriggerEngine 引擎骨架测试（离线，内存 DB）。

覆盖：
- 谓词为真才执行 action；为假不执行。
- 相同 dedup_key 不重复触发；模拟重启（丢弃旧 engine、同一 DB session 新建）仍去重（P9）。
- cooldown_sec 内跳过、超过后可再触发。
- 一个 trigger 的 action 抛异常，不影响同批其他 trigger，且不外抛（P10）。
- on_pulse 遍历多个 runtime，按各自 idle_sec 传入。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.agent.runtime import ElderRuntime
from app.agent.triggers import TriggerContext, TriggerEngine
from app.models import TriggerFire


# ---------- 桩 brain ----------
class StubBrain:
    """只提供 TriggerEngine 依赖的鸭子类型接口。"""

    def __init__(self):
        self._runtimes: dict[int, ElderRuntime] = {}

    def add_runtime(self, elder_id: int, last_activity_at: datetime) -> ElderRuntime:
        rt = ElderRuntime(elder_id=elder_id, system_prompt="")
        rt.last_activity_at = last_activity_at
        self._runtimes[elder_id] = rt
        return rt

    def iter_runtimes(self):
        return list(self._runtimes.values())


# ---------- 桩 Trigger ----------
class RecordingTrigger:
    """记录被调用次数与见到的 ctx；谓词/去重键/冷却可配置。"""

    def __init__(self, name, *, cooldown_sec=0, predicate=None, key="k"):
        self.name = name
        self.cooldown_sec = cooldown_sec
        self._predicate = predicate or (lambda ctx: True)
        self._key = key
        self.calls: list[TriggerContext] = []

    def when(self, ctx):
        return self._predicate(ctx)

    def action(self, ctx):
        self.calls.append(ctx)

    def dedup_key(self, ctx):
        # 允许按 elder 区分，便于 on_pulse 多 runtime 用例。
        if callable(self._key):
            return self._key(ctx)
        return self._key


class BoomTrigger:
    """action 必抛异常，用于验证失败隔离。"""

    def __init__(self, name="boom"):
        self.name = name
        self.cooldown_sec = 0
        self.fired = 0

    def when(self, ctx):
        return True

    def action(self, ctx):
        self.fired += 1
        raise RuntimeError("boom")

    def dedup_key(self, ctx):
        return "boom-key"


def _fire_count(session, name) -> int:
    return (
        session.query(TriggerFire)
        .filter(TriggerFire.trigger_name == name)
        .count()
    )


# ---------- 用例 ----------
def test_predicate_gates_action(session):
    brain = StubBrain()
    brain.add_runtime(1, datetime.now())
    fires = RecordingTrigger("fires", predicate=lambda ctx: True)
    skips = RecordingTrigger("skips", predicate=lambda ctx: False)
    engine = TriggerEngine(session, brain, [fires, skips])

    engine.on_turn(1)

    assert len(fires.calls) == 1
    assert len(skips.calls) == 0
    # 只有真正执行的 trigger 记录 TriggerFire。
    assert _fire_count(session, "fires") == 1
    assert _fire_count(session, "skips") == 0


def test_on_turn_sets_idle_zero_and_elder_id(session):
    brain = StubBrain()
    brain.add_runtime(7, datetime.now())
    t = RecordingTrigger("t")
    engine = TriggerEngine(session, brain, [t])

    engine.on_turn(7)

    ctx = t.calls[0]
    assert ctx.elder_id == 7
    assert ctx.idle_sec == 0.0
    assert ctx.runtime is brain._runtimes[7]


def test_dedup_same_key_not_refired(session):
    brain = StubBrain()
    brain.add_runtime(1, datetime.now())
    t = RecordingTrigger("dedup", cooldown_sec=0, key="same")
    engine = TriggerEngine(session, brain, [t])

    engine.on_turn(1)
    engine.on_turn(1)
    engine.on_turn(1)

    # 永久去重：只执行一次。
    assert len(t.calls) == 1
    assert _fire_count(session, "dedup") == 1


def test_idempotent_across_restart(session):
    """模拟重启：丢弃旧 engine，用同一 DB session 新建 engine，仍不重复触发（P9）。"""
    brain = StubBrain()
    brain.add_runtime(1, datetime.now())

    t1 = RecordingTrigger("persisted", cooldown_sec=0, key="same")
    engine1 = TriggerEngine(session, brain, [t1])
    engine1.on_turn(1)
    assert len(t1.calls) == 1

    # 丢弃 engine1 与其内存中的 trigger 实例，模拟进程重启。
    del engine1, t1

    t2 = RecordingTrigger("persisted", cooldown_sec=0, key="same")
    engine2 = TriggerEngine(session, brain, [t2])
    engine2.on_turn(1)

    # 去重只依赖 DB 里的 TriggerFire，重启后仍不重复触发。
    assert len(t2.calls) == 0
    assert _fire_count(session, "persisted") == 1


def test_cooldown_skips_then_allows(session):
    brain = StubBrain()
    brain.add_runtime(1, datetime.now())
    t = RecordingTrigger("cooling", cooldown_sec=60, key="same")
    engine = TriggerEngine(session, brain, [t])

    base = datetime(2024, 1, 1, 12, 0, 0)

    def ctx_at(now):
        return TriggerContext(
            now=now, session=session, elder_id=1,
            runtime=brain._runtimes[1], idle_sec=0.0,
        )

    # 首次触发。
    engine._evaluate(ctx_at(base))
    # 冷却期内（+30s）跳过。
    engine._evaluate(ctx_at(base + timedelta(seconds=30)))
    assert len(t.calls) == 1
    # 超过冷却（+61s）可再次触发。
    engine._evaluate(ctx_at(base + timedelta(seconds=61)))
    assert len(t.calls) == 2
    assert _fire_count(session, "cooling") == 2


def test_action_exception_isolated(session):
    """一个 trigger 的 action 抛异常，不影响其他 trigger，也不外抛（P10）。"""
    brain = StubBrain()
    brain.add_runtime(1, datetime.now())
    boom = BoomTrigger("boom")
    after = RecordingTrigger("after")
    engine = TriggerEngine(session, brain, [boom, after])

    # 不应抛出。
    engine.on_turn(1)

    assert boom.fired == 1
    # 异常 trigger 之后的 trigger 仍被评估执行。
    assert len(after.calls) == 1
    # 抛异常的 trigger 未记录 TriggerFire（未走到 _record_fire）。
    assert _fire_count(session, "boom") == 0
    assert _fire_count(session, "after") == 1


def test_on_pulse_iterates_runtimes_with_idle(session):
    now = datetime(2024, 1, 1, 12, 0, 0)
    brain = StubBrain()
    brain.add_runtime(1, now - timedelta(seconds=100))
    brain.add_runtime(2, now - timedelta(seconds=10))

    # 每个 elder 用各自的 dedup_key，避免相互去重。
    t = RecordingTrigger("pulse", key=lambda ctx: f"e{ctx.elder_id}")
    engine = TriggerEngine(session, brain, [t])

    engine.on_pulse(now)

    seen = {c.elder_id: c.idle_sec for c in t.calls}
    assert seen == {1: 100.0, 2: 10.0}
    assert _fire_count(session, "pulse") == 2
