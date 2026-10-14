"""属性测试：P1/P3/P4/P6/P9（记忆模块 Phase 3）。

⚠️ hypothesis 不可用（环境未安装，且本项目坚持「优先零新增依赖」），故改用
   **固定随机种子 + 普通 pytest 断言的随机化循环测试**覆盖同样的属性。
   每个测试用固定 seed 生成多组随机输入，对每组断言不变量，兼顾覆盖面与确定性
   （可复现）。若将来引入 hypothesis，可平滑替换这些循环为 @given 生成器。

覆盖（对应 design.md Correctness Properties）：
  P1 窗口有界：任意 process_turn 后窗口 token 有界（token 兜底 + EpisodeCut 缩窗）。
  P3 切分不丢段：每次切分恰好产生一条 Episode，切出的轮次全部计入、窗口精确缩短。
  P4 Foresight 单调失活：active 只会 True→False，过期后不复活、不再被 active_valid 召回。
  P6 召回有界：recall().items 条数 <= recall_max_items 且注入文本长度 <= recall_token_budget。
  P9 Trigger 幂等：同 (name, dedup_key) 不重复触发，重建 engine（模拟重启）读库仍去重。
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import app.config as config_module
from app.agent.runtime import ElderRuntime, Turn
from app.agent.triggers import (
    EpisodeCutTrigger,
    TriggerContext,
    TriggerEngine,
)
from app.config import config
from app.memory.boundary import BoundaryDecision
from app.memory.conflict import MemoryConflictResolver
from app.memory.episode import EpisodeStore
from app.memory.extractor import MemCell
from app.memory.foresight import ForesightDraft, ForesightStore
from app.memory.notes import ExtractedNote, NoteStore
from app.memory.person import PersonStore
from app.memory.recall import FtsRecaller
from app.models import Elder, Episode, Foresight, MemScene


# ══════════════════════════════════════════════════════════════════════
# 桩替身（复用 test_trigger_catalog 的思路，保持离线确定性）
# ══════════════════════════════════════════════════════════════════════
class StubBrain:
    def __init__(self):
        self._runtimes: dict[int, ElderRuntime] = {}

    def iter_runtimes(self):
        return list(self._runtimes.values())


class StubDetector:
    def __init__(self, decision: BoundaryDecision):
        self._decision = decision

    def detect(self, window):
        return self._decision


class StubExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, elder_id, turns):
        self.calls += 1
        # 结构合法、事实/前瞻/人物为空的 MemCell，摘要非空。
        return MemCell(episode_summary="一段对话", mood="平静", topics=[])


def _make_window(rng, n, base=None):
    base = base or datetime(2024, 1, 1, 9, 0, 0)
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = "".join(rng.choice("对话内容一二三四五") for _ in range(rng.randint(3, 12)))
        turns.append(Turn(role, content, base + timedelta(minutes=i)))
    return turns


# ══════════════════════════════════════════════════════════════════════
# P1 窗口有界
# ══════════════════════════════════════════════════════════════════════
def test_p1_window_bounded_after_process_turn(brain, started, monkeypatch):
    """随机对话序列下，任意 process_turn 后窗口保持有界。

    有界性的真实保证（见 WorkingMemory.fold_if_over_budget）：折叠会一直挤出
    直到 window_tokens <= token_budget，或窗口已缩到 <= fold_batch（单批下限）。
    因此不变量为二者之一成立；再叠加 EpisodeCut 的轮数兜底，窗口不会无界增长。
    用较小 token_budget 逼出频繁折叠以增强覆盖。
    """
    monkeypatch.setattr(config_module.config, "token_budget", 60)
    elder_id, _ = started()

    rng = random.Random(20240115)
    vocab = ["你好", "今天天气不错", "我有点累", "孙子来看我了",
             "血压有点高", "我女儿叫李梅", "想出去走走", "谢谢你陪我聊天"]
    for _ in range(60):
        # 偶尔来一句长文本，逼出折叠路径
        if rng.random() < 0.25:
            text = "".join(rng.choice("春夏秋冬风花雪月") for _ in range(rng.randint(30, 80)))
        else:
            text = rng.choice(vocab)
        brain.process_turn(elder_id, text)
        runtime = brain.get_runtime(elder_id)
        assert (
            runtime.window_tokens <= config.token_budget
            or len(runtime.window) <= config.fold_batch
        ), (
            f"窗口无界：tokens={runtime.window_tokens} "
            f"budget={config.token_budget} len={len(runtime.window)}"
        )
        # window_tokens 应与窗口内容估算一致（无漂移）
        assert runtime.window_tokens == sum(len(t.content) for t in runtime.window)


# ══════════════════════════════════════════════════════════════════════
# P3 切分不丢段
# ══════════════════════════════════════════════════════════════════════
def test_p3_cut_produces_exactly_one_episode_no_loss(session):
    """随机 window + 随机语义边界：每次切分恰好产生一条 Episode，且切出的轮次
    全部计入、窗口精确缩短（不丢段、不重复）。"""
    elder = Elder(name="P3老人", age=80)
    session.add(elder)
    session.flush()
    elder_id = elder.id

    rng = random.Random(2024031)
    episodes_expected = 0
    turns_cut_total = 0

    for _ in range(120):
        n = rng.randint(1, config.episode_max_turns - 1)  # 保持在轮数兜底之下
        window = _make_window(rng, n)
        rt = ElderRuntime(elder_id=elder_id, system_prompt="")
        rt.window = list(window)
        rt.window_tokens = sum(len(t.content) for t in rt.window)

        # 随机决定是否有语义边界及切分点
        if rng.random() < 0.7:
            cut = rng.randint(1, n)
            detector = StubDetector(BoundaryDecision(True, cut))
        else:
            detector = StubDetector(BoundaryDecision(False, 0))

        trig = EpisodeCutTrigger(
            detector,
            StubExtractor(),
            EpisodeStore(session),
            MemoryConflictResolver(session),
            ForesightStore(session),
            PersonStore(session),
        )
        ctx = TriggerContext(
            now=datetime.now(), session=session,
            elder_id=elder_id, runtime=rt, idle_sec=0.0,
        )

        before = session.query(Episode).filter(Episode.elder_id == elder_id).count()
        fired = trig.when(ctx)
        if fired:
            cut_index = trig._cut_index
            trig.action(ctx)
            after = session.query(Episode).filter(Episode.elder_id == elder_id).count()
            # 恰好一条 Episode
            assert after == before + 1
            # 窗口精确缩短 cut_index 轮，其余原样保留（不丢段、顺序不变）
            assert len(rt.window) == n - cut_index
            assert rt.window == window[cut_index:]
            assert rt.window_tokens == sum(len(t.content) for t in rt.window)
            episodes_expected += 1
            turns_cut_total += cut_index
        else:
            after = session.query(Episode).filter(Episode.elder_id == elder_id).count()
            assert after == before  # 未切分则不产生 Episode

    total = session.query(Episode).filter(Episode.elder_id == elder_id).count()
    assert total == episodes_expected
    assert episodes_expected > 0  # 随机序列确实触发过切分
    assert turns_cut_total > 0


# ══════════════════════════════════════════════════════════════════════
# P4 Foresight 单调失活
# ══════════════════════════════════════════════════════════════════════
def test_p4_foresight_monotonic_expiry(session):
    """随机前瞻集合 + 单调推进的「今天」：active 只减不增，过期不复活、不被召回。"""
    store = ForesightStore(session)
    rng = random.Random(77)

    for trial in range(20):
        elder_id = 1000 + trial  # 每轮独立 elder，互不干扰（FK 未强制）
        base = date(2024, 6, 1)
        # 造 8~15 条随机到期日的前瞻（相对 base 前后 10 天，含少量无期限）
        for _ in range(rng.randint(8, 15)):
            offset = rng.randint(-10, 10)
            if rng.random() < 0.15:
                due = ""  # 无期限
            else:
                due = (base + timedelta(days=offset)).isoformat()
            store.add(elder_id, ForesightDraft(content=f"事项{rng.random()}", kind="todo", due_date=due))
        session.flush()

        prev_active_ids: set[int] | None = None
        # 单调推进「今天」
        for day in range(-12, 12):
            today = base + timedelta(days=day)
            store.expire_due(elder_id, today)
            # 幂等：同一天二次调用不再处理
            assert store.expire_due(elder_id, today) == 0

            active_rows = (
                session.query(Foresight)
                .filter(Foresight.elder_id == elder_id, Foresight.active == True)  # noqa: E712
                .all()
            )
            active_ids = {f.id for f in active_rows}

            # active 集合单调非增（只失活、不复活）
            if prev_active_ids is not None:
                assert active_ids <= prev_active_ids
            prev_active_ids = active_ids

            # active_valid 里绝不含已过期项（due < today）
            for f in store.active_valid(elder_id, today):
                assert not (f.due_date and date.fromisoformat(f.due_date) < today)

            # 任一 active row 的 due 若已过期，说明失活不彻底 —— 不允许
            for f in active_rows:
                if f.due_date:
                    assert date.fromisoformat(f.due_date) >= today


# ══════════════════════════════════════════════════════════════════════
# P6 召回有界
# ══════════════════════════════════════════════════════════════════════
def test_p6_recall_bounded(session, monkeypatch):
    """随机、海量、可命中的多源记忆下，召回条数与注入长度严格有界。"""
    rng = random.Random(9090)

    for trial in range(12):
        elder_id = 2000 + trial
        max_items = rng.randint(1, 6)
        token_budget = rng.randint(20, 300)
        monkeypatch.setattr(config_module.config, "recall_max_items", max_items)
        monkeypatch.setattr(config_module.config, "recall_token_budget", token_budget)

        keyword = "遛弯"
        episodes = EpisodeStore(session)
        notes = NoteStore(session)
        foresights = ForesightStore(session)
        persons = PersonStore(session)
        now = datetime(2024, 5, 1, 9, 0, 0)

        n = rng.randint(20, 60)  # 远超上限的可命中记忆
        for i in range(n):
            episodes.add(
                elder_id, summary=f"{keyword}相关情景{i}那天聊了很久很久",
                mood="", topics=[keyword], started_at=now, ended_at=now,
            )
            notes.add_note(elder_id, ExtractedNote(True, f"{keyword}事实{i}", "", [], "other"))
            foresights.add(
                elder_id,
                ForesightDraft(content=f"{keyword}待办{i}", kind="todo",
                               due_date=(now.date() + timedelta(days=5)).isoformat()),
            )
            persons.upsert_mention(elder_id, name=f"{keyword}搭子{i}", relation="邻居",
                                   snippet=f"一起{keyword}", at=now)
            scene = MemScene(elder_id=elder_id, theme=keyword,
                             summary=f"经常一起{keyword}锻炼{i}", member_episode_ids=[], active=True)
            session.add(scene)
        session.flush()

        result = FtsRecaller(session).recall(elder_id, keyword, now)

        assert len(result.items) <= max_items, (
            f"条数超界：{len(result.items)} > {max_items}"
        )
        total_len = sum(len(it.text) for it in result.items)
        assert total_len <= token_budget, f"注入长度超界：{total_len} > {token_budget}"


# ══════════════════════════════════════════════════════════════════════
# P9 Trigger 幂等（含模拟重启）
# ══════════════════════════════════════════════════════════════════════
class _CountingTrigger:
    """总为真的计数触发器：cooldown_sec=0 → 同 dedup_key 永久去重。"""

    name = "counting"

    def __init__(self, key: str, counter: dict):
        self.cooldown_sec = 0
        self._key = key
        self._counter = counter

    def when(self, ctx):
        return True

    def action(self, ctx):
        self._counter[self._key] = self._counter.get(self._key, 0) + 1

    def dedup_key(self, ctx):
        return self._key


def test_p9_trigger_idempotent_across_restart(session):
    """随机地多次评估同一批 dedup_key（含重建 engine 模拟重启），每个 key 至多触发一次。"""
    brain = StubBrain()
    rng = random.Random(5150)

    for trial in range(15):
        elder_id = 3000 + trial
        counter: dict[str, int] = {}
        keys = [f"{elder_id}:key{k}" for k in range(rng.randint(1, 5))]

        # 反复评估：随机挑一个 key 构造 trigger，多轮触发；中途「重启」重建 engine。
        for _ in range(rng.randint(10, 30)):
            key = rng.choice(keys)
            trig = _CountingTrigger(key, counter)
            engine = TriggerEngine(session, brain, [trig])  # 每次重建 = 模拟重启
            ctx = TriggerContext(
                now=datetime.now(), session=session,
                elder_id=elder_id, runtime=None, idle_sec=0.0,
            )
            engine._evaluate(ctx)

        # 每个 key 至多触发一次（跨「重启」仍去重，依据 DB 里的 TriggerFire）
        for key in keys:
            assert counter.get(key, 0) <= 1, f"{key} 触发 {counter.get(key)} 次，应 <=1"
        # 至少有一个 key 真的触发过（否则说明去重把首次也拦了，属回归）
        assert sum(counter.values()) >= 1
