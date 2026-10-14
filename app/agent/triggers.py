"""统一条件触发引擎（TriggerEngine）——引擎骨架。

把记忆类周期性工作（情景切分/巩固/前瞻过期/原文清理/主动关心）统一抽象为
`Trigger`：谓词 `when` + 动作 `action` + 去重键 `dedup_key` + 冷却 `cooldown_sec`。
两个时钟源：
  - `on_turn(elder_id)`：每轮对话 reply 产出后触发（idle_sec=0）。
  - `on_pulse(now)`：低频被动脉冲，遍历所有常驻 runtime，按各自空闲时长评估。

关键性质：
  - **跨重启幂等**（需求 14.4 / P9）：去重/冷却判定只依赖 DB 里的 `TriggerFire`，
    不依赖内存状态；进程重启后重建 engine 仍不会重复触发。
  - **失败隔离**（需求 14.6 / P10）：单个 trigger 的 `action` 抛异常被捕获记录，
    不影响同批其他 trigger，也不影响主对话流程。

本文件只做引擎骨架，不实现具体触发器（见任务 12）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Protocol, runtime_checkable

from app.agent.runtime import ElderRuntime
from app.config import config
from app.models import TriggerFire

log = logging.getLogger(__name__)


@dataclass
class TriggerContext:
    """一次触发评估的上下文。见 design.md Core Types / 函数 7。"""

    now: datetime
    session: object                       # DB session（SQLAlchemy Session）
    elder_id: Optional[int]               # on_turn 时有值；on_pulse 逐个 runtime 填入
    runtime: Optional[ElderRuntime]       # 对应 elder 的常驻运行态
    idle_sec: Optional[float]             # 距该 elder 最近活动的秒数（on_turn 记 0.0）


@runtime_checkable
class Trigger(Protocol):
    """条件触发器协议。具体触发器实现于任务 12。"""

    name: str
    cooldown_sec: int

    def when(self, ctx: TriggerContext) -> bool:
        """谓词：本次是否应触发。纯读，不应有副作用。"""
        ...

    def action(self, ctx: TriggerContext) -> None:
        """动作：执行实际工作（可写 DB）。异常由引擎隔离。"""
        ...

    def dedup_key(self, ctx: TriggerContext) -> str:
        """去重键：同 (name, dedup_key) 视为同一次触发意图。"""
        ...


class TriggerEngine:
    """遍历触发器目录，按谓词/去重/冷却评估并执行动作，失败隔离。"""

    def __init__(self, session, brain, triggers: list[Trigger]):
        # brain 只用鸭子类型：需要 `iter_runtimes()` 与 `_runtimes` dict。
        self._session = session
        self._brain = brain
        self._triggers = list(triggers)

    # ---- 时钟源 1：每轮对话 ----
    def on_turn(self, elder_id: int) -> None:
        """reply 产出后调用：对单个 elder 评估触发（idle_sec=0）。"""
        runtime = self._brain._runtimes.get(elder_id)
        ctx = TriggerContext(
            now=datetime.now(),
            session=self._session,
            elder_id=elder_id,
            runtime=runtime,
            idle_sec=0.0,
        )
        self._evaluate(ctx)

    # ---- 时钟源 2：低频脉冲 ----
    def on_pulse(self, now: datetime) -> None:
        """遍历所有常驻 runtime，按各自 idle_sec 评估触发。"""
        for rt in self._brain.iter_runtimes():
            idle = (now - rt.last_activity_at).total_seconds()
            ctx = TriggerContext(
                now=now,
                session=self._session,
                elder_id=rt.elder_id,
                runtime=rt,
                idle_sec=idle,
            )
            self._evaluate(ctx)

    # ---- 核心评估 ----
    def _evaluate(self, ctx: TriggerContext) -> None:
        for trig in self._triggers:
            try:
                if not trig.when(ctx):
                    continue
                key = trig.dedup_key(ctx)
                if self._already_fired(trig.name, key, ctx.now, trig.cooldown_sec):
                    continue
                trig.action(ctx)
                self._record_fire(trig.name, key, ctx.now, ctx.elder_id)
            except Exception:
                # 失败隔离（需求 14.6 / P10）：记录并吞掉，不影响其他 trigger 与主流程。
                log.exception("trigger %s failed", getattr(trig, "name", "<unknown>"))

    # ---- 去重 / 冷却（只依赖 DB，跨重启幂等，需求 14.4 / P9）----
    def _already_fired(
        self, name: str, key: str, now: datetime, cooldown_sec: int
    ) -> bool:
        """判断 (name, key) 是否已触发过（永久去重）或仍在冷却期内。"""
        last = (
            self._session.query(TriggerFire)
            .filter(
                TriggerFire.trigger_name == name,
                TriggerFire.dedup_key == key,
            )
            .order_by(TriggerFire.fired_at.desc())
            .first()
        )
        if last is None:
            return False
        if cooldown_sec <= 0:
            # cooldown_sec<=0：一旦触发过即永久去重。
            return True
        # 冷却期内跳过；超过冷却窗口可再次触发。
        elapsed = (now - last.fired_at).total_seconds()
        return elapsed < cooldown_sec

    def _record_fire(
        self, name: str, key: str, now: datetime, elder_id: Optional[int]
    ) -> None:
        """落库一条 TriggerFire，作为后续去重/冷却与跨重启幂等的依据。"""
        fire = TriggerFire(
            elder_id=elder_id,
            trigger_name=name,
            dedup_key=key,
            fired_at=now,
        )
        self._session.add(fire)
        self._session.commit()


# ══════════════════════════════════════════════════════════════════════
# 具体触发器目录（任务 12）
#
# 每个触发器实现 Trigger 协议（name / cooldown_sec / when / action / dedup_key），
# 依赖注入所需 store/detector/extractor/consolidator，便于离线确定性测试。
# ══════════════════════════════════════════════════════════════════════

from app.memory.boundary import SemanticBoundaryDetector
from app.memory.conflict import MemoryConflictResolver
from app.memory.consolidate import Consolidator
from app.memory.episode import EpisodeStore
from app.memory.extractor import MemCellExtractor
from app.memory.foresight import ForesightStore
from app.memory.person import PersonStore
from app.memory.recent import RecentTurnLog
from app.memory.working import estimate_tokens


class EpisodeCutTrigger:
    """情景切分：把窗口靠前、已聊完的一段切出 → 抽取 MemCell → 落库，并缩短窗口。

    切分条件（满足其一即触发，先判廉价条件再调 LLM 边界检测以降成本）：
      - 窗口轮数 >= config.episode_max_turns（廉价兜底，保证窗口有界）；
      - SemanticBoundaryDetector.detect(window).is_boundary（语义边界，LLM）。

    action：取 turns = window[:cut_index]，抽取为 MemCell，写入 Episode + 原子事实
    （走冲突解析）+ 有到期日的前瞻 + 人物提及；随后从窗口移除已切出的轮次并用
    estimate_tokens 重算 window_tokens（保持与 WorkingMemory 一致的算法，见函数 3）。

    cooldown_sec=0：靠 dedup_key（elder + 切分点 + 边界轮时间）保证同一次切分不重复；
    不同切分点因 key 不同可再次触发。action 抛错时引擎不落 TriggerFire，下次可重试。

    注意：RecentTurn 原文的 mark_consolidated 需要 turn 的 DB id，而窗口里的 Turn 是
    内存对象没有 DB id，本触发器不在此关联 recent_turns（留给任务 13 的 Brain 集成，
    届时有落盘 id）。这里聚焦「窗口 → Episode/事实/前瞻/人物 + 缩短窗口」。
    """

    name = "episode_cut"

    def __init__(
        self,
        detector: SemanticBoundaryDetector,
        extractor: MemCellExtractor,
        episodes: EpisodeStore,
        conflict: MemoryConflictResolver,
        foresights: ForesightStore,
        persons: PersonStore,
        *,
        cooldown_sec: int = 0,
    ) -> None:
        self._detector = detector
        self._extractor = extractor
        self._episodes = episodes
        self._conflict = conflict
        self._foresights = foresights
        self._persons = persons
        self.cooldown_sec = cooldown_sec
        # 在 when() 里算好切分点，供 dedup_key/action 复用（引擎对单个 ctx 串行
        # 走 when → dedup_key → action，故用实例暂存是确定且安全的）。
        self._cut_index = 0

    def when(self, ctx: TriggerContext) -> bool:
        rt = ctx.runtime
        self._cut_index = 0
        if rt is None or not rt.window:
            return False
        window = rt.window
        # 廉价条件优先：轮数上限兜底，切出整窗，保证窗口有界。
        if len(window) >= config.episode_max_turns:
            self._cut_index = len(window)
            return True
        # 再调 LLM 语义边界检测（失败时 detector 自身降级为非边界，不抛出）。
        decision = self._detector.detect(window)
        if decision.is_boundary and 1 <= decision.cut_index <= len(window):
            self._cut_index = decision.cut_index
            return True
        return False

    def dedup_key(self, ctx: TriggerContext) -> str:
        # 基于 elder + 切分点 + 边界轮的对话时间，使同一次切分不重复；
        # 不同切分（窗口已前移）边界轮时间不同，可再次触发。
        window = ctx.runtime.window if ctx.runtime else []
        idx = self._cut_index
        if 0 < idx <= len(window):
            at = window[idx - 1].at.isoformat()
        else:
            at = ctx.now.isoformat()
        return f"{ctx.elder_id}:{idx}:{at}"

    def action(self, ctx: TriggerContext) -> None:
        rt = ctx.runtime
        cut_index = self._cut_index
        if rt is None or cut_index <= 0:
            return
        turns = rt.window[:cut_index]
        if not turns:
            return

        cell = self._extractor.extract(ctx.elder_id, turns)
        ep = self._episodes.add(
            ctx.elder_id,
            summary=cell.episode_summary,
            mood=cell.mood,
            topics=cell.topics,
            started_at=turns[0].at,
            ended_at=turns[-1].at,
        )
        # 原子事实走冲突解析（recency-wins + 显式失效），带来源情景 id。
        for note in cell.atomic_facts:
            self._conflict.upsert_note(
                ctx.elder_id, note, source_episode_id=ep.id
            )
        # 只有带到期日的前瞻才作为 Foresight 落库（无期限的不写，见 ForesightDraft 约定）。
        for draft in cell.foresights:
            if draft.due_date:
                self._foresights.add(
                    ctx.elder_id, draft, source_episode_id=ep.id
                )
        # 人物提及归集（同名更新而非新建）。跳过无名条目。
        for person in cell.persons:
            name = str(person.get("name", "")).strip()
            if not name:
                continue
            self._persons.upsert_mention(
                ctx.elder_id,
                name=name,
                relation=str(person.get("relation", "")).strip(),
                snippet=str(person.get("snippet", "")).strip(),
                at=turns[-1].at,
            )

        # 从窗口移除已切出的轮次，并用 estimate_tokens 重算 window_tokens
        # （与 WorkingMemory._fold_batch 相同的重算方式，保持窗口 token 一致）。
        rt.window = rt.window[cut_index:]
        rt.window_tokens = sum(estimate_tokens(t.content) for t in rt.window)


class ConsolidateTrigger:
    """语义巩固：空闲够久或未巩固 Episode 累积到阈值时，归并 Episode → MemScene/画像/趋势。

    when：ctx.idle_sec >= config.consolidate_idle_sec，或未巩固 Episode 数
          >= config.consolidate_min_episodes。
    action：Consolidator.consolidate(elder_id)（自身幂等、失败降级，不抛错）。
    dedup_key：elder + 日期；配合 cooldown_sec（默认 = consolidate_idle_sec）避免频繁巩固。
    """

    name = "consolidate"

    def __init__(
        self,
        consolidator: Consolidator,
        episodes: EpisodeStore,
        *,
        cooldown_sec: Optional[int] = None,
    ) -> None:
        self._consolidator = consolidator
        self._episodes = episodes
        self.cooldown_sec = (
            cooldown_sec if cooldown_sec is not None else config.consolidate_idle_sec
        )

    def when(self, ctx: TriggerContext) -> bool:
        if ctx.elder_id is None:
            return False
        if ctx.idle_sec is not None and ctx.idle_sec >= config.consolidate_idle_sec:
            return True
        count = len(self._episodes.unconsolidated(ctx.elder_id))
        return count >= config.consolidate_min_episodes

    def action(self, ctx: TriggerContext) -> None:
        self._consolidator.consolidate(ctx.elder_id)

    def dedup_key(self, ctx: TriggerContext) -> str:
        return f"{ctx.elder_id}:{ctx.now.date().isoformat()}"


class ForesightExpireTrigger:
    """前瞻过期失活：每次评估都检查越过到期日的 active 前瞻并失活（留审计）。

    when：总为真（elder 有值时）。
    action：ForesightStore.expire_due(elder_id, now.date())（幂等、不改 due_date）。
    dedup_key：elder + 日期（每天一次即可）；cooldown_sec=0 靠日期 key 永久去重当天不重扫。
    """

    name = "foresight_expire"

    def __init__(self, foresights: ForesightStore, *, cooldown_sec: int = 0) -> None:
        self._foresights = foresights
        self.cooldown_sec = cooldown_sec

    def when(self, ctx: TriggerContext) -> bool:
        return ctx.elder_id is not None

    def action(self, ctx: TriggerContext) -> None:
        self._foresights.expire_due(ctx.elder_id, ctx.now.date())

    def dedup_key(self, ctx: TriggerContext) -> str:
        return f"{ctx.elder_id}:{ctx.now.date().isoformat()}"


class RecentTurnCleanupTrigger:
    """近期原文清理：清除已巩固且超期的 recent_turns（未巩固原文保留，重启才能恢复）。

    when：总为真（elder 有值时）。
    action：RecentTurnLog.purge_consolidated(elder_id, config.recent_keep_days)。
    dedup_key：elder + 日期（每天清理一次）；cooldown_sec=0 靠日期 key 去重。
    """

    name = "recent_cleanup"

    def __init__(self, recent: RecentTurnLog, *, cooldown_sec: int = 0) -> None:
        self._recent = recent
        self.cooldown_sec = cooldown_sec

    def when(self, ctx: TriggerContext) -> bool:
        return ctx.elder_id is not None

    def action(self, ctx: TriggerContext) -> None:
        self._recent.purge_consolidated(ctx.elder_id, config.recent_keep_days)

    def dedup_key(self, ctx: TriggerContext) -> str:
        return f"{ctx.elder_id}:{ctx.now.date().isoformat()}"


class ProactiveCheckinTrigger:
    """主动关心（占位）：空闲极久且功能启用时，记录一次「应主动关心」的信号。

    **占位说明**：主动开口的语气/频率需产品确认，此处仅占位——不真正对老人发话，
    只记录日志并（可选）调用注入的回调，供任务 13 或宿主接线决定后续行为。

    when：config.proactive_idle_sec > 0（0 表示关闭）且 ctx.idle_sec >= 该阈值。
    action：记录信号（log）+ 可选回调。
    dedup_key：elder + 日期；cooldown_sec 用 proactive_idle_sec，避免频繁打扰。
    """

    name = "proactive_checkin"

    def __init__(
        self,
        callback: Optional[Callable[[TriggerContext], None]] = None,
        *,
        cooldown_sec: Optional[int] = None,
    ) -> None:
        self._callback = callback
        self.cooldown_sec = (
            cooldown_sec if cooldown_sec is not None else config.proactive_idle_sec
        )

    def when(self, ctx: TriggerContext) -> bool:
        # 动态读 config：便于运行期/测试期开关（proactive_idle_sec=0 即关闭）。
        threshold = config.proactive_idle_sec
        if threshold <= 0:
            return False
        if ctx.idle_sec is None:
            return False
        return ctx.idle_sec >= threshold

    def action(self, ctx: TriggerContext) -> None:
        # 占位：仅记录「应主动关心」信号，不真正发话（语气/频率待产品确认）。
        log.info(
            "proactive checkin signal for elder %s (idle=%.0fs)",
            ctx.elder_id,
            ctx.idle_sec or 0.0,
        )
        if self._callback is not None:
            self._callback(ctx)

    def dedup_key(self, ctx: TriggerContext) -> str:
        return f"{ctx.elder_id}:{ctx.now.date().isoformat()}"


# ══════════════════════════════════════════════════════════════════════
# 工厂：装配默认触发器目录（依赖注入各 store，供任务 13 的 Brain 使用）
# ══════════════════════════════════════════════════════════════════════

def build_default_triggers(
    session,
    llm,
    brain=None,
    *,
    proactive_callback: Optional[Callable[[TriggerContext], None]] = None,
) -> list[Trigger]:
    """装配默认触发器目录。

    Args:
        session: DB session（各 store 共用）。
        llm: LLM 客户端（边界检测/抽取/巩固共用）。
        brain: 预留（当前触发器不直接依赖 brain；引擎另行持有 brain）。
        proactive_callback: 可选，主动关心信号回调（占位用）。

    Returns:
        Trigger 列表，顺序即评估顺序：切分 → 巩固 → 前瞻过期 → 原文清理 → 主动关心。
    """
    detector = SemanticBoundaryDetector(llm)
    extractor = MemCellExtractor(llm)
    episodes = EpisodeStore(session)
    foresights = ForesightStore(session)
    persons = PersonStore(session)
    conflict = MemoryConflictResolver(session)
    recent = RecentTurnLog(session)
    consolidator = Consolidator(llm, session)

    return [
        EpisodeCutTrigger(
            detector, extractor, episodes, conflict, foresights, persons
        ),
        ConsolidateTrigger(consolidator, episodes),
        ForesightExpireTrigger(foresights),
        RecentTurnCleanupTrigger(recent),
        ProactiveCheckinTrigger(callback=proactive_callback),
    ]
