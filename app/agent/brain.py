"""对话大脑：整合紧急检测 / 用药 / 记忆，供 CLI 与语音层共用同一入口。

处理顺序（安全优先）：
  1. 若正在等待紧急确认 -> 先解读确认回复
  2. 紧急检测（规则）
  3. 用药意图（确认/推迟/查询）
  4. 健康数据抽取与记录
  5. 其余走 LLM 日常对话
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy.orm import Session

from app.agent.llm import LLMClient
from app.agent.nlu import Intent, build_nlu_prompt, parse_nlu_response
from app.agent.prompts import SUMMARY_PROMPT, build_system_prompt
from app.agent.profile_slots import SLOT_REGISTRY, SlotApplyContext, resolve_field
from app.agent.runtime import ElderRuntime, PendingSlot, Turn
from app.agent.timectx import build_time_context
from app.appointment.service import AppointmentService
from app.config import config
from app.emergency import detector
from app.emergency.detector import Level
from app.emergency.service import EmergencyService
from app.medication.service import MedicationService
from app.agent.reflection import ReflectionEngine
from app.agent.triggers import TriggerEngine, build_default_triggers
from app.memory.conflict import MemoryConflictResolver
from app.memory.conversation import ConversationMemory
from app.memory.events import EventStore, parse_health
from app.memory.folder import SummaryFolder
from app.memory.notes import NoteStore, ExtractedNote, date_status_label
from app.memory.profile import ProfileRepository
from app.memory.recall import FtsRecaller
from app.memory.recent import RecentTurnLog
from app.memory.running import RunningSummaryRepo
from app.memory.working import WorkingMemory, estimate_tokens
from app.models import Elder
from app.notify.notifier import Notifier
_HEALTH_KINDS_CN = {
    "blood_pressure": "血压",
    "blood_sugar": "血糖",
    "heart_rate": "心率",
    "temperature": "体温",
    "weight": "体重",
}

# 档案复述确认的 yes/no 解读（与紧急确认无关，独立于 distress 词表）
_YES_WORDS = ["对", "是", "嗯", "好的", "没错", "正确", "就是", "行", "可以", "确认"]
_NO_WORDS = ["不对", "不是", "错了", "不", "重来", "改一下", "换", "不行"]


def _interpret_yes_no(text: str) -> str:
    """解读复述确认回复：'yes' / 'no' / 'unclear'。否定词优先，避免'不对'被'对'误判。"""
    t = (text or "").strip()
    if not t:
        return "unclear"
    if any(w in t for w in _NO_WORDS):
        return "no"
    if any(w in t for w in _YES_WORDS):
        return "yes"
    return "unclear"


@dataclass
class TurnResult:
    reply: str
    kind: str = "chat"  # chat / emergency / medication / health / confirm
    level: str | None = None
    actions: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    # 流式回复迭代器（仅 _handle_chat / _comfort 填充，其余为 None）。
    # CLI 消费后 reply 已是完整文本；语音层直接接此迭代器做 TTS。
    stream: object = None  # Iterator[str] | None





class Brain:
    def __init__(self, session: Session, llm: LLMClient, notifier: Notifier) -> None:
        self._session = session
        self._llm = llm
        self._notifier = notifier
        self._profiles = ProfileRepository(session)
        self._events = EventStore(session)
        self._notes = NoteStore(session)
        self._conv = ConversationMemory(session)
        self._emergency = EmergencyService(session, notifier)
        self._medication = MedicationService(session, notifier)
        self._appointments = AppointmentService(session)
        self._running_repo = RunningSummaryRepo(session)
        self._folder = SummaryFolder(llm)
        self._wm = WorkingMemory(self._folder, self._running_repo)
        self._conflict = MemoryConflictResolver(session)
        self._reflection = ReflectionEngine(llm, self._conflict)
        self._runtimes: dict[int, ElderRuntime] = {}  # elder_id -> ElderRuntime
        # ---- Phase 3：近期原文落盘 / 检索式回忆 / 统一条件触发 ----
        self._recent = RecentTurnLog(session)
        self._recaller = FtsRecaller(session)
        # 循环依赖：TriggerEngine 需要 brain（读 _runtimes / iter_runtimes），
        # brain 又持有 engine。此处 brain 其它字段已就绪，把 self 传入即可。
        self._triggers = TriggerEngine(
            session, self, build_default_triggers(session, llm, brain=self)
        )

    # ---- 惰性运行态获取 ----
    def get_runtime(self, elder_id: int) -> ElderRuntime:
        if elder_id in self._runtimes:
            return self._runtimes[elder_id]
        elder = self._profiles.get(elder_id)
        if elder is None:
            raise ValueError(f"未找到老人 id={elder_id}")
        today = date.today()
        notes_text = self._notes.build_context(elder_id, today)
        appt_text = self._appointments_context(elder_id, today)
        running_summary = self._running_repo.build_context(elder_id)
        combined = "\n".join(t for t in [notes_text, appt_text] if t)
        # running summary 不再烘焙进 system_prompt（它会随折叠持续增长），
        # 改为每轮在 _handle_chat 动态注入最新值，避免 system_prompt 里的摘要过期。
        system_prompt = build_system_prompt(
            self._profiles.build_context(elder), "", combined
        )
        runtime = ElderRuntime(
            elder_id=elder_id,
            system_prompt=system_prompt,
            running_summary=running_summary,
        )
        # 重启恢复近期窗口（需求 8.2）：从 RecentTurn 落盘按时间恢复最近若干轮，
        # 并用 estimate_tokens 累加窗口 token。best-effort——恢复失败绝不阻断对话。
        try:
            recovered = self._recent.load_recent(elder_id, limit=config.fold_batch * 4)
            if recovered:
                runtime.window.extend(recovered)
                runtime.window_tokens += sum(
                    estimate_tokens(t.content) for t in recovered
                )
                runtime.last_activity_at = recovered[-1].at
        except Exception:
            pass
        self._runtimes[elder_id] = runtime
        return runtime

    def iter_runtimes(self):
        return list(self._runtimes.values())

    # ---- 低频被动脉冲：转发到 TriggerEngine（宿主主循环 / CLI /tick / 测试注入 now）----
    def on_pulse(self, now: datetime) -> None:
        """低频脉冲驱动记忆类触发（巩固/前瞻过期/清理/主动关心）。

        失败隔离：引擎内部已隔离单个 trigger 异常，这里再兜一层，
        保证脉冲永不外抛影响宿主循环。
        """
        try:
            self._triggers.on_pulse(now)
        except Exception:
            pass

    def _elder(self, elder_id: int) -> Elder:
        return self._profiles.get(elder_id)

    # ---- 主入口 ----
    def process_turn(self, elder_id: int, user_text: str, on_token=None) -> TurnResult:
        """处理一轮对话。

        on_token: Callable[[str], None] | None
            流式回调，仅对 chat（闲聊）路径有效；每收到一个 LLM token 即调用。
            CLI 传入打印函数以实现逐字显示；None 则静默收集（默认，兼容旧调用）。
        """
        runtime = self.get_runtime(elder_id)
        elder = self._elder(elder_id)

        with runtime.lock:
            now = datetime.now()
            last_at = runtime.window[-1].at if runtime.window else None
            tctx = build_time_context(now, last_at)

            # Episode 边界折叠（质量优化）：久别重逢时先把上一段整体折叠，
            # 让摘要贴合话题边界，再开始新一段。
            if (
                runtime.window
                and tctx.gap_since_last is not None
                and tctx.gap_since_last >= config.episode_gap_sec
            ):
                self._wm.fold_all(runtime, self._session)

            # 追加用户轮次
            user_turn = Turn("user", user_text, now)
            self._wm.append(runtime, user_turn)
            self._recent.append(elder_id, user_turn)  # 落盘 best-effort（需求 8.3）
            runtime.last_activity_at = now

            # ---- 安全语义不变（顺序严格保持） ----
            if runtime.pending_emergency_id is not None:
                result = self._handle_emergency_confirmation(runtime, elder, user_text)
            else:
                decision = detector.detect(user_text)
                if decision.is_incident:
                    result = self._handle_emergency(runtime, elder, user_text, decision)
                else:
                    # 设置会话超时：到时先回 daily（最小权限），再处理本轮（安全语义之后）
                    self._expire_setup_if_idle(runtime, now)
                    if runtime.pending_profile_slot is not None:
                        # 复述确认挂起中：本轮解读确认/否认/修正（仿紧急确认状态机）
                        runtime.setup_last_at = now
                        result = self._handle_profile_confirmation(runtime, elder, user_text)
                    else:
                        intents = self._run_nlu(elder, user_text)
                        result = self._route(runtime, elder, user_text, intents, on_token=on_token)

            # 追加 assistant 轮次
            assistant_turn = Turn("assistant", result.reply, datetime.now())
            self._wm.append(runtime, assistant_turn)
            self._recent.append(elder_id, assistant_turn)  # 落盘 best-effort（需求 8.3）

            # 超预算则增量折叠（唯一硬触发）
            self._wm.fold_if_over_budget(runtime, self._session)

            # 重要性反思：累积分数，超阈值时触发一次洞察
            self._reflection.observe(runtime, user_text, result.kind, self._session)

        # reply 已产出、锁已释放后再跑记忆类触发（语义边界/抽取/巩固/清理等），
        # 避免持锁做重活拖慢对话。引擎内部已做失败隔离，这里再兜一层，
        # 保证触发器异常绝不影响 process_turn 的返回（需求 12 / P10）。
        try:
            self._triggers.on_turn(elder_id)
        except Exception:
            pass

        return result

    # ---- NLU + 路由 ----
    def _run_nlu(self, elder: Elder, text: str) -> list[Intent]:
        has_pending = bool(self._medication.pending_today(elder.id))
        prompt = build_nlu_prompt(text, date.today(), has_pending)
        raw = self._llm.chat([{"role": "user", "content": prompt}])
        return parse_nlu_response(raw)

    def _route(
        self, runtime: ElderRuntime, elder: Elder, text: str, intents: list[Intent],
        on_token=None,
    ) -> TurnResult:
        """把意图路由到确定性处理函数。工具即工程内的普通方法。"""
        dispatch = {
            "confirm_medication": self._do_confirm_medication,
            "defer_medication": self._do_defer_medication,
            "deny_medication": self._do_deny_medication,
            "record_health": self._do_record_health,
            "record_appointment": self._do_record_appointment,
            "remember_note": self._do_remember_note,
            "query_medication": self._do_query_medication,
            "query_health": self._do_query_health,
            "query_appointment": self._do_query_appointment,
            "query_emergency": self._do_query_emergency,
            # 模式控制在两个模式下都可用（进入需口令校验）
            "enter_setup": self._do_enter_setup,
            "exit_setup": self._do_exit_setup,
        }
        # 能力过滤：写档案意图仅在设置模式开放（需求 1.2/1.3）
        if runtime.mode == "setup":
            dispatch["update_profile"] = self._do_update_profile

        # 纯闲聊（或未识别）→ 走 LLM 对话
        structured = [i for i in intents if i.name in dispatch]
        if not structured:
            # 日常模式下想改档案：温和提示需先进设置模式，绝不静默写入
            if runtime.mode != "setup" and any(i.name == "update_profile" for i in intents):
                return TurnResult(
                    reply="要修改资料得先进入设置模式哦。您可以说“进入设置模式”，再把口令告诉我。",
                    kind="mode", data={"mode": runtime.mode},
                )
            return self._handle_chat(runtime, on_token=on_token)

        # 执行所有结构化意图（写操作 + 查询），收集片段
        fragments: list[TurnResult] = []
        for intent in structured:
            handler = dispatch[intent.name]
            fragments.append(handler(runtime, elder, text, intent.slots))

        if len(fragments) == 1:
            return fragments[0]

        # 多意图：合并回复与数据
        reply = " ".join(f.reply for f in fragments if f.reply)
        merged_data = {"multi": [{"kind": f.kind, "data": f.data} for f in fragments]}
        actions = [a for f in fragments for a in f.actions]
        return TurnResult(reply=reply, kind="mixed", actions=actions, data=merged_data)

    # ---- 紧急 ----
    # 安全动作（通知/升级）永远先于话术执行，且不依赖 LLM；
    # 话术由 LLM 结合对话上下文生成，失败则回退到确定性模板。
    def _handle_emergency(
        self, runtime: ElderRuntime, elder: Elder, text: str, decision
    ) -> TurnResult:
        event = self._emergency.open_event(elder, decision.level, text)
        if decision.level == Level.EMERGENCY:
            reply = self._comfort(
                runtime, elder, "emergency", text,
                fallback=f"{elder.name}别怕，我已经通知您的家人了，您先坐稳，我一直陪着您。",
            )
            return TurnResult(
                reply=reply, kind="emergency", level=event.level,
                actions=list(event.actions or []), data={"event_id": event.id},
            )
        if decision.level == Level.ALERT:
            runtime.pending_emergency_id = event.id
            reply = self._comfort(
                runtime, elder, "alert", text,
                fallback=f"{elder.name}，您现在是不是哪里不舒服？要不要我联系家里人？",
            )
            return TurnResult(
                reply=reply, kind="emergency", level=event.level,
                actions=list(event.actions or []), data={"event_id": event.id},
            )
        # OBSERVE
        reply = self._comfort(
            runtime, elder, "observe", text,
            fallback="嗯，我记下了，您多注意身体。要是不舒服随时跟我说。",
        )
        return TurnResult(
            reply=reply, kind="emergency", level=event.level,
            actions=list(event.actions or []), data={"event_id": event.id},
        )

    def _handle_emergency_confirmation(
        self, runtime: ElderRuntime, elder: Elder, text: str
    ) -> TurnResult:
        from app.models import EmergencyEvent

        event = self._session.get(EmergencyEvent, runtime.pending_emergency_id)
        verdict = detector.interpret_confirmation(text)
        if verdict == "cancel":
            self._emergency.cancel(event, text)
            runtime.pending_emergency_id = None
            reply = self._comfort(
                runtime, elder, "cancel", text,
                fallback="那就好，您没事我就放心了。要是有哪里不舒服，一定马上告诉我。",
            )
            return TurnResult(
                reply=reply, kind="confirm", level=event.level,
                data={"event_id": event.id, "resolved": True},
            )
        if verdict == "confirm":
            self._emergency.escalate(event, elder, f"老人确认不适：{text}")
            runtime.pending_emergency_id = None
            reply = self._comfort(
                runtime, elder, "escalate", text,
                fallback=f"{elder.name}您别急，我这就通知家里人，救护也在安排，我一直陪着您。",
            )
            return TurnResult(
                reply=reply, kind="emergency", level=event.level,
                data={"event_id": event.id, "escalated": True},
            )
        # unclear -> 再问一次
        reply = self._comfort(
            runtime, elder, "alert", text,
            fallback="您能再说一下现在是哪里不舒服吗？要是难受，我马上帮您联系家人。",
        )
        return TurnResult(
            reply=reply, kind="confirm", level=event.level,
            data={"event_id": event.id},
        )

    # ---- 紧急话术生成（带上下文，安全动作已先行完成）----
    def _comfort(
        self, runtime: ElderRuntime, elder: Elder, kind: str, trigger: str, fallback: str
    ) -> str:
        """结合对话上下文生成安抚话术。安全动作已在调用前完成，此处只影响措辞。

        LLM 失败或离线时回退到 fallback 模板，保证任何情况下都有得体回复。
        """
        situations = {
            "emergency": "系统已判定为紧急情况，并且已经立刻通知了所有家属、建议拨打120。",
            "alert": "系统检测到老人可能身体不适，已经告警家属。你需要温和确认老人现在是否不舒服。",
            "escalate": "老人刚确认了确实不舒服，系统已升级为紧急并通知了家属，救护正在安排。",
            "cancel": "老人刚确认自己没事，虚惊一场。",
            "observe": "老人提到了一点不适，但暂不紧急，先安抚并叮嘱留意。",
        }
        ask = "请在最后自然地问一句她现在是不是哪里不舒服、要不要联系家人。" if kind in ("alert",) else ""
        directive = (
            "【安抚回复】" + situations.get(kind, "")
            + f"老人刚说的是：「{trigger}」。"
            + "请结合前面的对话上下文，用一到两句话温柔、镇定地安抚老人，"
            + "让她有安全感；不要提供任何医疗诊断或用药建议，不要重复通知家属的系统措辞。"
            + ask
        )
        messages = [{"role": "system", "content": runtime.system_prompt}]
        messages.extend(t.to_llm_message() for t in runtime.window[-6:])
        messages.append({"role": "user", "content": directive})
        try:
            reply = "".join(self._llm.stream_chat(messages))
            return reply.strip() or fallback
        except Exception:
            return fallback

    def handle_silence(self, elder_id: int) -> TurnResult | None:
        """老人被问话后长时间无响应：静默即信号，升级紧急确认。"""
        runtime = self._runtimes.get(elder_id)
        if runtime is None or runtime.pending_emergency_id is None:
            return None
        from app.models import EmergencyEvent

        elder = self._elder(elder_id)
        event = self._session.get(EmergencyEvent, runtime.pending_emergency_id)
        self._emergency.escalate(event, elder, "老人被询问后长时间无响应")
        runtime.pending_emergency_id = None
        return TurnResult(
            reply="（无人应答，已按紧急情况通知家属）",
            kind="emergency",
            level=Level.EMERGENCY.value,
            data={"event_id": event.id, "escalated": True, "reason": "silence"},
        )

    # ══════════════════════════════════════════════
    # 意图处理函数（"工具"= 工程内的普通方法，确定性执行）
    # ══════════════════════════════════════════════

    # ---- 用药 ----
    def _do_confirm_medication(self, runtime, elder, text, slots) -> TurnResult:
        from app.models import Medication

        pending = self._medication.pending_today(elder.id)
        if pending:
            names = "、".join({
                self._session.get(Medication, p.medication_id).name for p in pending
            })
            for log in pending:
                self._medication.confirm_taken(log)
            return TurnResult(
                reply=f"好的，我给您记下了，您今天的{names}都吃过了，对吗？按时吃药身体棒。",
                kind="medication",
                data={"answer": "confirmed", "count": len(pending)},
            )
        # 无待服用：可能已吃过或还没到点
        taken = [m for m in elder.medications if self._medication.taken_today(elder.id, m.id)]
        if taken:
            return TurnResult(
                reply="嗯，您今天的药都吃过了，记得了，真好。",
                kind="medication",
                data={"answer": "already_taken"},
            )
        return TurnResult(
            reply="好的，我记下了。现在还没到吃药的点，到点我再提醒您。",
            kind="medication",
            data={"answer": "none"},
        )

    def _do_defer_medication(self, runtime, elder, text, slots) -> TurnResult:
        pending = self._medication.pending_today(elder.id)
        for log in pending:
            self._medication.defer(log)
        return TurnResult(
            reply="那您先忙，过一小会儿我再提醒您吃药，别忘了哦。",
            kind="medication",
            data={"answer": "deferred" if pending else "none"},
        )

    def _do_deny_medication(self, runtime, elder, text, slots) -> TurnResult:
        pending = self._medication.pending_today(elder.id)
        return TurnResult(
            reply="没关系，现在吃也来得及。您吃好了记得跟我说一声。",
            kind="medication",
            data={"answer": "not_taken" if pending else "none"},
        )

    def _do_query_medication(self, runtime, elder, text, slots) -> TurnResult:
        scope = slots.get("scope", "today")
        if scope == "schedule":
            from app.agent.tools import execute_tool
            result = execute_tool(
                "query_medication_schedule", "{}", self._session, elder.id
            )
            return TurnResult(reply=result, kind="medication", data={"answer": "schedule"})
        if scope == "history":
            from app.agent.tools import execute_tool
            days = int(slots.get("days", 3))
            result = execute_tool(
                "query_medication_history", f'{{"days": {days}}}',
                self._session, elder.id,
            )
            return TurnResult(reply=result, kind="medication", data={"answer": "history"})

        # 今天：防重复服用
        taken = [m for m in elder.medications if self._medication.taken_today(elder.id, m.id)]
        if taken:
            names = "、".join(m.name for m in taken)
            return TurnResult(
                reply=f"您今天已经吃过{names}了，不用再吃啦。",
                kind="medication", data={"answer": "already_taken"},
            )
        if self._medication.pending_today(elder.id):
            return TurnResult(
                reply="您现在还没吃药，要不趁着我提醒您，现在吃了吧？",
                kind="medication", data={"answer": "pending"},
            )
        return TurnResult(
            reply="现在还没到吃药的时间，到点我会提醒您的。",
            kind="medication", data={"answer": "none"},
        )

    # ---- 健康数据 ----
    def _do_record_health(self, runtime, elder, text, slots) -> TurnResult:
        kind = slots.get("kind", "")
        value = slots.get("value", "")
        unit = slots.get("unit", "")
        if not (kind and value):
            # 槽位缺失时用正则兜底
            parsed = parse_health(text)
            if parsed is None:
                return TurnResult(reply="嗯，我记着了。", kind="health", data={})
            kind, value, unit = parsed.kind, parsed.value, parsed.unit
        self._events.add_health_reading(elder.id, kind, value, unit, note=text)
        cn = _HEALTH_KINDS_CN.get(kind, kind)
        return TurnResult(
            reply=f"好的，{cn}{value}{unit}我帮您记下了。要是不舒服记得告诉家里人。",
            kind="health",
            data={"kind": kind, "value": value, "unit": unit},
        )

    # ---- 事实记忆 ----
    def _do_remember_note(self, runtime, elder, text, slots) -> TurnResult:
        today = date.today()
        content = slots.get("content") or text
        date_iso = slots.get("date_iso", "")
        persons = slots.get("persons", [])
        extracted = ExtractedNote(
            is_memory_request=True, content=content,
            date_iso=date_iso, persons=persons, kind="commitment" if date_iso else "other",
        )
        note = self._conflict.upsert_note(elder.id, extracted)
        label = date_status_label(date_iso, today)
        runtime.system_prompt += f"\n【新记住的事】\n- {label} {content}".rstrip()
        if date_iso:
            d = date.fromisoformat(date_iso)
            reply = f"好的，我记住了：{d.month}月{d.day}日{content}，到时候我会提醒您。"
        else:
            reply = f"好的，我记住了：{content}。"
        return TurnResult(
            reply=reply, kind="memory",
            data={"note_id": note.id, "content": content,
                  "remind_date": date_iso, "persons": persons},
        )

    # ---- 预约/日程 ----
    def _do_record_appointment(self, runtime, elder, text, slots) -> TurnResult:
        today = date.today()
        title = slots.get("title") or text
        date_iso = slots.get("date_iso", "")
        time = slots.get("time", "")
        persons = slots.get("persons", [])
        location = slots.get("location", "")
        appt = self._appointments.create(
            elder.id, title=title, date_iso=date_iso, time=time,
            persons=persons, location=location,
        )
        # 同时把预约作为事实记忆写入（走冲突解析）
        appt_note = ExtractedNote(
            is_memory_request=True, content=title,
            date_iso=date_iso, persons=persons, kind="commitment",
        )
        self._conflict.upsert_note(elder.id, appt_note)
        label = date_status_label(date_iso, today)
        runtime.system_prompt += f"\n【新记住的预约】\n- {label} {title}".rstrip()
        if date_iso:
            d = date.fromisoformat(date_iso)
            when = f"{d.month}月{d.day}日" + (f" {time}" if time else "")
            reply = f"好的，我记下了：{when}{title}。到时候我提醒您。"
        else:
            reply = f"好的，我记下了：{title}。"
        return TurnResult(
            reply=reply, kind="appointment",
            data={"appointment_id": appt.id, "title": title,
                  "date_iso": date_iso, "persons": persons},
        )

    def _do_query_appointment(self, runtime, elder, text, slots) -> TurnResult:
        appts = self._appointments.upcoming(elder.id, date.today())
        if not appts:
            return TurnResult(reply="您最近没有安排什么事，安心休息就好。",
                              kind="appointment", data={"count": 0})
        lines = []
        for a in appts:
            label = date_status_label(a.date_iso, date.today()) if a.date_iso else ""
            lines.append(f"{label} {a.title}".strip())
        reply = "您接下来的安排有：" + "；".join(lines) + "。"
        return TurnResult(reply=reply, kind="appointment", data={"count": len(appts)})

    # ---- 查询：健康 / 紧急 ----
    def _do_query_health(self, runtime, elder, text, slots) -> TurnResult:
        from app.agent.tools import execute_tool
        import json as _json
        args = {"kind": slots.get("kind", "all"), "limit": int(slots.get("limit", 5))}
        result = execute_tool("query_health_records", _json.dumps(args),
                              self._session, elder.id)
        return TurnResult(reply=result, kind="query", data={"query": "health"})

    def _do_query_emergency(self, runtime, elder, text, slots) -> TurnResult:
        from app.agent.tools import execute_tool
        result = execute_tool("query_emergency_history", "{}",
                              self._session, elder.id)
        return TurnResult(reply=result, kind="query", data={"query": "emergency"})

    # ══════════════════════════════════════════════
    # 设置模式：模式控制 + 槽位填充 + 复述确认（profile-setup-mode）
    # 安全语义之后运行；紧急待确认时不会进入（见 process_turn）。
    # ══════════════════════════════════════════════

    def _expire_setup_if_idle(self, runtime: ElderRuntime, now: datetime) -> None:
        """设置会话超时：闲置超阈值则回 daily，丢弃未确认槽位（P7 / 需求 3.2/3.3）。"""
        if runtime.mode != "setup" or runtime.setup_last_at is None:
            return
        if (now - runtime.setup_last_at).total_seconds() >= config.setup_session_timeout_sec:
            runtime.mode = "daily"
            runtime.pending_profile_slot = None
            runtime.setup_last_at = None

    def _do_enter_setup(self, runtime, elder, text, slots) -> TurnResult:
        """进入设置模式：口令校验（占位口令，来自 config）。不回显口令明文。"""
        passcode = str(slots.get("passcode", "")).strip()
        if not passcode or passcode != config.setup_passcode:
            return TurnResult(
                reply="进入设置需要正确的口令哦，您把口令告诉我就行。",
                kind="mode", data={"mode": runtime.mode, "entered": False},
            )
        runtime.mode = "setup"
        runtime.setup_last_at = datetime.now()
        return TurnResult(
            reply="已进入设置模式。您可以告诉我要修改的资料，比如住址、过敏史、紧急联系人或用药。说“退出设置”就回到平常聊天。",
            kind="mode", data={"mode": "setup", "entered": True},
        )

    def _do_exit_setup(self, runtime, elder, text, slots) -> TurnResult:
        """退出设置模式，回到日常陪护；丢弃未确认槽位。"""
        runtime.mode = "daily"
        runtime.pending_profile_slot = None
        runtime.setup_last_at = None
        return TurnResult(
            reply="好的，已退回日常陪护。您有什么想聊的都可以跟我说。",
            kind="mode", data={"mode": "daily"},
        )

    def _do_update_profile(self, runtime, elder, text, slots) -> TurnResult:
        """设置模式写档案：解析候选 → 校验 → 复述确认（不落库，需求 4.2）。"""
        runtime.setup_last_at = datetime.now()
        field = resolve_field(slots.get("field", ""))
        if field is None:
            return TurnResult(
                reply="您想设置哪一项呢？住址、过敏史、慢性病、紧急联系人，还是用药？",
                kind="profile",
            )
        spec = SLOT_REGISTRY[field]
        value = spec.parse(slots, text)
        if value is None:
            return TurnResult(reply=f"好的，我们来设置{spec.label}。您具体说说？", kind="profile")
        ok, err = spec.validate(value)
        if not ok:
            return TurnResult(reply=err, kind="profile", data={"field": field, "need_more": True})
        runtime.pending_profile_slot = PendingSlot(field, value, spec.display(value))
        return TurnResult(
            reply=f"您是说{spec.label}是「{spec.display(value)}」，对吗？对的话我就记下了。",
            kind="profile", data={"field": field, "pending": True},
        )

    def _handle_profile_confirmation(self, runtime, elder, text) -> TurnResult:
        """复述确认解读：确认→落库；否认→重采；修正值→以新值再确认（需求 4.3/4.4）。"""
        pend = runtime.pending_profile_slot
        spec = SLOT_REGISTRY[pend.field]
        t = (text or "").strip()

        # 退出设置优先于确认：挂起中说退出 → 丢弃未确认槽位并回 daily（需求 3.3 / P7）
        if any(k in t for k in ["退出设置", "设置好了", "设置完了", "完成设置", "结束设置", "不设置了", "退出"]):
            return self._do_exit_setup(runtime, elder, text, {})
        # 取消本条：丢弃当前槽位，但留在设置模式
        if any(k in t for k in ["取消", "算了", "不改了", "不记了"]):
            runtime.pending_profile_slot = None
            return TurnResult(
                reply=f"好的，那{spec.label}先不改了。还要设置别的吗？",
                kind="profile", data={"field": spec.field, "pending": False},
            )

        verdict = _interpret_yes_no(text)

        if verdict == "yes":
            saved = self._apply_slot(elder, spec, pend.value)
            display = pend.display
            runtime.pending_profile_slot = None
            if saved:
                return TurnResult(
                    reply=f"好的，{spec.label}我记好了：{display}。",
                    kind="profile", data={"field": spec.field, "saved": True},
                )
            return TurnResult(
                reply=f"哎呀，这条{spec.label}没能存上，麻烦您稍后再说一遍。",
                kind="profile", data={"field": spec.field, "saved": False},
            )

        if verdict == "no":
            runtime.pending_profile_slot = None
            return TurnResult(
                reply=f"好的，那{spec.label}我们重新来，您再说一遍。",
                kind="profile", data={"field": spec.field, "pending": False},
            )

        # 不确定：可能是在给修正值，尝试按同字段重新解析
        new_val = spec.parse({}, text)
        if new_val is not None:
            ok, _ = spec.validate(new_val)
            if ok:
                runtime.pending_profile_slot = PendingSlot(pend.field, new_val, spec.display(new_val))
                return TurnResult(
                    reply=f"那{spec.label}是「{spec.display(new_val)}」，对吗？",
                    kind="profile", data={"field": spec.field, "pending": True},
                )
        return TurnResult(
            reply=f"您是说{spec.label}是「{pend.display}」吗？对的话说“对”，不对就再说一遍。",
            kind="profile", data={"field": spec.field, "pending": True},
        )

    def _apply_slot(self, elder, spec, value) -> bool:
        """确定性 setter 落库；写成功（flush 无异常）才算合同成立（C1）。"""
        try:
            ctx = SlotApplyContext(session=self._session, medication=self._medication)
            spec.apply(ctx, elder, value)
            self._session.flush()
            return True
        except Exception:
            try:
                self._session.rollback()
            except Exception:
                pass
            return False

    # ---- 闲聊上下文组装（_handle_chat 使用）----
    def _build_chat_messages(self, runtime: ElderRuntime) -> list[dict]:
        """组装闲聊上下文：档案+召回+回顾+时间+窗口。"""
        tctx = build_time_context(
            datetime.now(),
            runtime.window[-2].at if len(runtime.window) >= 2 else None,
        )
        messages: list[dict] = [{"role": "system", "content": runtime.system_prompt}]
        try:
            query = runtime.window[-1].content if runtime.window else ""
            recall_result = self._recaller.recall(
                runtime.elder_id, query, datetime.now()
            )
            injection = recall_result.to_injection()
            if injection:
                messages.append({"role": "system", "content": injection})
        except Exception:
            pass
        if runtime.running_summary:
            messages.append({
                "role": "system",
                "content": (
                    "【最近对话回顾（较早内容的大致印象，具体细节以结构化记录为准，"
                    f"记不清的不要编造）】\n{runtime.running_summary}"
                ),
            })
        messages.append({"role": "system", "content": tctx.to_injection()})
        messages.extend(t.to_llm_message() for t in runtime.window)
        return messages

    # ---- 日常对话（LLM）----
    def _handle_chat(
        self,
        runtime: ElderRuntime,
        on_token=None,   # Callable[[str], None] | None，每个 token 到达时调用
    ) -> TurnResult:
        """纯闲聊：LLM 流式，边收 token 边调 on_token 回调（供 CLI 逐字打印）。

        reply 收集完整文本；on_token=None 时静默收集。
        """
        messages = self._build_chat_messages(runtime)
        buf: list[str] = []
        for tok in self._llm.stream_chat(messages):
            buf.append(tok)
            if on_token is not None:
                on_token(tok)
        reply = "".join(buf)
        return TurnResult(reply=reply, kind="chat")

    # ---- 预约上下文（供 get_runtime 注入）----
    def _appointments_context(self, elder_id: int, today: date) -> str:
        appts = self._appointments.upcoming(elder_id, today)
        if not appts:
            return ""
        lines = []
        for a in appts:
            label = date_status_label(a.date_iso, today) if a.date_iso else ""
            lines.append(f"- {label} {a.title}".rstrip())
        return "【预约/日程】\n" + "\n".join(lines)

    def _summarize(self, transcript: str) -> tuple[str, str, list[str]]:
        try:
            raw = self._llm.chat(
                [{"role": "user", "content": SUMMARY_PROMPT + transcript}]
            )
            data = json.loads(raw)
            return (
                data.get("summary", "")[:200],
                data.get("mood", ""),
                list(data.get("topics", []))[:5],
            )
        except Exception:
            # FakeLLM 或解析失败时的兜底
            lines = transcript.split("\n")
            first_user = next((l[3:] for l in lines if l.startswith("老人：")), "")
            return (f"闲聊了几句，{first_user[:20]}", "平静", [])
