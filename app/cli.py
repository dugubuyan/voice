"""文本对话模拟器（离线演示入口）。

运行：
  python -m app.cli
特殊指令：
  /help              查看帮助
  /who               查看当前老人档案
  /time 08:00        模拟到点，触发该时间的用药提醒
  /tick [HH:MM]      手动驱动一次低频记忆脉冲（巩固/前瞻过期/清理等条件触发）
                     无参数用当前时间；带 HH:MM 则注入「今天该时刻」，便于离线演示
  /silence           模拟老人沉默（紧急确认中会升级）
  /events            查看已记录的健康/紧急/通知
  /mem               查看当前记忆状态（running summary + 窗口 + 情景/场景/前瞻/画像/召回预览）
  /mode              查看当前运行模式（daily 日常 / setup 设置）——仅演示查看
  /quit              把残留窗口折叠入记忆并退出

设置模式演示：说"进入设置模式，口令0000"进入，再自然说出住址/过敏史等，
机器人会复述确认，您说"对"即保存；说"退出设置"回到日常陪护。
"""
from __future__ import annotations

import sys
from datetime import datetime

from sqlalchemy import select

from app.agent.brain import Brain
from app.agent.llm import get_llm
from app.channels import ConsoleChannel
from app.db import SessionLocal, init_db
from app.medication.service import MedicationService
from app.memory.episode import EpisodeStore
from app.memory.events import EventStore
from app.memory.foresight import ForesightStore
from app.memory.profile import ProfileRepository
from app.models import ElderProfile, Episode, Medication, MemScene, Notification
from app.notify.notifier import ConsoleNotifier
from seed.seed_data import seed

BOT = ConsoleChannel(name="小暖")


def _choose_elder(profiles: ProfileRepository) -> int:
    elders = profiles.list_all()
    print("\n请选择要陪护的老人：")
    for e in elders:
        cond = "、".join(e.conditions or [])
        print(f"  [{e.id}] {e.name}  {e.age}岁  {cond}")
    while True:
        raw = input("输入编号（回车默认第一位）：").strip()
        if not raw:
            return elders[0].id
        if raw.isdigit() and any(e.id == int(raw) for e in elders):
            return int(raw)
        print("编号无效，请重试。")


def _trigger_time(session, elder, hhmm: str) -> None:
    med_svc = MedicationService(session, ConsoleNotifier(session))
    meds = (
        session.query(Medication)
        .filter(Medication.elder_id == elder.id)
        .all()
    )
    fired = 0
    for med in meds:
        if hhmm in (med.times or []):
            result = med_svc.trigger_reminder(elder, med.id, hhmm)
            BOT.speak(elder.id, result.text)
            fired += 1
    session.commit()
    if fired == 0:
        print(f"（{hhmm} 没有安排用药）")


def _tick(brain, hhmm: str | None) -> None:
    """手动驱动一次记忆脉冲：brain.on_pulse(now)。

    - 无参数：用当前真实时间 datetime.now()。
    - 带 HH:MM：构造「今天该时刻」的 datetime 注入 now，便于离线演示
      （例如把某老人 last_activity 设远后，用一个较晚的 now 让空闲类触发器命中）。
      解析失败则回退到 now，并给出提示。
    """
    if hhmm:
        try:
            hour, minute = (int(x) for x in hhmm.split(":", 1))
            base = datetime.now()
            now = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except (ValueError, TypeError):
            print(f"（时间格式无效：{hhmm}，已改用当前时间）")
            now = datetime.now()
    else:
        now = datetime.now()
    brain.on_pulse(now)
    print(f"（已触发一次记忆脉冲 @ {now:%H:%M}：记忆巩固 / 前瞻过期 / 清理等条件检查）")


def _mem_stats(session, elder_id: int, now: datetime) -> None:
    """展示 Phase 3 结构化记忆概览：情景 / 场景 / 前瞻 / 画像。"""
    # Episode：全部数量 + 未巩固数量 + 最近 1–2 条摘要
    episodes = EpisodeStore(session)
    all_eps = list(
        session.execute(
            select(Episode)
            .where(Episode.elder_id == elder_id)
            .order_by(Episode.ended_at.desc())
        ).scalars().all()
    )
    unconsolidated = episodes.unconsolidated(elder_id)
    print(
        f"\n—— 情景 Episode：共 {len(all_eps)} 条，未巩固 {len(unconsolidated)} 条 ——"
    )
    if all_eps:
        for ep in all_eps[:2]:
            mark = "" if ep.consolidated else "（未巩固）"
            print(f"  [{ep.ended_at:%m-%d %H:%M}]{mark} {ep.summary or '（无摘要）'}")
    else:
        print("  （暂无情景，还没切分/抽取）")

    # active MemScene 的 theme 列表
    scenes = list(
        session.execute(
            select(MemScene).where(
                MemScene.elder_id == elder_id,
                MemScene.active == True,  # noqa: E712
            )
        ).scalars().all()
    )
    print("\n—— 主题场景 MemScene（active）——")
    if scenes:
        print("  " + "、".join(s.theme or "（无主题）" for s in scenes))
    else:
        print("  （暂无 active 场景，还没触发巩固）")

    # 有效（未过期）Foresight：content + due_date
    foresights = ForesightStore(session).active_valid(elder_id, now.date())
    print("\n—— 有效前瞻 Foresight（未过期）——")
    if foresights:
        for fs in foresights:
            due = fs.due_date or "无期限"
            print(f"  [{due}] {fs.content}")
    else:
        print("  （暂无有效前瞻）")

    # ElderProfile 画像（截断展示）
    profile = session.execute(
        select(ElderProfile).where(ElderProfile.elder_id == elder_id)
    ).scalars().first()
    print("\n—— 老人画像 ElderProfile ——")
    if profile and profile.content:
        content = profile.content.strip()
        preview = content[:200] + ("…" if len(content) > 200 else "")
        print(f"  {preview}")
    else:
        print("  （暂无画像，还没触发巩固重写）")


def _mem_recall_preview(brain, rt) -> None:
    """按最近一条用户话语跑一次 recall，展示召回条数与注入预览。"""
    last_user = next(
        (t for t in reversed(rt.window) if t.role == "user"), None
    )
    print("\n—— 最近一次按用户话语的召回预览 ——")
    if last_user is None:
        print("  （窗口内暂无用户话语）")
        return
    try:
        result = brain._recaller.recall(rt.elder_id, last_user.content, datetime.now())
    except Exception as exc:  # 检索失败绝不影响演示
        print(f"  （召回失败，已忽略：{exc}）")
        return
    print(f"  query：{last_user.content[:30]}")
    print(f"  召回 {len(result.items)} 条")
    injection = result.to_injection()
    if injection:
        preview = injection[:200] + ("…" if len(injection) > 200 else "")
        print(f"  注入预览：{preview}")
    else:
        print("  （无命中，正常闲聊、不编造）")


def _show_memory(brain, session, elder_id: int) -> None:
    """展示当前记忆状态：running summary + 窗口 + 情景/场景/前瞻/画像 + 召回预览。"""
    rt = brain._runtimes.get(elder_id)
    if rt is None:
        print("（运行态尚未创建）")
        return
    print("\n—— running summary（远期模糊记忆 / 即时 gist 兜底）——")
    print(f"  {rt.running_summary or '（暂无，还没触发折叠）'}")
    print(f"\n—— 工作记忆窗口（近期原文）: {len(rt.window)} 轮 / ~{rt.window_tokens} tokens ——")
    for t in rt.window:
        who = "老人" if t.role == "user" else "助手"
        print(f"  [{t.at:%H:%M}] {who}：{t.content}")

    # Phase 3 结构化记忆概览 + 召回预览
    _mem_stats(session, elder_id, datetime.now())
    _mem_recall_preview(brain, rt)
    print()


def _show_events(session, elder_id: int) -> None:
    events = EventStore(session)
    print("\n—— 健康记录 ——")
    for r in events.recent_health(elder_id):
        print(f"  {r.recorded_at:%m-%d %H:%M}  {r.kind} {r.value}{r.unit}")
    print("—— 紧急事件 ——")
    for e in events.recent_emergencies(elder_id):
        print(f"  {e.created_at:%m-%d %H:%M}  [{e.level}] {e.trigger_text}  状态:{e.status}")
    print("—— 家属通知 ——")
    notes = (
        session.query(Notification)
        .filter(Notification.elder_id == elder_id)
        .order_by(Notification.created_at.desc())
        .limit(10)
        .all()
    )
    for n in notes:
        print(f"  {n.created_at:%m-%d %H:%M}  -> {n.recipient} [{n.kind}] {n.content}")
    print()


def _force_utf8_io() -> None:
    """把标准流强制为 UTF-8 且对坏字节容错。

    终端/输入法偶尔送来非 UTF-8 字节（或 stdin 被开成非 UTF-8 编码）时，
    默认会让 input() 抛 UnicodeDecodeError 直接崩溃。这里用 errors="replace"
    保证单个坏字节最多变成占位符，而不会拖垮整个对话进程。
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # 某些被重定向的流不支持 reconfigure，忽略即可


def main() -> None:
    _force_utf8_io()
    init_db()
    seed()  # 确保有数据

    session = SessionLocal()
    profiles = ProfileRepository(session)
    elder_id = _choose_elder(profiles)
    elder = profiles.get(elder_id)

    brain = Brain(session, get_llm(), ConsoleNotifier(session))
    brain.get_runtime(elder_id)
    session.commit()

    print("\n" + "=" * 50)
    print(f"开始陪护【{elder.name}】。输入 /help 查看指令，/quit 结束。")
    print("=" * 50)
    BOT.speak(elder_id, f"{elder.name}，我在呢。今天感觉怎么样呀？")

    while True:
        try:
            text = input(f"\n👵 {elder.name}：").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/quit"
        except UnicodeDecodeError:
            # 输入含无法解码的字节：跳过这一行、提示重输，不崩溃、不退出。
            print("（这句没听清，像是编码有点乱，麻烦您再说一遍～）")
            continue

        if not text:
            continue

        if text in ("/quit", "/exit"):
            brain._wm.fold_all(brain._runtimes[elder_id], session)
            session.commit()
            print(f"\n📝 本次对话已折叠入记忆。")
            break
        if text == "/help":
            print(__doc__)
            continue
        if text == "/who":
            print("\n" + profiles.build_context(elder))
            continue
        if text.startswith("/time"):
            parts = text.split()
            _trigger_time(session, elder, parts[1] if len(parts) > 1 else "08:00")
            continue
        if text.startswith("/tick"):
            parts = text.split()
            _tick(brain, parts[1] if len(parts) > 1 else None)
            session.commit()  # 触发器可能落库（巩固/前瞻失活/清理），提交持久化
            continue
        if text == "/silence":
            result = brain.handle_silence(elder_id)
            session.commit()
            if result:
                BOT.speak(elder_id, result.reply)
            else:
                print("（当前没有待确认的紧急情况）")
            continue
        if text == "/events":
            _show_events(session, elder_id)
            continue
        if text == "/mem":
            _show_memory(brain, session, elder_id)
            continue
        if text == "/mode":
            rt = brain._runtimes.get(elder_id)
            mode = rt.mode if rt else "daily"
            label = "设置模式（可修改档案）" if mode == "setup" else "日常陪护"
            pending = rt.pending_profile_slot.field if (rt and rt.pending_profile_slot) else "无"
            print(f"\n当前模式：{mode}（{label}）；待确认字段：{pending}")
            continue

        import sys
        # 流式回调：首个 token 前先打印前缀，之后逐字输出
        _prefix_printed = [False]
        def _on_token(tok: str):
            if not _prefix_printed[0]:
                print(f"🤖 {BOT._name}：", end="", flush=True)
                _prefix_printed[0] = True
            print(tok, end="", flush=True)

        result = brain.process_turn(elder_id, text, on_token=_on_token)
        if _prefix_printed[0]:
            print()  # 换行
        elif result.stream is not None:
            BOT.stream_speak(elder_id, result.stream)
        else:
            BOT.speak(elder_id, result.reply)
        session.commit()
        if result.actions:
            print(f"   [系统动作] {'; '.join(result.actions)}")

    session.close()


if __name__ == "__main__":
    main()
