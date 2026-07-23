"""文本对话模拟器（离线演示入口）。

运行：
  python -m app.cli
特殊指令：
  /help              查看帮助
  /who               查看当前老人档案
  /time 08:00        模拟到点，触发该时间的用药提醒
  /silence           模拟老人沉默（紧急确认中会升级）
  /events            查看已记录的健康/紧急/通知
  /mem               查看当前记忆状态（running summary + 工作记忆窗口）
  /quit              把残留窗口折叠入记忆并退出
"""
from __future__ import annotations

from app.agent.brain import Brain
from app.agent.llm import get_llm
from app.channels import ConsoleChannel
from app.db import SessionLocal, init_db
from app.medication.service import MedicationService
from app.memory.events import EventStore
from app.memory.profile import ProfileRepository
from app.models import Medication, Notification
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


def _show_memory(brain, elder_id: int) -> None:
    """展示当前工作记忆状态：running summary + 窗口，便于演示折叠。"""
    rt = brain._runtimes.get(elder_id)
    if rt is None:
        print("（运行态尚未创建）")
        return
    print("\n—— running summary（远期模糊记忆）——")
    print(f"  {rt.running_summary or '（暂无，还没触发折叠）'}")
    print(f"\n—— 工作记忆窗口（近期原文）: {len(rt.window)} 轮 / ~{rt.window_tokens} tokens ——")
    for t in rt.window:
        who = "老人" if t.role == "user" else "助手"
        print(f"  [{t.at:%H:%M}] {who}：{t.content}")
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


def main() -> None:
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
            _show_memory(brain, elder_id)
            continue

        result = brain.process_turn(elder_id, text)
        session.commit()
        BOT.speak(elder_id, result.reply)
        if result.actions:
            print(f"   [系统动作] {'; '.join(result.actions)}")

    session.close()


if __name__ == "__main__":
    main()
