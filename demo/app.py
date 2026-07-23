"""老人看护对话机器人 — 极简场景测试界面（Gradio）。

给测试者用：选老人 → 说话或随机来一句 → 看机器人回应 + 系统记住/检测到了什么。
完全独立于 app/ 核心代码，删掉 demo/ 对生产零影响。

运行：python demo/app.py          （本地 http://localhost:7860）
云端：python demo/app.py --server_name 0.0.0.0 --server_port 7860
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from sqlalchemy import select

from app.agent.brain import Brain
from app.agent.llm import get_llm
from app.db import SessionLocal, init_db
from app.memory.events import EventStore
from app.memory.profile import ProfileRepository
from app.models import MemoryNote, Notification
from app.notify.notifier import ConsoleNotifier
from seed.seed_data import seed
from demo.scenarios import random_scenario

# ── 初始化（只做一次）──
init_db()
seed()
_session = SessionLocal()
_brain = Brain(_session, get_llm(), ConsoleNotifier(_session, echo=False))
_profiles = ProfileRepository(_session)
_elders = {e.name: e for e in _profiles.list_all()}
_elder_names = list(_elders.keys())


# ── 状态面板：把"系统记住/检测到了什么"汇总成一段易读文本 ──
def _status(elder_name: str) -> str:
    elder = _elders[elder_name]
    rt = _brain._runtimes.get(elder.id)
    parts = []

    summary = (rt.running_summary if rt else "") or "（还没有）"
    parts.append(f"🧠 长期记忆（聊过的大致印象）\n{summary}")

    # 记住的事 + 反思洞察
    notes = _session.execute(
        select(MemoryNote)
        .where(MemoryNote.elder_id == elder.id, MemoryNote.active == True)  # noqa: E712
        .order_by(MemoryNote.created_at.desc()).limit(8)
    ).scalars().all()
    if notes:
        lines = []
        for n in notes:
            icon = "🔮洞察" if n.source == "reflection" else "📌"
            d = f"[{n.remind_date}] " if n.remind_date else ""
            lines.append(f"  {icon} {d}{n.content}")
        parts.append("📝 记住的事 / 洞察\n" + "\n".join(lines))

    # 事件（健康 / 紧急 / 家属通知）
    store = EventStore(_session)
    ev = []
    for r in store.recent_health(elder.id, limit=3):
        ev.append(f"  🩺 {r.recorded_at:%m-%d %H:%M} {r.kind} {r.value}{r.unit}")
    for e in store.recent_emergencies(elder.id, limit=3):
        ev.append(f"  🚨 {e.created_at:%m-%d %H:%M} [{e.level}] {e.trigger_text[:16]} ({e.status})")
    notis = _session.execute(
        select(Notification).where(Notification.elder_id == elder.id)
        .order_by(Notification.created_at.desc()).limit(3)
    ).scalars().all()
    for n in notis:
        ev.append(f"  📞 通知{n.recipient}：{n.content[:28]}")
    if ev:
        parts.append("📋 健康 / 紧急 / 通知\n" + "\n".join(ev))

    return "\n\n".join(parts)


def _reply(elder_name: str, user_text: str, history: list):
    elder = _elders[elder_name]
    _brain.get_runtime(elder.id)
    result = _brain.process_turn(elder.id, user_text)
    _session.commit()
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": result.reply},
    ]
    return history, "", _status(elder_name)


def send_message(elder_name, user_input, history):
    if not user_input.strip():
        return history, "", _status(elder_name)
    return _reply(elder_name, user_input, history)


def send_random(elder_name, history):
    _, text = random_scenario()
    return _reply(elder_name, text, history)


def on_elder_change(elder_name):
    elder = _elders[elder_name]
    _brain.get_runtime(elder.id)
    greeting = f"{elder.name}，我在呢。今天感觉怎么样呀？"
    return [{"role": "assistant", "content": greeting}], "", _status(elder_name)


# ── 极简 UI ──
with gr.Blocks(title="老人看护机器人 · 场景测试") as demo:
    gr.Markdown("## 🤖 老人看护机器人 · 场景测试")
    gr.Markdown("选一位老人，直接说话，或点「🎲 随机来一句」模拟生活里随机发生的小事。右侧看系统记住和检测到了什么。")

    with gr.Row():
        with gr.Column(scale=3):
            elder = gr.Dropdown(_elder_names, value=_elder_names[0], label="👵 陪护对象")
            chatbot = gr.Chatbot(label="对话", height=460)
            with gr.Row():
                box = gr.Textbox(placeholder="输入老人说的话…", label="", scale=5, lines=1)
                send = gr.Button("发送", variant="primary", scale=1)
            dice = gr.Button("🎲 随机来一句（模拟生活小事）")

        with gr.Column(scale=2):
            status = gr.Textbox(label="📊 系统记住 / 检测到了什么", lines=22, interactive=False)

    outputs = [chatbot, box, status]
    elder.change(on_elder_change, [elder], outputs)
    send.click(send_message, [elder, box, chatbot], outputs)
    box.submit(send_message, [elder, box, chatbot], outputs)
    dice.click(send_random, [elder, chatbot], outputs)
    demo.load(on_elder_change, [elder], outputs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server_name", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    print(f"\n🤖 场景测试界面启动… http://{args.server_name}:{args.server_port}\n")
    demo.launch(server_name=args.server_name, server_port=args.server_port,
                share=args.share, theme=gr.themes.Soft())
