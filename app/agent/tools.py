"""LLM 可调用的工具：层2事件记忆的按需查询入口。

工具定义（schema）供 LLM 决策用，工具执行（executor）负责真正查 DB。
LLM 只决定"要不要查、查什么参数"，精确数字永远来自 DB，不靠 LLM 背诵。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Callable

from sqlalchemy.orm import Session

from app.memory.events import EventStore

# ──────────────────────────────────────────────
# 工具 schema（OpenAI function-calling 格式）
# ──────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_health_records",
            "description": (
                "查询老人最近的健康数据，包括血压、血糖、心率、体温、体重。"
                "当老人询问自己的健康指标或历史趋势时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "blood_pressure", "blood_sugar",
                            "heart_rate", "temperature", "weight", "all"
                        ],
                        "description": "要查询的指标类型，all 表示查所有",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回最近几条记录，默认 5",
                        "default": 5,
                    },
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_medication_history",
            "description": (
                "查询老人的服药记录，包括今天/最近是否按时服药、漏服情况。"
                "当老人询问自己的服药情况时调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "查最近几天的记录，默认 3",
                        "default": 3,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_emergency_history",
            "description": (
                "查询老人最近的紧急事件记录。"
                "当老人或家属询问最近有没有发生过不适/紧急情况时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回最近几条，默认 3",
                        "default": 3,
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# 工具执行层
# ──────────────────────────────────────────────

_KIND_CN = {
    "blood_pressure": "血压",
    "blood_sugar": "血糖",
    "heart_rate": "心率",
    "temperature": "体温",
    "weight": "体重",
}

_STATUS_CN = {
    "taken": "已服用",
    "pending": "待服用",
    "deferred": "已推迟",
    "missed": "漏服",
    "skipped": "跳过",
}

_LEVEL_CN = {
    "observe": "观察",
    "alert": "告警",
    "emergency": "紧急",
}


def _exec_query_health(session: Session, elder_id: int, args: dict) -> str:
    store = EventStore(session)
    kind = args.get("kind", "all")
    limit = int(args.get("limit", 5))
    records = store.recent_health(elder_id, limit=limit * 5 if kind == "all" else limit)
    if kind != "all":
        records = [r for r in records if r.kind == kind][:limit]
    if not records:
        return "暂无相关健康记录。"
    lines = []
    for r in records:
        cn = _KIND_CN.get(r.kind, r.kind)
        lines.append(f"{r.recorded_at.strftime('%m月%d日 %H:%M')}  {cn} {r.value}{r.unit}")
    return "\n".join(lines)


def _exec_query_medication(session: Session, elder_id: int, args: dict) -> str:
    from datetime import timedelta

    from sqlalchemy import select

    from app.models import Medication, MedicationLog

    days = int(args.get("days", 3))
    today = date.today()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days)]

    results = []
    for day in dates:
        logs = session.execute(
            select(MedicationLog)
            .where(MedicationLog.elder_id == elder_id, MedicationLog.scheduled_date == day)
            .order_by(MedicationLog.scheduled_time)
        ).scalars().all()
        for log in logs:
            med = session.get(Medication, log.medication_id)
            name = med.name if med else "未知药物"
            status = _STATUS_CN.get(log.status, log.status)
            results.append(f"{day} {log.scheduled_time}  {name}  {status}")

    if not results:
        return f"最近 {days} 天暂无服药记录。"
    return "\n".join(results)


def _exec_query_emergency(session: Session, elder_id: int, args: dict) -> str:
    store = EventStore(session)
    limit = int(args.get("limit", 3))
    events = store.recent_emergencies(elder_id, limit=limit)
    if not events:
        return "最近没有紧急事件记录。"
    lines = []
    for e in events:
        level = _LEVEL_CN.get(e.level, e.level)
        lines.append(
            f"{e.created_at.strftime('%m月%d日 %H:%M')}  [{level}] {e.trigger_text}  状态:{e.status}"
        )
    return "\n".join(lines)


# 工具名 -> 执行函数
_EXECUTORS: dict[str, Callable] = {
    "query_health_records": _exec_query_health,
    "query_medication_history": _exec_query_medication,
    "query_emergency_history": _exec_query_emergency,
}


def execute_tool(name: str, args_json: str, session: Session, elder_id: int) -> str:
    """执行工具调用，返回文字结果（塞回 messages 给 LLM 看）。"""
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError:
        args = {}
    executor = _EXECUTORS.get(name)
    if executor is None:
        return f"未知工具：{name}"
    try:
        return executor(session, elder_id, args)
    except Exception as e:
        return f"查询失败：{e}"
