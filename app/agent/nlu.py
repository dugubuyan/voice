"""意图识别层（NLU）：一次 LLM 调用，把老人的话解析成结构化意图 + 槽位。

设计：LLM 只负责"听懂人话"，不负责编排。返回结构化意图后，
由 Brain 的确定性代码路由到各处理函数。工具即工程里的普通函数，不暴露给 LLM。

支持多意图（一句话里既报血压又说吃了药）。
紧急检测不在此层，由规则前置处理。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date


# ──────────────────────────────────────────────
# 意图定义
# ──────────────────────────────────────────────

INTENT_SPECS = {
    "confirm_medication": "老人表示已经吃了药（如'药吃了''两种都喝了''按时吃的'）",
    "defer_medication": "老人表示待会儿再吃药（如'等会儿吃''这就去吃'）",
    "deny_medication": "老人表示还没吃药/忘了吃（如'还没吃''忘了'）",
    "record_health": "老人报告了健康数值，槽位 kind(blood_pressure/blood_sugar/heart_rate/temperature/weight)、value、unit",
    "record_appointment": "老人提到医疗相关的预约/安排（复查/体检/门诊/看病），槽位 title、date_iso、time、persons、location",
    "remember_note": "老人明确要求记住某件事（非医疗预约），如'记一下''别忘了提醒我'，槽位 content、date_iso、persons",
    "query_medication": "老人询问自己的服药情况，槽位 scope(today（是否已服）/ history（服药记录）/ schedule（查询计划服药时间表））、days；schedule 触发例句：我几点吃药 / 我一般几点吃药 / 什么时候该吃药 / 我的吃药时间",
    "query_health": "老人询问自己的健康数据/趋势，槽位 kind、days",
    "query_appointment": "老人询问自己的预约/日程安排",
    "query_emergency": "老人或家属询问最近的紧急/不适事件记录",
    "enter_setup": "请求进入设置模式来修改档案，槽位 passcode（口令）；例句：进入设置模式 口令1234 / 我要设置资料 密码1234",
    "exit_setup": "结束/退出设置模式，回到日常陪护；例句：退出设置 / 设置好了 / 完成设置",
    "update_profile": "在设置模式里为老人档案的某个字段赋值，槽位 field(address住址/allergies过敏史/conditions慢性病/contacts紧急联系人/medication用药) + value 及相应子槽位(name/relation/phone/dose/times/timing)；例句：家住浦东新区滨江大道4600号 / 他对青霉素过敏 / 紧急联系人女儿李梅电话13800000001",
    "chitchat": "以上都不是，普通闲聊/情感陪伴",
}


@dataclass
class Intent:
    name: str
    slots: dict = field(default_factory=dict)


_SCHEMA = """{
  "intents": [
    {"name": "意图名", "slots": {"槽位名": "值"}}
  ]
}"""


def build_nlu_prompt(text: str, today: date, has_pending_med: bool) -> str:
    intent_lines = "\n".join(f"  - {name}: {desc}" for name, desc in INTENT_SPECS.items())
    pending_hint = "（提示：老人今天有待服用的药）" if has_pending_med else ""
    return (
        f"【意图识别】今天是 {today.isoformat()}（{today.strftime('%Y年%m月%d日')}）。{pending_hint}\n"
        f"你是老人陪护系统的意图识别模块。判断老人这句话包含哪些意图（可多个），并抽取槽位。\n"
        f"可选意图：\n{intent_lines}\n\n"
        f"规则：\n"
        f"- 相对日期（今天/明天/后天/大后天/下周三等）请换算成绝对日期 date_iso（YYYY-MM-DD）。\n"
        f"- 一句话可能有多个意图（如同时报血压又说吃了药），都要列出。\n"
        f"- 询问句（含'吗''吧''是不是'且是在问状态）归为 query_*，不要归为陈述。\n"
        f"- 拿不准就归为 chitchat。\n"
        f"严格输出 JSON，不要任何多余文字：\n{_SCHEMA}\n\n"
        f"老人说：{text}"
    )


def parse_nlu_response(raw: str) -> list[Intent]:
    """解析 LLM 返回，失败兜底为 [chitchat]。"""
    try:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.startswith("```")
            ).strip()
        data = json.loads(text)
        intents = []
        for item in data.get("intents", []):
            name = item.get("name", "")
            if name in INTENT_SPECS:
                intents.append(Intent(name=name, slots=dict(item.get("slots", {}))))
        return intents or [Intent("chitchat", {})]
    except Exception:
        return [Intent("chitchat", {})]
