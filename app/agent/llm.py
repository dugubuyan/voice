"""LLM 接口：OpenAI 兼容客户端 + 离线 Fake 实现。

- chat() : 普通文本对话（摘要、记忆提取、意图识别、回复生成等任务用）
"""
from __future__ import annotations

from typing import Protocol

from app.config import config


# ──────────────────────────────────────────────
# 协议
# ──────────────────────────────────────────────

class LLMClient(Protocol):
    def chat(self, messages: list[dict]) -> str: ...
    def stream_chat(self, messages: list[dict]):
        """流式生成，yield str token。非流式实现可直接 yield chat() 整体结果。"""
        ...


# ──────────────────────────────────────────────
# 离线 Fake（测试 / 无 key 时使用）
# ──────────────────────────────────────────────

class FakeLLMClient:
    """离线模板回复。不触发 tool calling，保证测试确定性。"""

    def chat(self, messages: list[dict]) -> str:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        # 紧急安抚话术（离线回退模板）
        if "【安抚回复】" in user:
            if "已经立刻通知了所有家属" in user:
                return "您别怕，我已经通知您的家人了，您先坐稳，我一直陪着您。"
            if "升级为紧急" in user:
                return "您别急，家里人我已经通知了，救护也在安排，我一直陪着您。"
            if "虚惊一场" in user or "确认自己没事" in user:
                return "那就好，您没事我就放心了。有哪里不舒服一定马上告诉我。"
            if "确认老人现在是否不舒服" in user:
                return "您现在是不是哪里不舒服？要不要我联系家里人？"
            return "我在您身边呢，您慢慢说，别着急。"

        # 增量折叠摘要 prompt
        if "【折叠摘要任务】" in user:
            return self._fake_fold(user)

        # 语义边界检测 prompt（默认判定为非边界，保证离线确定性）
        if "【边界检测】" in user:
            return '{"is_boundary": false, "cut_index": 0}'

        # 记忆抽取 prompt（离线返回结构合法 JSON，事实/前瞻/人物可为空）
        if "【记忆抽取】" in user:
            return self._fake_memcell(user)

        # 记忆巩固 prompt（离线返回结构合法 JSON：至少一个 scene + profile + trends）
        if "【记忆巩固】" in user:
            return self._fake_consolidate(user)

        # 反思任务（ReflectionEngine）
        if "【反思任务】" in user:
            return '{"insight": "近期有情绪或健康方面的关注点，建议家属留意", "kind": "other"}'

        # 意图识别 prompt 识别（build_nlu_prompt 特征）
        if "【意图识别】" in user and "intents" in user:
            return self._fake_nlu(user)

        # 摘要 prompt 识别
        if "summary" in user and "mood" in user and "topics" in user and "老人" in user:
            return '{"summary": "闲聊了几句", "mood": "平静", "topics": []}'

        if any(k in user for k in ["你好", "在吗", "早", "中午好", "晚上好"]):
            return "哎，我在呢。今天感觉怎么样呀？"
        if "谢谢" in user:
            return "不客气，能帮到您我就高兴。"
        if "?" in user or "？" in user:
            return "这个我记下了，回头帮您问问家里人，您先别急。"
        return "嗯，我听着呢，您慢慢说。"

    def stream_chat(self, messages: list[dict]):
        """FakeLLM 流式：整体结果一次 yield，保测试简单。"""
        yield self.chat(messages)

    def _fake_memcell(self, prompt: str) -> str:
        """离线记忆抽取存根：取本段对话首句作情景摘要，返回结构合法 JSON。

        事实/前瞻/人物离线一律为空，保证可解析且结构合法（需求 13）。
        """
        import json as _json
        import re
        m = re.search(r"【本段对话】\n(.+)$", prompt, re.DOTALL)
        lines = (m.group(1).strip() if m else "").splitlines()
        first_line = lines[0][:60] if lines else ""
        summary = (first_line or "闲聊了几句").strip()
        return _json.dumps(
            {
                "episode_summary": summary,
                "mood": "平静",
                "topics": [],
                "atomic_facts": [],
                "foresights": [],
                "persons": [],
            },
            ensure_ascii=False,
        )

    def _fake_consolidate(self, prompt: str) -> str:
        """离线记忆巩固存根：把全部待巩固情景归并到一个固定主题场景，
        重写一份简短画像，并给出不含编造数值的趋势，保证离线可解析（需求 13）。

        - scene：固定主题（便于测试「同 theme 唯一 active」），member_indices 覆盖
          prompt 中出现的全部情景下标 [i]。
        - profile：固定短文本（整体重写，长度有界）。
        - trends：mood 趋势恒有；仅当 prompt 含结构化健康读数时再给 health 趋势，
          且描述不含任何编造数值（数值溯源交由结构化源）。
        """
        import json as _json
        import re

        indices = [int(m) for m in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)]
        trends = [{"kind": "mood", "content": "情绪总体平稳，愿意聊天。"}]
        # prompt 的【健康读数】小节若含结构化读数行（以 "- " 开头），才给健康趋势
        health_m = re.search(r"【健康读数】\n(.+)$", prompt, re.DOTALL)
        health_section = health_m.group(1) if health_m else ""
        if re.search(r"^- ", health_section, re.MULTILINE):
            trends.append(
                {"kind": "health", "content": "近期健康读数总体平稳，具体数值以结构化记录为准。"}
            )
        return _json.dumps(
            {
                "scenes": [
                    {
                        "theme": "日常点滴",
                        "summary": "近期与老人的若干段日常对话，情绪平稳。",
                        "member_indices": indices,
                    }
                ],
                "profile": "老人近期状态平稳，喜欢聊天，作息规律。（离线巩固存根）",
                "trends": trends,
            },
            ensure_ascii=False,
        )

    def _fake_fold(self, prompt: str) -> str:
        """离线折叠摘要存根：拼接已有摘要与新对话首句。"""
        import json as _json
        import re
        existing_m = re.search(r"【已有摘要】\n(.+?)\n\n【新发生的对话】", prompt, re.DOTALL)
        dialogue_m = re.search(r"【新发生的对话】\n(.+)$", prompt, re.DOTALL)
        existing = (existing_m.group(1).strip() if existing_m else "").replace("（暂无）", "")
        dialogue_lines = (dialogue_m.group(1).strip() if dialogue_m else "").splitlines()
        first_line = dialogue_lines[0][:30] if dialogue_lines else ""
        summary = (existing + (" " if existing else "") + first_line).strip()[:150]
        return _json.dumps({"summary": summary or "闲聊了几句", "mood": "平静", "topics": []}, ensure_ascii=False)

    def _fake_nlu(self, prompt: str) -> str:
        """离线意图识别存根：用启发式规则模拟 LLM，保证测试确定性。

        生产用真实 LLM；此处仅供无 key 的离线演示与单元测试。
        """
        import json as _json
        import re
        from datetime import date, timedelta

        from app.memory.events import parse_health

        m = re.search(r"老人说：(.+)$", prompt, re.DOTALL)
        text = m.group(1).strip() if m else ""

        # ---- 设置模式意图（profile-setup-mode）：高信号，命中即单意图返回 ----
        if any(k in text for k in ["进入设置", "设置模式", "我要设置", "修改资料", "修改档案"]):
            mp = re.search(r"(?:口令|密码)\s*[:：]?\s*([0-9A-Za-z]+)", text) or re.search(r"([0-9]{3,})", text)
            return _json.dumps(
                {"intents": [{"name": "enter_setup", "slots": {"passcode": mp.group(1) if mp else ""}}]},
                ensure_ascii=False,
            )
        if any(k in text for k in ["退出设置", "设置好了", "设置完了", "完成设置", "结束设置", "不设置了"]):
            return _json.dumps({"intents": [{"name": "exit_setup", "slots": {}}]}, ensure_ascii=False)

        up_slots = None
        ma = re.search(r"对(.+?)过敏", text)
        if ma:
            up_slots = {"field": "allergies", "value": ma.group(1)}
        elif "过敏" in text:
            up_slots = {"field": "allergies", "value": text.replace("过敏", "").strip("，。 、的")}
        elif any(k in text for k in ["家住", "住址", "地址", "家在", "住在"]):
            val = text
            for cue in ["我家住在", "家庭住址是", "家住址是", "家住址", "家住", "住址是",
                        "住址", "地址是", "地址", "家在", "住在"]:
                val = val.replace(cue, "")
            up_slots = {"field": "address", "value": val.strip("，。 、的是")}
        elif "联系人" in text:
            mp = re.search(r"(1[3-9]\d{9})", text)
            mr = re.search(r"(女儿|儿子|老伴|护工|邻居|社区|孙子|孙女|外孙)", text)
            mn = re.search(r"(?:联系人|女儿|儿子|护工|邻居)\s*(?:是|叫)?\s*([\u4e00-\u9fa5]{2,3})", text)
            up_slots = {
                "field": "contacts",
                "name": mn.group(1) if mn else "",
                "relation": mr.group(1) if mr else "",
                "phone": mp.group(1) if mp else "",
            }
        if up_slots is not None:
            return _json.dumps(
                {"intents": [{"name": "update_profile", "slots": up_slots}]},
                ensure_ascii=False,
            )

        today = date.today()
        date_iso = ""
        for term, delta in [("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0)]:
            if term in text:
                date_iso = (today + timedelta(days=delta)).isoformat()
                break
        persons = [
            p for p in re.findall(r'[\u4e00-\u9fa5]{2,3}(?=同学|朋友|老师|医生|叔|姨|哥|姐)', text)
            if p not in {"后天", "明天", "今天", "大后"}
        ]
        m_name = re.search(r'叫([\u4e00-\u9fa5]{2,3})', text)
        if m_name and m_name.group(1) not in persons:
            persons.append(m_name.group(1))

        intents: list[dict] = []

        # 健康数值
        parsed = parse_health(text)
        if parsed is not None:
            intents.append({
                "name": "record_health",
                "slots": {"kind": parsed.kind, "value": parsed.value, "unit": parsed.unit},
            })

        # 用药
        is_question = any(q in text for q in ["吗", "是不是", "？", "?"])
        confirm_kw = any(k in text for k in ["吃了", "吃过", "服了", "喝了", "吃完", "咽下"])
        defer_kw = any(k in text for k in ["待会", "等会", "一会儿", "过会", "稍后", "等下", "这就", "马上吃"])
        deny_kw = any(k in text for k in ["没吃", "还没", "忘了", "没顾上", "没有吃"])
        schedule_kw = any(k in text for k in ["几点", "什么时候", "啥时候", "吃药时间", "时间表"])
        if "药" in text and schedule_kw:
            intents.append({"name": "query_medication", "slots": {"scope": "schedule"}})
        elif "药" in text and is_question:
            intents.append({"name": "query_medication", "slots": {"scope": "today"}})
        elif confirm_kw and "药" in text:
            intents.append({"name": "confirm_medication", "slots": {}})
        elif defer_kw and "药" in text:
            intents.append({"name": "defer_medication", "slots": {}})
        elif deny_kw and "药" in text:
            intents.append({"name": "deny_medication", "slots": {}})

        # 预约（医疗类）
        medical_kw = any(k in text for k in ["医院", "复查", "体检", "门诊", "看病", "挂号"])
        memory_cue = any(k in text for k in ["记一下", "记下", "记住", "记着", "帮我记", "别忘", "提醒我", "记录"])
        if medical_kw:
            intents.append({
                "name": "record_appointment",
                "slots": {"title": text, "date_iso": date_iso, "persons": persons},
            })
        elif memory_cue:
            content = text
            for cue in ["你记一下", "帮我记住", "帮我记", "记一下", "记下", "记住", "记着", "别忘了", "提醒我", "记录一下"]:
                content = content.replace(cue, "")
            content = content.strip("，。 、你您")
            intents.append({
                "name": "remember_note",
                "slots": {"content": content or text, "date_iso": date_iso, "persons": persons},
            })

        # 预约/日程查询
        if not medical_kw and any(k in text for k in ["安排", "日程", "什么事", "有什么约", "预约"]):
            intents.append({"name": "query_appointment", "slots": {}})

        # 历史查询
        history_kw = any(k in text for k in ["最近", "这周", "上周", "这几天", "趋势", "情况"])
        if history_kw:
            if any(k in text for k in ["血压", "血糖", "心率", "体温", "体重"]):
                intents.append({"name": "query_health", "slots": {"kind": "all"}})
            elif "药" in text and not any(i["name"].startswith("confirm") for i in intents):
                intents.append({"name": "query_medication", "slots": {"scope": "history"}})

        if not intents:
            intents.append({"name": "chitchat", "slots": {}})

        return _json.dumps({"intents": intents}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 真实客户端（OpenAI / DeepSeek 兼容协议）
# ──────────────────────────────────────────────

class OpenAICompatibleClient:
    def __init__(self) -> None:
        from openai import OpenAI

        api_key, model, base_url = config.resolve_llm()
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def chat(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream_chat(self, messages: list[dict]):
        """流式生成：逐 token yield str。供面向用户的回复使用。"""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ──────────────────────────────────────────────
# 工厂
# ──────────────────────────────────────────────

def get_llm() -> LLMClient:
    if config.has_real_llm:
        try:
            return OpenAICompatibleClient()
        except Exception as exc:
            if config.allow_fake_llm:
                print(f"[warn] 初始化真实 LLM 失败({exc})，回退到离线模板回复。")
                return FakeLLMClient()
            raise
    if config.allow_fake_llm:
        return FakeLLMClient()
    raise RuntimeError("未配置 LLM API key 且未允许 FakeLLM，请检查 .env")
