"""重要性驱动的反思引擎（ReflectionEngine）。

触发方式：每轮 process_turn 结束后，把本轮的重要性分数累积到
ElderRuntime.reflection_score；一旦超过阈值就触发一次反思——
分析近期对话中的情绪/健康线索，生成一条洞察型 MemoryNote（source="reflection"），
然后归零 reflection_score。

重要性打分：规则驱动，不调 LLM，零延迟。
反思生成：一次 LLM 调用，廉价。失败时不阻塞对话，静默跳过。
"""
from __future__ import annotations

import json
import re

from app.agent.runtime import ElderRuntime, Turn
from app.config import config
from app.memory.conflict import MemoryConflictResolver
from app.memory.notes import ExtractedNote
from sqlalchemy.orm import Session

# ──────────────────────────────────────────────
# 重要性打分
# ──────────────────────────────────────────────

_EMOTION_WORDS = [
    "失眠", "睡不着", "睡不好", "难受", "难过", "痛苦", "孤独", "寂寞",
    "想念", "思念", "担心", "害怕", "悲伤", "伤心", "烦躁", "焦虑",
    "高兴", "开心", "快乐", "幸福", "欣慰", "骄傲",
]
_HEALTH_WORDS = [
    "疼", "痛", "头晕", "眩晕", "胸闷", "喘不上气", "心慌", "恶心",
    "发烧", "发热", "腿软", "摔", "跌",
]
_PERSON_RE = re.compile(
    r"[\u4e00-\u9fa5]{2,3}(?:同学|朋友|老师|医生|叔|姨|哥|姐|嫂|弟|妹)"  # 姓名+关系
    r"|(?:女儿|儿子|孙子|孙女|外孙|外孙女|老伴|老公|老婆|邻居)"  # 直接关系词
)


def importance_score(user_text: str, result_kind: str) -> float:
    """对一轮对话估算重要性分数（纯规则，零延迟）。"""
    score = 0.5  # 基础分（普通闲聊）
    text = user_text or ""

    if any(w in text for w in _EMOTION_WORDS):
        score += 1.5
    if any(w in text for w in _HEALTH_WORDS):
        score += 2.0
    if result_kind == "health":
        score += 1.0
    if result_kind == "emergency":
        score += 3.0
    if _PERSON_RE.search(text):
        score += 1.0

    return score


# ──────────────────────────────────────────────
# 反思 Prompt
# ──────────────────────────────────────────────

REFLECT_PROMPT_MARKER = "【反思任务】"

REFLECT_PROMPT = """\
【反思任务】你是老人陪护助手的记忆分析模块。
根据下面的【近期对话摘要】和【最近原文窗口】，识别出值得长期记住的情绪/健康模式洞察。
输出严格为 JSON，不要额外文字：
{{"insight": "一句话洞察（如：近期多次提到睡不好，情绪偏低落）", "kind": "emotional/health/social/other"}}

【近期对话摘要】
{summary}

【最近原文窗口（最新若干轮）】
{window}
"""


# ──────────────────────────────────────────────
# ReflectionEngine
# ──────────────────────────────────────────────

class ReflectionEngine:
    def __init__(self, llm, conflict: MemoryConflictResolver) -> None:
        self._llm = llm
        self._conflict = conflict

    def observe(
        self,
        runtime: ElderRuntime,
        user_text: str,
        result_kind: str,
        session: Session,
    ) -> bool:
        """累积本轮重要性分；超阈值时触发反思并归零。

        Returns:
            True 表示本轮触发了反思；False 表示只是累积。
        """
        runtime.reflection_score += importance_score(user_text, result_kind)
        if runtime.reflection_score < config.reflect_threshold:
            return False
        runtime.reflection_score = 0.0
        self._reflect(runtime, session)
        return True

    def _reflect(self, runtime: ElderRuntime, session: Session) -> None:
        """一次 LLM 调用，生成洞察型 MemoryNote（失败静默跳过）。"""
        window_text = "\n".join(
            f"{'老人' if t.role == 'user' else '助手'}：{t.content}"
            for t in runtime.window[-6:]
        )
        prompt = REFLECT_PROMPT.format(
            summary=runtime.running_summary or "（暂无）",
            window=window_text or "（窗口为空）",
        )
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            insight = str(data.get("insight", "")).strip()
            kind = str(data.get("kind", "other")).strip()
            if not insight:
                return
        except Exception:
            return  # 失败静默跳过，不阻塞对话

        # 通过冲突解析器写入，相似洞察自动合并/替换
        note = self._conflict.upsert_note(
            runtime.elder_id,
            ExtractedNote(
                is_memory_request=True,
                content=insight,
                date_iso="",
                persons=[],
                kind=kind,
            ),
        )
        # 源标记为 reflection
        note.source = "reflection"
        session.flush()

        # 追加到运行期 system_prompt，让下一轮闲聊可感知
        runtime.system_prompt += f"\n【近期情况洞察】\n- {insight}"
