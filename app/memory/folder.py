"""增量摘要器：把被挤出窗口的轮次 fold 进已有 running summary。

一次廉价 LLM 调用；LLM 失败时降级为"旧摘要 + 新对话首句"，
保证窗口仍能正常挤出，不因 LLM 故障无限增长。
"""
from __future__ import annotations

import json

from app.agent.runtime import Turn

FOLD_PROMPT_MARKER = "【折叠摘要任务】"

FOLD_PROMPT = """\
【折叠摘要任务】你在维护一段"老人陪护对话的持续记忆摘要"。
下面是【已有摘要】和【新发生的对话】，请把新对话并入摘要，输出更新后的摘要。
要求：保留重要的人名/事件/情绪线索；控制在150字内；只输出 JSON，不要额外文字：
{{"summary": "...", "mood": "平静/开心/低落/焦虑/...", "topics": ["..."]}}

【已有摘要】
{existing}

【新发生的对话】
{dialogue}
"""


class SummaryFolder:
    """增量摘要折叠器（stateless，依赖注入 LLM 客户端）。"""

    def __init__(self, llm) -> None:
        self._llm = llm

    def fold(
        self, existing: str, evicted: list[Turn]
    ) -> tuple[str, str, list[str]]:
        """把 evicted 轮次并入 existing 摘要。

        Returns:
            (summary, mood, topics)

        Raises:
            不抛出异常：LLM 失败时返回降级摘要。
        """
        dialogue = "\n".join(
            f"{'老人' if t.role == 'user' else '助手'}：{t.content}"
            for t in evicted
        )
        prompt = FOLD_PROMPT.format(
            existing=existing or "（暂无）",
            dialogue=dialogue,
        )
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            summary = str(data.get("summary", "")).strip()[:200]
            mood = str(data.get("mood", "")).strip()
            topics = list(data.get("topics", []))[:5]
            if not summary:
                raise ValueError("empty summary")
            return summary, mood, topics
        except Exception:
            # 降级：旧摘要 + 新对话首句，保证不阻塞折叠
            tail = evicted[0].content[:30] if evicted else ""
            degraded = (existing + f" 又聊到：{tail}").strip()[:200]
            return degraded or tail[:200], "", []
