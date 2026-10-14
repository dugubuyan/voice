"""语义边界检测器（SemanticBoundaryDetector）。

判断窗口最近轮次是否发生话题切换，给出切分点，供 EpisodeCutTrigger 切出
raw episode。一次廉价 LLM 调用；纯读，不写 DB。

降级：LLM 调用或 JSON 解析失败 → 返回 BoundaryDecision(False, 0)，交给上层
token 兜底 / 久别兜底处理，绝不抛出（需求 1.4）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app.agent.runtime import Turn

BOUNDARY_PROMPT_MARKER = "【边界检测】"

BOUNDARY_PROMPT = """\
【边界检测】你在维护一段"老人陪护对话"，需要判断最近的对话里是否发生了话题切换。
下面是【最近对话窗口】（按时间从早到晚编号，从 1 开始）。请判断：窗口靠前的若干轮
是否属于"上一个已经聊完的话题"，而后面的轮次开启了新话题。

只输出 JSON，不要额外文字：
{{"is_boundary": true/false, "cut_index": 整数}}

字段说明：
- is_boundary：是否检测到话题切换。
- cut_index：从窗口头部起、属于"上一话题"的轮数（即应切出 window[:cut_index]）。
  取值范围 [1, 窗口轮数]；is_boundary 为 false 时该值无意义，填 0 即可。

【最近对话窗口】
{dialogue}
"""


@dataclass
class BoundaryDecision:
    """边界检测结果。"""

    is_boundary: bool
    cut_index: int          # 切出 window[:cut_index]；is_boundary=False 时无意义


class SemanticBoundaryDetector:
    """语义边界检测器（stateless，依赖注入 LLM 客户端）。"""

    def __init__(self, llm) -> None:
        self._llm = llm

    def detect(self, window: list[Turn]) -> BoundaryDecision:
        """判断窗口最近轮次是否发生话题切换。

        Args:
            window: 最近对话窗口（Turn 列表）。

        Returns:
            BoundaryDecision(is_boundary, cut_index)。
            cut_index 被裁剪到 [1, len(window)]；is_boundary=False 时返回 0。

        Raises:
            不抛出异常：LLM 调用或 JSON 解析失败时降级为 (False, 0)。
        """
        if not window:
            return BoundaryDecision(False, 0)

        dialogue = "\n".join(
            f"{i}. {'老人' if t.role == 'user' else '助手'}：{t.content}"
            for i, t in enumerate(window, start=1)
        )
        prompt = BOUNDARY_PROMPT.format(dialogue=dialogue)

        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            is_boundary = bool(data.get("is_boundary", False))
            if not is_boundary:
                # is_boundary 为假时 cut_index 无意义，统一返回 0
                return BoundaryDecision(False, 0)
            # 边界裁剪到 [1, len(window)]
            cut_index = int(data.get("cut_index", 0))
            cut_index = max(1, min(cut_index, len(window)))
            return BoundaryDecision(True, cut_index)
        except Exception:
            # 降级：视为非边界，交给上层 token/久别兜底，绝不抛出（需求 1.4）
            return BoundaryDecision(False, 0)
