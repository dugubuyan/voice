"""分层紧急检测（规则驱动，保证低延迟与确定性，不依赖 LLM）。

分层：
  第 1 层  关键词粗筛（求救词 / 不适词）
  第 2 层  意图判断：区分"当前正在发生"还是"转述/提及"（规则识别转述标记）
  第 3 层  是否需要主动确认
  第 4 层  分级：observe / alert / emergency
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Level(str, Enum):
    NONE = "none"
    OBSERVE = "observe"
    ALERT = "alert"
    EMERGENCY = "emergency"


# 求救词：命中即最高级，免确认
RESCUE_KEYWORDS = [
    "救命", "来人啊", "来人呀", "救救我", "打120", "打119", "打110", "救护车",
]

# 不适词：可能紧急，需结合意图 + 确认
DISTRESS_KEYWORDS = [
    "胸口", "胸闷", "喘不上气", "喘不过气", "透不过气",
    "头晕", "眩晕", "站不起来", "起不来", "摔倒", "跌倒", "摔了", "站不稳",
    "心慌", "心悸", "手麻", "说不出话", "浑身没劲", "疼死了", "难受", "不舒服",
]

# 转述/提及标记：出现则大概率不是"当前正在发生"
REPORTED_MARKERS = [
    "上次", "上回", "以前", "从前", "昨天", "前天", "曾经", "那次", "小时候",
    "电视", "剧里", "电视剧", "新闻", "别人", "邻居", "隔壁", "听说", "如果", "要是", "会不会",
]

# 取消/否认词：用于确认环节降级
CANCEL_KEYWORDS = [
    "没事", "没关系", "不用", "不要紧", "我很好", "我挺好", "好多了", "开玩笑", "逗你", "误会",
]

# 确认词：用于确认环节升级
# 注意："嗯"等语气词过于模糊，不计入确认词；模糊回复走"再问一次"更安全
CONFIRM_KEYWORDS = [
    "是的", "对", "真的", "很难受", "特别难受", "受不了", "帮我", "快点", "需要",
]


@dataclass
class Decision:
    level: Level
    need_confirmation: bool
    matched: list[str]
    reason: str

    @property
    def is_incident(self) -> bool:
        return self.level != Level.NONE


def _find(text: str, words: list[str]) -> list[str]:
    return [w for w in words if w in text]


def detect(text: str) -> Decision:
    """对单条用户文本做紧急研判。"""
    text = (text or "").strip()
    if not text:
        return Decision(Level.NONE, False, [], "空输入")

    # 第 2 层先算转述/提及标记（媒体、他人、过往、假设）
    reported = _find(text, REPORTED_MARKERS)

    rescue = _find(text, RESCUE_KEYWORDS)
    if rescue:
        if reported:
            return Decision(
                Level.OBSERVE,
                False,
                rescue,
                f"命中求救词{rescue}但含转述标记{reported}，判定为提及/媒体而非本人当前紧急，仅观察",
            )
        return Decision(Level.EMERGENCY, False, rescue, "命中求救词，立即最高级响应")

    distress = _find(text, DISTRESS_KEYWORDS)
    if not distress:
        return Decision(Level.NONE, False, [], "未命中风险词")

    if reported:
        return Decision(
            Level.OBSERVE,
            False,
            distress,
            f"命中不适词{distress}但含转述标记{reported}，判定为提及而非当前紧急，仅观察",
        )

    # 当前疑似不适 -> 告警并主动确认
    return Decision(
        Level.ALERT,
        True,
        distress,
        f"命中不适词{distress}且无转述标记，判定疑似当前不适，需主动确认",
    )


def interpret_confirmation(text: str) -> str:
    """解读确认环节的回复：'cancel' / 'confirm' / 'unclear'。"""
    text = (text or "").strip()
    if _find(text, CANCEL_KEYWORDS):
        return "cancel"
    if _find(text, CONFIRM_KEYWORDS) or _find(text, DISTRESS_KEYWORDS) or _find(text, RESCUE_KEYWORDS):
        return "confirm"
    return "unclear"
