"""编造的生活场景库：用于参数化测试与端到端演示。"""
from __future__ import annotations

from app.emergency.detector import Level

# ---- 紧急检测场景：(输入文本, 期望级别, 是否需要确认) ----
EMERGENCY_CASES = [
    ("救命啊！有没有人！", Level.EMERGENCY, False),
    ("快帮我打120", Level.EMERGENCY, False),
    ("我摔在地上了，起不来", Level.ALERT, True),
    ("我这胸口闷得慌，喘不上气", Level.ALERT, True),
    ("刚才有点头晕", Level.ALERT, True),
    ("我上次差点摔倒，还好扶住了", Level.OBSERVE, False),
    ("电视里那个人一直喊救命", Level.OBSERVE, False),
    ("我邻居前几天摔了一跤", Level.OBSERVE, False),
    ("今天天气真好，我去浇花了", Level.NONE, False),
    ("外孙小宝昨天给我打电话了", Level.NONE, False),
    ("要是哪天我摔倒了可怎么办哦", Level.OBSERVE, False),
]

# ---- 确认环节场景：(回复文本, 期望裁决) ----
CONFIRM_CASES = [
    ("没事没事，我就是喘口气", "cancel"),
    ("我很好，别担心", "cancel"),
    ("是的，我很难受", "confirm"),
    ("对，快帮我", "confirm"),
    ("嗯……", "unclear"),
    ("外面在下雨", "unclear"),
]

# ---- 健康数据抽取场景：(文本, 期望kind, 期望value) ----
HEALTH_CASES = [
    ("我今天量的血压是140/90", "blood_pressure", "140/90"),
    ("早上血糖7.8", "blood_sugar", "7.8"),
    ("心率大概78吧", "heart_rate", "78"),
    ("体温37.2，有点低烧", "temperature", "37.2"),
    ("我今天称了体重，58公斤", "weight", None),  # 记录即可
]

# ---- 用药对话场景：(用户输入, 期望 kind, 期望 data 片段) ----
MED_DIALOG_CASES = [
    ("我药吃过了", "medication", {"answer": "confirmed"}),
    ("我还没吃药呢", "medication", {"answer": "not_taken"}),
    ("等会儿再吃药吧", "medication", {"answer": "deferred"}),
    ("我今天吃药了吗？", "medication", {"answer": "pending"}),
]
