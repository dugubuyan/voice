"""网页服务配置（环境变量）。

先 import app.config 触发 .env 加载，再读取下面这些变量，保证 .env 里的值生效。
"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import config  # noqa: F401  先加载 .env

# 所有持久化数据的根目录：accounts.db（账号）+ users/<id>.db（每个用户独立的记忆库）
DATA_DIR = Path(
    os.getenv("DATA_DIR") or Path(__file__).resolve().parent.parent / "data"
).resolve()

# 注册策略：默认需要一次性邀请码（python -m demo.invites new 生成，用过即作废）。
# REGISTER_OPEN=true 才会开放注册（不推荐：任何人可注册并消耗额度）。
REGISTER_OPEN = (os.getenv("REGISTER_OPEN") or "").strip().lower() in {"1", "true", "yes", "on"}
REGISTRATION_MODE = "open" if REGISTER_OPEN else "invite"

SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))           # 登录态有效期
USER_DAILY_TURNS = int(os.getenv("USER_DAILY_TURNS", "300"))  # 单用户每天对话轮数上限
WORKSPACE_TTL_SEC = int(os.getenv("WORKSPACE_TTL_MIN", "60")) * 60  # 内存中保留时间
MAX_WORKSPACES = int(os.getenv("MAX_WORKSPACES", "200"))      # 内存中同时保留的用户数
MAX_TEXT = 200                                                # 单条消息长度上限
