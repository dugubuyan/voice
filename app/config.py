"""全局配置：从环境变量 / .env 读取。"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv 可选
    pass


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """运行期配置。"""

    # openai / deepseek，二者均走 OpenAI 兼容协议
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")

    # DeepSeek（OpenAI 兼容）。未单独配置 DEEPSEEK_API_KEY 时回退到 OPENAI_API_KEY。
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///elder_care.db")
    allow_fake_llm: bool = _bool(os.getenv("ALLOW_FAKE_LLM"), True)

    # 用药提醒未确认时的重复提醒次数与间隔（分钟）
    medication_max_reminders: int = int(os.getenv("MED_MAX_REMINDERS", "3"))
    medication_retry_minutes: int = int(os.getenv("MED_RETRY_MINUTES", "10"))

    # 工作记忆滚动窗口（token 预算 + 每次折叠批次）
    token_budget: int = int(os.getenv("MEM_TOKEN_BUDGET", "1500"))
    fold_batch: int = int(os.getenv("MEM_FOLD_BATCH", "4"))
    # episode 边界折叠：久别重逢时先折叠上一段，让摘要贴合话题边界（质量优化）
    episode_gap_sec: int = int(os.getenv("MEM_EPISODE_GAP_SEC", "1800"))
    # 重要性反思（ReflectionEngine）
    reflect_threshold: float = float(os.getenv("MEM_REFLECT_THRESHOLD", "5.0"))

    def resolve_llm(self) -> tuple[str | None, str, str | None]:
        """按 provider 返回 (api_key, model, base_url)。"""
        if self.llm_provider.lower() == "deepseek":
            return self.deepseek_api_key, self.deepseek_model, self.deepseek_base_url
        return self.openai_api_key, self.openai_model, self.openai_base_url

    @property
    def has_real_llm(self) -> bool:
        api_key, _, _ = self.resolve_llm()
        return bool(api_key)


config = Config()
