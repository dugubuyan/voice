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

    # 联网搜索（火山联网搜索 API，非方舟 LLM 接口）。缺 key 或关闭时对话如实告知"查不了"。
    volcano_api_key: str | None = os.getenv("VOLCANO_API_KEY")
    volcano_search_url: str = os.getenv(
        "VOLCANO_SEARCH_URL", "https://open.feedcoopapi.com/search_api/web_search"
    )
    web_search_enabled: bool = _bool(os.getenv("WEB_SEARCH_ENABLED"), True)
    web_search_timeout_sec: float = float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8"))
    web_search_count: int = int(os.getenv("WEB_SEARCH_COUNT", "3"))

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

    # ===== Phase 3：情景记忆 + 检索式回忆 + 统一条件触发 =====
    # 情景切分：轮数上限兜底
    episode_max_turns: int = int(os.getenv("MEM_EPISODE_MAX_TURNS", "12"))
    # 巩固触发：空闲 > 15min，或未巩固 Episode 累积量超阈值
    consolidate_idle_sec: int = int(os.getenv("MEM_CONSOLIDATE_IDLE_SEC", "900"))
    consolidate_min_episodes: int = int(os.getenv("MEM_CONSOLIDATE_MIN_EPISODES", "3"))
    # 检索：召回条数 / token 上界
    recall_max_items: int = int(os.getenv("MEM_RECALL_MAX_ITEMS", "6"))
    recall_token_budget: int = int(os.getenv("MEM_RECALL_TOKEN_BUDGET", "600"))
    # pulse：低频 tick 间隔（秒）
    pulse_interval_sec: int = int(os.getenv("MEM_PULSE_INTERVAL_SEC", "30"))
    # 保留/清理：近期原文保留天数
    recent_keep_days: int = int(os.getenv("MEM_RECENT_KEEP_DAYS", "3"))
    # 主动关心：空闲极久阈值（秒），6h，0=关
    proactive_idle_sec: int = int(os.getenv("MEM_PROACTIVE_IDLE_SEC", "21600"))

    # ===== 设置模式（profile-setup-mode）=====
    # 进入设置模式的占位口令（后续可替换为家属端 App / PIN / 硬件按键）
    setup_passcode: str = os.getenv("SETUP_PASSCODE", "0000")
    # 设置会话超时（秒），超时自动回到 daily 模式，默认 5min
    setup_session_timeout_sec: int = int(os.getenv("SETUP_TIMEOUT_SEC", "300"))


    # 关闭思考型模型的思考过程（请求体加 enable_thinking=false）。
    # 空=自动（base_url 含 dashscope 时开启）；true/false 强制。
    llm_disable_thinking: str = os.getenv("LLM_DISABLE_THINKING", "")

    def should_disable_thinking(self, base_url: str | None) -> bool:
        if self.llm_disable_thinking.strip():
            return _bool(self.llm_disable_thinking)
        return "dashscope" in (base_url or "").lower()

    def resolve_llm(self) -> tuple[str | None, str, str | None]:
        """按 provider 返回 (api_key, model, base_url)。"""
        if self.llm_provider.lower() == "deepseek":
            return self.deepseek_api_key, self.deepseek_model, self.deepseek_base_url
        return self.openai_api_key, self.openai_model, self.openai_base_url

    @property
    def has_web_search(self) -> bool:
        return self.web_search_enabled and bool(self.volcano_api_key)

    @property
    def has_real_llm(self) -> bool:
        api_key, _, _ = self.resolve_llm()
        return bool(api_key)


config = Config()
