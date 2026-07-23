"""工作记忆管理：token 有界滚动窗口 + 增量 running summary 折叠。

唯一的硬触发：window_tokens 超预算时，从头部挤出 FOLD_BATCH 轮，
增量 fold 进 running summary，直到回到预算内。
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.agent.runtime import ElderRuntime, Turn
from app.config import config
from app.memory.folder import SummaryFolder
from app.memory.running import RunningSummaryRepo


def estimate_tokens(text: str) -> int:
    """字符数近似估算 token（中文 1 字≈1 token，英文 4 字符≈1 token）。MVP 够用。"""
    return len(text or "")


class WorkingMemory:
    """工作记忆管理器（stateless，依赖注入 SummaryFolder + RunningSummaryRepo）。"""

    def __init__(self, folder: SummaryFolder, repo: RunningSummaryRepo) -> None:
        self._folder = folder
        self._repo = repo

    # ---- 主接口 ----

    def append(self, runtime: ElderRuntime, turn: Turn) -> None:
        """追加一轮到窗口，更新 token 估算。"""
        runtime.window.append(turn)
        runtime.window_tokens += estimate_tokens(turn.content)

    def fold_if_over_budget(self, runtime: ElderRuntime, session: Session) -> None:
        """token 超预算则循环挤出 FOLD_BATCH 轮并增量折叠，直到回预算内。

        设计保证终止：每次迭代至少挤出一轮（FOLD_BATCH >= 1），窗口严格缩小。
        """
        budget = config.token_budget
        batch = config.fold_batch
        while runtime.window_tokens > budget and len(runtime.window) > batch:
            self._fold_batch(runtime, session, batch)

    def fold_all(self, runtime: ElderRuntime, session: Session) -> None:
        """整窗折叠（关机 flush / episode 边界可选调用）。"""
        if not runtime.window:
            return
        self._fold_batch(runtime, session, len(runtime.window))

    # ---- 内部 ----

    def _fold_batch(
        self, runtime: ElderRuntime, session: Session, n: int
    ) -> None:
        """把 window[:n] 增量折叠进 running summary，并从窗口移除。"""
        evicted = runtime.window[:n]
        summary, mood, topics = self._folder.fold(runtime.running_summary, evicted)

        # 同步内存镜像
        runtime.running_summary = summary

        # 落库
        repo = RunningSummaryRepo(session)
        repo.upsert(
            runtime.elder_id,
            summary=summary,
            mood=mood,
            topics=topics,
            covers_from=evicted[0].at,
            covers_to=evicted[-1].at,
        )

        # 截断窗口，重算 token（基于剩余内容）
        runtime.window = runtime.window[n:]
        runtime.window_tokens = sum(estimate_tokens(t.content) for t in runtime.window)
