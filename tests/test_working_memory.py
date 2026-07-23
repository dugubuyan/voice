"""WorkingMemory + RunningSummaryRepo + SummaryFolder 测试。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.agent.llm import FakeLLMClient
from app.agent.runtime import ElderRuntime, Turn
from app.memory.folder import SummaryFolder
from app.memory.running import RunningSummaryRepo
from app.memory.working import WorkingMemory, estimate_tokens


# ──────────────────────────────────────────────
# 夹具
# ──────────────────────────────────────────────

@pytest.fixture()
def folder():
    return SummaryFolder(FakeLLMClient())


@pytest.fixture()
def repo(session):
    return RunningSummaryRepo(session)


@pytest.fixture()
def wm(folder, repo):
    return WorkingMemory(folder, repo)


def _runtime(elder_id: int = 1) -> ElderRuntime:
    return ElderRuntime(elder_id=elder_id, system_prompt="")


def _turn(role: str, content: str, offset_sec: int = 0) -> Turn:
    return Turn(role=role, content=content, at=datetime(2025, 1, 1) + timedelta(seconds=offset_sec))


def _fill_window(runtime: ElderRuntime, n_turns: int, chars_each: int = 200) -> None:
    """往窗口里塞 n_turns 轮，每轮 chars_each 字符。"""
    for i in range(n_turns):
        content = "字" * chars_each
        t = _turn("user", content, offset_sec=i * 10)
        runtime.window.append(t)
        runtime.window_tokens += estimate_tokens(content)


# ──────────────────────────────────────────────
# estimate_tokens
# ──────────────────────────────────────────────

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_chars():
    assert estimate_tokens("hello") == 5
    assert estimate_tokens("你好") == 2


# ──────────────────────────────────────────────
# P5: 空窗/预算内 no-op，不调 LLM
# ──────────────────────────────────────────────

def test_empty_window_noop(session, wm):
    rt = _runtime()
    called = []
    original_fold = wm._folder.fold

    def spy(*a, **kw):
        called.append(1)
        return original_fold(*a, **kw)

    wm._folder.fold = spy
    wm.fold_if_over_budget(rt, session)
    assert not called         # 窗口空，不应调用 LLM

    wm.fold_all(rt, session)
    assert not called


def test_within_budget_noop(session, wm):
    from app.config import config

    rt = _runtime()
    rt.window.append(_turn("user", "短句", 0))
    rt.window_tokens = 1   # 远低于预算

    calls = []
    orig = wm._folder.fold
    wm._folder.fold = lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1]

    wm.fold_if_over_budget(rt, session)
    assert not calls


# ──────────────────────────────────────────────
# P2: 折叠后 window_tokens <= TOKEN_BUDGET
# ──────────────────────────────────────────────

def test_fold_brings_within_budget(session, elders, wm):
    from app.config import config

    elder = elders["李桂芳"]
    rt = _runtime(elder.id)
    # 塞入超预算内容（每轮200字，塞10轮 = 2000 token，超默认1500）
    _fill_window(rt, n_turns=10, chars_each=200)
    assert rt.window_tokens > config.token_budget

    wm.fold_if_over_budget(rt, session)
    session.commit()

    assert rt.window_tokens <= config.token_budget  # P2


# ──────────────────────────────────────────────
# P3: 挤出的内容进入 running summary
# ──────────────────────────────────────────────

def test_evicted_content_in_summary(session, elders, wm):
    from app.config import config

    elder = elders["李桂芳"]
    rt = _runtime(elder.id)
    _fill_window(rt, n_turns=10, chars_each=200)

    wm.fold_if_over_budget(rt, session)
    session.commit()

    assert rt.running_summary  # 内存镜像非空（P3）
    row = RunningSummaryRepo(session).get_active(elder.id)
    assert row is not None
    assert row.summary  # 落库非空


# ──────────────────────────────────────────────
# P4: covers_to 单调不减
# ──────────────────────────────────────────────

def test_covers_to_monotone(session, elders, wm):
    from app.config import config

    elder = elders["李桂芳"]
    rt = _runtime(elder.id)
    _fill_window(rt, n_turns=10, chars_each=200)
    wm.fold_if_over_budget(rt, session)
    session.commit()

    row1 = RunningSummaryRepo(session).get_active(elder.id)
    to1 = row1.covers_to

    # 再塞一批超预算
    _fill_window(rt, n_turns=10, chars_each=200)
    wm.fold_if_over_budget(rt, session)
    session.commit()

    row2 = RunningSummaryRepo(session).get_active(elder.id)
    assert row2.covers_to >= to1  # P4


# ──────────────────────────────────────────────
# RunningSummaryRepo: 每 elder 始终一条 active
# ──────────────────────────────────────────────

def test_upsert_keeps_single_active_row(session, elders):
    elder = elders["李桂芳"]
    repo = RunningSummaryRepo(session)
    t0 = datetime(2025, 1, 1, 8, 0)
    t1 = datetime(2025, 1, 1, 9, 0)
    t2 = datetime(2025, 1, 1, 10, 0)

    repo.upsert(elder.id, summary="第一次", mood="平静", topics=[], covers_from=t0, covers_to=t0)
    repo.upsert(elder.id, summary="第二次", mood="开心", topics=["聊天"], covers_from=t0, covers_to=t1)
    repo.upsert(elder.id, summary="第三次", mood="低落", topics=[], covers_from=t0, covers_to=t2)
    session.commit()

    from sqlalchemy import select
    from app.models import RunningSummary

    rows = session.execute(
        select(RunningSummary).where(RunningSummary.elder_id == elder.id, RunningSummary.active == True)  # noqa
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].summary == "第三次"


# ──────────────────────────────────────────────
# SummaryFolder 降级路径
# ──────────────────────────────────────────────

def test_folder_degrades_on_bad_llm():
    """LLM 返回无效 JSON 时，降级仍返回非空摘要，不阻塞折叠。"""

    class BadLLM:
        def chat(self, messages): raise RuntimeError("LLM 挂了")

    folder = SummaryFolder(BadLLM())
    evicted = [_turn("user", "今天心情不好", 0)]
    summary, mood, topics = folder.fold("旧摘要", evicted)
    assert summary  # 降级产出非空
    assert "旧摘要" in summary or "今天心情不好" in summary


def test_folder_degraded_on_empty_json():
    """LLM 返回空 summary 字段时同样降级。"""

    class EmptyLLM:
        def chat(self, messages): return '{"summary": "", "mood": "", "topics": []}'

    folder = SummaryFolder(EmptyLLM())
    evicted = [_turn("user", "聊聊天", 0)]
    summary, _, _ = folder.fold("旧有内容", evicted)
    assert summary  # 仍非空
