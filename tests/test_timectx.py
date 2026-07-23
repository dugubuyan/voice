"""TimeContext 单元测试：间隔分档边界 + 时段映射边界。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.agent.timectx import build_time_context, day_period_of


# ──────────────────────────────────────────────
# day_period_of 边界
# ──────────────────────────────────────────────

@pytest.mark.parametrize("hour,expected", [
    (0, "深夜"), (4, "深夜"),
    (5, "清晨"), (7, "清晨"),
    (8, "上午"), (10, "上午"),
    (11, "中午"), (12, "中午"),  # hour<11 → 上午, 11/12 → 中午
    (13, "下午"), (16, "下午"),
    (17, "傍晚"), (18, "傍晚"),
    (19, "晚上"), (22, "晚上"),
    (23, "深夜"),
])
def test_day_period(hour, expected):
    assert day_period_of(hour) == expected


# ──────────────────────────────────────────────
# 首句（last_at=None）
# ──────────────────────────────────────────────

def test_first_turn_has_none_gap():
    now = datetime(2025, 6, 12, 9, 0, 0)
    ctx = build_time_context(now, None)
    assert ctx.gap_since_last is None
    assert "第一句话" in ctx.to_injection()


# ──────────────────────────────────────────────
# 间隔分档边界（<90s / <1h / <1d / >1d）
# ──────────────────────────────────────────────

def _ctx(gap_seconds: float) -> "TimeContext":
    now = datetime(2025, 6, 12, 15, 0, 0)
    last = now - timedelta(seconds=gap_seconds)
    return build_time_context(now, last)


def test_gap_under_90s_is_continuous():
    assert "连续对话" in _ctx(89).to_injection()


def test_gap_90s_is_minutes():
    inj = _ctx(90).to_injection()
    assert "分钟" in inj
    assert "连续" not in inj


def test_gap_under_1h_shows_minutes():
    inj = _ctx(3599).to_injection()
    assert "分钟" in inj


def test_gap_60min_is_hours():
    inj = _ctx(3600).to_injection()
    assert "小时" in inj
    assert "分钟" not in inj


def test_gap_under_24h_shows_hours():
    inj = _ctx(86399).to_injection()
    assert "小时" in inj


def test_gap_24h_is_days():
    inj = _ctx(86400).to_injection()
    assert "天" in inj
    assert "小时" not in inj


def test_gap_over_1day_shows_days():
    inj = _ctx(86400 * 3).to_injection()
    assert "3" in inj
    assert "天" in inj


# ──────────────────────────────────────────────
# 时间注入格式完整性
# ──────────────────────────────────────────────

def test_injection_contains_datetime_and_weekday():
    now = datetime(2025, 6, 12, 21, 40, 0)   # 周四
    ctx = build_time_context(now, None)
    inj = ctx.to_injection()
    assert "2025年06月12日" in inj
    assert "21:40" in inj
    assert "四" in inj
    assert "晚上" in inj
