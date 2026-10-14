"""profile-setup-mode：双模式 + 槽位填充 + 复述确认 + 结构化字段 的测试。

覆盖 Correctness Properties P1–P7 与需求 1–8 的关键验收点，全程 FakeLLM 离线。
"""
from __future__ import annotations

import pytest

from app.agent.profile_slots import (
    SLOT_REGISTRY,
    SlotApplyContext,
    resolve_field,
)
from app.config import config
from app.emergency.detector import Level
from app.medication.service import MedicationService
from app.models import Elder


# ──────────────────────────────────────────────
# 模式与入口把关（需求 1/2，P1/P2）
# ──────────────────────────────────────────────

def test_default_mode_is_daily(brain, started):
    elder_id, _ = started()
    assert brain._runtimes[elder_id].mode == "daily"


def test_enter_setup_requires_correct_passcode(brain, started):
    elder_id, _ = started()
    # 错误口令：保持 daily
    r = brain.process_turn(elder_id, "进入设置模式，口令9999")
    assert brain._runtimes[elder_id].mode == "daily"
    assert r.data.get("entered") is False
    # 正确口令：进入 setup
    r2 = brain.process_turn(elder_id, f"进入设置模式，口令{config.setup_passcode}")
    assert brain._runtimes[elder_id].mode == "setup"
    assert r2.data.get("mode") == "setup"


def test_passcode_not_echoed(brain, started):
    elder_id, _ = started()
    r = brain.process_turn(elder_id, f"进入设置模式，口令{config.setup_passcode}")
    assert config.setup_passcode not in r.reply


def test_exit_setup_returns_to_daily(brain, started):
    elder_id, _ = started()
    brain.process_turn(elder_id, f"进入设置模式，口令{config.setup_passcode}")
    r = brain.process_turn(elder_id, "退出设置")
    assert brain._runtimes[elder_id].mode == "daily"
    assert r.data.get("mode") == "daily"


def test_update_profile_blocked_in_daily(brain, started, session):
    """P2：daily 模式下不派发写意图，走温和提示，且不落库。"""
    elder_id, _ = started()
    elder = brain._profiles.get(elder_id)
    before = elder.address
    r = brain.process_turn(elder_id, "家住北京朝阳区幸福路1号")
    assert r.kind == "mode"  # 提示进设置模式
    session.refresh(elder)
    assert elder.address == before  # 未写入


# ──────────────────────────────────────────────
# 复述确认与落库（需求 4/5，P3/P4）
# ──────────────────────────────────────────────

def _enter_setup(brain, elder_id):
    brain.process_turn(elder_id, f"进入设置模式，口令{config.setup_passcode}")


def test_address_readback_then_commit(brain, started, session):
    elder_id, _ = started("王秀英")  # 无预置住址
    _enter_setup(brain, elder_id)
    elder = brain._profiles.get(elder_id)

    # 给住址 → 复述确认，尚未落库（P3）
    r1 = brain.process_turn(elder_id, "我家住在浦东新区滨江大道4600号")
    assert brain._runtimes[elder_id].pending_profile_slot is not None
    assert "4600号" in r1.reply  # 复述念回原值（P4）
    session.refresh(elder)
    assert elder.address == ""  # 确认前不落库

    # 确认 → 落库
    r2 = brain.process_turn(elder_id, "对")
    session.refresh(elder)
    assert elder.address == "浦东新区滨江大道4600号"
    assert r2.data.get("saved") is True
    assert brain._runtimes[elder_id].pending_profile_slot is None


def test_deny_then_recollect(brain, started, session):
    elder_id, _ = started("王秀英")
    _enter_setup(brain, elder_id)
    elder = brain._profiles.get(elder_id)
    brain.process_turn(elder_id, "我家住在浦东新区滨江大道4600号")
    r = brain.process_turn(elder_id, "不对")
    session.refresh(elder)
    assert elder.address == ""
    assert brain._runtimes[elder_id].pending_profile_slot is None


def test_allergies_list_commit(brain, started, session):
    elder_id, _ = started("张建国")  # 张建国无过敏记录
    _enter_setup(brain, elder_id)
    elder = brain._profiles.get(elder_id)
    brain.process_turn(elder_id, "他对青霉素和海鲜过敏")
    brain.process_turn(elder_id, "对")
    session.refresh(elder)
    assert "青霉素" in elder.allergies and "海鲜" in elder.allergies


# ──────────────────────────────────────────────
# 设置会话超时（需求 3，P7）
# ──────────────────────────────────────────────

def test_setup_session_times_out(brain, started, monkeypatch):
    elder_id, _ = started()
    _enter_setup(brain, elder_id)
    rt = brain._runtimes[elder_id]
    assert rt.mode == "setup"
    # 把超时设为 0，下一轮任何交互都触发超时回 daily
    monkeypatch.setattr(config, "setup_session_timeout_sec", 0)
    brain.process_turn(elder_id, "今天天气不错")
    assert brain._runtimes[elder_id].mode == "daily"


def test_exit_discards_pending(brain, started, session):
    """P7：退出时未确认槽位丢弃、不落库。"""
    elder_id, _ = started("王秀英")
    _enter_setup(brain, elder_id)
    elder = brain._profiles.get(elder_id)
    brain.process_turn(elder_id, "我家住在测试路9号")
    assert brain._runtimes[elder_id].pending_profile_slot is not None
    brain.process_turn(elder_id, "退出设置")
    session.refresh(elder)
    assert brain._runtimes[elder_id].pending_profile_slot is None
    assert elder.address == ""


# ──────────────────────────────────────────────
# 读取侧：档案上下文注入（需求 5.3/5.4）
# ──────────────────────────────────────────────

def test_address_injected_into_profile_context(brain, started, session):
    elder_id, _ = started("王秀英")
    _enter_setup(brain, elder_id)
    brain.process_turn(elder_id, "我家住在浦东新区滨江大道4600号")
    brain.process_turn(elder_id, "对")
    elder = brain._profiles.get(elder_id)
    ctx = brain._profiles.build_context(elder)
    assert "浦东新区滨江大道4600号" in ctx


def test_empty_field_says_no_record(brain, started):
    elder_id, _ = started("王秀英")  # 无住址
    elder = brain._profiles.get(elder_id)
    ctx = brain._profiles.build_context(elder)
    assert "暂无记录" in ctx  # 住址空 → 暂无记录，不编造


# ──────────────────────────────────────────────
# 槽位契约单测（需求 4/5，覆盖结构化对象字段）
# ──────────────────────────────────────────────

def test_slot_registry_all_fields():
    assert set(SLOT_REGISTRY) == {"address", "allergies", "conditions", "contacts", "medication"}


def test_resolve_field_aliases():
    assert resolve_field("住址") == "address"
    assert resolve_field("过敏史") == "allergies"
    assert resolve_field("紧急联系人") == "contacts"
    assert resolve_field("用药") == "medication"
    assert resolve_field("不存在的") is None


def test_contact_slot_validate_and_apply(session, elders):
    elder = elders["王秀英"]
    spec = SLOT_REGISTRY["contacts"]
    # 缺电话 → 校验失败（追问补槽）
    v = spec.parse({"name": "王芳", "relation": "女儿"}, "")
    ok, msg = spec.validate(v)
    assert ok is False and "电话" in msg
    # 补齐电话 → 通过并落库
    v2 = spec.parse({"name": "王芳", "relation": "女儿", "phone": "13800000009"}, "")
    ok2, _ = spec.validate(v2)
    assert ok2 is True
    svc = MedicationService(session, __import__("app.notify.notifier", fromlist=["ConsoleNotifier"]).ConsoleNotifier(session, echo=False))
    ctx = SlotApplyContext(session=session, medication=svc)
    spec.apply(ctx, elder, v2)
    session.flush()
    assert any(c["name"] == "王芳" and c["phone"] == "13800000009" for c in elder.emergency_contacts)


def test_medication_slot_applies_via_service(session, elders):
    elder = elders["李桂芳"]
    before = len(elder.medications)
    spec = SLOT_REGISTRY["medication"]
    v = spec.parse({"name": "钙片", "dose": "1片", "times": "08:00", "timing": "after_meal"}, "")
    ok, _ = spec.validate(v)
    assert ok is True
    from app.notify.notifier import ConsoleNotifier
    ctx = SlotApplyContext(session=session, medication=MedicationService(session, ConsoleNotifier(session, echo=False)))
    spec.apply(ctx, elder, v)
    session.flush()
    session.refresh(elder)
    assert len(elder.medications) == before + 1
    assert any(m.name == "钙片" for m in elder.medications)


def test_medication_slot_requires_dose_and_time(session):
    spec = SLOT_REGISTRY["medication"]
    v = spec.parse({"name": "钙片"}, "")
    ok, msg = spec.validate(v)
    assert ok is False


# ──────────────────────────────────────────────
# 安全语义不变（需求 7，P5）
# ──────────────────────────────────────────────

def test_emergency_takes_priority_over_setup(brain, started, notifier):
    """P5：设置模式中出现求救词，仍走紧急，不被设置流程抢占。"""
    elder_id, _ = started("李桂芳")
    _enter_setup(brain, elder_id)
    r = brain.process_turn(elder_id, "救命，我喘不上气了")
    assert r.level == Level.EMERGENCY.value


def test_pending_emergency_not_preempted_by_setup(brain, started):
    """紧急待确认优先于设置模式的槽位确认。"""
    elder_id, _ = started("李桂芳")
    _enter_setup(brain, elder_id)
    # 触发告警 → 进入紧急待确认
    r1 = brain.process_turn(elder_id, "我这胸口有点闷")
    assert r1.level == Level.ALERT.value
    assert brain._runtimes[elder_id].pending_emergency_id is not None
    # 下一轮确认不适：走紧急确认，而非设置流程
    r2 = brain.process_turn(elder_id, "是的，特别难受")
    assert r2.data.get("escalated") is True


# ──────────────────────────────────────────────
# 高可信保护：uncertain 不取代 certain/setup（需求 6，P6）
# ──────────────────────────────────────────────

def test_certain_not_superseded_by_uncertain(session, elders):
    from app.memory.conflict import MemoryConflictResolver
    from app.memory.notes import ExtractedNote

    elder = elders["李桂芳"]
    resolver = MemoryConflictResolver(session)
    certain = ExtractedNote(True, "喜欢听越剧", "", [], "other")
    row_certain = resolver.upsert_note(elder.id, certain, confidence="certain", source="setup")
    # 存疑抽取，内容高度重叠
    uncertain = ExtractedNote(True, "喜欢听越剧吗", "", [], "other")
    resolver.upsert_note(elder.id, uncertain, confidence="uncertain")
    session.refresh(row_certain)
    assert row_certain.active is True  # certain/setup 未被 uncertain 取代
