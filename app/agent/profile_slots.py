"""一等结构化字段的槽位注册表（profile-setup-mode）。

每个一等字段对应一个 SlotSpec，统一契约：
  parse    从 NLU 槽位 + 原文解析出候选值（拿不准返回 None）
  validate 校验候选值，返回 (ok, error_msg)
  display  复述念回文本（用 NLU 抽取的原值，不经 LLM 改写，C1）
  apply    确定性 setter：把值写入结构化存储（合同的落库点）

新增一等字段 = 在此注册一个 SlotSpec + 加测试，不改模式/确认内核（C5）。

apply 的落库以调用方 session.commit() 的成功为回执；写成功后才回"已记好"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.medication.service import MedicationService
from app.models import Elder


@dataclass
class SlotApplyContext:
    """apply 落库所需资源。medication 走既有服务，其余直写 Elder。"""

    session: Session
    medication: MedicationService


@dataclass
class SlotSpec:
    field: str                                              # NLU 里的字段名
    label: str                                              # 中文名，用于话术
    parse: Callable[[dict, str], object]                    # (slots, text) -> value | None
    validate: Callable[[object], tuple[bool, str]]          # value -> (ok, error_msg)
    display: Callable[[object], str]                        # value -> 复述文本
    apply: Callable[["SlotApplyContext", Elder, object], None]  # 落库


# ──────────────────────────────────────────────
# 通用解析工具
# ──────────────────────────────────────────────

def _as_list(raw) -> list[str]:
    """把字符串（顿号/逗号分隔）或列表规整成去空去重的字符串列表。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw]
    else:
        s = str(raw)
        for sep in ("、", "，", ",", "和", "跟", "以及"):
            s = s.replace(sep, "|")
        items = [x.strip() for x in s.split("|")]
    out: list[str] = []
    for it in items:
        if it and it not in out:
            out.append(it)
    return out


def _slot_val(slots: dict, *keys: str):
    """按优先级从 slots 里取第一个非空键值。"""
    for k in keys:
        v = slots.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


# ──────────────────────────────────────────────
# 各字段的 parse / validate / display / apply
# ──────────────────────────────────────────────

# ---- 住址（标量 str）----
def _parse_address(slots: dict, text: str):
    v = _slot_val(slots, "value", "address")
    return str(v).strip() if v is not None else None


def _validate_address(v) -> tuple[bool, str]:
    s = str(v or "").strip()
    if not s:
        return False, "我没太听清地址，您再说一遍具体到门牌号好吗？"
    if len(s) > 256:
        return False, "这个地址有点太长了，您说个简短些的住址就行。"
    return True, ""


def _apply_address(ctx: SlotApplyContext, elder: Elder, v) -> None:
    elder.address = str(v).strip()
    ctx.session.flush()


# ---- 过敏史 / 慢性病（list[str]）----
def _parse_list(slots: dict, text: str):
    v = _slot_val(slots, "value", "items")
    items = _as_list(v if v is not None else text)
    return items or None


def _validate_list(v) -> tuple[bool, str]:
    if not isinstance(v, list) or not v:
        return False, "这个我没记全，您再说一遍具体是哪些？"
    return True, ""


def _display_list(v) -> str:
    return "、".join(str(x) for x in v)


def _apply_allergies(ctx: SlotApplyContext, elder: Elder, v) -> None:
    elder.allergies = list(v)
    ctx.session.flush()


def _apply_conditions(ctx: SlotApplyContext, elder: Elder, v) -> None:
    elder.conditions = list(v)
    ctx.session.flush()


# ---- 紧急联系人（结构化对象，追加/按姓名更新）----
def _parse_contact(slots: dict, text: str):
    name = _slot_val(slots, "name", "contact_name")
    if not name:
        return None
    return {
        "name": str(name).strip(),
        "relation": str(_slot_val(slots, "relation") or "").strip(),
        "phone": str(_slot_val(slots, "phone") or "").strip(),
        "priority": int(_slot_val(slots, "priority") or 1),
    }


def _validate_contact(v) -> tuple[bool, str]:
    if not isinstance(v, dict) or not v.get("name"):
        return False, "这位联系人叫什么名字呢？"
    if not v.get("phone"):
        return False, "还需要联系电话，您把号码也告诉我好吗？"
    return True, ""


def _display_contact(v) -> str:
    rel = f"（{v['relation']}）" if v.get("relation") else ""
    return f"{v['name']}{rel}，电话{v['phone']}"


def _apply_contact(ctx: SlotApplyContext, elder: Elder, v) -> None:
    contacts = list(elder.emergency_contacts or [])
    # 按姓名更新（recency-wins），否则追加
    contacts = [c for c in contacts if c.get("name") != v["name"]]
    contacts.append(v)
    contacts.sort(key=lambda c: c.get("priority", 99))
    elder.emergency_contacts = contacts
    ctx.session.flush()


# ---- 用药计划（结构化对象，经 MedicationService 落库）----
def _parse_medication(slots: dict, text: str):
    name = _slot_val(slots, "name", "med_name")
    if not name:
        return None
    times = _as_list(_slot_val(slots, "times", "time"))
    return {
        "name": str(name).strip(),
        "dose": str(_slot_val(slots, "dose") or "").strip(),
        "times": times,
        "timing": str(_slot_val(slots, "timing") or "anytime").strip() or "anytime",
    }


def _validate_medication(v) -> tuple[bool, str]:
    if not isinstance(v, dict) or not v.get("name"):
        return False, "这是什么药呢？您把药名告诉我。"
    if not v.get("dose"):
        return False, "每次吃多少量呢？比如一片、一粒。"
    if not v.get("times"):
        return False, "每天几点吃呢？您说个时间，比如早上八点。"
    return True, ""


def _display_medication(v) -> str:
    times = "、".join(v["times"]) if v.get("times") else "未定时间"
    timing = {"before_meal": "饭前", "after_meal": "饭后", "anytime": ""}.get(v.get("timing", ""), "")
    tail = f"，{timing}吃" if timing else ""
    return f"{v['name']}，{v['dose']}，每天{times}{tail}"


def _apply_medication(ctx: SlotApplyContext, elder: Elder, v) -> None:
    ctx.medication.create_plan(
        elder.id,
        name=v["name"],
        dose=v["dose"],
        times=v.get("times", []),
        timing=v.get("timing", "anytime"),
    )


# ──────────────────────────────────────────────
# 注册表
# ──────────────────────────────────────────────

SLOT_REGISTRY: dict[str, SlotSpec] = {
    "address": SlotSpec(
        field="address", label="家庭住址",
        parse=_parse_address, validate=_validate_address,
        display=lambda v: str(v), apply=_apply_address,
    ),
    "allergies": SlotSpec(
        field="allergies", label="过敏史",
        parse=_parse_list, validate=_validate_list,
        display=_display_list, apply=_apply_allergies,
    ),
    "conditions": SlotSpec(
        field="conditions", label="慢性病",
        parse=_parse_list, validate=_validate_list,
        display=_display_list, apply=_apply_conditions,
    ),
    "contacts": SlotSpec(
        field="contacts", label="紧急联系人",
        parse=_parse_contact, validate=_validate_contact,
        display=_display_contact, apply=_apply_contact,
    ),
    "medication": SlotSpec(
        field="medication", label="用药计划",
        parse=_parse_medication, validate=_validate_medication,
        display=_display_medication, apply=_apply_medication,
    ),
}


# 常用别名 → 规范字段名（NLU 或用户可能用不同说法）
_FIELD_ALIASES = {
    "住址": "address", "地址": "address", "家庭住址": "address", "家住": "address",
    "过敏": "allergies", "过敏史": "allergies",
    "慢病": "conditions", "慢性病": "conditions", "疾病": "conditions",
    "联系人": "contacts", "紧急联系人": "contacts", "家属": "contacts",
    "用药": "medication", "药": "medication", "吃药": "medication",
}


def resolve_field(name: str) -> str | None:
    """把 NLU 给的 field 名规整为注册表键；无法识别返回 None。"""
    if not name:
        return None
    key = str(name).strip()
    if key in SLOT_REGISTRY:
        return key
    return _FIELD_ALIASES.get(key)
