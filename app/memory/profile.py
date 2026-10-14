"""档案记忆（静态）：加载老人画像，构建注入 system prompt 的上下文。"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Elder


class ProfileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, elder_id: int) -> Elder | None:
        return self._session.get(Elder, elder_id)

    def get_by_name(self, name: str) -> Elder | None:
        return self._session.execute(
            select(Elder).where(Elder.name == name)
        ).scalars().first()

    def list_all(self) -> list[Elder]:
        return list(self._session.execute(select(Elder)).scalars().all())

    def build_context(self, elder: Elder) -> str:
        """把档案整理成简洁的中文上下文文本。"""
        conditions = "、".join(elder.conditions or []) or "无明确记录"
        contacts = elder.emergency_contacts or []
        contact_text = "、".join(
            f"{c.get('name')}（{c.get('relation')}）" for c in contacts
        ) or "无"
        prefs = elder.preferences or {}
        pref_text = "；".join(f"{k}：{v}" for k, v in prefs.items()) or "暂无"
        meds = elder.medications or []
        _timing_map = {"before_meal": "饭前", "after_meal": "饭后", "anytime": "随时"}

        def _med_str(m) -> str:
            times = m.times or []
            timing_cn = _timing_map.get(m.timing or "", "")
            if not times:
                return f"{m.name}({m.dose})"
            parts = [m.dose, f"每天{'、'.join(times)}"]
            if timing_cn:
                parts.append(timing_cn)
            return f"{m.name}({', '.join(parts)})"

        med_text = "、".join(_med_str(m) for m in meds) or "无"
        address_text = elder.address or "暂无记录"
        allergies_text = "、".join(elder.allergies or []) or "暂无已知过敏"
        return (
            f"老人姓名：{elder.name}\n"
            f"年龄：{elder.age}岁，性别：{elder.gender or '未知'}\n"
            f"家庭住址：{address_text}\n"
            f"慢性病：{conditions}\n"
            f"过敏史：{allergies_text}\n"
            f"常用药物：{med_text}\n"
            f"家属联系人：{contact_text}\n"
            f"方言/语言：{elder.dialect}\n"
            f"偏好与习惯：{pref_text}\n"
            f"备注：{elder.notes or '无'}\n"
            f"（以上为结构化档案记录，回答住址/过敏史/用药等务必以此为准，没有记录就如实说暂无，不要编造）"
        )
