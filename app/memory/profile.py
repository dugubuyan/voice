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
        med_text = "、".join(f"{m.name}({m.dose})" for m in meds) or "无"
        return (
            f"老人姓名：{elder.name}\n"
            f"年龄：{elder.age}岁，性别：{elder.gender or '未知'}\n"
            f"慢性病：{conditions}\n"
            f"常用药物：{med_text}\n"
            f"家属联系人：{contact_text}\n"
            f"方言/语言：{elder.dialect}\n"
            f"偏好与习惯：{pref_text}\n"
            f"备注：{elder.notes or '无'}"
        )
