"""种子数据：虚构多位老人档案与用药计划，贯穿演示与测试。"""
from __future__ import annotations

from app.db import init_db, session_scope
from app.models import Elder, Medication

# 每位老人：(档案, [用药计划])
SEED = [
    {
        "elder": dict(
            name="李桂芳",
            age=72,
            gender="女",
            conditions=["高血压", "2型糖尿病"],
            emergency_contacts=[
                {"name": "李梅", "relation": "女儿", "phone": "13800000001", "priority": 1},
                {"name": "社区卫生站", "relation": "社区", "phone": "0571-88880000", "priority": 2},
            ],
            dialect="普通话",
            preferences={"爱好": "养花、听越剧", "牵挂": "外孙小宝在上海读大学", "口味": "爱吃甜, 需控糖"},
            notes="独居，女儿每周探望一次。听力略下降，说话需慢些。",
        ),
        "meds": [
            dict(name="氨氯地平", dose="1片", times=["08:00"], timing="after_meal", color_or_shape="白色小圆片", notes="降压药"),
            dict(name="二甲双胍", dose="1片", times=["08:00", "18:00"], timing="after_meal", color_or_shape="白色椭圆片", notes="降糖药"),
        ],
    },
    {
        "elder": dict(
            name="张建国",
            age=78,
            gender="男",
            conditions=["冠心病", "轻度阿尔茨海默"],
            emergency_contacts=[
                {"name": "张伟", "relation": "儿子", "phone": "13800000002", "priority": 1},
                {"name": "王护工", "relation": "护工", "phone": "13800000003", "priority": 2},
            ],
            dialect="普通话",
            preferences={"爱好": "下象棋、看新闻", "牵挂": "老伴前年过世", "习惯": "早睡早起"},
            notes="有记忆减退，需反复温和提醒。随身有硝酸甘油。",
        ),
        "meds": [
            dict(name="阿司匹林", dose="1片", times=["08:00"], timing="after_meal", color_or_shape="肠溶片", notes="抗血小板"),
            dict(name="硝酸甘油", dose="遵医嘱", times=[], timing="anytime", color_or_shape="小白片", notes="胸痛时舌下含服，仅提示不代替医嘱"),
        ],
    },
    {
        "elder": dict(
            name="王秀英",
            age=68,
            gender="女",
            conditions=["骨质疏松", "失眠"],
            emergency_contacts=[
                {"name": "王芳", "relation": "女儿", "phone": "13800000004", "priority": 1},
            ],
            dialect="四川话",
            preferences={"爱好": "跳广场舞、打麻将", "牵挂": "小孙女", "口味": "喜辣"},
            notes="性格开朗，社交活跃。近来夜里常起夜，需注意防跌倒。",
        ),
        "meds": [
            dict(name="碳酸钙", dose="1片", times=["12:00"], timing="after_meal", color_or_shape="大白片", notes="补钙"),
            dict(name="阿法骨化醇", dose="1粒", times=["08:00"], timing="after_meal", color_or_shape="软胶囊", notes="促钙吸收"),
        ],
    },
    {
        "elder": dict(
            name="陈明远",
            age=81,
            gender="男",
            conditions=["慢阻肺", "高血压"],
            emergency_contacts=[
                {"name": "陈晓", "relation": "儿子", "phone": "13800000005", "priority": 1},
                {"name": "李邻居", "relation": "邻居", "phone": "13800000006", "priority": 2},
            ],
            dialect="普通话",
            preferences={"爱好": "书法、养鸟", "牵挂": "老家的宅子", "习惯": "午后小睡"},
            notes="呼吸不畅时易焦虑，需重点关注气促相关表述。",
        ),
        "meds": [
            dict(name="缬沙坦", dose="1粒", times=["08:00"], timing="before_meal", color_or_shape="胶囊", notes="降压"),
            dict(name="沙丁胺醇吸入剂", dose="按需", times=["09:00", "21:00"], timing="anytime", color_or_shape="蓝色吸入器", notes="平喘"),
        ],
    },
]


def seed(reset: bool = False) -> list[int]:
    """写入种子数据，返回创建的老人 id 列表。"""
    init_db()
    ids: list[int] = []
    with session_scope() as session:
        if reset:
            session.query(Medication).delete()
            session.query(Elder).delete()
            session.flush()
        for item in SEED:
            existing = session.query(Elder).filter_by(name=item["elder"]["name"]).first()
            if existing and not reset:
                ids.append(existing.id)
                continue
            elder = Elder(**item["elder"])
            session.add(elder)
            session.flush()
            for m in item["meds"]:
                session.add(Medication(elder_id=elder.id, **m))
            ids.append(elder.id)
    return ids


if __name__ == "__main__":
    created = seed(reset=True)
    print(f"已写入 {len(created)} 位老人档案：id={created}")
