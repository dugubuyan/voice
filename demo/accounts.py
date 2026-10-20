"""账号与登录态：注册 / 登录 / 会话令牌。

- 密码：scrypt 加盐哈希（标准库，无额外依赖），常量时间比较。
- 登录态：随机令牌放 HttpOnly Cookie；库里只存令牌的 SHA-256，泄库也无法直接冒用。
- 账号库独立于每个用户的记忆库（用独立的 DeclarativeBase，不会混进用户库的建表）。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Integer, String, create_engine, delete, select, update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from demo import settings


class AccountError(Exception):
    """可直接展示给用户的账号类错误。"""


class AccountBase(DeclarativeBase):
    pass


class User(AccountBase):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)  # 小写存储
    display_name: Mapped[str] = mapped_column(String(32))
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AuthSession(AccountBase):
    __tablename__ = "auth_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class InviteCode(AccountBase):
    """一次性邀请码：库里只存哈希，明文只在生成时显示一次。"""

    __tablename__ = "invite_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    note: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    used_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # users.id
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


@dataclass(frozen=True)
class Account:
    id: int
    username: str
    display_name: str


settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
_engine = create_engine(
    f"sqlite:///{settings.DATA_DIR / 'accounts.db'}",
    connect_args={"check_same_thread": False, "timeout": 15},
    future=True,
)
AccountBase.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


# ---------------- 密码 ----------------
_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        expected = bytes.fromhex(dk)
        cand = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
        return hmac.compare_digest(cand, expected)
    except Exception:
        return False


# 用户不存在时也做一次等价的哈希校验，避免通过响应时间探测用户名是否存在
_DUMMY_HASH = hash_password("dummy-password")


# ---------------- 登录失败限流（按用户名，进程内） ----------------
class AttemptLimiter:
    """滑动窗口失败计数：窗口内失败达到上限即拦截。"""

    def __init__(self, max_fail: int = 5, window_sec: int = 900) -> None:
        self._max = max_fail
        self._window = window_sec
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> list[float]:
        cutoff = time.monotonic() - self._window
        recent = [t for t in self._fails.get(key, []) if t > cutoff]
        if recent:
            self._fails[key] = recent
        else:
            self._fails.pop(key, None)
        return recent

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self._max

    def fail(self, key: str) -> None:
        with self._lock:
            recent = self._recent(key)
            recent.append(time.monotonic())
            self._fails[key] = recent

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


_limiter = AttemptLimiter(max_fail=5)                 # 登录：按用户名
invite_limiter = AttemptLimiter(max_fail=10)          # 邀请码猜测：按来源 IP


class TooManyAttempts(AccountError):
    pass


# ---------------- 注册 / 登录 ----------------
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def clean_name(name: str) -> str:
    """称呼会进入 LLM 的系统提示：压缩空白、去掉换行与控制字符，限长。"""
    cleaned = "".join(ch for ch in " ".join((name or "").split()) if ch.isprintable())
    return cleaned[:20]


def create_user(username: str, password: str, display_name: str) -> Account:
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise AccountError("用户名需 3~32 位，只能包含字母、数字、下划线、点和短横线")
    if not 8 <= len(password or "") <= 128:
        raise AccountError("密码长度需为 8~128 位")
    display_name = clean_name(display_name)
    if not display_name:
        raise AccountError("请填写称呼")
    with _Session() as s:
        user = User(
            username=username.lower(),
            display_name=display_name,
            password_hash=hash_password(password),
        )
        s.add(user)
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            raise AccountError("用户名已被占用") from None
        return Account(user.id, user.username, user.display_name)


def delete_user(user_id: int) -> None:
    """注册中途失败时回滚账号（同时清理其登录态）。"""
    with _Session() as s:
        s.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
        s.execute(delete(User).where(User.id == user_id))
        s.commit()


def authenticate(username: str, password: str) -> Account | None:
    key = (username or "").strip().lower()
    if _limiter.blocked(key):
        raise TooManyAttempts("尝试次数过多，请 15 分钟后再试")
    with _Session() as s:
        user = s.execute(select(User).where(User.username == key)).scalars().first()
        ok = verify_password(password or "", user.password_hash if user else _DUMMY_HASH)
        if user is None or not ok:
            _limiter.fail(key)
            return None
        _limiter.reset(key)
        return Account(user.id, user.username, user.display_name)


# ---------------- 登录态 ----------------
def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _Session() as s:
        now = datetime.now()
        s.execute(delete(AuthSession).where(AuthSession.expires_at < now))  # 顺手清理过期
        s.add(
            AuthSession(
                token_hash=_digest(token),
                user_id=user_id,
                expires_at=now + timedelta(days=settings.SESSION_DAYS),
            )
        )
        s.commit()
    return token


def account_for_token(token: str | None) -> Account | None:
    if not token:
        return None
    with _Session() as s:
        row = s.execute(
            select(AuthSession).where(AuthSession.token_hash == _digest(token))
        ).scalars().first()
        if row is None:
            return None
        if row.expires_at < datetime.now():
            s.delete(row)
            s.commit()
            return None
        user = s.get(User, row.user_id)
        if user is None:
            return None
        return Account(user.id, user.username, user.display_name)


def revoke_session(token: str | None) -> None:
    if not token:
        return
    with _Session() as s:
        s.execute(delete(AuthSession).where(AuthSession.token_hash == _digest(token)))
        s.commit()


# ---------------- 一次性邀请码 ----------------
# 去掉易混字符（0/O、1/I/L），12 位 31 进制 ≈ 59 bit，叠加按 IP 的猜错限流，无法暴力枚举
_INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_INVITE_LEN = 12


def _normalize_code(code: str) -> str:
    """忽略大小写、连字符和空格，方便手输 / 粘贴。"""
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def _format_code(raw: str) -> str:
    return "-".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def generate_invites(count: int, note: str = "") -> list[str]:
    """生成 count 个新邀请码，返回明文（仅此一次可见）。"""
    codes: list[str] = []
    with _Session() as s:
        while len(codes) < count:
            raw = "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(_INVITE_LEN))
            s.add(InviteCode(code_hash=_digest(raw), note=note[:64]))
            try:
                s.flush()
            except IntegrityError:  # 极小概率撞库：重抽
                s.rollback()
                continue
            codes.append(_format_code(raw))
        s.commit()
    return codes


def consume_invite(code: str) -> int | None:
    """原子地占用一个未使用且未作废的邀请码，成功返回其 id；无效 / 已用返回 None。

    用单条条件 UPDATE 保证并发下同一个码只可能被一个请求占到。
    """
    norm = _normalize_code(code)
    if len(norm) != _INVITE_LEN:
        return None
    h = _digest(norm)
    with _Session() as s:
        res = s.execute(
            update(InviteCode)
            .where(
                InviteCode.code_hash == h,
                InviteCode.used_at.is_(None),
                InviteCode.revoked.is_(False),
            )
            .values(used_at=datetime.now())
        )
        s.commit()
        if res.rowcount != 1:
            return None
        return s.execute(select(InviteCode.id).where(InviteCode.code_hash == h)).scalar_one()


def release_invite(invite_id: int) -> None:
    """注册没成功（用户名重复、建库失败等）时把码还回去，用户可以直接重试。"""
    with _Session() as s:
        s.execute(
            update(InviteCode)
            .where(InviteCode.id == invite_id, InviteCode.used_by.is_(None))
            .values(used_at=None)
        )
        s.commit()


def bind_invite(invite_id: int, user_id: int) -> None:
    with _Session() as s:
        s.execute(update(InviteCode).where(InviteCode.id == invite_id).values(used_by=user_id))
        s.commit()


def revoke_invite(invite_id: int) -> bool:
    """作废一个尚未使用的邀请码；已使用的不能作废。"""
    with _Session() as s:
        res = s.execute(
            update(InviteCode)
            .where(InviteCode.id == invite_id, InviteCode.used_at.is_(None))
            .values(revoked=True)
        )
        s.commit()
        return res.rowcount == 1


def list_invites() -> list[dict]:
    with _Session() as s:
        rows = s.execute(select(InviteCode).order_by(InviteCode.id)).scalars().all()
        names = {u.id: u.username for u in s.execute(select(User)).scalars()}
        out = []
        for r in rows:
            status = "已使用" if r.used_at else ("已作废" if r.revoked else "未使用")
            out.append({
                "id": r.id, "status": status, "note": r.note,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
                "used_by": names.get(r.used_by, "") if r.used_by else "",
                "used_at": r.used_at.strftime("%Y-%m-%d %H:%M") if r.used_at else "",
            })
        return out


def unused_invite_count() -> int:
    with _Session() as s:
        return len(
            s.execute(
                select(InviteCode.id).where(
                    InviteCode.used_at.is_(None), InviteCode.revoked.is_(False)
                )
            ).all()
        )
