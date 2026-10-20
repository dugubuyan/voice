"""陪伴机器人网页服务（FastAPI）：注册 / 登录 + 每个用户独立的会话与记忆。

启动（仓库根目录，务必单进程，不要加 --workers）：
  python3 -m uvicorn demo.server:app --host 0.0.0.0 --port 8080

环境变量（均可选，详见 demo/README.md）：
  DATA_DIR              数据目录，默认 <仓库>/data（accounts.db + users/<id>.db）
  注册需要一次性邀请码，用 `python -m demo.invites new -n 5` 生成（REGISTER_OPEN=true 可开放注册，不推荐）
  SESSION_DAYS          登录有效期（天），默认 30
  USER_DAILY_TURNS      单用户每天对话上限，默认 300
  WORKSPACE_TTL_MIN     用户记忆在内存中的保留时间（分钟），默认 60
  LLM_PROVIDER / DEEPSEEK_API_KEY / OPENAI_API_KEY / VOLCANO_API_KEY ...  与主程序一致
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.llm import FakeLLMClient, get_llm
from demo import accounts, settings, workspace
from demo.accounts import Account, AccountError, TooManyAttempts

_STATIC = Path(__file__).parent / "static"
_COOKIE = "session"

_llm = get_llm()
_pool = workspace.WorkspacePool(_llm)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    if settings.REGISTRATION_MODE == "open":
        if not isinstance(_llm, FakeLLMClient):
            print("[warn] 开放注册且已接入真实 LLM：任何人都能注册并消耗你的额度，建议去掉 REGISTER_OPEN。")
    else:
        print(
            f"[info] 注册需一次性邀请码，当前未使用 {accounts.unused_invite_count()} 个；"
            "生成：python -m demo.invites new -n 5"
        )
    stop = _pool.start_pulse_thread()
    try:
        yield
    finally:
        stop.set()
        _pool.close_all()


app = FastAPI(
    title="陪伴机器人", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif request.url.path == "/":
        # 页面每次都向服务器校验，部署新版本后用户不会继续看到旧页面
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# ---------------- 登录态 ----------------
def _set_cookie(request: Request, response: Response, token: str) -> None:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        _COOKIE, token,
        max_age=settings.SESSION_DAYS * 86400,
        httponly=True, samesite="lax", secure=(proto == "https"), path="/",
    )


def current_user(request: Request) -> Account:
    acct = accounts.account_for_token(request.cookies.get(_COOKIE))
    if acct is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "请先登录")
    return acct


def _me(acct: Account) -> dict:
    return {
        "username": acct.username,
        "name": acct.display_name,
        "llm": "offline" if isinstance(_llm, FakeLLMClient) else "online",
        "max_text": settings.MAX_TEXT,
    }


# ---------------- 请求体 ----------------
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=20)
    age: int = Field(ge=1, le=120)
    invite_code: str = Field(default="", max_length=128)


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=settings.MAX_TEXT)


# ---------------- 页面 ----------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


# ---------------- 账号 ----------------
@app.get("/api/auth-config")
def auth_config() -> dict:
    """前端据此决定是否显示注册、邀请码是否必填（不泄露邀请码本身）。"""
    return {"registration": settings.REGISTRATION_MODE}


@app.post("/api/register")
def register(body: RegisterIn, request: Request, response: Response) -> dict:
    invite_id: int | None = None
    if settings.REGISTRATION_MODE == "invite":
        ip = request.client.host if request.client else "unknown"
        if accounts.invite_limiter.blocked(ip):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "尝试次数过多，请稍后再试")
        # 原子占用：同一个码并发提交也只会有一个请求成功
        invite_id = accounts.consume_invite(body.invite_code)
        if invite_id is None:
            accounts.invite_limiter.fail(ip)
            raise HTTPException(status.HTTP_403_FORBIDDEN, "邀请码无效或已被使用")

    def _abort(code: int, msg: str, acct: Account | None = None) -> HTTPException:
        # 注册没成功：回滚半成品账号，并把邀请码还回去，用户改正后可直接重试
        if acct is not None:
            accounts.delete_user(acct.id)
            workspace.remove_data(acct.id)
        if invite_id is not None:
            accounts.release_invite(invite_id)
        return HTTPException(code, msg)

    try:
        acct = accounts.create_user(body.username, body.password, body.name)
    except AccountError as exc:
        raise _abort(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception:
        raise _abort(status.HTTP_500_INTERNAL_SERVER_ERROR, "注册失败，请稍后再试") from None
    try:
        workspace.provision(acct.id, acct.display_name, body.age)
    except Exception:
        raise _abort(status.HTTP_500_INTERNAL_SERVER_ERROR, "注册失败，请稍后再试", acct) from None
    if invite_id is not None:
        accounts.bind_invite(invite_id, acct.id)
    _set_cookie(request, response, accounts.create_session(acct.id))
    return _me(acct)


@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response) -> dict:
    try:
        acct = accounts.authenticate(body.username, body.password)
    except TooManyAttempts as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc
    if acct is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    _set_cookie(request, response, accounts.create_session(acct.id))
    return _me(acct)


@app.post("/api/logout", status_code=204, response_class=Response)
def logout(request: Request) -> Response:
    accounts.revoke_session(request.cookies.get(_COOKIE))
    resp = Response(status_code=204)
    resp.delete_cookie(_COOKIE, path="/")
    return resp


@app.get("/api/me")
def me(acct: Account = Depends(current_user)) -> dict:
    return _me(acct)


# ---------------- 对话 ----------------
def _workspace(acct: Account) -> workspace.Workspace:
    try:
        return _pool.get(acct.id)
    except workspace.WorkspaceError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc


@app.post("/api/chat")
def chat(body: ChatIn, acct: Account = Depends(current_user)) -> dict:
    ws = _workspace(acct)
    try:
        return ws.chat(body.text.strip())
    except workspace.DailyLimitReached as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc


@app.get("/api/history")
def history(
    before: int | None = None, limit: int = 50, acct: Account = Depends(current_user)
) -> dict:
    limit = max(1, min(limit, 100))
    return {"messages": _workspace(acct).history(limit, before)}


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080"))
    )
