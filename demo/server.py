"""网页演示服务（FastAPI）。

本地：  python -m demo.server            -> http://127.0.0.1:8000
部署：  uvicorn demo.server:app --host 0.0.0.0 --port 8000   （在仓库根目录执行）

环境变量（均可选）：
  DEMO_ACCESS_CODE      设置后整站启用 HTTP Basic 认证（用户名任意，密码=该值）。
                        公网部署且配置了真实 LLM key 时强烈建议设置，否则任何人都能消耗你的额度。
  DEMO_MAX_SESSIONS     同时在线的访客沙箱上限，默认 30
  DEMO_SESSION_TTL_MIN  沙箱空闲回收时间（分钟），默认 30
  DEMO_MAX_TURNS        单个沙箱最多对话轮数，默认 80
  LLM_PROVIDER / DEEPSEEK_API_KEY / OPENAI_API_KEY ...  与主程序一致，未配置则用离线模板回复
"""
from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from app.agent.llm import FakeLLMClient, get_llm
from demo.sandbox import (
    Sandbox,
    SandboxError,
    SandboxFull,
    SandboxPool,
    TurnLimitReached,
)
from demo.scenarios import SCENARIOS

_STATIC = Path(__file__).parent / "static"
_ACCESS_CODE = os.getenv("DEMO_ACCESS_CODE") or ""
_MAX_TEXT = 200
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

_llm = get_llm()
_pool = SandboxPool(
    _llm,
    max_sessions=int(os.getenv("DEMO_MAX_SESSIONS", "30")),
    ttl_sec=int(os.getenv("DEMO_SESSION_TTL_MIN", "30")) * 60,
    max_turns=int(os.getenv("DEMO_MAX_TURNS", "80")),
)

_basic = HTTPBasic(auto_error=False)


def _require_auth(cred: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    if not _ACCESS_CODE:
        return
    ok = cred is not None and secrets.compare_digest(
        cred.password.encode("utf-8"), _ACCESS_CODE.encode("utf-8")
    )
    if not ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "需要访问口令",
            headers={"WWW-Authenticate": 'Basic realm="demo"'},
        )


app = FastAPI(
    title="老人看护机器人演示",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
# 除 /healthz 外的所有路由都挂在带鉴权的 router 上
router = APIRouter(dependencies=[Depends(_require_auth)])


def _sandbox(sid: str) -> Sandbox:
    sb = _pool.get(sid)
    if sb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "演示会话已过期，请重新开始")
    return sb


def _guard(fn, *args):
    try:
        return fn(*args)
    except TurnLimitReached as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc
    except SandboxError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except ValueError as exc:  # 如 Brain.get_runtime 找不到老人
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


class ChatIn(BaseModel):
    elder_id: int
    text: str = Field(min_length=1, max_length=_MAX_TEXT)


class ElderIn(BaseModel):
    elder_id: int


class RemindIn(ElderIn):
    time: str


class PulseIn(BaseModel):
    minutes: int = Field(default=60, ge=1, le=24 * 60)


# ---- 页面 ----
@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


# 健康检查：不鉴权，供部署平台探活（不泄露任何信息）
@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


# ---- API ----
@router.get("/api/scenarios")
def scenarios() -> dict:
    return SCENARIOS


@router.post("/api/sessions")
def create_session() -> dict:
    try:
        sid, sb = _pool.create()
    except SandboxFull as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return {
        "sid": sid,
        "llm": "offline" if isinstance(_llm, FakeLLMClient) else "online",
        "max_text": _MAX_TEXT,
        "elders": sb.elders(),
    }


@router.delete("/api/sessions/{sid}", status_code=204, response_class=Response)
def delete_session(sid: str) -> Response:
    _pool.drop(sid)
    return Response(status_code=204)


@router.post("/api/sessions/{sid}/chat")
def chat(sid: str, body: ChatIn) -> dict:
    return _guard(_sandbox(sid).chat, body.elder_id, body.text.strip())


@router.post("/api/sessions/{sid}/remind")
def remind(sid: str, body: RemindIn) -> dict:
    if not _HHMM.match(body.time):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "时间格式应为 HH:MM")
    texts = _guard(_sandbox(sid).remind, body.elder_id, body.time)
    return {"texts": texts}


@router.post("/api/sessions/{sid}/silence")
def silence(sid: str, body: ElderIn) -> dict:
    return {"reply": _guard(_sandbox(sid).silence, body.elder_id)}


@router.post("/api/sessions/{sid}/pulse", status_code=204, response_class=Response)
def pulse(sid: str, body: PulseIn) -> Response:
    _guard(_sandbox(sid).pulse, body.minutes)
    return Response(status_code=204)


@router.get("/api/sessions/{sid}/state")
def state(sid: str, elder_id: int) -> dict:
    return _guard(_sandbox(sid).state, elder_id)


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
