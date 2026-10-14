# 网页演示（临时，演示结束后整体删除）

把 `python -m app.cli` 的对话演示搬到浏览器：选老人、聊天、模拟到点吃药 / 沉默 / 过一小时，
右侧实时显示系统记住了什么、检测到了什么（紧急、通知、健康记录、用药、记忆）。

## 本地运行（仓库根目录）

```bash
pip install -r requirements.txt -r demo/requirements.txt
python -m demo.server            # http://127.0.0.1:8000
```

未配置 LLM key 时自动使用离线模板回复，页面顶部会标明。

## 部署

```bash
docker build -f demo/Dockerfile -t eldercare-demo .
docker run -p 8000:8000 \
  -e DEMO_ACCESS_CODE=<访问口令> \
  -e LLM_PROVIDER=deepseek -e DEEPSEEK_API_KEY=<key> \
  eldercare-demo
```

不用 Docker：在仓库根目录执行 `uvicorn demo.server:app --host 0.0.0.0 --port 8000`。
健康检查路径：`/healthz`。放在 HTTPS 反向代理（Nginx / Caddy / 平台自带）后面。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DEMO_ACCESS_CODE` | 空（不鉴权） | 设置后整站 HTTP Basic 认证，用户名任意，密码为该值 |
| `DEMO_MAX_SESSIONS` | 30 | 同时在线的访客沙箱数 |
| `DEMO_SESSION_TTL_MIN` | 30 | 沙箱空闲回收时间（分钟） |
| `DEMO_MAX_TURNS` | 80 | 单个沙箱最多对话轮数 |
| `LLM_PROVIDER` 等 | 同主程序 | 见 `.env.example` |

**公网部署且配置了真实 LLM key 时务必设置 `DEMO_ACCESS_CODE`**，否则任何知道网址的人都能消耗你的额度。
需要单进程运行（沙箱存放在进程内存里）；不要加 `--workers`。

## 结构与隔离

```
demo/
  server.py      FastAPI：页面 + JSON API，可选 Basic 认证
  sandbox.py     每位访客一个沙箱：独立内存库 + 种子老人 + Brain；容量上限 / 空闲回收 / 轮数上限
  scenarios.py   随机生活场景话术
  static/index.html  单页前端（原生 JS，无构建步骤）
  Dockerfile  requirements.txt  README.md
```

- 依赖方向单向：`demo -> app / seed`。`app/`、`seed/`、`tests/` 不引用 `demo`，也没有为演示改动过。
- 不读写 `elder_care.db`；访客之间互不可见，沙箱过期或点「重新开始」即清除。

## 演示结束后删除

```bash
rm -rf demo
```

仅此一步，无其它残留：核心代码、测试、依赖文件都没有演示相关改动。
若部署过，同时下线容器 / 服务并撤销为演示配置的 LLM key。
