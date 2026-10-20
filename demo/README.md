# 陪伴机器人网页服务

浏览器里注册 / 登录后和机器人对话。每个用户有自己独立的对话历史和记忆（档案、情景、巩固、前瞻等），互相不可见。

## 运行（仓库根目录）

```bash
pip install -r requirements.txt -r demo/requirements.txt
python -m uvicorn demo.server:app --host 0.0.0.0 --port 8080
```

**必须单进程，不要加 `--workers`**：用户记忆在进程内缓存并用进程内锁串行，后台线程每 30 秒驱动一次记忆脉冲（巩固 / 前瞻过期 / 清理）。
健康检查：`/healthz`。请放在 HTTPS 反向代理后面（Cookie 会在 https 下自动带 `Secure`；代理需转发 `X-Forwarded-Proto`）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_DIR` | `<仓库>/data` | 数据目录，**务必持久化 / 备份** |
| `REGISTER_OPEN` | 空 | 设为 `true` 才开放注册（任何人可注册并消耗额度，不推荐）。默认需要一次性邀请码 |
| `SETUP_PASSCODE` | `0000` | 设置模式口令，对话里说“进入设置模式，口令xxxx”。**部署前务必修改** |
| `SESSION_DAYS` | 30 | 登录有效期（天） |
| `USER_DAILY_TURNS` | 300 | 单用户每天对话上限 |
| `WORKSPACE_TTL_MIN` | 60 | 用户记忆在内存中的保留时间（分钟），之后自动释放，下次对话从磁盘恢复 |
| `MAX_WORKSPACES` | 200 | 内存中同时保留的用户数 |
| `LLM_PROVIDER` / `DEEPSEEK_API_KEY` / `VOLCANO_API_KEY` 等 | 同主程序 | 见 `.env.example` |

## 邀请码

注册需要一次性邀请码：每个码只能注册一个账号，用过即失效。在服务器上、仓库根目录执行（与服务使用同一个 `DATA_DIR`，服务不用停）：

```bash
python -m demo.invites new -n 5 --note "内测第一批"   # 生成 5 个，明文只显示这一次
python -m demo.invites list                            # 查看 未使用 / 已使用(谁用的) / 已作废
python -m demo.invites revoke 3                        # 作废第 3 号（仅限未使用的）
```

- 库里只存哈希，没发出去的码丢了只能重新生成。
- 输入时忽略大小写和连字符；注册失败（如用户名重复）不会烧掉码，可直接重试。
- 同一来源 IP 猜错 10 次后 15 分钟内被拦截。

## 数据布局

```
data/
  accounts.db        账号（密码 scrypt 哈希）与登录态（只存令牌的 SHA-256）
  users/<id>.db      每个用户一个独立库：档案 + 全部记忆表 + chat_messages（页面对话历史，永久保存）
```

- 一个用户对应一位被陪护的人，注册时填写称呼和年龄，其余资料通过对话中的“设置模式”补充。
- 备份：停服务或用 `sqlite3 <file> ".backup <dest>"`，拷走整个 `data/`。
- 注销某个用户：删除 `users/<id>.db`，并在 `accounts.db` 里删除对应记录。
- 导出某用户的对话：`sqlite3 -header -csv data/users/<id>.db "select role,content,kind,level,at from chat_messages order by id"`。

## 结构

```
demo/
  server.py     FastAPI：页面 + 注册/登录/登出/对话/历史 API
  accounts.py   账号、密码哈希、登录态、登录限流
  workspace.py  每用户一个库 + Brain；内存缓存、空闲回收、后台记忆脉冲
  settings.py   环境变量
  static/index.html  单页前端（原生 JS，无构建步骤）
  sandbox.py scenarios.py  旧的内存演示实现，已不再使用
```

依赖方向单向：`demo -> app`。`app/`、`seed/`、`tests/` 不引用 `demo`。
