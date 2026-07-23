# 老人看护对话机器人（Pipecat + ElevenLabs）

本地单机、文本可演示的老人看护对话机器人业务闭环。核心三大模块：
**紧急检测 / 用药提醒 / 记忆**。语音层（Pipecat + ElevenLabs）已留骨架，
业务逻辑与交互层解耦，后续上语音时无需改动业务代码。

## 设计要点

- **业务与交互解耦**：`Brain.process_turn()` 是 CLI 与语音层共用的唯一入口。
  CLI 用打字，语音层用 STT/TTS，行为完全一致。
- **安全优先**：紧急检测、用药确认为**规则驱动**，不依赖 LLM，保证低延迟与确定性。
  LLM 只用于日常闲聊与对话摘要。
- **事件记忆是地基**：用药记录、健康读数、紧急事件都写入结构化表，供三个模块共享。
- **离线可跑**：未配置 LLM key 时用模板回复（FakeLLM），业务闭环与测试不受影响。

## 目录结构

```
app/
  config.py            配置（.env）
  db.py                SQLite + SQLAlchemy
  models.py            数据模型（档案/用药/事件/记忆/通知）
  channels.py          交互输出通道抽象（CLI/语音共用）
  emergency/           分层紧急检测 + 分级响应
  medication/          用药提醒调度 + 确认闭环 + 防重复
  memory/              档案 / 事件 / 对话摘要 三层记忆
  notify/              家属通知（控制台，可替换短信/微信）
  agent/               LLM 接口 + 系统 Prompt + 对话大脑
  cli.py               文本演示入口
voice/server.py        语音层骨架（Pipecat + ElevenLabs，待启用）
seed/seed_data.py      4 位虚构老人档案与用药计划
tests/                 44 个测试，覆盖编造的生活场景
```

## 快速开始

```bash
# 1. 创建并激活环境
conda create -y -n eldercare python=3.11
conda activate eldercare

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置（可选：不配 key 也能跑，用离线模板回复）
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY 以启用真实对话

# 4. 写入种子数据
python -m seed.seed_data

# 5. 启动文本演示
python -m app.cli
```

## 演示指令（CLI 内）

| 指令 | 作用 |
|------|------|
| `/who` | 查看当前老人档案 |
| `/time 08:00` | 模拟到点，触发该时间的用药提醒 |
| `/silence` | 模拟老人沉默（紧急确认中会升级为紧急并通知家属）|
| `/events` | 查看已记录的健康 / 紧急 / 家属通知 |
| `/mem` | 查看当前记忆状态（running summary + 工作记忆窗口）|
| `/quit` | 把残留窗口折叠入记忆并退出 |

## 记忆模型（常驻在线）

设备是长期常驻的，没有"开始/结束会话"的概念。记忆按**事件驱动**流动，不依赖时钟：

- **工作记忆滚动窗口**：最近若干轮原文常驻上下文；一旦超过 token 预算
  （`MEM_TOKEN_BUDGET`，默认 1500），最老的几轮就被**增量折叠**进一条持续演进的
  running summary（远期模糊记忆），窗口保持有界。
- **结构化事实**（用药/健康/预约/约定）由 NLU 当场写库，是精确的长期记忆。
- **时间感**：每轮把"当前时刻 / 时段 / 距上一句多久"注入上下文，让机器人有昼夜与间隔感。

用 `/mem` 可直观看到"聊着聊着，较早的原文被折叠进 running summary、窗口缩小"。

## 推荐演示脚本

```
/time 08:00              # 主动提醒吃药（含药名/形状/剂量/饭前后）
药吃过了                  # 确认服药，闭环
我今天吃药了吗            # 防重复服用：回答"已经吃过"
我血压量了是150/95        # 健康数据自动记录
今天去公园遇到王阿姨了    # 闲聊（多聊几轮后较早内容会被折叠进 running summary）
/mem                     # 查看 running summary 与工作记忆窗口
你还记得我今天遇到谁吗    # 验证折叠后仍能回忆（来自 running summary）
我这胸口有点闷            # 触发告警 + 主动确认 + 通知家属
没事没事我就是喘口气      # 确认无事 -> 降级取消
救命啊我不行了            # 最高级 -> 通知全部联系人 + 建议120
/events                  # 查看全程留痕
/quit                    # 折叠残留窗口并退出
```

> 演示折叠效果时可临时调小预算：`MEM_TOKEN_BUDGET=40 MEM_FOLD_BATCH=2 python -m app.cli`。
> 数据结构有变更时直接删库重建：`rm -f elder_care.db && python -m seed.seed_data`。

## 运行测试

```bash
python -m pytest
```

测试覆盖：紧急分级与误报抑制、确认升级/降级、沉默升级、用药确认闭环与防重复、
健康数据抽取、跨会话记忆、以及端到端生活场景（独居的一天 / 深夜求救 /
反复漏药通知家属 / 认知障碍防误服）。

## 后续上语音

业务闭环测试通过后：

```bash
pip install -r requirements-voice.txt
# .env 配置 ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID
```

按 `voice/server.py` 顶部说明，把 `BrainProcessor` 接入 Pipecat pipeline
（STT → Brain → TTS），即可复用全部业务逻辑。
```
transport.input() -> ElevenLabs STT -> BrainProcessor -> ElevenLabs TTS -> transport.output()
```
```
