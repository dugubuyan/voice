"""语音层骨架（Pipecat + ElevenLabs）。

当前为骨架，未启用。业务闭环已在 app/ 中跑通并可用文本 CLI 演示。
上语音时：
  1. pip install -r requirements-voice.txt
  2. 在 .env 配置 ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID
  3. 实现 BrainProcessor（把 STT 文本交给 Brain，回复交给 TTS），接入下方 pipeline
  4. python -m voice.server

关键设计：语音层不重写任何业务逻辑，只是把 Brain 接到 STT/TTS 之间。
CLI 与语音共用同一个 Brain.process_turn，行为一致。
"""
from __future__ import annotations

# 说明：以下为参考骨架，依赖 requirements-voice.txt 安装后方可运行。
# 为避免未安装语音依赖时报错，这里不做顶层 import，改为运行时说明。

REFERENCE_PIPELINE = '''
import os
import asyncio
import aiohttp

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.network.websocket_server import (
    WebsocketServerTransport, WebsocketServerParams,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from app.db import SessionLocal
from app.agent.brain import Brain
from app.agent.llm import get_llm
from app.notify.notifier import ConsoleNotifier


class BrainProcessor(FrameProcessor):
    """把 STT 文本交给业务大脑，产出回复文本给 TTS。

    宿主循环（pipeline runner）除了把每轮 STT 文本喂给 brain.process_turn 外，
    还应每 config.pulse_interval_sec 秒调用一次 brain.on_pulse(datetime.now())，
    以驱动沉默期的记忆巩固 / 前瞻过期 / 近期原文清理 / 主动关心等条件触发
    （TriggerEngine 的低频时钟源）。on_turn 只覆盖「有对话时」的触发，
    沉默期的周期性记忆工作必须靠 on_pulse 推动。

    参考接线（骨架，未启用）：起一个 asyncio 后台任务，
        while True:
            await asyncio.sleep(config.pulse_interval_sec)
            await asyncio.to_thread(brain.on_pulse, datetime.now())
    on_pulse 内部已做失败隔离，异常不会影响语音主循环；CLI 的 /tick 指令
    与测试注入固定 now 走的是同一入口，便于离线确定性验证。
    """

    def __init__(self, elder_id: int):
        super().__init__()
        self._session = SessionLocal()
        self._brain = Brain(self._session, get_llm(), ConsoleNotifier(self._session))
        self._elder_id = elder_id
        # Phase 3：惰性初始化该老人的常驻运行态（首次加载会从 RecentTurn 落盘
        # 恢复近期窗口，需求 8.2）。不再有 start_session/会话 id 概念——记忆按
        # elder_id 常驻，CLI 与语音共用同一 Brain.process_turn(elder_id, text)。
        self._brain.get_runtime(elder_id)
        self._session.commit()
        # 问候语用固定招呼；如需更个性化可从 runtime/画像取。
        self._greeting = "哎，我在呢。今天感觉怎么样呀？"

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            # 业务逻辑是同步的，放到线程池避免阻塞事件循环
            result = await asyncio.to_thread(
                self._brain.process_turn, self._elder_id, frame.text
            )
            self._session.commit()
            await self.push_frame(TextFrame(result.reply), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


async def run(elder_id: int = 1):
    async with aiohttp.ClientSession() as http:
        transport = WebsocketServerTransport(
            host="0.0.0.0",
            port=8765,
            params=WebsocketServerParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=ProtobufFrameSerializer(),
            ),
        )
        stt = ElevenLabsSTTService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            aiohttp_session=http, model="scribe_v2", language="zh",
        )
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=os.environ["ELEVENLABS_VOICE_ID"],
            model="eleven_turbo_v2_5",
        )
        pipeline = Pipeline([
            transport.input(),
            stt,
            BrainProcessor(elder_id),  # ← 复用与 CLI 相同的业务大脑
            tts,
            transport.output(),
        ])
        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
        await PipelineRunner().run(task)
'''


def main() -> None:
    print(
        "语音层为骨架，尚未启用。\n"
        "启用步骤见本文件顶部说明；参考 pipeline 见 REFERENCE_PIPELINE 常量。\n"
        "业务闭环请先用文本演示：python -m app.cli"
    )


if __name__ == "__main__":
    main()
