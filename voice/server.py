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
import uuid
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
    """把 STT 文本交给业务大脑，产出回复文本给 TTS。"""

    def __init__(self, elder_id: int):
        super().__init__()
        self._session = SessionLocal()
        self._brain = Brain(self._session, get_llm(), ConsoleNotifier(self._session))
        self._sid = uuid.uuid4().hex
        greeting = self._brain.start_session(elder_id, self._sid)
        self._session.commit()
        self._greeting = greeting

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            # 业务逻辑是同步的，放到线程池避免阻塞事件循环
            result = await asyncio.to_thread(
                self._brain.process_turn, self._sid, frame.text
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
