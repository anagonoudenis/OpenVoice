"""Bridges OpenVoice's own pluggable STT/LLM/TTS providers into a LiveKit
`Agent`, by overriding its `stt_node` / `llm_node` / `tts_node` pipeline
hooks rather than subclassing `stt.STT` / `llm.LLM` / `tts.TTS` directly.

Why nodes, not full plugin classes: `stt.STT`/`llm.LLM`/`tts.TTS` also
require implementing their internal streaming primitives (`ChunkedStream`,
`LLMStream`), whose exact internals are not fully documented and are
materially riskier to get right without a live LiveKit deployment to test
against (none is available in this environment). The node-override
surface used here — `stt.SpeechEvent`/`SpeechData`, `llm.ChatChunk`/
`ChoiceDelta`, `rtc.AudioFrame`, `vad.VADEvent` — was verified by
introspecting the actual installed `livekit-agents==1.6.9` classes at
authoring time (constructor signatures, abstract methods), not guessed
from memory. What was NOT verified: end-to-end behavior against a real
LiveKit server + SIP trunk. Smoke-test a real call before production use.

VAD-based turn detection and barge-in/interruption handling are left to
`AgentSession` itself (via the `vad=`/`allow_interruptions=` it's built
with in `openvoice.telephony.worker`) — this class only bridges the
STT/LLM/TTS legs of the pipeline. `BaseSTTProvider.transcribe_stream`
expects one VAD-delimited utterance per call, so `stt_node` runs its own
`VADStream` to segment the continuous call audio into utterances before
handing each one to the configured STT provider.
"""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import Any

import structlog
from livekit import rtc
from livekit.agents import Agent, ModelSettings, llm, stt, vad
from livekit.agents.language import LanguageCode

from openvoice.agent.conversation import ConversationManager
from openvoice.stt.base import BaseSTTProvider
from openvoice.tts.base import BaseTTSProvider

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000
_NUM_CHANNELS = 1


class OpenVoiceAgent(Agent):
    """A LiveKit `Agent` whose STT/LLM/TTS legs are OpenVoice's own providers."""

    def __init__(
        self,
        *,
        conversation: ConversationManager,
        stt_provider: BaseSTTProvider,
        tts_provider: BaseTTSProvider,
        vad_provider: vad.VAD,
        system_prompt: str,
        on_transfer_to_human: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        super().__init__(instructions=system_prompt)
        self._conversation = conversation
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider
        self._vad_provider = vad_provider
        self._on_transfer_to_human = on_transfer_to_human
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        """Segment `audio` with VAD, then batch-transcribe each utterance."""
        vad_stream = self._vad_provider.stream()

        async def _push_frames() -> None:
            async for frame in audio:
                vad_stream.push_frame(frame)
            vad_stream.end_input()

        pusher = asyncio.create_task(_push_frames())
        try:
            async for event in vad_stream:
                if event.type is not vad.VADEventType.END_OF_SPEECH or not event.frames:
                    continue

                async def _utterance_frames(
                    frames: list[rtc.AudioFrame] = event.frames,
                ) -> AsyncIterator[bytes]:
                    for frame in frames:
                        yield bytes(frame.data)

                async for segment in self._stt_provider.transcribe_stream(
                    _utterance_frames(), sample_rate=_SAMPLE_RATE
                ):
                    if segment.is_final and segment.text:
                        yield stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                stt.SpeechData(
                                    language=LanguageCode(segment.language or "en"),
                                    text=segment.text,
                                )
                            ],
                        )
        finally:
            pusher.cancel()
            await vad_stream.aclose()

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """Route the latest caller message through `ConversationManager`."""
        messages = chat_ctx.messages()
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return

        caller_text = user_messages[-1].text_content or ""
        reply = await self._conversation.handle_utterance(caller_text)

        if reply.transfer_to_human and self._on_transfer_to_human is not None:
            task: asyncio.Task[None] = asyncio.create_task(self._on_transfer_to_human())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        yield llm.ChatChunk(
            id=f"openvoice-{len(messages)}",
            delta=llm.ChoiceDelta(role="assistant", content=reply.text),
        )

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Join the (already-complete) reply text and synthesize it in one shot."""
        full_text = "".join([chunk async for chunk in text])
        if not full_text:
            return

        async for pcm_chunk in self._tts_provider.synthesize(full_text, sample_rate=_SAMPLE_RATE):
            if len(pcm_chunk) % 2 != 0:
                # PCM16 requires an even byte count; a stray trailing byte
                # would otherwise crash AudioFrame construction and drop
                # the whole reply, so trim it (1 sample is inaudible).
                logger.warning("tts_chunk_odd_byte_length_truncated", length=len(pcm_chunk))
                pcm_chunk = pcm_chunk[:-1]

            samples_per_channel = len(pcm_chunk) // 2  # 16-bit mono samples
            if samples_per_channel == 0:
                continue
            yield rtc.AudioFrame(
                data=pcm_chunk,
                sample_rate=_SAMPLE_RATE,
                num_channels=_NUM_CHANNELS,
                samples_per_channel=samples_per_channel,
            )
