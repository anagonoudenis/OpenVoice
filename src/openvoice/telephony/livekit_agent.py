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

`_NullLLM` below works around a real bug found by actually talking to the
agent (see CHANGELOG): overriding `llm_node` is meant to fully replace
how a reply is generated, but `AgentActivity._user_turn_completed_task`
(livekit-agents' own turn-taking code, not ours) still gates the whole
generation step on `self.llm is not None` *before* it ever calls
`llm_node` -- with no LLM plugin configured, that check silently
`return`s with no exception logged. The caller's speech was transcribed
correctly (proof VAD+STT worked), but no reply was ever generated because
of this gate, not because of anything in our `llm_node`. Passing a
present-but-inert `LLM` instance satisfies the gate; its `chat()` is
never actually invoked because `llm_node` is fully overridden.
"""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import Any

import structlog
from livekit import rtc
from livekit.agents import Agent, APIConnectOptions, ModelSettings, llm, stt, vad
from livekit.agents.language import LanguageCode

from openvoice.agent.conversation import ConversationManager
from openvoice.stt.base import BaseSTTProvider
from openvoice.tts.base import BaseTTSProvider

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16000
_NUM_CHANNELS = 1
_DEFAULT_CONN_OPTIONS = APIConnectOptions()


class _NullLLM(llm.LLM[Any]):
    """Satisfies `AgentActivity`'s `self.llm is not None` gate without ever
    being called: `OpenVoiceAgent.llm_node` fully replaces reply
    generation, so `chat()` here is unreachable in practice.
    """

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = _DEFAULT_CONN_OPTIONS,
        parallel_tool_calls: Any = None,
        tool_choice: Any = None,
        extra_kwargs: Any = None,
    ) -> llm.LLMStream:
        raise NotImplementedError(
            "_NullLLM.chat should be unreachable: OpenVoiceAgent overrides llm_node"
        )


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
        super().__init__(instructions=system_prompt, llm=_NullLLM())
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

                # The frame's own `sample_rate` is the actual capture
                # rate -- not necessarily `_SAMPLE_RATE` (16 kHz), which
                # is only what *we* ask providers to speak/listen at, not
                # a guarantee about what the mic/call audio arrives as.
                # Passing the wrong value here silently made Whisper hear
                # the caller's speech sped up or slowed down, which reads
                # as empty or garbled transcripts, not an obvious error.
                utterance_sample_rate = event.frames[0].sample_rate
                total_samples = sum(f.samples_per_channel for f in event.frames)
                logger.info(
                    "stt_utterance_captured",
                    sample_rate=utterance_sample_rate,
                    num_frames=len(event.frames),
                    duration_seconds=round(total_samples / utterance_sample_rate, 3)
                    if utterance_sample_rate
                    else None,
                )
                async for segment in self._stt_provider.transcribe_stream(
                    _utterance_frames(), sample_rate=utterance_sample_rate
                ):
                    logger.info(
                        "stt_segment_transcribed",
                        text=segment.text,
                        is_final=segment.is_final,
                        language=segment.language,
                    )
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
        """Route the latest caller message through `ConversationManager`,
        streaming its reply (see `ConversationManager.handle_utterance_stream`)
        so `tts_node` below can start synthesizing speech before the whole
        reply is ready.
        """
        messages = chat_ctx.messages()
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return

        caller_text = user_messages[-1].text_content or ""
        logger.info("llm_node_caller_text", caller_text=caller_text, num_messages=len(messages))
        chunk_id = f"openvoice-{len(messages)}"
        async for delta in self._conversation.handle_utterance_stream(caller_text):
            yield llm.ChatChunk(id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=delta))

        reply = self._conversation.last_reply
        if reply is not None and reply.transfer_to_human and self._on_transfer_to_human is not None:
            task: asyncio.Task[None] = asyncio.create_task(self._on_transfer_to_human())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Synthesize speech sentence-by-sentence as text streams in from
        `llm_node`, instead of waiting for the whole reply.

        Piper/ElevenLabs (like most TTS engines) need a full clause of
        context for coherent prosody, so a sentence is the right unit to
        stream at here, not literally per-token -- letting playback start
        after the first sentence is ready is most of what streaming
        `llm_node` buys on a live call; waiting for the *entire* reply
        before synthesizing anything (the previous behavior) threw that
        latency win away at the very next pipeline stage.
        """
        buffer = ""
        async for chunk in text:
            buffer += chunk
            while (cut := _find_sentence_end(buffer)) is not None:
                sentence, buffer = buffer[:cut], buffer[cut:]
                # `cut` lands right after the punctuation mark, not past
                # the whitespace that confirmed it -- strip it here so
                # that leading space doesn't linger at the start of
                # `buffer` (and thus the next sentence).
                sentence = sentence.strip()
                if not sentence:
                    continue
                async for frame in self._synthesize_to_frames(sentence):
                    yield frame

        if buffer.strip():
            async for frame in self._synthesize_to_frames(buffer.strip()):
                yield frame

    async def _synthesize_to_frames(self, text: str) -> AsyncIterable[rtc.AudioFrame]:
        async for pcm_chunk in self._tts_provider.synthesize(text, sample_rate=_SAMPLE_RATE):
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


def _find_sentence_end(buffer: str) -> int | None:
    """Index just after the first complete sentence in `buffer`, or `None`
    if there isn't one yet.

    A sentence is considered complete at a '.', '!', '?', or newline that
    is *followed* by whitespace -- deliberately requiring one character of
    lookahead, not just a trailing punctuation mark, so a period that
    might turn out to be part of "3.14" or an abbreviation followed
    immediately by more text isn't mistaken for a sentence boundary
    (imperfect -- "Dr. Smith" still splits after "Dr." -- but wrong in a
    way that only costs a little prosody, never a dropped or garbled
    word, which is the right tradeoff for real streaming here). Whatever
    never finds a boundary is flushed once by the caller after the input
    stream ends, so no text is ever lost, just synthesized without the
    streaming benefit.
    """
    for i, char in enumerate(buffer):
        if char in ".!?\n" and i + 1 < len(buffer) and buffer[i + 1] in " \n":
            return i + 1
    return None
