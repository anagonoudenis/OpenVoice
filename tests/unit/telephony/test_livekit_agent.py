"""Tests for OpenVoiceAgent's stt_node/llm_node/tts_node bridges.

Uses the real `livekit.agents`/`livekit.rtc` types (installed via the
`voice` extra) so these tests exercise the actual object shapes the
production code constructs, not a hand-rolled approximation of them. Only
VAD is faked, since `silero.VAD.load()` needs a real ONNX model file.
"""

from collections.abc import AsyncIterable, AsyncIterator

from livekit import rtc
from livekit.agents import ModelSettings, llm, stt, vad

from openvoice.agent.conversation import ConversationManager
from openvoice.stt.base import TranscriptSegment
from openvoice.telephony.livekit_agent import OpenVoiceAgent
from tests.unit.agent.fakes import FakeLLMProvider
from tests.unit.telephony.fakes import FakeSTTProvider, FakeTTSProvider


class _FakeVADStream:
    def __init__(self, events: list[vad.VADEvent]) -> None:
        self._events = events
        self.pushed_frames: list[rtc.AudioFrame] = []
        self.ended = False
        self.closed = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self.pushed_frames.append(frame)

    def end_input(self) -> None:
        self.ended = True

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[vad.VADEvent]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[vad.VADEvent]:
        for event in self._events:
            yield event


class _FakeVAD:
    def __init__(self, events: list[vad.VADEvent]) -> None:
        self._events = events

    def stream(self) -> _FakeVADStream:
        return _FakeVADStream(self._events)


def _frame(pcm: bytes) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm, sample_rate=16000, num_channels=1, samples_per_channel=len(pcm) // 2
    )


async def _audio_stream(frames: list[rtc.AudioFrame]) -> AsyncIterable[rtc.AudioFrame]:
    for frame in frames:
        yield frame


def _agent(
    *,
    stt_provider: FakeSTTProvider,
    tts_provider: FakeTTSProvider,
    llm_provider: FakeLLMProvider,
    vad_events: list[vad.VADEvent] | None = None,
) -> OpenVoiceAgent:
    conversation = ConversationManager(
        llm=llm_provider, system_prompt="You are helpful.", call_id="call-1"
    )
    return OpenVoiceAgent(
        conversation=conversation,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        vad_provider=_FakeVAD(vad_events or []),  # type: ignore[arg-type]
        system_prompt="You are helpful.",
    )


async def test_stt_node_emits_final_transcript_for_vad_segmented_utterance() -> None:
    frame = _frame(b"\x00\x01" * 160)
    event = vad.VADEvent(
        type=vad.VADEventType.END_OF_SPEECH,
        samples_index=0,
        timestamp=0.0,
        speech_duration=1.0,
        silence_duration=0.0,
        frames=[frame],
    )
    stt_provider = FakeSTTProvider([TranscriptSegment(text="hello there", is_final=True)])
    agent = _agent(
        stt_provider=stt_provider,
        tts_provider=FakeTTSProvider(),
        llm_provider=FakeLLMProvider(),
        vad_events=[event],
    )

    events = [
        e async for e in agent.stt_node(_audio_stream([frame]), model_settings=ModelSettings())
    ]

    assert len(events) == 1
    assert isinstance(events[0], stt.SpeechEvent)
    assert events[0].type is stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[0].alternatives[0].text == "hello there"


async def test_stt_node_ignores_non_end_of_speech_events() -> None:
    frame = _frame(b"\x00\x01" * 160)
    start_event = vad.VADEvent(
        type=vad.VADEventType.START_OF_SPEECH,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
        frames=[],
    )
    stt_provider = FakeSTTProvider([TranscriptSegment(text="unused", is_final=True)])
    agent = _agent(
        stt_provider=stt_provider,
        tts_provider=FakeTTSProvider(),
        llm_provider=FakeLLMProvider(),
        vad_events=[start_event],
    )

    events = [
        e async for e in agent.stt_node(_audio_stream([frame]), model_settings=ModelSettings())
    ]

    assert events == []


async def test_llm_node_routes_last_user_message_through_conversation() -> None:
    llm_provider = FakeLLMProvider(responses=["general", "We're open 9 to 5."])
    agent = _agent(
        stt_provider=FakeSTTProvider([]), tts_provider=FakeTTSProvider(), llm_provider=llm_provider
    )
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="What are your hours?")

    chunks = [c async for c in agent.llm_node(chat_ctx, tools=[], model_settings=ModelSettings())]

    assert len(chunks) == 1
    assert chunks[0].delta is not None
    assert chunks[0].delta.content == "We're open 9 to 5."


async def test_llm_node_yields_nothing_without_a_user_message() -> None:
    agent = _agent(
        stt_provider=FakeSTTProvider([]),
        tts_provider=FakeTTSProvider(),
        llm_provider=FakeLLMProvider(),
    )
    chat_ctx = llm.ChatContext.empty()

    chunks = [c async for c in agent.llm_node(chat_ctx, tools=[], model_settings=ModelSettings())]

    assert chunks == []


async def test_tts_node_synthesizes_joined_text_into_audio_frames() -> None:
    tts_provider = FakeTTSProvider()
    agent = _agent(
        stt_provider=FakeSTTProvider([]), tts_provider=tts_provider, llm_provider=FakeLLMProvider()
    )

    async def text_stream() -> AsyncIterable[str]:
        yield "Hi "
        yield "there"

    frames = [f async for f in agent.tts_node(text_stream(), model_settings=ModelSettings())]

    assert tts_provider.synthesized_text == ["Hi there"]
    assert len(frames) == 1
    assert isinstance(frames[0], rtc.AudioFrame)
    assert bytes(frames[0].data) == b"audio:Hi there"
    assert frames[0].sample_rate == 16000
    assert frames[0].num_channels == 1


async def test_tts_node_truncates_odd_length_pcm_chunk() -> None:
    # "audio:H" is 7 bytes (odd) -- not valid PCM16, which AudioFrame
    # would otherwise reject outright and drop the whole reply.
    tts_provider = FakeTTSProvider()
    agent = _agent(
        stt_provider=FakeSTTProvider([]), tts_provider=tts_provider, llm_provider=FakeLLMProvider()
    )

    async def text_stream() -> AsyncIterable[str]:
        yield "H"

    frames = [f async for f in agent.tts_node(text_stream(), model_settings=ModelSettings())]

    assert len(frames) == 1
    assert bytes(frames[0].data) == b"audio:H"[:-1]
    assert frames[0].samples_per_channel == 3


async def test_tts_node_yields_nothing_for_empty_text() -> None:
    tts_provider = FakeTTSProvider()
    agent = _agent(
        stt_provider=FakeSTTProvider([]), tts_provider=tts_provider, llm_provider=FakeLLMProvider()
    )

    async def empty_stream() -> AsyncIterable[str]:
        return
        yield  # pragma: no cover -- makes this an async generator

    frames = [f async for f in agent.tts_node(empty_stream(), model_settings=ModelSettings())]

    assert frames == []
    assert tts_provider.synthesized_text == []
