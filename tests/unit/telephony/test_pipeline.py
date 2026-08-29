"""Tests for CallPipeline (the STT -> agent -> TTS loop), fully mocked."""

from collections.abc import AsyncIterator

from openvoice.agent.conversation import ConversationManager
from openvoice.stt.base import TranscriptSegment
from openvoice.telephony.pipeline import CallPipeline
from tests.unit.agent.fakes import FakeLLMProvider
from tests.unit.telephony.fakes import FakeSTTProvider, FakeTTSProvider


async def _frames() -> AsyncIterator[bytes]:
    yield b"\x00\x01\x02\x03"


def _pipeline(*, stt: FakeSTTProvider, tts: FakeTTSProvider, llm: FakeLLMProvider) -> CallPipeline:
    conversation = ConversationManager(llm=llm, system_prompt="You are helpful.", call_id="call-1")
    return CallPipeline(stt=stt, tts=tts, conversation=conversation, call_id="call-1")


def _structured(intent: str, reply: str) -> str:
    return f"{reply}\n###INTENT: {intent}"


async def test_handle_utterance_audio_runs_full_loop() -> None:
    stt = FakeSTTProvider([TranscriptSegment(text="What are your hours?", is_final=True)])
    tts = FakeTTSProvider()
    llm = FakeLLMProvider(responses=[_structured("general", "We're open 9 to 5.")])
    pipeline = _pipeline(stt=stt, tts=tts, llm=llm)

    result = await pipeline.handle_utterance_audio(_frames())

    assert result is not None
    reply, audio = result
    assert reply.text == "We're open 9 to 5."
    assert reply.transfer_to_human is False
    assert [c async for c in audio] == [b"audio:We're open 9 to 5."]
    assert tts.synthesized_text == ["We're open 9 to 5."]


async def test_handle_utterance_audio_returns_none_for_empty_transcript() -> None:
    stt = FakeSTTProvider([TranscriptSegment(text="", is_final=True)])
    tts = FakeTTSProvider()
    llm = FakeLLMProvider()
    pipeline = _pipeline(stt=stt, tts=tts, llm=llm)

    result = await pipeline.handle_utterance_audio(_frames())

    assert result is None
    assert tts.synthesized_text == []


async def test_handle_utterance_audio_ignores_non_final_segments() -> None:
    stt = FakeSTTProvider(
        [
            TranscriptSegment(text="What are", is_final=False),
            TranscriptSegment(text="What are your hours?", is_final=True),
        ]
    )
    tts = FakeTTSProvider()
    llm = FakeLLMProvider(responses=[_structured("general", "9 to 5.")])
    pipeline = _pipeline(stt=stt, tts=tts, llm=llm)

    result = await pipeline.handle_utterance_audio(_frames())

    assert result is not None
    assert llm.calls[-1][-1].content == "What are your hours?"


async def test_human_transfer_still_synthesizes_a_reply() -> None:
    stt = FakeSTTProvider([TranscriptSegment(text="Give me a human", is_final=True)])
    tts = FakeTTSProvider()
    llm = FakeLLMProvider(responses=[_structured("human_transfer", "Let me transfer you.")])
    pipeline = _pipeline(stt=stt, tts=tts, llm=llm)

    result = await pipeline.handle_utterance_audio(_frames())

    assert result is not None
    reply, _audio = result
    assert reply.transfer_to_human is True
