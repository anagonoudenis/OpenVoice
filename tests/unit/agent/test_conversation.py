"""Tests for ConversationManager."""

from datetime import UTC, datetime, timedelta

import pytest

from openvoice.agent.base import Intent
from openvoice.agent.conversation import ConversationManager
from openvoice.llm.base import LLMMessageRole, LLMResponse, ToolCall, ToolDefinition
from tests.unit.agent.fakes import FakeLLMProvider

_ECHO_TOOL = ToolDefinition(
    name="echo", description="Echoes its input.", parameters={"type": "object", "properties": {}}
)


def _manager(llm: FakeLLMProvider, *, max_history_turns: int = 20) -> ConversationManager:
    return ConversationManager(
        llm=llm,
        system_prompt="You are a helpful agent.",
        call_id="call-1",
        max_history_turns=max_history_turns,
    )


def _structured(intent: str, reply: str) -> str:
    return f"{reply}\n###INTENT: {intent}"


async def test_handle_utterance_returns_llm_reply_and_updates_history() -> None:
    llm = FakeLLMProvider(responses=[_structured("general", "Sure, I can help with that.")])
    manager = _manager(llm)

    reply = await manager.handle_utterance("What are your hours?")

    assert reply.text == "Sure, I can help with that."
    assert reply.intent is Intent.GENERAL
    assert reply.transfer_to_human is False
    assert len(llm.calls) == 1  # intent + reply come from a single call
    assert [m.role for m in manager.history] == [LLMMessageRole.USER, LLMMessageRole.ASSISTANT]


async def test_human_transfer_intent_transfers_with_llm_reply() -> None:
    llm = FakeLLMProvider(
        responses=[_structured("human_transfer", "Of course, let me transfer you.")]
    )
    manager = _manager(llm)

    reply = await manager.handle_utterance("I want to talk to a person")

    assert reply.transfer_to_human is True
    assert reply.intent is Intent.HUMAN_TRANSFER
    assert reply.text == "Of course, let me transfer you."
    assert len(llm.calls) == 1


async def test_urgent_intent_transfers_to_human_with_llm_reply() -> None:
    llm = FakeLLMProvider(
        responses=[_structured("urgent", "I understand, let me get someone right away.")]
    )
    manager = _manager(llm)

    reply = await manager.handle_utterance("My pipe is bursting!")

    assert reply.transfer_to_human is True
    assert reply.text == "I understand, let me get someone right away."


async def test_llm_failure_falls_back_to_human_transfer_and_drops_user_turn() -> None:
    llm = FakeLLMProvider(fail=True)
    manager = _manager(llm)

    reply = await manager.handle_utterance("Help me")

    assert reply.transfer_to_human is True
    assert "trouble" in reply.text
    assert manager.history == []


async def test_history_is_trimmed_to_max_turns() -> None:
    llm = FakeLLMProvider()
    manager = _manager(llm, max_history_turns=2)

    for i in range(5):
        llm.responses.append(_structured("general", f"reply {i}"))
        await manager.handle_utterance(f"utterance {i}")

    assert len(manager.history) == 4  # 2 turns * 2 messages


async def test_llm_generate_is_called_with_the_structured_system_prompt() -> None:
    llm = FakeLLMProvider(responses=[_structured("general", "Hi.")])
    manager = _manager(llm)

    await manager.handle_utterance("Hello")

    [system_prompt] = llm.system_prompts
    assert system_prompt is not None
    assert system_prompt.startswith("You are a helpful agent.")
    assert "###INTENT:" in system_prompt


def test_constructor_rejects_tools_without_executor() -> None:
    llm = FakeLLMProvider()

    with pytest.raises(ValueError, match="tools and tool_executor"):
        ConversationManager(
            llm=llm, system_prompt="You are helpful.", call_id="call-1", tools=[_ECHO_TOOL]
        )


def test_constructor_rejects_executor_without_tools() -> None:
    llm = FakeLLMProvider()

    async def executor(_call: ToolCall) -> tuple[str, bool]:
        return "unused", False

    with pytest.raises(ValueError, match="tools and tool_executor"):
        ConversationManager(
            llm=llm, system_prompt="You are helpful.", call_id="call-1", tool_executor=executor
        )


async def test_tool_call_is_executed_and_result_fed_back() -> None:
    tool_call = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
    llm = FakeLLMProvider(
        responses=[
            LLMResponse(content="", model="fake-model", tool_calls=[tool_call]),
            _structured("general", "Done: hi"),
        ]
    )
    executed: list[ToolCall] = []

    async def executor(call: ToolCall) -> tuple[str, bool]:
        executed.append(call)
        return "echoed: hi", False

    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        tools=[_ECHO_TOOL],
        tool_executor=executor,
    )

    reply = await manager.handle_utterance("please echo hi")

    assert executed == [tool_call]
    assert reply.text == "Done: hi"
    assert len(llm.calls) == 2  # one tool-call turn, one final-reply turn
    assert [m.role for m in manager.history] == [
        LLMMessageRole.USER,
        LLMMessageRole.ASSISTANT,
        LLMMessageRole.TOOL,
        LLMMessageRole.ASSISTANT,
    ]
    tool_result = manager.history[2]
    assert tool_result.content == "echoed: hi"
    assert tool_result.tool_call_id == "call_1"
    assert tool_result.tool_call_is_error is False


async def test_tools_are_passed_to_generate() -> None:
    llm = FakeLLMProvider(responses=[_structured("general", "Hi.")])

    async def executor(_call: ToolCall) -> tuple[str, bool]:
        return "unused", False

    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        tools=[_ECHO_TOOL],
        tool_executor=executor,
    )

    await manager.handle_utterance("Hello")

    assert llm.tools_passed == [[_ECHO_TOOL]]


async def test_tool_executor_exception_is_caught_and_fed_back_as_error() -> None:
    tool_call = ToolCall(id="call_1", name="echo", arguments={})
    llm = FakeLLMProvider(
        responses=[
            LLMResponse(content="", model="fake-model", tool_calls=[tool_call]),
            _structured("general", "Sorry about that."),
        ]
    )

    async def executor(_call: ToolCall) -> tuple[str, bool]:
        raise RuntimeError("boom")

    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        tools=[_ECHO_TOOL],
        tool_executor=executor,
    )

    reply = await manager.handle_utterance("please echo")

    assert reply.text == "Sorry about that."
    tool_result = manager.history[2]
    assert tool_result.tool_call_is_error is True


async def test_tool_loop_exceeding_max_iterations_falls_back_to_human() -> None:
    tool_call = ToolCall(id="call_1", name="echo", arguments={})
    llm = FakeLLMProvider(
        responses=[LLMResponse(content="", model="fake-model", tool_calls=[tool_call])] * 10
    )

    async def executor(_call: ToolCall) -> tuple[str, bool]:
        return "still going", False

    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        tools=[_ECHO_TOOL],
        tool_executor=executor,
        max_tool_iterations=2,
    )

    reply = await manager.handle_utterance("loop forever")

    assert reply.transfer_to_human is True
    assert manager.history == []  # the whole failed turn is dropped, not left dangling
    assert len(llm.calls) == 2


async def test_history_trim_never_splits_a_tool_exchange() -> None:
    """A turn with tool calls has 4 messages, not the usual 2 -- trimming
    must always cut on a USER-message boundary, never mid-turn, or the
    next request would contain a dangling tool result with no matching
    call (which both Anthropic's and OpenAI's wire formats reject).
    """
    tool_call = ToolCall(id="call_1", name="echo", arguments={})

    async def executor(_call: ToolCall) -> tuple[str, bool]:
        return "ok", False

    llm = FakeLLMProvider(
        responses=[
            LLMResponse(content="", model="fake-model", tool_calls=[tool_call]),
            _structured("general", "reply with tool"),
            _structured("general", "plain reply"),
        ]
    )
    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        tools=[_ECHO_TOOL],
        tool_executor=executor,
        max_history_turns=1,
    )

    await manager.handle_utterance("first, uses a tool")
    await manager.handle_utterance("second, plain")

    assert [m.role for m in manager.history] == [LLMMessageRole.USER, LLMMessageRole.ASSISTANT]
    assert manager.history[0].content == "second, plain"


async def test_max_conversation_turns_forces_transfer_without_calling_llm() -> None:
    llm = FakeLLMProvider(responses=[_structured("general", "reply 1")])
    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        max_conversation_turns=1,
    )

    first = await manager.handle_utterance("hello")
    second = await manager.handle_utterance("still here")

    assert first.transfer_to_human is False
    assert second.transfer_to_human is True
    assert second.intent is Intent.HUMAN_TRANSFER
    # The limit is checked *before* calling the LLM -- the second turn
    # must not have cost an API call.
    assert len(llm.calls) == 1


async def test_max_call_duration_forces_transfer() -> None:
    llm = FakeLLMProvider(responses=[_structured("general", "reply")])
    call_started_at = datetime.now(UTC) - timedelta(seconds=1000)
    manager = ConversationManager(
        llm=llm,
        system_prompt="You are helpful.",
        call_id="call-1",
        max_call_duration_seconds=900.0,
        call_started_at=call_started_at,
    )

    reply = await manager.handle_utterance("hello")

    assert reply.transfer_to_human is True
    assert reply.intent is Intent.HUMAN_TRANSFER
    assert len(llm.calls) == 0


async def test_no_call_limits_by_default() -> None:
    llm = FakeLLMProvider()
    manager = ConversationManager(llm=llm, system_prompt="You are helpful.", call_id="call-1")

    for i in range(50):
        llm.responses.append(_structured("general", f"reply {i}"))
        reply = await manager.handle_utterance(f"utterance {i}")
        assert reply.transfer_to_human is False


class TestHandleUtteranceStream:
    async def test_yields_reply_text_incrementally_and_sets_last_reply(self) -> None:
        llm = FakeLLMProvider(
            responses=[_structured("general", "Sure, I can help with that today.")],
            stream_chunk_size=5,
        )
        manager = _manager(llm)

        chunks = [c async for c in manager.handle_utterance_stream("What are your hours?")]
        joined = "".join(chunks)

        assert len(chunks) > 1  # actually streamed in multiple pieces, not one shot
        # A trailing whitespace character right before the marker may or
        # may not already have streamed by the time the marker itself is
        # found -- an inherent, harmless ambiguity (see
        # StreamingReplyExtractor's docstring/tests) -- so normalized here.
        assert joined.rstrip() == "Sure, I can help with that today."
        assert "###INTENT" not in joined
        assert manager.last_reply is not None
        assert manager.last_reply.text == joined  # exactly what was streamed, nothing more/less
        assert manager.last_reply.intent is Intent.GENERAL
        assert manager.last_reply.transfer_to_human is False

    async def test_streamed_reply_is_recorded_in_history(self) -> None:
        llm = FakeLLMProvider(responses=[_structured("general", "Noted.")], stream_chunk_size=2)
        manager = _manager(llm)

        async for _ in manager.handle_utterance_stream("Hello"):
            pass

        # A trailing whitespace character right before the marker may or
        # may not already have been released by the time the marker is
        # found, depending on exactly where chunk boundaries fall -- an
        # inherent, harmless streaming ambiguity (see
        # StreamingReplyExtractor's docstring/tests), so normalized here
        # rather than asserted on exactly.
        assert manager.history[0].content == "Hello"
        assert manager.history[1].content.strip() == "Noted."

    async def test_human_transfer_intent_streams_correctly(self) -> None:
        llm = FakeLLMProvider(
            responses=[_structured("human_transfer", "Of course, let me transfer you.")],
            stream_chunk_size=3,
        )
        manager = _manager(llm)

        chunks = [c async for c in manager.handle_utterance_stream("get me a person")]

        assert "".join(chunks) == "Of course, let me transfer you."
        assert manager.last_reply is not None
        assert manager.last_reply.transfer_to_human is True
        assert manager.last_reply.intent is Intent.HUMAN_TRANSFER

    async def test_llm_stream_failure_falls_back_to_human_and_drops_turn(self) -> None:
        llm = FakeLLMProvider(fail=True)
        manager = _manager(llm)

        chunks = [c async for c in manager.handle_utterance_stream("Help me")]

        assert "".join(chunks) == manager.last_reply.text  # type: ignore[union-attr]
        assert manager.last_reply is not None
        assert manager.last_reply.transfer_to_human is True
        assert "trouble" in manager.last_reply.text
        assert manager.history == []

    async def test_call_limit_reached_yields_message_without_calling_llm(self) -> None:
        llm = FakeLLMProvider(responses=[_structured("general", "reply")])
        manager = ConversationManager(
            llm=llm, system_prompt="You are helpful.", call_id="call-1", max_conversation_turns=0
        )

        chunks = [c async for c in manager.handle_utterance_stream("hello")]

        assert "".join(chunks) == manager.last_reply.text  # type: ignore[union-attr]
        assert manager.last_reply is not None
        assert manager.last_reply.transfer_to_human is True
        assert len(llm.stream_calls) == 0
        assert len(llm.calls) == 0

    async def test_falls_back_to_non_streaming_handle_utterance_when_tools_configured(
        self,
    ) -> None:
        llm = FakeLLMProvider(responses=[_structured("general", "Booked.")])

        async def executor(_call: ToolCall) -> tuple[str, bool]:
            return "unused", False

        manager = ConversationManager(
            llm=llm,
            system_prompt="You are helpful.",
            call_id="call-1",
            tools=[_ECHO_TOOL],
            tool_executor=executor,
        )

        chunks = [c async for c in manager.handle_utterance_stream("book me a slot")]

        # No tool call requested this turn, so the underlying non-streaming
        # generate() path is used and its single complete reply is yielded
        # as one chunk -- generate_stream() must never be called.
        assert chunks == ["Booked."]
        assert llm.stream_calls == []
        assert manager.last_reply is not None
        assert manager.last_reply.text == "Booked."

    async def test_marker_split_one_character_at_a_time_still_parses_intent(self) -> None:
        """End-to-end regression test (the extractor itself has an
        exhaustive per-boundary test in test_structured_reply.py) for a
        real bug: the marker landing at the very edge of a streamed delta
        used to silently lose the intent label. Worst-case fragmentation
        here to exercise every possible split through the real
        ConversationManager -> generate_stream -> extractor path.
        """
        llm = FakeLLMProvider(
            responses=[_structured("booking", "Let's do that.")], stream_chunk_size=1
        )
        manager = _manager(llm)

        chunks = [c async for c in manager.handle_utterance_stream("book me a slot")]

        # See TestHandleUtteranceStream.test_streamed_reply_is_recorded_in_history
        # re: trailing-whitespace normalization.
        assert "".join(chunks).rstrip() == "Let's do that."
        assert manager.last_reply is not None
        assert manager.last_reply.intent is Intent.BOOKING
