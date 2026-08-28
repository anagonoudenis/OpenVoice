"""Tests for ConversationManager."""

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
    return f'{{"intent": "{intent}", "reply": "{reply}"}}'


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
    assert '"intent"' in system_prompt


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
