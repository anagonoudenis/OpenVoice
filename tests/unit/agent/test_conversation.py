"""Tests for ConversationManager."""

from openvoice.agent.base import Intent
from openvoice.agent.conversation import ConversationManager
from openvoice.llm.base import LLMMessageRole, LLMProviderError
from tests.unit.agent.fakes import FakeLLMProvider


def _manager(llm: FakeLLMProvider, **kwargs: int) -> ConversationManager:
    return ConversationManager(
        llm=llm, system_prompt="You are a helpful agent.", call_id="call-1", **kwargs
    )


async def test_handle_utterance_returns_llm_reply_and_updates_history() -> None:
    llm = FakeLLMProvider(responses=["general", "Sure, I can help with that."])
    manager = _manager(llm)

    reply = await manager.handle_utterance("What are your hours?")

    assert reply.text == "Sure, I can help with that."
    assert reply.intent is Intent.GENERAL
    assert reply.transfer_to_human is False
    assert [m.role for m in manager.history] == [LLMMessageRole.USER, LLMMessageRole.ASSISTANT]


async def test_human_transfer_intent_skips_llm_generation() -> None:
    llm = FakeLLMProvider(responses=["human_transfer"])
    manager = _manager(llm)

    reply = await manager.handle_utterance("I want to talk to a person")

    assert reply.transfer_to_human is True
    assert reply.intent is Intent.HUMAN_TRANSFER
    assert len(llm.calls) == 1  # only the intent-classification call was made


async def test_urgent_intent_transfers_to_human_with_llm_reply() -> None:
    llm = FakeLLMProvider(responses=["urgent", "I understand, let me get someone right away."])
    manager = _manager(llm)

    reply = await manager.handle_utterance("My pipe is bursting!")

    assert reply.transfer_to_human is True
    assert reply.text == "I understand, let me get someone right away."


async def test_llm_failure_falls_back_to_human_transfer_and_drops_user_turn() -> None:
    llm = FakeLLMProvider(responses=["general"])
    manager = _manager(llm)
    original_generate = llm.generate
    call_count = 0

    async def flaky_generate(messages: list, **kwargs: object) -> object:  # type: ignore[type-arg]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await original_generate(messages, **kwargs)  # type: ignore[arg-type]
        raise LLMProviderError("down")

    llm.generate = flaky_generate  # type: ignore[method-assign]

    reply = await manager.handle_utterance("Help me")

    assert reply.transfer_to_human is True
    assert "trouble" in reply.text
    assert manager.history == []


async def test_history_is_trimmed_to_max_turns() -> None:
    llm = FakeLLMProvider()
    manager = _manager(llm, max_history_turns=2)

    for i in range(5):
        llm.responses.extend(["general", f"reply {i}"])
        await manager.handle_utterance(f"utterance {i}")

    assert len(manager.history) == 4  # 2 turns * 2 messages
