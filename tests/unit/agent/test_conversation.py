"""Tests for ConversationManager."""

from openvoice.agent.base import Intent
from openvoice.agent.conversation import ConversationManager
from openvoice.llm.base import LLMMessageRole
from tests.unit.agent.fakes import FakeLLMProvider


def _manager(llm: FakeLLMProvider, **kwargs: int) -> ConversationManager:
    return ConversationManager(
        llm=llm, system_prompt="You are a helpful agent.", call_id="call-1", **kwargs
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
