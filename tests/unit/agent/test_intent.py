"""Tests for intent detection."""

import pytest

from openvoice.agent.base import Intent
from openvoice.agent.intent import detect_intent
from tests.unit.agent.fakes import FakeLLMProvider


@pytest.mark.parametrize("label", [i.value for i in Intent])
async def test_detect_intent_parses_valid_labels(label: str) -> None:
    llm = FakeLLMProvider(responses=[label])
    assert await detect_intent(llm, "some utterance") == Intent(label)


async def test_detect_intent_is_case_insensitive_and_strips_whitespace() -> None:
    llm = FakeLLMProvider(responses=["  Booking \n"])
    assert await detect_intent(llm, "I want an appointment") == Intent.BOOKING


async def test_detect_intent_falls_back_to_general_on_unparseable_response() -> None:
    llm = FakeLLMProvider(responses=["not-a-real-category"])
    assert await detect_intent(llm, "hi") == Intent.GENERAL


async def test_detect_intent_falls_back_to_general_on_provider_error() -> None:
    llm = FakeLLMProvider(fail=True)
    assert await detect_intent(llm, "hi") == Intent.GENERAL
