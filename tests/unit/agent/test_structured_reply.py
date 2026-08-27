"""Tests for structured reply parsing."""

import pytest

from openvoice.agent.base import Intent
from openvoice.agent.structured_reply import build_structured_system_prompt, parse_structured_reply


def test_build_structured_system_prompt_includes_base_and_schema() -> None:
    prompt = build_structured_system_prompt("You are a helpful agent.")
    assert "You are a helpful agent." in prompt
    assert '"reply"' in prompt
    assert '"intent"' in prompt


@pytest.mark.parametrize("label", [i.value for i in Intent])
def test_parse_structured_reply_parses_valid_json(label: str) -> None:
    raw = f'{{"intent": "{label}", "reply": "Sure, one moment."}}'
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent(label)
    assert reply == "Sure, one moment."


def test_parse_structured_reply_is_case_insensitive_on_intent() -> None:
    raw = '{"intent": "  Booking \\n", "reply": "Let\'s find you a slot."}'
    intent, _reply = parse_structured_reply(raw)
    assert intent is Intent.BOOKING


def test_parse_structured_reply_strips_markdown_fences() -> None:
    raw = '```json\n{"intent": "general", "reply": "Hi there."}\n```'
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == "Hi there."


def test_parse_structured_reply_falls_back_on_invalid_intent_label() -> None:
    raw = '{"intent": "not-a-real-category", "reply": "Hello."}'
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == "Hello."


def test_parse_structured_reply_falls_back_on_missing_reply_key() -> None:
    raw = '{"intent": "general"}'
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == raw


def test_parse_structured_reply_falls_back_on_unparseable_json() -> None:
    raw = "Sure, I can help with that -- not JSON at all."
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == raw
