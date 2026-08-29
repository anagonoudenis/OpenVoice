"""Tests for structured reply parsing, including the streaming extractor."""

import pytest

from openvoice.agent.base import Intent
from openvoice.agent.structured_reply import (
    StreamingReplyExtractor,
    build_structured_system_prompt,
    parse_structured_reply,
)


def test_build_structured_system_prompt_includes_base_and_marker_instructions() -> None:
    prompt = build_structured_system_prompt("You are a helpful agent.")
    assert "You are a helpful agent." in prompt
    assert "###INTENT:" in prompt


@pytest.mark.parametrize("label", [i.value for i in Intent])
def test_parse_structured_reply_parses_valid_response(label: str) -> None:
    raw = f"Sure, one moment.\n###INTENT: {label}"
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent(label)
    assert reply == "Sure, one moment."


def test_parse_structured_reply_is_case_insensitive_and_strips_whitespace_on_intent() -> None:
    raw = "Let's find you a slot.\n###INTENT:   Booking  \n"
    intent, _reply = parse_structured_reply(raw)
    assert intent is Intent.BOOKING


def test_parse_structured_reply_falls_back_on_invalid_intent_label() -> None:
    raw = "Hello.\n###INTENT: not-a-real-category"
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == "Hello."


def test_parse_structured_reply_falls_back_on_missing_marker() -> None:
    raw = "Sure, I can help with that -- no marker at all."
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == raw


def test_parse_structured_reply_falls_back_on_empty_reply_before_marker() -> None:
    raw = "###INTENT: general"
    intent, reply = parse_structured_reply(raw)
    assert intent is Intent.GENERAL
    assert reply == raw  # no usable reply text -- use the raw response verbatim


class TestStreamingReplyExtractor:
    def test_marker_arriving_in_one_delta(self) -> None:
        extractor = StreamingReplyExtractor()

        spoken = extractor.feed("Sure, I can help.\n###INTENT: general")
        full_text, trailing, intent = extractor.finalize()

        assert spoken == "Sure, I can help."
        assert full_text == "Sure, I can help."
        assert trailing == ""
        assert intent is Intent.GENERAL

    def test_marker_split_across_every_possible_delta_boundary(self) -> None:
        """The marker can land split across streaming deltas at any
        character offset -- provider chunk boundaries have no relation to
        it. Try every possible split point of the full response and
        confirm the reply text comes out correct (content-preserving)
        every time.

        Trailing whitespace right before the marker may or may not
        already have been released by the time the marker itself is
        found, depending on exactly where the split falls -- that's an
        inherent, harmless streaming ambiguity (whitespace mid-stream is
        indistinguishable from soon-to-be-trailing whitespace until more
        text arrives), not data loss or corruption, so it's normalized
        out of this comparison rather than asserted on exactly.
        """
        reply_text = "Let's get you booked in for Tuesday."
        full_response = f"{reply_text}\n###INTENT: booking"

        for split_at in range(len(full_response) + 1):
            extractor = StreamingReplyExtractor()
            first, second = full_response[:split_at], full_response[split_at:]

            spoken = extractor.feed(first) + extractor.feed(second)
            full_text, trailing, intent = extractor.finalize()

            assert spoken.rstrip() == reply_text, f"failed at split {split_at}"
            assert full_text.rstrip() == reply_text, f"failed at split {split_at}"
            assert trailing == ""
            assert intent is Intent.BOOKING

    def test_marker_split_one_character_at_a_time(self) -> None:
        """Worst-case fragmentation: every character its own delta."""
        reply_text = "One moment please."
        full_response = f"{reply_text}\n###INTENT: general"
        extractor = StreamingReplyExtractor()

        spoken = "".join(extractor.feed(ch) for ch in full_response)
        full_text, trailing, intent = extractor.finalize()

        # See the boundary test above re: trailing-whitespace normalization.
        assert spoken.rstrip() == reply_text
        assert full_text.rstrip() == reply_text
        assert trailing == ""
        assert intent is Intent.GENERAL

    def test_marker_never_appears_recovers_trailing_text_at_finalize(self) -> None:
        """A malformed response with no marker at all must not silently
        lose the tail end of the reply that `feed()` was holding back
        just in case a marker was starting there.
        """
        extractor = StreamingReplyExtractor()

        spoken = extractor.feed("This response forgot the marker entirely")
        full_text, trailing, intent = extractor.finalize()

        assert spoken + trailing == "This response forgot the marker entirely"
        assert full_text == "This response forgot the marker entirely"
        assert intent is Intent.GENERAL

    def test_reply_containing_hash_characters_not_part_of_the_marker(self) -> None:
        """A reply that happens to contain '#' characters (e.g. "#1") must
        not be mistaken for the start of the marker and held back forever.
        """
        reply_text = "We're rated #1 in the area! Give us a call."
        full_response = f"{reply_text}\n###INTENT: general"
        extractor = StreamingReplyExtractor()

        spoken = "".join(extractor.feed(ch) for ch in full_response)
        full_text, _trailing, _intent = extractor.finalize()

        assert spoken.rstrip() == reply_text
        assert full_text.rstrip() == reply_text

    def test_empty_reply_text_before_marker(self) -> None:
        extractor = StreamingReplyExtractor()

        spoken = extractor.feed("###INTENT: human_transfer")
        full_text, trailing, intent = extractor.finalize()

        assert spoken == ""
        assert full_text == ""
        assert trailing == ""
        assert intent is Intent.HUMAN_TRANSFER

    def test_empty_deltas_are_ignored(self) -> None:
        extractor = StreamingReplyExtractor()

        spoken = extractor.feed("") + extractor.feed("Hi.\n###INTENT: general") + extractor.feed("")

        assert spoken == "Hi."

    def test_invalid_intent_label_falls_back_to_general(self) -> None:
        extractor = StreamingReplyExtractor()

        extractor.feed("Hello there.\n###INTENT: not-a-real-category")
        _full_text, _trailing, intent = extractor.finalize()

        assert intent is Intent.GENERAL

    def test_feed_after_marker_found_returns_nothing(self) -> None:
        extractor = StreamingReplyExtractor()
        extractor.feed("Hi.\n###INTENT: general")

        # A misbehaving caller feeding more deltas after the marker was
        # already found must not resurrect anything.
        assert extractor.feed("more text") == ""
