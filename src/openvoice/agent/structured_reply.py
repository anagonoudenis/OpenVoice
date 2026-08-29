"""Structured reply parsing: intent classification + natural-language reply
from a single LLM call, in a format that streams naturally.

`ConversationManager` used to make two sequential LLM round-trips per
caller turn (a dedicated intent-classification call, then a separate
reply-generation call). On a live phone call that doubles the dead air
between the caller finishing a sentence and the agent starting to speak
-- the single biggest lever on how natural a voice agent feels. Merging
both into one call cut that in half; this module defines the shape of
that merged response and both ways of consuming it.

The reply comes back as plain text with a trailing marker, not JSON:

    <natural spoken reply>
    ###INTENT: <label>

JSON was the first design (`{"intent": ..., "reply": ...}`), but it
doesn't stream: TTS can only start speaking text that's known to be
final, and a streamed JSON string value needs unescaping (`\"`, `\n`,
...) that isn't safe to do on an arbitrary partial suffix -- you'd have
to buffer until you can prove no escape sequence is mid-flight. A trailing
plain-text marker sidesteps that entirely: every character read before
the marker is exactly what should be spoken, no decoding step at all.
`StreamingReplyExtractor` below is the piece that makes this safe when
the marker itself might arrive split across multiple streaming deltas.
"""

import structlog

from openvoice.agent.base import Intent

logger = structlog.get_logger(__name__)

_INTENT_LABELS = ", ".join(f'"{intent.value}"' for intent in Intent)

_INTENT_MARKER = "###INTENT:"

_STRUCTURED_OUTPUT_INSTRUCTIONS = f"""

Additionally, classify the caller's latest message. First write your \
natural spoken reply to the caller, in the caller's own language. Then, \
on its own new line, write exactly "{_INTENT_MARKER} <label>" where \
<label> is one of {_INTENT_LABELS}. Do not write anything after the \
intent line. Do not use markdown code fences or any other formatting \
around either part.

Example:
Sure, I can help with that.
{_INTENT_MARKER} general
"""


def build_structured_system_prompt(base_prompt: str) -> str:
    """Append structured-output instructions to `base_prompt`."""
    return base_prompt + _STRUCTURED_OUTPUT_INSTRUCTIONS


def parse_structured_reply(raw: str) -> tuple[Intent, str]:
    """Parse one complete LLM completion into `(intent, reply_text)`.

    Never raises: a model forgetting the marker, adding stray text after
    it, or misspelling the label must not be the reason a live call
    drops. Falls back to `Intent.GENERAL` with the raw response used
    verbatim as the reply when the marker is missing entirely, so the
    caller still hears *something* coherent instead of silence or a
    crash.
    """
    marker_index = raw.find(_INTENT_MARKER)
    if marker_index == -1:
        logger.warning("structured_reply_missing_intent_marker", raw_response=raw)
        return Intent.GENERAL, raw.strip()

    reply = raw[:marker_index].strip()
    label = raw[marker_index + len(_INTENT_MARKER) :].strip().lower()
    try:
        intent = Intent(label)
    except ValueError:
        logger.warning("structured_reply_invalid_intent_label", raw_response=raw)
        intent = Intent.GENERAL

    if not reply:
        # A marker with no reply text before it isn't usable -- the
        # caller still needs to hear something.
        return intent, raw.strip()
    return intent, reply


class StreamingReplyExtractor:
    """Incrementally extracts speakable reply text from a stream of text
    deltas shaped like `parse_structured_reply` expects, without ever
    emitting a prefix of `###INTENT:` itself.

    That precaution matters because a provider's streaming chunk
    boundaries have no relationship to where the marker falls in the
    text -- it can arrive split across any number of deltas, at any
    character offset, not just as one clean piece. Feeding "the last few
    characters look like they *might* be the start of the marker" text
    straight to TTS would risk speaking a stray "#" or "##" out loud, or
    (worse) speaking the first few characters of "###INTENT:" itself.

    Usage: call `feed(delta)` per streamed delta, yielding newly speakable
    text as it becomes safe to release. Call `finalize()` exactly once,
    after the stream ends, to get the complete reply text, any trailing
    text that was held back and never released via `feed()`, and the
    parsed intent.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._released = ""
        self._marker_found = False
        self._after_marker = ""

    def feed(self, delta: str) -> str:
        """Newly speakable text extracted from this delta, if any."""
        if not delta:
            return ""
        if self._marker_found:
            # The intent label itself can arrive split across further
            # deltas once the marker has already been found (e.g. the
            # marker lands at the very end of one delta with the label
            # only starting in the next) -- keep accumulating it rather
            # than silently dropping it, even though nothing more is ever
            # speakable from this point on.
            self._after_marker += delta
            return ""

        self._pending += delta
        marker_index = self._pending.find(_INTENT_MARKER)
        if marker_index != -1:
            # Trailing whitespace right before the marker (typically the
            # newline the prompt asks for) is formatting, not something
            # to speak -- safe to trim here since this is, by definition,
            # the last piece of reply text there is.
            newly_safe = self._pending[:marker_index].rstrip()
            self._after_marker = self._pending[marker_index + len(_INTENT_MARKER) :]
            self._marker_found = True
            self._pending = ""
            self._released += newly_safe
            return newly_safe

        # Not found yet -- but the tail of `_pending` might be a partial
        # prefix of the marker. Hold back just enough characters that a
        # marker split across the *next* delta can still be caught whole.
        hold_back = len(_INTENT_MARKER) - 1
        if len(self._pending) <= hold_back:
            return ""
        cut = len(self._pending) - hold_back
        newly_safe = self._pending[:cut]
        self._pending = self._pending[cut:]
        self._released += newly_safe
        return newly_safe

    def finalize(self) -> tuple[str, str, Intent]:
        """Call once, after the stream has ended.

        Returns `(full_reply_text, trailing_unspoken_text, intent)`.
        `trailing_unspoken_text` is non-empty only if the marker never
        appeared at all (a malformed response): text `feed()` held back
        "just in case" that never got released. The caller must yield it
        itself to actually speak it -- this method only returns values,
        it isn't a generator.
        """
        if not self._marker_found:
            logger.warning("streaming_reply_missing_intent_marker", partial_text=self._released)
            trailing = self._pending
            self._released += trailing
            self._pending = ""
            return self._released, trailing, Intent.GENERAL

        label = self._after_marker.strip().lower()
        try:
            intent = Intent(label)
        except ValueError:
            logger.warning("structured_reply_invalid_intent_label", raw_response=self._after_marker)
            intent = Intent.GENERAL
        return self._released, "", intent
