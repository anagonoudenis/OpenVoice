"""Structured reply parsing: intent classification + natural-language reply
from a single LLM call.

Previously `ConversationManager` made two sequential LLM round-trips per
caller turn: a dedicated intent-classification call (see the old
`agent.intent.detect_intent`), then a separate reply-generation call. On
a live phone call that doubles the dead air between the caller finishing
a sentence and the agent starting to speak -- the single biggest lever on
how "natural" a voice agent feels. Merging both into one call (the model
returns a small JSON envelope containing both) roughly halves that
latency, at the cost of needing to parse the model's output instead of
trusting a clean single-label response.
"""

import json

import structlog

from openvoice.agent.base import Intent

logger = structlog.get_logger(__name__)

_INTENT_LABELS = ", ".join(f'"{intent.value}"' for intent in Intent)

_STRUCTURED_OUTPUT_INSTRUCTIONS = f"""

Additionally, classify the caller's latest message. Respond with ONLY a \
single JSON object -- no markdown code fences, no text before or after \
it -- with exactly two keys:
  "intent": one of {_INTENT_LABELS}
  "reply": your natural spoken reply to the caller, in the caller's own \
language

Example: {{"intent": "general", "reply": "Sure, I can help with that."}}
"""


def build_structured_system_prompt(base_prompt: str) -> str:
    """Append structured-output instructions to `base_prompt`."""
    return base_prompt + _STRUCTURED_OUTPUT_INSTRUCTIONS


def parse_structured_reply(raw: str) -> tuple[Intent, str]:
    """Parse one LLM completion into `(intent, reply_text)`.

    Never raises: a model occasionally wrapping its JSON in markdown
    fences, adding stray text, or misspelling a label must not be the
    reason a live call drops. Falls back to `Intent.GENERAL` with the
    raw response used verbatim as the reply when the JSON can't be
    parsed at all, so the caller still hears *something* coherent
    instead of silence or a crash.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    try:
        data = json.loads(text)
        reply = str(data["reply"])
    except (json.JSONDecodeError, KeyError, TypeError):
        # No usable reply text at all -- the caller still needs to hear
        # something, so use the raw completion verbatim rather than drop it.
        logger.warning("structured_reply_unparseable", raw_response=raw)
        return Intent.GENERAL, raw.strip()

    try:
        intent = Intent(str(data.get("intent", "")).strip().lower())
    except ValueError:
        # A valid reply with an unrecognized/missing intent label is not
        # worth discarding the reply over -- just default the intent.
        logger.warning("structured_reply_invalid_intent_label", raw_response=raw)
        intent = Intent.GENERAL

    return intent, reply
