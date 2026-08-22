"""Intent detection: classifies a caller utterance using the configured LLM.

A focused classification prompt rather than a separate trained model —
consistent with the pluggable-provider architecture, whichever LLM
provider is configured handles this too.
"""

import structlog

from openvoice.agent.base import Intent
from openvoice.llm.base import BaseLLMProvider, LLMMessage, LLMMessageRole, LLMProviderError

logger = structlog.get_logger(__name__)

_INTENT_LABELS = ", ".join(intent.value for intent in Intent)

_CLASSIFICATION_PROMPT = f"""\
Classify the caller's message below into exactly one of these categories: \
{_INTENT_LABELS}.

Respond with only the category label, nothing else.

Caller message: {{utterance}}
"""


async def detect_intent(llm: BaseLLMProvider, utterance: str) -> Intent:
    """Classify `utterance`.

    Falls back to `Intent.GENERAL` on any provider failure or unparseable
    response — intent detection must never be the reason a call drops.
    """
    try:
        response = await llm.generate(
            [
                LLMMessage(
                    role=LLMMessageRole.USER,
                    content=_CLASSIFICATION_PROMPT.format(utterance=utterance),
                )
            ],
            temperature=0.0,
            max_tokens=16,
        )
    except LLMProviderError as exc:
        logger.warning("intent_detection_failed", error=str(exc))
        return Intent.GENERAL

    label = response.content.strip().lower()
    try:
        return Intent(label)
    except ValueError:
        logger.warning("intent_detection_unparseable", raw_response=response.content)
        return Intent.GENERAL
