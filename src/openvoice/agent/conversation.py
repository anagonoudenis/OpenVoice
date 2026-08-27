"""Per-call conversation orchestration: history, LLM turn-taking, fallback."""

import structlog

from openvoice.agent.base import AgentReply, Intent
from openvoice.agent.structured_reply import build_structured_system_prompt, parse_structured_reply
from openvoice.llm.base import BaseLLMProvider, LLMMessage, LLMMessageRole, LLMProviderError

logger = structlog.get_logger(__name__)

_FALLBACK_MESSAGE = (
    "I'm sorry, I'm having trouble processing that right now. "
    "Let me transfer you to a member of our team."
)

# Both mean "a human should pick this up": urgent issues still get a
# reassuring LLM-generated reply first, but are always escalated rather
# than resolved by the bot.
_TRANSFER_INTENTS = frozenset({Intent.HUMAN_TRANSFER, Intent.URGENT})


class ConversationManager:
    """Owns one call's conversation state and produces the agent's replies.

    Constructed once per call. `handle_utterance` never raises
    `LLMProviderError`: on an unrecoverable LLM failure it returns the
    defined human-transfer fallback reply instead, so a live call never
    hangs on an unhandled exception.

    Intent classification and reply generation happen in a single LLM
    call (see `openvoice.agent.structured_reply`) rather than two
    sequential ones -- this roughly halves per-turn latency on a live
    call, the dominant driver of how natural the agent feels to talk to.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProvider,
        system_prompt: str,
        call_id: str,
        max_history_turns: int = 20,
    ) -> None:
        self._llm = llm
        self._system_prompt = build_structured_system_prompt(system_prompt)
        self._call_id = call_id
        self._max_history_turns = max_history_turns
        self._history: list[LLMMessage] = []

    @property
    def history(self) -> list[LLMMessage]:
        """A copy of the conversation history so far (user/assistant turns only)."""
        return list(self._history)

    async def handle_utterance(self, caller_text: str) -> AgentReply:
        """Produce the agent's reply to one caller utterance, updating history."""
        log = logger.bind(call_id=self._call_id)
        log.info("utterance_received", text=caller_text)

        self._history.append(LLMMessage(role=LLMMessageRole.USER, content=caller_text))
        try:
            response = await self._llm.generate(self._history, system_prompt=self._system_prompt)
        except LLMProviderError as exc:
            log.error("llm_generate_failed_falling_back_to_human", error=str(exc))
            self._history.pop()  # don't keep an unanswered user turn in context
            return AgentReply(text=_FALLBACK_MESSAGE, intent=Intent.GENERAL, transfer_to_human=True)

        intent, reply_text = parse_structured_reply(response.content)
        log.info("intent_detected", intent=intent.value)

        self._history.append(LLMMessage(role=LLMMessageRole.ASSISTANT, content=reply_text))
        self._trim_history()
        log.info("reply_generated", text=reply_text)

        return AgentReply(
            text=reply_text, intent=intent, transfer_to_human=intent in _TRANSFER_INTENTS
        )

    def _trim_history(self) -> None:
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
