"""Per-call conversation orchestration: history, LLM turn-taking, tool-calling, fallback."""

from collections.abc import Sequence

import structlog

from openvoice.agent.base import AgentReply, Intent
from openvoice.agent.structured_reply import build_structured_system_prompt, parse_structured_reply
from openvoice.agent.tools.base import ToolExecutor
from openvoice.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMMessageRole,
    LLMProviderError,
    ToolCall,
    ToolDefinition,
)

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

    When `tools`/`tool_executor` are given (see `openvoice.agent.tools`),
    `handle_utterance` runs an agentic loop: if the model asks to call a
    tool, the result is executed and fed back, and the model is asked
    again, up to `max_tool_iterations` times, before it must produce a
    final natural-language reply. The iteration cap exists so a
    misbehaving model (or a tool that keeps failing) can't turn one
    caller utterance into an unbounded, ever-billing loop on a live call.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProvider,
        system_prompt: str,
        call_id: str,
        max_history_turns: int = 20,
        tools: Sequence[ToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_iterations: int = 4,
    ) -> None:
        if bool(tools) != bool(tool_executor):
            raise ValueError("tools and tool_executor must be provided together, or not at all")

        self._llm = llm
        self._system_prompt = build_structured_system_prompt(system_prompt)
        self._call_id = call_id
        self._max_history_turns = max_history_turns
        self._tools = list(tools) if tools else None
        self._tool_executor = tool_executor
        self._max_tool_iterations = max_tool_iterations
        self._history: list[LLMMessage] = []

    @property
    def history(self) -> list[LLMMessage]:
        """A copy of the conversation history so far (user/assistant turns only)."""
        return list(self._history)

    async def handle_utterance(self, caller_text: str) -> AgentReply:
        """Produce the agent's reply to one caller utterance, updating history."""
        log = logger.bind(call_id=self._call_id)
        log.info("utterance_received", text=caller_text)

        turn_start = len(self._history)
        self._history.append(LLMMessage(role=LLMMessageRole.USER, content=caller_text))

        for iteration in range(self._max_tool_iterations):
            try:
                response = await self._llm.generate(
                    self._history, system_prompt=self._system_prompt, tools=self._tools
                )
            except LLMProviderError as exc:
                log.error("llm_generate_failed_falling_back_to_human", error=str(exc))
                self._history = self._history[:turn_start]  # drop this whole unanswered turn
                return AgentReply(
                    text=_FALLBACK_MESSAGE, intent=Intent.GENERAL, transfer_to_human=True
                )

            if response.tool_calls:
                log.info(
                    "tool_calls_requested",
                    tools=[tc.name for tc in response.tool_calls],
                    iteration=iteration,
                )
                self._history.append(
                    LLMMessage(
                        role=LLMMessageRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                for tool_call in response.tool_calls:
                    result_text, is_error = await self._execute_tool(tool_call)
                    self._history.append(
                        LLMMessage(
                            role=LLMMessageRole.TOOL,
                            content=result_text,
                            tool_call_id=tool_call.id,
                            tool_call_is_error=is_error,
                        )
                    )
                continue

            intent, reply_text = parse_structured_reply(response.content)
            log.info("intent_detected", intent=intent.value)

            self._history.append(LLMMessage(role=LLMMessageRole.ASSISTANT, content=reply_text))
            self._trim_history()
            log.info("reply_generated", text=reply_text)

            return AgentReply(
                text=reply_text, intent=intent, transfer_to_human=intent in _TRANSFER_INTENTS
            )

        log.error("tool_loop_exceeded_max_iterations", max_iterations=self._max_tool_iterations)
        self._history = self._history[:turn_start]
        return AgentReply(text=_FALLBACK_MESSAGE, intent=Intent.GENERAL, transfer_to_human=True)

    async def _execute_tool(self, tool_call: ToolCall) -> tuple[str, bool]:
        if self._tool_executor is None:
            return f"Tool '{tool_call.name}' is not available.", True
        try:
            return await self._tool_executor(tool_call)
        except Exception as exc:  # a tool must never crash a live call
            logger.error("tool_executor_raised", tool=tool_call.name, error=str(exc))
            return "An internal error occurred while executing this action.", True

    def _trim_history(self) -> None:
        """Keep at most `max_history_turns` full turns.

        Trims on user-message boundaries, not a raw message count: a turn
        that involved tool calls has more than the usual two messages
        (user, assistant), and cutting at an arbitrary message count could
        slice a tool_use/tool_result pair apart, leaving a dangling tool
        result with no matching call in the next request -- which both
        the Anthropic and OpenAI wire formats reject.
        """
        user_indices = [i for i, m in enumerate(self._history) if m.role is LLMMessageRole.USER]
        if len(user_indices) > self._max_history_turns:
            cutoff = user_indices[-self._max_history_turns]
            self._history = self._history[cutoff:]
