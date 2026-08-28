"""Abstract LLM provider interface.

Business logic (intent detection, conversation handling, post-call
summaries) depends only on the types and interface defined here — never
on a vendor SDK. See `openvoice.llm.factory.get_llm_provider` for how a
concrete implementation is selected from configuration.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class LLMMessageRole(StrEnum):
    """Role of a message in a conversation, per the common chat-completion shape."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDefinition(BaseModel):
    """One callable tool exposed to the model, in provider-agnostic form.

    `parameters` is a JSON Schema object (the `{"type": "object",
    "properties": {...}, "required": [...]}` shape both Anthropic's
    `input_schema` and OpenAI's `parameters` expect natively) -- each
    provider passes it through with no translation needed.
    """

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """One invocation of a tool the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]


class LLMMessage(BaseModel):
    """One turn of conversation history passed to the LLM.

    `tool_calls` is set on an `ASSISTANT` message that requested tool
    calls (reconstructed from a prior `LLMResponse.tool_calls` so the
    next call in the loop has full context). `tool_call_id` and
    `tool_call_is_error` are set on a `TOOL` message: which call this is
    the result of, and whether it represents a failure -- providers that
    support a structured error channel (Anthropic's `tool_result.is_error`)
    use it directly; providers that don't (OpenAI-style `tool` messages)
    fold it into the content text instead.
    """

    role: LLMMessageRole
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    tool_call_is_error: bool = False


class LLMResponse(BaseModel):
    """A completed LLM generation.

    `tool_calls` is non-empty exactly when the model chose to call one or
    more tools instead of (or before) replying in natural language --
    callers must check it before treating `content` as the final reply.
    """

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[ToolCall] | None = None


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails after exhausting its retry budget.

    Callers (e.g. the call-handling loop) catch this specifically to trigger
    the agent's defined fallback (transfer to human, apology + callback)
    instead of letting a live call hang on an unhandled exception.
    """


class BaseLLMProvider(ABC):
    """Abstract LLM backend. One instance is built per process by the factory."""

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given conversation history.

        Implementations must apply a timeout and a retry-with-backoff
        policy internally, and raise `LLMProviderError` (never a raw
        vendor exception) once that budget is exhausted. When `tools` is
        given and the model chooses to use one, the returned
        `LLMResponse.tool_calls` is non-empty and `content` may be empty
        -- callers must not treat it as a final reply in that case.
        """
        raise NotImplementedError
