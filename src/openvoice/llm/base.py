"""Abstract LLM provider interface.

Business logic (intent detection, conversation handling, post-call
summaries) depends only on the types and interface defined here — never
on a vendor SDK. See `openvoice.llm.factory.get_llm_provider` for how a
concrete implementation is selected from configuration.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel


class LLMMessageRole(StrEnum):
    """Role of a message in a conversation, per the common chat-completion shape."""

    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """One turn of conversation history passed to the LLM."""

    role: LLMMessageRole
    content: str


class LLMResponse(BaseModel):
    """A completed LLM generation."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


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
    ) -> LLMResponse:
        """Generate a completion for the given conversation history.

        Implementations must apply a timeout and a retry-with-backoff
        policy internally, and raise `LLMProviderError` (never a raw
        vendor exception) once that budget is exhausted.
        """
        raise NotImplementedError
