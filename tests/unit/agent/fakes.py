"""Test double for `BaseLLMProvider`, shared across agent-core tests."""

from collections.abc import Sequence

from openvoice.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    ToolDefinition,
)


class FakeLLMProvider(BaseLLMProvider):
    """Returns queued canned responses in order, or always raises if `fail=True`.

    A queued response is either a plain string (wrapped into a final,
    tool-call-free `LLMResponse`) or a full `LLMResponse` -- pass the
    latter to simulate the model requesting a tool call.
    """

    def __init__(
        self, responses: list[str | LLMResponse] | None = None, *, fail: bool = False
    ) -> None:
        self.responses = list(responses or [])
        self.fail = fail
        self.calls: list[list[LLMMessage]] = []
        self.system_prompts: list[str | None] = []
        self.tools_passed: list[Sequence[ToolDefinition] | None] = []

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.system_prompts.append(system_prompt)
        self.tools_passed.append(tools)
        if self.fail:
            raise LLMProviderError("fake provider failure")
        if not self.responses:
            return LLMResponse(content="ok", model="fake-model")
        queued = self.responses.pop(0)
        if isinstance(queued, LLMResponse):
            return queued
        return LLMResponse(content=queued, model="fake-model")
