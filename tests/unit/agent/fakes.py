"""Test double for `BaseLLMProvider`, shared across agent-core tests."""

from collections.abc import AsyncIterator, Sequence

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
    latter to simulate the model requesting a tool call. `generate_stream`
    draws from the same queue (using just the text), split into
    `stream_chunk_size`-character pieces (default: the whole text as one
    chunk) so tests can exercise real multi-delta streaming behavior.
    """

    def __init__(
        self,
        responses: list[str | LLMResponse] | None = None,
        *,
        fail: bool = False,
        stream_chunk_size: int | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.fail = fail
        self.stream_chunk_size = stream_chunk_size
        self.calls: list[list[LLMMessage]] = []
        self.system_prompts: list[str | None] = []
        self.tools_passed: list[Sequence[ToolDefinition] | None] = []
        self.stream_calls: list[list[LLMMessage]] = []

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

    async def generate_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(list(messages))
        if self.fail:
            raise LLMProviderError("fake provider failure")
        if not self.responses:
            text = "ok"
        else:
            queued = self.responses.pop(0)
            text = queued.content if isinstance(queued, LLMResponse) else queued

        if self.stream_chunk_size is None:
            yield text
            return
        for i in range(0, len(text), self.stream_chunk_size):
            yield text[i : i + self.stream_chunk_size]
