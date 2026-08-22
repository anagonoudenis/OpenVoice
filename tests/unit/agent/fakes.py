"""Test double for `BaseLLMProvider`, shared across agent-core tests."""

from collections.abc import Sequence

from openvoice.llm.base import BaseLLMProvider, LLMMessage, LLMProviderError, LLMResponse


class FakeLLMProvider(BaseLLMProvider):
    """Returns queued canned responses in order, or always raises if `fail=True`."""

    def __init__(self, responses: list[str] | None = None, *, fail: bool = False) -> None:
        self.responses = list(responses or [])
        self.fail = fail
        self.calls: list[list[LLMMessage]] = []

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.fail:
            raise LLMProviderError("fake provider failure")
        content = self.responses.pop(0) if self.responses else "ok"
        return LLMResponse(content=content, model="fake-model")
