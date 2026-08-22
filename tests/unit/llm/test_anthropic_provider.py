"""Tests for AnthropicLLMProvider, mocking the Anthropic SDK client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import APIConnectionError, AuthenticationError

from openvoice.llm.base import LLMMessage, LLMMessageRole, LLMProviderError
from openvoice.llm.providers.anthropic import AnthropicLLMProvider


def _fake_response(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def provider() -> AnthropicLLMProvider:
    return AnthropicLLMProvider(
        api_key="test-key", model="claude-sonnet-5", timeout_seconds=5.0, max_retries=3
    )


async def test_generate_returns_response(provider: AnthropicLLMProvider) -> None:
    provider._client.messages.create = AsyncMock(return_value=_fake_response("Hello!"))  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "Hello!"
    assert result.model == "claude-sonnet-5"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


async def test_generate_passes_system_prompt_when_given(provider: AnthropicLLMProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    await provider.generate(
        [LLMMessage(role=LLMMessageRole.USER, content="Hi")],
        system_prompt="You are a voice assistant.",
    )

    assert mock_create.call_args.kwargs["system"] == "You are a voice assistant."


async def test_generate_omits_system_key_when_not_given(provider: AnthropicLLMProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert "system" not in mock_create.call_args.kwargs


async def test_generate_retries_transient_error_then_succeeds(
    provider: AnthropicLLMProvider,
) -> None:
    mock_create = AsyncMock(
        side_effect=[APIConnectionError(request=_request()), _fake_response("ok")]
    )
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "ok"
    assert mock_create.call_count == 2


async def test_generate_raises_provider_error_after_exhausting_retries(
    provider: AnthropicLLMProvider,
) -> None:
    mock_create = AsyncMock(side_effect=APIConnectionError(request=_request()))
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    with pytest.raises(LLMProviderError):
        await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert mock_create.call_count == 3


async def test_generate_does_not_retry_non_retryable_error(provider: AnthropicLLMProvider) -> None:
    response = httpx.Response(401, request=_request())
    mock_create = AsyncMock(
        side_effect=AuthenticationError(message="bad key", response=response, body=None)
    )
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    with pytest.raises(LLMProviderError):
        await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert mock_create.call_count == 1
