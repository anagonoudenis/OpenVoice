"""Tests for OpenAICompatibleLLMProvider (backs both `openai` and `self_hosted`)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from openvoice.llm.base import LLMMessage, LLMMessageRole, LLMProviderError
from openvoice.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def _fake_response(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


@pytest.fixture
def provider() -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        api_key="test-key", model="gpt-4o", timeout_seconds=5.0, max_retries=3
    )


async def test_generate_returns_response(provider: OpenAICompatibleLLMProvider) -> None:
    provider._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_fake_response("Hello!")
    )

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "Hello!"
    assert result.model == "gpt-4o"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


async def test_generate_prepends_system_message_when_given(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]

    await provider.generate(
        [LLMMessage(role=LLMMessageRole.USER, content="Hi")],
        system_prompt="You are a voice assistant.",
    )

    sent_messages = mock_create.call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are a voice assistant."}


async def test_generate_retries_transient_error_then_succeeds(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    mock_create = AsyncMock(
        side_effect=[APIConnectionError(request=_request()), _fake_response("ok")]
    )
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "ok"
    assert mock_create.call_count == 2


async def test_generate_raises_provider_error_after_exhausting_retries(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    mock_create = AsyncMock(side_effect=APIConnectionError(request=_request()))
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]

    with pytest.raises(LLMProviderError):
        await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert mock_create.call_count == 3


async def test_generate_does_not_retry_non_retryable_error(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    response = httpx.Response(401, request=_request())
    mock_create = AsyncMock(
        side_effect=AuthenticationError(message="bad key", response=response, body=None)
    )
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]

    with pytest.raises(LLMProviderError):
        await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert mock_create.call_count == 1


async def test_self_hosted_style_construction_without_api_key() -> None:
    provider = OpenAICompatibleLLMProvider(
        api_key=None,
        model="mistral-7b",
        timeout_seconds=5.0,
        max_retries=1,
        base_url="http://localhost:8001/v1",
    )
    provider._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_fake_response("local response")
    )

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "local response"
