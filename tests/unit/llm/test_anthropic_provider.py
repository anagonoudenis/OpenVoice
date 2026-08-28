"""Tests for AnthropicLLMProvider, mocking the Anthropic SDK client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import APIConnectionError, AuthenticationError

from openvoice.llm.base import (
    LLMMessage,
    LLMMessageRole,
    LLMProviderError,
    ToolCall,
    ToolDefinition,
)
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


async def test_generate_awaits_a_sync_wrapper_that_returns_a_coroutine(
    provider: AnthropicLLMProvider,
) -> None:
    """Regression test for a real bug: the Anthropic/OpenAI SDKs' `create`
    methods are plain (non-`async def`) functions that return a coroutine
    when called -- the correct calling convention is still `await
    client...create(...)`, but `inspect.iscoroutinefunction(create)` is
    False, which used to make tenacity's `AsyncRetrying` skip awaiting it
    and return the coroutine object itself instead of the response. Caught
    by an actual DeepSeek API call in production testing, not by any
    AsyncMock-based test above (AsyncMock is *always* correctly detected
    as async, which is exactly why it masked this bug).
    """

    def _sync_wrapper_returning_coroutine(**kwargs: object) -> object:
        async def _inner() -> SimpleNamespace:
            return _fake_response("really awaited")

        return _inner()

    provider._client.messages.create = _sync_wrapper_returning_coroutine  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "really awaited"


async def test_generate_passes_tool_definitions(provider: AnthropicLLMProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.messages.create = mock_create  # type: ignore[method-assign]
    tool = ToolDefinition(
        name="check_availability",
        description="Find open appointment slots.",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")], tools=[tool])

    sent_tools = mock_create.call_args.kwargs["tools"]
    assert sent_tools == [
        {
            "name": "check_availability",
            "description": "Find open appointment slots.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]


async def test_generate_parses_tool_use_blocks_from_response(
    provider: AnthropicLLMProvider,
) -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="check_availability",
                input={"duration_minutes": 30},
            ),
        ],
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    provider._client.messages.create = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "Let me check."
    assert result.tool_calls == [
        ToolCall(id="toolu_1", name="check_availability", arguments={"duration_minutes": 30})
    ]


async def test_generate_merges_consecutive_tool_results_into_one_user_message(
    provider: AnthropicLLMProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.messages.create = mock_create  # type: ignore[method-assign]
    history = [
        LLMMessage(role=LLMMessageRole.USER, content="Book me two things"),
        LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="On it.",
            tool_calls=[
                ToolCall(id="toolu_1", name="check_availability", arguments={}),
                ToolCall(
                    id="toolu_2", name="check_availability", arguments={"duration_minutes": 60}
                ),
            ],
        ),
        LLMMessage(role=LLMMessageRole.TOOL, content="slot A", tool_call_id="toolu_1"),
        LLMMessage(
            role=LLMMessageRole.TOOL,
            content="no slots",
            tool_call_id="toolu_2",
            tool_call_is_error=True,
        ),
    ]

    await provider.generate(history)

    sent = mock_create.call_args.kwargs["messages"]
    assert sent[1]["content"] == [
        {"type": "text", "text": "On it."},
        {"type": "tool_use", "id": "toolu_1", "name": "check_availability", "input": {}},
        {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "check_availability",
            "input": {"duration_minutes": 60},
        },
    ]
    assert sent[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "slot A",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "no slots",
                "is_error": True,
            },
        ],
    }


async def test_generate_does_not_retry_non_retryable_error(provider: AnthropicLLMProvider) -> None:
    response = httpx.Response(401, request=_request())
    mock_create = AsyncMock(
        side_effect=AuthenticationError(message="bad key", response=response, body=None)
    )
    provider._client.messages.create = mock_create  # type: ignore[method-assign]

    with pytest.raises(LLMProviderError):
        await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert mock_create.call_count == 1
