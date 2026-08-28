"""Tests for OpenAICompatibleLLMProvider (backs both `openai` and `openai_compatible`)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from openvoice.llm.base import (
    LLMMessage,
    LLMMessageRole,
    LLMProviderError,
    ToolCall,
    ToolDefinition,
)
from openvoice.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def _fake_response(
    text: str = "hello", *, tool_calls: list[object] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tool_calls))],
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


async def test_generate_awaits_a_sync_wrapper_that_returns_a_coroutine(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    """Regression test for a real bug: the OpenAI/Anthropic SDKs' `create`
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

    provider._client.chat.completions.create = _sync_wrapper_returning_coroutine  # type: ignore[method-assign]

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.content == "really awaited"


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


async def test_local_server_style_construction_without_api_key() -> None:
    """A self-hosted vLLM/Ollama server typically needs no auth at all."""
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


async def test_generate_passes_tool_definitions(provider: OpenAICompatibleLLMProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response())
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]
    tool = ToolDefinition(
        name="check_availability",
        description="Find open appointment slots.",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")], tools=[tool])

    sent_tools = mock_create.call_args.kwargs["tools"]
    assert sent_tools == [
        {
            "type": "function",
            "function": {
                "name": "check_availability",
                "description": "Find open appointment slots.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]


async def test_generate_parses_tool_calls_from_response(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    raw_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="check_availability", arguments='{"duration_minutes": 30}'),
    )
    provider._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_fake_response("", tool_calls=[raw_call])
    )

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.tool_calls == [
        ToolCall(id="call_1", name="check_availability", arguments={"duration_minutes": 30})
    ]


async def test_generate_handles_malformed_tool_call_arguments_without_raising(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    raw_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="check_availability", arguments="not json")
    )
    provider._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_fake_response("", tool_calls=[raw_call])
    )

    result = await provider.generate([LLMMessage(role=LLMMessageRole.USER, content="Hi")])

    assert result.tool_calls is not None
    assert result.tool_calls[0].arguments == {"_parse_error": "not json"}


async def test_generate_sends_assistant_tool_call_and_tool_result_messages(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_response("Done"))
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]
    history = [
        LLMMessage(role=LLMMessageRole.USER, content="Book me a slot"),
        LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="check_availability", arguments={"duration_minutes": 30})
            ],
        ),
        LLMMessage(role=LLMMessageRole.TOOL, content='{"slots": []}', tool_call_id="call_1"),
    ]

    await provider.generate(history)

    sent = mock_create.call_args.kwargs["messages"]
    assert sent[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "check_availability", "arguments": '{"duration_minutes": 30}'},
        }
    ]
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"slots": []}'}


async def test_generate_marks_tool_error_in_content_for_openai_style(
    provider: OpenAICompatibleLLMProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_response("Sorry"))
    provider._client.chat.completions.create = mock_create  # type: ignore[method-assign]
    history = [
        LLMMessage(
            role=LLMMessageRole.TOOL,
            content="slot no longer available",
            tool_call_id="call_1",
            tool_call_is_error=True,
        )
    ]

    await provider.generate(history)

    sent = mock_create.call_args.kwargs["messages"]
    assert sent[0]["content"] == "Error: slot no longer available"


async def test_hosted_compatible_provider_style_construction_with_api_key() -> None:
    """DeepSeek/Kimi/Qwen/Groq-style: a hosted endpoint that requires a real key."""
    provider = OpenAICompatibleLLMProvider(
        api_key="sk-real-key",
        model="deepseek-chat",
        timeout_seconds=5.0,
        max_retries=1,
        base_url="https://api.deepseek.com/v1",
    )

    assert provider._client.api_key == "sk-real-key"
    assert str(provider._client.base_url) == "https://api.deepseek.com/v1/"
