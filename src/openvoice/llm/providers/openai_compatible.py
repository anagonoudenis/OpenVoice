"""OpenAI Chat Completions provider.

Also backs the `openai_compatible` provider (`LLMProvider.OPENAI_COMPATIBLE`):
any endpoint that speaks the OpenAI Chat Completions protocol is reachable
by pointing `base_url` at it — DeepSeek, Moonshot/Kimi, Alibaba Qwen
(DashScope), Groq, Together AI, or a self-hosted vLLM/llama.cpp/Ollama
server. There is no separate class per vendor because the wire protocol
and error handling are identical — only the endpoint, model name, and
whether an API key is required differ, all plain constructor parameters.
"""

import json
from collections.abc import Sequence
from typing import Any

import openai
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from openvoice.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMMessageRole,
    LLMProviderError,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)

logger = structlog.get_logger(__name__)

_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API (or a compatible server)."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        base_url: str | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "not-required",
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self._model = model
        self._retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        )

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> LLMResponse:
        openai_messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend(_to_openai_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        async def _call() -> openai.types.chat.ChatCompletion:
            return await self._client.chat.completions.create(**kwargs)  # type: ignore[no-any-return]

        try:
            # `self._client.chat.completions.create` is not passed to the
            # retryer directly: the OpenAI SDK wraps its methods in a way
            # that `inspect.iscoroutinefunction` (which tenacity's
            # AsyncRetrying uses to decide whether to await the call) fails
            # to recognize as async, silently returning an unawaited
            # coroutine instead of the response. Wrapping it in our own
            # plain `async def` closure sidesteps that misdetection.
            response: openai.types.chat.ChatCompletion = await self._retryer(_call)
        except openai.APIError as exc:
            logger.error("openai_call_failed", error=str(exc), model=self._model)
            raise LLMProviderError(f"OpenAI-compatible API call failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=usage.completion_tokens if usage is not None else None,
            tool_calls=_parse_tool_calls(choice.message.tool_calls),
        )


def _to_openai_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Convert our provider-agnostic history into OpenAI's wire format."""
    openai_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.role is LLMMessageRole.TOOL:
            content = message.content
            if message.tool_call_is_error:
                # OpenAI-style `tool` messages have no structured error
                # channel (unlike Anthropic's `tool_result.is_error`) --
                # fold the failure into the text itself so the model still
                # knows the call didn't succeed.
                content = f"Error: {content}"
            openai_messages.append(
                {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
            )
            continue

        if message.role is LLMMessageRole.ASSISTANT and message.tool_calls:
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            continue

        openai_messages.append({"role": message.role.value, "content": message.content})

    return openai_messages


def _parse_tool_calls(raw_tool_calls: Sequence[Any] | None) -> list[ToolCall] | None:
    """Parse the SDK's `message.tool_calls` into our provider-agnostic form.

    A tool call whose `arguments` isn't valid JSON (a model output error,
    not a network failure) doesn't raise -- it's carried through as a
    `_parse_error` marker so the tool dispatcher can feed a clear error
    back to the model instead of the whole turn crashing.
    """
    if not raw_tool_calls:
        return None

    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        function = getattr(tc, "function", None)
        if function is None:
            continue
        try:
            arguments = json.loads(function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {"_parse_error": function.arguments}
        tool_calls.append(ToolCall(id=tc.id, name=function.name, arguments=arguments))

    return tool_calls or None
