"""Anthropic Claude LLM provider."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
import structlog
from anthropic.lib.streaming import AsyncMessageStream
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
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class AnthropicLLMProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_anthropic_messages(messages),
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        async def _call() -> anthropic.types.Message:
            return await self._client.messages.create(**kwargs)  # type: ignore[no-any-return]

        try:
            # `self._client.messages.create` is not passed to the retryer
            # directly: the Anthropic SDK wraps its methods in a way that
            # `inspect.iscoroutinefunction` (which tenacity's AsyncRetrying
            # uses to decide whether to await the call) fails to recognize
            # as async, silently returning an unawaited coroutine instead
            # of the response. Wrapping it in our own plain `async def`
            # closure sidesteps that misdetection.
            response: anthropic.types.Message = await self._retryer(_call)
        except anthropic.APIError as exc:
            logger.error("anthropic_call_failed", error=str(exc), model=self._model)
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            content="".join(text_parts),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=tool_calls or None,
        )

    async def generate_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _to_anthropic_messages(messages),
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        # `.messages.stream(...)` itself makes no network call -- it just
        # builds a context-manager object; the request happens on
        # `__aenter__`. Retrying is done by hand here (rather than via
        # `self._retryer` on a whole `async with` block) specifically so
        # only *that* connection-establishing step is retried, never the
        # token iteration after it: some text may already have reached
        # the caller (and been spoken) by the time a later chunk fails,
        # so silently restarting mid-stream risks duplicated or
        # out-of-order speech.
        stream_manager = self._client.messages.stream(**kwargs)

        async def _enter() -> AsyncMessageStream[Any]:
            return await stream_manager.__aenter__()

        try:
            stream: AsyncMessageStream[Any] = await self._retryer(_enter)
        except anthropic.APIError as exc:
            logger.error("anthropic_stream_connect_failed", error=str(exc), model=self._model)
            raise LLMProviderError(f"Anthropic streaming call failed: {exc}") from exc

        try:
            async for text in stream.text_stream:
                yield text
        except anthropic.APIError as exc:
            logger.error("anthropic_stream_failed", error=str(exc), model=self._model)
            raise LLMProviderError(f"Anthropic streaming call failed: {exc}") from exc
        finally:
            await stream_manager.__aexit__(None, None, None)


def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Convert our provider-agnostic history into Anthropic's wire format.

    Anthropic has no dedicated "tool" role: a tool result is a `user`
    message with `tool_result` content blocks, and *all* results
    answering one assistant turn's tool calls must land in a single such
    message (the API rejects one `tool_result` per message for a
    multi-tool turn) -- so consecutive `TOOL` messages in our history are
    merged into one Anthropic message here, not sent one-by-one.
    """
    anthropic_messages: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        message = messages[i]

        if message.role is LLMMessageRole.TOOL:
            tool_result_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].role is LLMMessageRole.TOOL:
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": messages[i].tool_call_id,
                        "content": messages[i].content,
                        "is_error": messages[i].tool_call_is_error,
                    }
                )
                i += 1
            anthropic_messages.append({"role": "user", "content": tool_result_blocks})
            continue

        if message.role is LLMMessageRole.ASSISTANT and message.tool_calls:
            content_blocks: list[dict[str, Any]] = []
            if message.content:
                content_blocks.append({"type": "text", "text": message.content})
            content_blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in message.tool_calls
            )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            i += 1
            continue

        anthropic_messages.append({"role": message.role.value, "content": message.content})
        i += 1

    return anthropic_messages
