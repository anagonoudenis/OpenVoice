"""OpenAI Chat Completions provider.

Also backs the `openai_compatible` provider (`LLMProvider.OPENAI_COMPATIBLE`):
any endpoint that speaks the OpenAI Chat Completions protocol is reachable
by pointing `base_url` at it — DeepSeek, Moonshot/Kimi, Alibaba Qwen
(DashScope), Groq, Together AI, or a self-hosted vLLM/llama.cpp/Ollama
server. There is no separate class per vendor because the wire protocol
and error handling are identical — only the endpoint, model name, and
whether an API key is required differ, all plain constructor parameters.
"""

from collections.abc import Sequence
from typing import Any

import openai
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from openvoice.llm.base import BaseLLMProvider, LLMMessage, LLMProviderError, LLMResponse

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
    ) -> LLMResponse:
        openai_messages: list[dict[str, str]] = []
        if system_prompt is not None:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend({"role": m.role.value, "content": m.content} for m in messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response: openai.types.chat.ChatCompletion = await self._retryer(
                self._client.chat.completions.create, **kwargs
            )
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
        )
