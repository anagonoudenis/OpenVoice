"""LLM provider factory.

The only place in the codebase that maps `Settings.llm_provider` to a
concrete implementation. Everything downstream (agent core, intent
detection, summaries) depends on `BaseLLMProvider`, obtained by calling
`get_llm_provider(settings)` — swapping providers, or switching to a
different model entirely (DeepSeek, Kimi, Qwen, Groq, a self-hosted
vLLM/Ollama server, ...) via `LLM_PROVIDER=openai_compatible`, is a
config change, never a code change.
"""

from openvoice.config import LLMProvider, Settings
from openvoice.llm.base import BaseLLMProvider
from openvoice.llm.providers.anthropic import AnthropicLLMProvider
from openvoice.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def get_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Build the LLM provider configured in `settings`.

    `Settings`' own validation already guarantees the credentials required
    by the selected provider are present, so the `RuntimeError`s below are
    unreachable in practice; they exist so a future change can't silently
    construct a misconfigured provider.
    """
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        if settings.anthropic_api_key is None:  # pragma: no cover -- guarded by Settings validation
            raise RuntimeError("ANTHROPIC_API_KEY missing despite LLM_PROVIDER=anthropic")
        return AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if settings.llm_provider is LLMProvider.OPENAI:
        if settings.openai_api_key is None:  # pragma: no cover -- guarded by Settings validation
            raise RuntimeError("OPENAI_API_KEY missing despite LLM_PROVIDER=openai")
        return OpenAICompatibleLLMProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if settings.llm_provider is LLMProvider.OPENAI_COMPATIBLE:
        if settings.openai_compatible_base_url is None:  # pragma: no cover
            raise RuntimeError(
                "OPENAI_COMPATIBLE_BASE_URL missing despite LLM_PROVIDER=openai_compatible"
            )
        if settings.openai_compatible_model is None:  # pragma: no cover
            raise RuntimeError(
                "OPENAI_COMPATIBLE_MODEL missing despite LLM_PROVIDER=openai_compatible"
            )
        return OpenAICompatibleLLMProvider(
            api_key=settings.openai_compatible_api_key,
            model=settings.openai_compatible_model,
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
            base_url=settings.openai_compatible_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")  # pragma: no cover
