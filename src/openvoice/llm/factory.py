"""LLM provider factory.

The only place in the codebase that maps `Settings.llm_provider` to a
concrete implementation. Everything downstream (agent core, intent
detection, summaries) depends on `BaseLLMProvider`, obtained by calling
`get_llm_provider(settings)` — swapping providers is a config change.
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

    if settings.llm_provider is LLMProvider.SELF_HOSTED:
        if settings.self_hosted_llm_base_url is None:  # pragma: no cover -- guarded by Settings
            raise RuntimeError("SELF_HOSTED_LLM_BASE_URL missing despite LLM_PROVIDER=self_hosted")
        return OpenAICompatibleLLMProvider(
            api_key=None,
            model=settings.self_hosted_llm_model or "default",
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
            base_url=settings.self_hosted_llm_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")  # pragma: no cover
