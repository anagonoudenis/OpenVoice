# 4. Generalize the "self-hosted" LLM provider to "OpenAI-compatible"

## Status

Accepted

## Context

The third LLM provider option was originally named `self_hosted`
(`SELF_HOSTED_LLM_BASE_URL`/`SELF_HOSTED_LLM_MODEL`), matching the original
spec's wording: "a self-hosted OpenAI-compatible model (e.g. vLLM +
Llama/Mistral)". Its factory hardcoded `api_key=None`, since a self-hosted
inference server typically needs no authentication.

In practice, the OpenAI Chat Completions wire protocol is also what a
number of *hosted, authenticated* model providers expose — DeepSeek,
Moonshot (Kimi), Alibaba Qwen (DashScope), Groq, Together AI among them.
`OpenAICompatibleLLMProvider` (the class backing this option) already
supported an optional API key at the constructor level, but the factory
never passed one through for this branch, and the naming ("self-hosted")
actively suggested this path was for private infrastructure only. A user
wanting to point OpenVoice at DeepSeek or Kimi had no working option
without editing code.

## Decision

- Renamed `LLMProvider.SELF_HOSTED` → `LLMProvider.OPENAI_COMPATIBLE`
  (env value `openai_compatible`), and its settings fields to
  `OPENAI_COMPATIBLE_BASE_URL` / `_MODEL` / `_API_KEY` (new).
- The factory now passes `settings.openai_compatible_api_key` through
  instead of hardcoding `None` — a real bug fix, not just a rename: hosted
  providers requiring auth were previously unreachable via this path.
- `Settings` now requires **both** base URL and model when this provider
  is selected (previously the model silently defaulted to the string
  `"default"`, which is meaningless for a real hosted API and would have
  surfaced as a confusing model-not-found error instead of a clear config
  error at boot).
- `.env.example`, the README, and `ARCHITECTURE.md` now document concrete
  base URLs for DeepSeek, Kimi, Qwen, and Groq alongside the still-fully-
  supported local vLLM/Ollama case (verified against each provider's
  current docs at authoring time, not recalled from memory).

This is a breaking rename (pre-1.0, no released version, so no migration
path is provided). The class backing this provider stays
`OpenAICompatibleLLMProvider` — that name was already accurate and didn't
need to change; only the config-facing provider enum/field names did.

## Consequences

- OpenVoice's "pluggable LLM" claim now actually covers any OpenAI-
  compatible model, hosted or local, not just literal self-hosted infra —
  closing a real gap between the architecture's stated goal and what was
  configurable.
- Anyone with an existing `.env` using `LLM_PROVIDER=self_hosted` /
  `SELF_HOSTED_LLM_*` must update to the new names; there are no external
  users of this pre-1.0 project yet, so no deprecation shim was added.
