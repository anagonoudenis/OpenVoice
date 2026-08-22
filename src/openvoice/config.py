"""Application configuration.

All configuration is loaded from environment variables (optionally via a
local ``.env`` file) and validated with Pydantic at process startup. There
are no hardcoded secrets: any field required for the app to function safely
has no default, so the app raises and refuses to start if it is missing.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Supported LLM backends, selected purely via configuration."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    SELF_HOSTED = "self_hosted"


class TTSProvider(StrEnum):
    """Supported text-to-speech backends."""

    PIPER = "piper"
    COQUI = "coqui"
    ELEVENLABS = "elevenlabs"


class STTProvider(StrEnum):
    """Supported speech-to-text backends."""

    FASTER_WHISPER = "faster_whisper"
    DEEPGRAM = "deepgram"


class Settings(BaseSettings):
    """Central application settings, validated once at startup.

    Instantiating this class with missing/invalid critical configuration
    (e.g. no API key for the selected LLM provider) raises a
    ``pydantic.ValidationError``, which is intentional: the app must fail
    fast at boot rather than fail mid-call.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = Environment.LOCAL
    debug: bool = False

    # --- Web / API -----------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 -- intentional default for a containerized service
    api_port: int = 8000
    secret_key: str = Field(..., min_length=16)

    # --- Database --------------------------------------------------------
    database_url: str = Field(
        ...,
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    database_pool_size: int = 10
    database_echo: bool = False

    # --- Redis -------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM (pluggable via provider factory) -------------------------------
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    self_hosted_llm_base_url: str | None = None
    self_hosted_llm_model: str | None = None
    llm_request_timeout_seconds: float = 15.0
    llm_max_retries: int = 3

    # --- STT / TTS (pluggable) ----------------------------------------------
    stt_provider: STTProvider = STTProvider.FASTER_WHISPER
    tts_provider: TTSProvider = TTSProvider.PIPER
    deepgram_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_turbo_v2_5"
    faster_whisper_model_size: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    piper_voice_model_path: str = "models/piper/en_US-amy-medium.onnx"

    # --- Agent (per-company, never hardcoded in business logic) --------------
    agent_company_name: str = "our company"
    agent_system_prompt: str | None = None
    agent_max_history_turns: int = 20
    agent_human_transfer_number: str | None = None

    # --- Observability -------------------------------------------------------
    sentry_dsn: str | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_llm_provider_credentials(self) -> "Settings":
        """Ensure the selected LLM provider has the credentials it needs."""
        if self.llm_provider is LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider is LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider is LLMProvider.SELF_HOSTED and not self.self_hosted_llm_base_url:
            raise ValueError("SELF_HOSTED_LLM_BASE_URL is required when LLM_PROVIDER=self_hosted")
        return self

    @model_validator(mode="after")
    def _validate_tts_provider_credentials(self) -> "Settings":
        """Ensure the selected TTS provider has the credentials it needs."""
        if self.tts_provider is TTSProvider.ELEVENLABS and not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.

    Cached so validation only runs once per process; tests that need a
    fresh instance should call ``get_settings.cache_clear()`` first.
    """
    return Settings()
