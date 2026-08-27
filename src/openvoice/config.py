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
    """Supported LLM backends, selected purely via configuration.

    ``OPENAI_COMPATIBLE`` is deliberately generic, not a single vendor:
    it covers any endpoint that speaks the OpenAI Chat Completions
    protocol, which in practice means self-hosted inference (vLLM,
    llama.cpp server, Ollama) *and* hosted providers that expose an
    OpenAI-compatible API — DeepSeek, Moonshot/Kimi, Alibaba Qwen
    (DashScope), Groq, Together AI, and others. Nothing in this codebase
    hardcodes a specific model; which one runs is entirely a config
    choice (``OPENAI_COMPATIBLE_BASE_URL`` / ``_MODEL`` / `_API_KEY``).
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class TTSProvider(StrEnum):
    """Supported text-to-speech backends."""

    PIPER = "piper"
    COQUI = "coqui"
    ELEVENLABS = "elevenlabs"


class STTProvider(StrEnum):
    """Supported speech-to-text backends."""

    FASTER_WHISPER = "faster_whisper"
    DEEPGRAM = "deepgram"


class CalendarProvider(StrEnum):
    """Supported calendar backends. Google first; Outlook/Cal.com later."""

    GOOGLE = "google"


class SMSProvider(StrEnum):
    """Supported SMS backends."""

    TWILIO = "twilio"


class EmailProvider(StrEnum):
    """Supported email backends."""

    RESEND = "resend"


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
        # `.env.example` (and any `.env` copied from it) ships every
        # optional field as `KEY=` with no value, by design, so a user can
        # see what's configurable without having to add it themselves. By
        # pydantic-settings' own default (`env_ignore_empty=False`), an
        # empty string is a real value, not "unset" -- it would satisfy an
        # `Optional[str]` field as `""` rather than falling through to the
        # field's `None` default, silently breaking every `is None` check
        # and any provider (LLM prompt, calendar credentials, ...) reading
        # that field. Setting this to True restores the intended "blank
        # means use the default" behavior everywhere, once, instead of
        # requiring every optional field's every reader to remember to
        # check falsiness instead of identity.
        env_ignore_empty=True,
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
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None
    openai_compatible_api_key: str | None = None
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

    # --- Voice turn-taking (silero VAD, tuned for phone conversation) --------
    # The default (0.55s) was tuned for generic use; phone callers expect a
    # snappier response, and 0.55s of dead air after they stop talking reads
    # as the agent being slow. Lowered a bit as a safe middle ground -- too
    # low risks cutting callers off mid-sentence during a natural pause.
    vad_min_silence_duration_seconds: float = 0.4

    # --- Calendar / booking (pluggable) ---------------------------------------
    calendar_provider: CalendarProvider = CalendarProvider.GOOGLE
    google_calendar_id: str = "primary"
    google_service_account_json_path: str | None = None
    booking_business_hours_start: int = 9
    booking_business_hours_end: int = 17
    booking_default_duration_minutes: int = 30
    booking_search_days_ahead: int = 7
    booking_timezone: str = "UTC"

    # --- SMS / email notifications (pluggable) --------------------------------
    sms_provider: SMSProvider = SMSProvider.TWILIO
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None
    email_provider: EmailProvider = EmailProvider.RESEND
    resend_api_key: str | None = None
    resend_from_email: str | None = None
    notification_request_timeout_seconds: float = 10.0
    notification_max_retries: int = 3

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
        if self.llm_provider is LLMProvider.OPENAI_COMPATIBLE and not (
            self.openai_compatible_base_url and self.openai_compatible_model
        ):
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are both required "
                "when LLM_PROVIDER=openai_compatible -- there's no universal default model "
                "across providers, so it must be set explicitly (e.g. base_url "
                "https://api.deepseek.com/v1 with model deepseek-chat, "
                "https://api.moonshot.ai/v1 with a kimi-* model, "
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1 with a qwen-* model, "
                "or a local vLLM/Ollama server with whatever model it's serving)"
            )
        return self

    @model_validator(mode="after")
    def _validate_tts_provider_credentials(self) -> "Settings":
        """Ensure the selected TTS provider has the credentials it needs."""
        if self.tts_provider is TTSProvider.ELEVENLABS and not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs")
        return self

    # Note: calendar/SMS/email credentials are deliberately NOT validated
    # here at startup, unlike the LLM/TTS providers. Booking and
    # notifications are an optional sub-feature (a CRM-only or
    # support-only deployment shouldn't be forced to configure Google
    # Calendar just to boot); their factories
    # (`openvoice.calendar.factory.get_calendar_provider`,
    # `openvoice.notifications.factory.get_sms_provider`/
    # `get_email_provider`) raise a clear `RuntimeError` if invoked
    # without the credentials the selected provider needs.


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.

    Cached so validation only runs once per process; tests that need a
    fresh instance should call ``get_settings.cache_clear()`` first.
    """
    return Settings()
