"""TTS provider factory. Selects the implementation from `Settings.tts_provider`."""

from openvoice.config import Settings, TTSProvider
from openvoice.tts.base import BaseTTSProvider
from openvoice.tts.providers.elevenlabs import ElevenLabsTTSProvider
from openvoice.tts.providers.piper import PiperTTSProvider


def get_tts_provider(settings: Settings) -> BaseTTSProvider:
    """Build the TTS provider configured in `settings`."""
    if settings.tts_provider is TTSProvider.PIPER:
        return PiperTTSProvider.from_settings(settings)

    if settings.tts_provider is TTSProvider.ELEVENLABS:
        if settings.elevenlabs_api_key is None:  # pragma: no cover -- guarded by Settings
            raise RuntimeError("ELEVENLABS_API_KEY missing despite TTS_PROVIDER=elevenlabs")
        return ElevenLabsTTSProvider(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model,
        )

    if settings.tts_provider is TTSProvider.COQUI:
        raise NotImplementedError(
            "The Coqui TTS provider is intentionally not implemented: coqui-ai/TTS is "
            "archived and its XTTS-v2 model is non-commercially licensed (see "
            "docs/ARCHITECTURE.md). Use TTS_PROVIDER=piper or elevenlabs."
        )

    raise ValueError(f"Unsupported TTS provider: {settings.tts_provider}")  # pragma: no cover
