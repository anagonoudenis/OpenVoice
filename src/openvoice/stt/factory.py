"""STT provider factory. Selects the implementation from `Settings.stt_provider`."""

from openvoice.config import Settings, STTProvider
from openvoice.stt.base import BaseSTTProvider
from openvoice.stt.providers.faster_whisper import FasterWhisperSTTProvider


def get_stt_provider(settings: Settings) -> BaseSTTProvider:
    """Build the STT provider configured in `settings`."""
    if settings.stt_provider is STTProvider.FASTER_WHISPER:
        return FasterWhisperSTTProvider.from_settings(settings)

    if settings.stt_provider is STTProvider.DEEPGRAM:
        raise NotImplementedError(
            "The Deepgram STT provider is not implemented yet (see docs/ROADMAP.md). "
            "Set STT_PROVIDER=faster_whisper."
        )

    raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")  # pragma: no cover
