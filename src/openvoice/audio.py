"""Shared PCM16 audio utilities.

Used on both ends of the voice pipeline: `tts.providers.piper` resamples
*outgoing* synthesized audio to whatever rate the call needs (a Piper
voice always synthesizes at its own native rate), and
`stt.providers.faster_whisper` resamples *incoming* caller audio to the
16 kHz Whisper's models are trained on (the caller's actual capture rate
depends on the audio source and isn't guaranteed to already be 16 kHz).
"""

import numpy as np


def resample_pcm16(pcm: bytes, *, from_rate: int, to_rate: int) -> bytes:
    """Linearly resample mono PCM16 `pcm` from `from_rate` to `to_rate`.

    Linear interpolation, not a proper band-limited resampler -- good
    enough for speech at these rates, and avoids pulling in a dedicated
    resampling dependency for two call sites.
    """
    if from_rate == to_rate or not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    target_count = max(1, round(samples.size * to_rate / from_rate))
    resampled = np.interp(
        np.linspace(0, samples.size - 1, target_count),
        np.arange(samples.size),
        samples,
    )
    return resampled.astype(np.int16).tobytes()
