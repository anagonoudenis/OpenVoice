"""The STT -> agent -> TTS loop for one call.

LiveKit-agnostic and fully unit-testable: it consumes audio via the
`BaseSTTProvider`/`BaseTTSProvider` interfaces and drives
`ConversationManager`. `openvoice.telephony.livekit_worker` wires this
into a real LiveKit room (VAD segmentation, barge-in, SIP transfer).
"""

from collections.abc import AsyncIterator

import structlog

from openvoice.agent.base import AgentReply
from openvoice.agent.conversation import ConversationManager
from openvoice.stt.base import BaseSTTProvider
from openvoice.tts.base import BaseTTSProvider

logger = structlog.get_logger(__name__)


class CallPipeline:
    """Turns one VAD-segmented utterance's audio into a spoken reply.

    The caller (the LiveKit worker) is responsible for segmenting the raw
    call audio into per-utterance frame streams via VAD before invoking
    this — this class handles exactly one utterance per call.
    """

    def __init__(
        self,
        *,
        stt: BaseSTTProvider,
        tts: BaseTTSProvider,
        conversation: ConversationManager,
        call_id: str,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._conversation = conversation
        self._call_id = call_id

    async def handle_utterance_audio(
        self, audio_frames: AsyncIterator[bytes], *, sample_rate: int = 16000
    ) -> tuple[AgentReply, AsyncIterator[bytes]] | None:
        """Transcribe `audio_frames`, generate a reply, and synthesize it.

        Returns `None` if STT produced no usable transcript (e.g. a pure
        silence/noise segment) — callers should simply stay silent in that
        case rather than treat it as an error.
        """
        log = logger.bind(call_id=self._call_id)
        transcript = ""
        async for segment in self._stt.transcribe_stream(audio_frames, sample_rate=sample_rate):
            if segment.is_final and segment.text:
                transcript = segment.text

        if not transcript:
            log.info("utterance_empty_transcript")
            return None

        reply = await self._conversation.handle_utterance(transcript)
        audio = self._tts.synthesize(reply.text, sample_rate=sample_rate)
        return reply, audio
