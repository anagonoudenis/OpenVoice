# 3. Bridge OpenVoice providers into LiveKit via node overrides, not plugin subclasses

## Status

Accepted

## Context

LiveKit Agents offers two ways to plug a custom STT/LLM/TTS backend into
an `AgentSession`: (a) subclass `stt.STT`/`llm.LLM`/`tts.TTS` directly, or
(b) override `Agent.stt_node`/`llm_node`/`tts_node` to intercept the
pipeline at a higher level. Since OpenVoice already has its own pluggable
provider interfaces (`BaseLLMProvider`, `BaseSTTProvider`,
`BaseTTSProvider`), either approach just needs to bridge into whichever
LiveKit surface is used.

Option (a) also requires implementing LiveKit's internal streaming
primitives (`stt.SpeechStream`/`tts.ChunkedStream`/`llm.LLMStream`), whose
exact internals were not available in the fetched documentation. Option
(b) has a narrow, three-method surface.

No LiveKit server or SIP trunk is available in this environment, so
nothing here could be tested end-to-end. To keep risk bounded, every type
actually used (`stt.SpeechEvent`, `stt.SpeechData`, `llm.ChatChunk`,
`llm.ChoiceDelta`, `rtc.AudioFrame`, `vad.VADEvent`, `vad.VADStream`,
`LanguageCode`) was verified by introspecting the real installed
`livekit-agents==1.6.9` / `livekit==1.1.14` packages (constructor
signatures, abstract methods) rather than recalled from training data —
see the module docstrings in `openvoice/telephony/livekit_agent.py` and
`worker.py` for exactly what was and wasn't confirmed this way.

## Decision

Bridge via `Agent.stt_node`/`llm_node`/`tts_node` overrides
(`OpenVoiceAgent` in `openvoice/telephony/livekit_agent.py`):

- `stt_node` runs its own `vad.VADStream` to segment continuous call
  audio into utterances (matching `BaseSTTProvider.transcribe_stream`'s
  one-utterance-per-call contract), then yields `stt.SpeechEvent`s.
- `llm_node` reads the latest user message off LiveKit's `ChatContext`,
  routes it through `ConversationManager` (which keeps its own
  provider-agnostic history), and yields a single `llm.ChatChunk`.
- `tts_node` joins the (already-complete) reply text and yields
  `rtc.AudioFrame`s from the configured `BaseTTSProvider`.

VAD-based turn detection and barge-in/interruption handling are left to
`AgentSession` itself (`vad=`, `allow_interruptions=True`), not
reimplemented here.

Human transfer uses `livekit.api`'s `SipService.transfer_sip_participant`
(`openvoice/telephony/worker.py::_transfer_to_human`), wrapped so any
failure is logged and swallowed rather than crashing the call — matching
the project's "every external call has a defined fallback" rule.

## Consequences

- Fully covered by unit tests using the *real* LiveKit types (only VAD is
  faked, since `silero.VAD.load()` needs a real ONNX model) — see
  `tests/unit/telephony/test_livekit_agent.py`. One real bug was caught
  this way: `tts_node` crashed on an odd-length PCM chunk (`AudioFrame`
  requires an even byte count); it now truncates the stray trailing byte
  and logs a warning instead of dropping the whole reply.
- `worker.entrypoint` itself (the `JobContext`/`AgentSession` wiring) is
  *not* unit tested — mocking the whole session lifecycle would test that
  functions are called in order more than any real behavior. It must be
  smoke-tested against a real LiveKit dev server + SIP trunk before
  production use.
- If LiveKit's node-override contract changes in a future SDK version,
  this integration breaks at those three methods specifically — re-run
  the introspection steps in the module docstrings against the new
  version rather than assuming the signatures still match.
