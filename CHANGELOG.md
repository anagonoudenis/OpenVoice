# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
doesn't use semantic version tags yet (pre-1.0, Phase 1 MVP).

## [Unreleased]

### Added — Phase 1.3: streaming LLM → TTS

The last big latency lever: previously `llm_node` awaited the *entire*
reply before yielding anything, and `tts_node` joined *all* of that text
before synthesizing a single audio chunk -- so a caller heard nothing at
all until the whole reply had both finished generating and finished
being spoken to the TTS engine, even though LiveKit's own pipeline was
built to stream both stages.

- **`BaseLLMProvider.generate_stream()`**: a new abstract method yielding
  incremental text deltas, implemented natively against both providers'
  real streaming APIs (Anthropic's `messages.stream()`/`text_stream`,
  OpenAI-compatible's `stream=True` chunks). Deliberately has no `tools`
  parameter -- reassembling incrementally fragmented tool-call deltas
  across both wire formats is real, separate complexity this doesn't
  take on; `ConversationManager` only streams a turn it already knows
  won't request a tool. Retries the connection-establishing step only,
  never mid-stream (some text may already be reaching the caller and
  being spoken by the time a later chunk fails, so silently restarting
  would risk duplicated or out-of-order speech).
- **The structured-reply format changed from JSON to a trailing plain-text
  marker** (`<reply>\n###INTENT: <label>`, see
  `openvoice.agent.structured_reply`): JSON doesn't stream safely --
  TTS can only speak text known to be final, and a streamed JSON string
  value needs unescaping that isn't safe to do on an arbitrary partial
  suffix. A trailing marker sidesteps that: every character read before
  it is exactly what should be spoken, no decoding step. This also
  simplified the non-streaming parser (no more JSON-repair fallback
  logic needed).
- **`StreamingReplyExtractor`**: incrementally extracts speakable text
  from a stream of deltas without ever emitting a prefix of the marker
  itself, since a provider's chunk boundaries have no relationship to
  where the marker falls in the text. Its own test suite tries *every*
  possible split point of a known response, which caught a real bug
  before it shipped: if the marker landed exactly at a delta boundary,
  the intent label arriving in the next delta was silently discarded
  instead of accumulated, always falling back to `Intent.GENERAL`.
- **`ConversationManager.handle_utterance_stream`**: the streaming
  counterpart to `handle_utterance`, used by `OpenVoiceAgent.llm_node`.
  Only actually streams when no tools are configured for the
  conversation; falls back to calling `handle_utterance` and yielding
  its result as one chunk otherwise, so callers can use it
  unconditionally and get real streaming exactly when it's safe to.
- **`OpenVoiceAgent.tts_node` now synthesizes sentence-by-sentence** as
  text arrives from `llm_node`, instead of buffering the whole reply
  first -- sentence, not token, because Piper/ElevenLabs (like most TTS
  engines) need a full clause of context for coherent prosody. The
  sentence-boundary heuristic intentionally trades a little prosody for
  simplicity and safety (e.g. "Dr. Smith" still splits after "Dr.") but
  never drops or garbles text -- anything left over once the input
  stream ends is flushed as a final sentence.

### Added — Phase 1.2: production-reliability hardening

Found by a full-project audit specifically looking for what could go
wrong once real calls and real bookings start happening, not by
incremental feature work.

- **Sentry + Prometheus, actually wired up.** `Settings.sentry_dsn`
  existed since Phase 1 but nothing ever called `sentry_sdk.init()`, and
  there was no `/metrics` endpoint despite `prometheus-client` being a
  dependency -- structured logging was the only real observability
  channel. New `openvoice.observability.configure_sentry`, called from
  each of the three process entrypoints this app runs as (API server,
  telephony worker, Celery worker) since each can crash independently.
  New `openvoice.metrics`: call volume/duration, LLM error count, and
  tool-call outcomes, mounted at `/metrics` on the API app.
- **Call duration/turn caps** (`AGENT_MAX_CONVERSATION_TURNS`,
  `AGENT_MAX_CALL_DURATION_SECONDS`, defaults 40 / 900s). Nothing
  previously stopped a single call from running indefinitely and paying
  for an LLM/TTS call on every turn. `ConversationManager` now hands the
  call to a human once either limit is hit -- checked *before* calling
  the LLM, so hitting the cap itself costs nothing further.

### Fixed

- **Double-booking.** `BookingService.book_appointment` never checked
  whether the client already had an overlapping appointment before
  creating a new one -- nothing stopped a caller (or a confused LLM
  tool-calling loop retrying a turn) from booking the same slot twice,
  creating two real calendar events. An exact repeat (identical
  start/end) is now idempotent, returning the existing appointment
  instead of duplicating it; a genuinely different but overlapping time
  now raises `BookingError` so the caller is told about the conflict.
  This only guards within one request path, not with a database-level
  exclusion constraint -- a true race between two simultaneous calls for
  the same client isn't fully closed.
- **Celery could stall call teardown for minutes.** `celery_app.py` never
  configured broker-connection retry behavior, so Celery/Kombu's default
  (up to 100 reconnection attempts) applied. `summarize_call_task.delay()`
  is called synchronously from the live call-teardown path
  (`openvoice.telephony.worker.finalize_call`) specifically wrapped in a
  try/except so a broker outage can't break call teardown -- but with the
  default retry budget, that `.delay()` call itself could block for
  minutes before the try/except ever got a chance to catch anything,
  stalling the whole worker process. Reduced to one retry with a 2s
  timeout: the post-call summary was already best-effort, so failing
  fast costs nothing a real Redis outage wasn't already going to cost.

### Added — Phase 1.1: voice-driven booking (native LLM tool-calling)

- **`BaseLLMProvider.generate()` now supports tools.** New
  `ToolDefinition`/`ToolCall` types in `openvoice.llm.base`, a `tools=`
  parameter, and a `tool_calls` field on `LLMResponse` and (for feeding
  results back) `LLMMessage`. Implemented natively in both providers
  against the real installed SDKs (verified via introspection, not
  guessed): Anthropic's `tool_use`/`tool_result` content blocks --
  including merging consecutive tool results into one `user` message,
  which Anthropic's API requires -- and OpenAI-compatible's
  `tool_calls`/`role: "tool"` messages (works the same for DeepSeek/
  Kimi/Qwen/Groq, since they all speak the same wire protocol).
- **`ConversationManager` runs an agentic tool loop.** When the model
  requests a tool call, it's executed and the result fed back, up to
  `max_tool_iterations` (default 4) before the turn falls back to a
  human-transfer message -- a misbehaving model or a failing tool can't
  turn one caller utterance into an unbounded loop on a live call.
  History trimming (`_trim_history`) now cuts on user-message
  boundaries, not a raw message count, since a tool-calling turn has
  more than the usual two messages and an arbitrary cut could split a
  tool-call/tool-result pair, producing a request the LLM APIs reject.
- **Booking tools** (`openvoice.agent.tools.booking`): `check_availability`,
  `list_my_appointments`, `book_appointment`, `cancel_appointment`,
  `reschedule_appointment`, mapped onto the existing `BookingService`.
  Mutating tools require `caller_confirmed: true` in their arguments,
  *enforced in code* (the dispatcher rejects the call otherwise) --
  not just requested via prompt text -- so the model can't silently book
  or cancel something the instant it resolves a date from a possibly
  mistranscribed request. Cancel/reschedule also re-verify the
  appointment belongs to the caller's own `client_id` before acting on
  it. Added `BookingService.list_upcoming_appointments` to support this
  (a caller has no appointment ID in hand the way a REST API client
  would).
- **The agent now knows the current date/time.**
  `openvoice.agent.prompts.build_temporal_context` tells the model the
  current date/time in the business's configured timezone
  (`BOOKING_TIMEZONE`), so it can resolve "tomorrow afternoon" or "next
  Monday" into a real, timezone-aware datetime for a booking tool call.
  Booking tools reject a datetime with no UTC offset outright rather
  than silently guessing a timezone.
- **Wired into the telephony worker**
  (`openvoice.telephony.worker._build_booking_tools`): booking tools are
  enabled only when a calendar provider is configured *and* the caller's
  phone number resolved to a client (a console/test-harness session with
  no SIP participant gets a fully working agent, just without booking --
  same graceful-degradation pattern already used for optional SMS/email).
  Tool-call/tool-result messages are excluded from the persisted
  call transcript (and the post-call LLM summary that reads it) --
  they aren't something either party "said".

### Added — Phase 1 MVP

- Project foundations: `uv`-managed `pyproject.toml`, ruff, mypy strict,
  pre-commit, GitHub Actions CI (lint, typecheck, test w/ real Postgres +
  Redis, Docker build).
- Pydantic Settings with fail-fast validation; structured JSON logging
  (structlog).
- Database models (`Client`, `Call`, `Appointment`, `CallTranscript`) +
  Alembic migration.
- Pluggable LLM architecture (Anthropic, OpenAI, and any OpenAI-compatible
  endpoint — DeepSeek, Kimi, Qwen, Groq, self-hosted vLLM/Ollama) behind a
  `BaseLLMProvider` interface + factory.
- Pluggable STT (faster-whisper) and TTS (Piper, ElevenLabs) behind
  `BaseSTTProvider`/`BaseTTSProvider` interfaces + factories.
- Conversational agent core: intent detection, per-call history,
  LLM-failure fallback to human transfer (`ConversationManager`).
- LiveKit call pipeline: `OpenVoiceAgent` bridges OpenVoice's providers
  into LiveKit via `stt_node`/`llm_node`/`tts_node`; VAD-based turn
  detection and barge-in delegated to `AgentSession`; SIP transfer-to-human
  fallback.
- Appointment booking: Google Calendar integration (availability,
  create/cancel/reschedule), alternative-slot suggestion within business
  hours, SMS (Twilio) / email (Resend) confirmations via direct REST calls.
- Lightweight CRM: caller recognition by phone number (SIP
  `sip.phoneNumber` attribute), call history.
- Post-call summaries generated by the LLM, dispatched via Celery so they
  never block call teardown.
- REST API (FastAPI + OpenAPI): `/clients`, `/calls`, `/appointments`,
  `/health` (real DB/Redis checks + best-effort LiveKit reachability).
- End-to-end integration test: a full simulated call from intake through
  post-call summary against a real Postgres, every external service faked.
- Community infrastructure: issue/PR templates, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, good-first-issue list in `CONTRIBUTING.md`.

### Changed — latency and turn-taking, for a more natural live conversation

- **Merged intent classification into the reply-generation call.**
  `ConversationManager.handle_utterance` used to make two sequential LLM
  round-trips per caller turn (`agent.intent.detect_intent`, then a
  separate reply generation) -- on a live call that's dead air twice
  over. Replaced with a single call: the system prompt now asks the
  model for a small JSON envelope (`{"intent": ..., "reply": ...}`),
  parsed by the new `openvoice.agent.structured_reply`. Roughly halves
  per-turn latency. Also fixes a latent bug: the old hardcoded
  `_HUMAN_TRANSFER_MESSAGE` was always English, contradicting the
  "always reply in the caller's language" instruction added earlier --
  the transfer acknowledgment is now LLM-generated like every other
  reply, so it's in the right language too.
- **Disabled `livekit-agents`' preemptive generation.** It's enabled by
  default and calls `llm_node` speculatively, before the caller's turn
  is confirmed final, against a transcript that may still change.
  `OpenVoiceAgent.llm_node` is stateful (`ConversationManager` mutates
  conversation history as a side effect of being called), so a
  speculative call invalidated by a changed transcript would leave a
  bogus exchange in history for words the caller never actually
  finished saying. Disabled via `turn_handling=TurnHandlingOptions(...)`
  in `openvoice.telephony.worker` for correctness, not performance --
  found by reading `agent_activity.py`'s `on_preemptive_generation`
  after realizing the `_NullLLM` fix (above) incidentally made this
  path reachable for the first time (it's gated on `self.llm is not
  None`, same gate as the no-reply bug).
- **Migrated off deprecated `allow_interruptions=True`** to
  `turn_handling=TurnHandlingOptions(interruption={"enabled": True})`,
  clearing the deprecation warning livekit-agents prints on every
  startup and avoiding a break when v2.0 removes the old kwarg.
- **Tuned VAD silence threshold for phone conversation.**
  `silero.VAD.load()`'s default `min_silence_duration` (0.55s) reads as
  the agent being slow to respond on a call. Lowered to 0.4s (new
  `VAD_MIN_SILENCE_DURATION_SECONDS` setting), a safe middle ground
  between responsiveness and cutting callers off mid-pause.
- **Switched the default local voice from Piper to ElevenLabs**
  (`TTS_PROVIDER=elevenlabs`) once a real API key was available --
  already fully implemented, just unused; dramatically more natural
  voice quality than a local Piper model, and multilingual by design.

### Fixed

- **Blank optional `.env` values were silently breaking config.** Every
  field `.env.example` ships as `KEY=` (no value) — by design, so
  contributors can see what's configurable — was being read by
  pydantic-settings as the literal empty string `""`, not "unset". That
  silently broke every `is None` check reading an unset-but-present key
  (e.g. `AGENT_SYSTEM_PROMPT=` produced an empty system prompt instead of
  the real default; `GOOGLE_SERVICE_ACCOUNT_JSON_PATH=` produced a
  confusing "package not installed" error instead of the intended
  "credential missing" one). Fixed at the root with
  `env_ignore_empty=True` on `Settings.model_config`, plus a regression
  test, rather than patching each affected call site individually.
- **Tenacity silently never awaited real Anthropic/OpenAI SDK calls.**
  `AsyncRetrying` decides whether to `await` a call based on
  `inspect.iscoroutinefunction(fn)`; both SDKs' `create` methods are
  plain (non-`async def`) functions that return a coroutine when called,
  so that check is `False` and the raw, unawaited coroutine was returned
  as the "response" instead of the real result. Every mocked unit test
  passed anyway, because `AsyncMock` *is* correctly detected as async —
  this only broke against a real API call (caught by actually calling
  DeepSeek for real). Fixed by wrapping each SDK call in a plain local
  `async def` closure, which tenacity detects correctly; added a
  regression test that reproduces the exact "sync wrapper returning a
  coroutine" shape rather than relying on `AsyncMock`.
- **`_caller_phone_number` could pass a `Mock` object into a SQL query.**
  In LiveKit's `console` mode (used for interactive local testing), the
  simulated participant's `attributes` isn't a real `dict`, so `.get(...)`
  returns a `Mock` instead of `None` when the key is absent. A plain
  truthiness check (`if phone_number:`) let that `Mock` through as if it
  were a real phone number, which then hit the database as a query
  parameter and crashed. Fixed with an explicit `isinstance(..., str)`
  check; caught by running the worker for real, not by any mocked test.
- **The agent transcribed the caller correctly but never replied, with no
  error logged.** `OpenVoiceAgent` fully overrides `llm_node` to route
  replies through OpenVoice's own `ConversationManager`, so no
  `livekit.agents.llm.LLM` plugin was ever configured on `AgentSession` or
  `Agent`. But `livekit-agents`' own turn-taking code
  (`AgentActivity._user_turn_completed_task`) gates the entire
  reply-generation step on `self.llm is not None` *before* it calls
  `llm_node` at all — with no LLM configured, it just returns, silently,
  turn after turn. STT worked (the caller's speech showed up as a
  transcript), which made this look like an `llm_node`/TTS bug; it was
  actually one line away, in `livekit-agents` itself, and only visible by
  reading its source (`agent_activity.py`), not by reasoning about our
  own code. Fixed by giving `OpenVoiceAgent` a `_NullLLM` stub — present
  so the gate passes, never actually called since `llm_node` bypasses it
  entirely.
- **The agent's voice sounded slow, deep, and robotic.** Piper voice
  models synthesize at their own native rate (22050 Hz for the bundled
  `en_US-amy-medium`), not whatever rate the caller asks for --
  `PiperTTSProvider.synthesize` accepted a `sample_rate` argument but
  silently ignored it, and the LiveKit bridge then labeled that 22050 Hz
  audio as 16000 Hz `AudioFrame`s. That plays back ~27% slower and pitched
  down: exactly what a real call sounded like. Fixed by resampling to the
  requested rate (linear interpolation, no new dependency needed --
  `numpy` was already required); regression test covers a 22050→16000
  conversion. Caught by actually listening to the agent speak, not by any
  prior test (every existing test used a fake voice at a matching rate).
- **The agent always replied in English, regardless of what language the
  caller spoke, and pronounced non-English replies badly.** Two separate
  gaps: the system prompt never told the LLM to match the caller's
  language, and the default local Piper voice (`en_US-amy-medium`) is
  English-only, so it would mispronounce any other language's text even
  if the LLM did reply in it. Fixed the prompt gap by adding an explicit
  "always reply in the caller's language" instruction. The voice-model
  gap isn't fixable in general (a single local Piper voice is one
  language) -- for local dev, added a French voice
  (`fr_FR-siwis-medium`) alongside the English one; which one loads is
  controlled by `PIPER_VOICE_MODEL_PATH`, per deployment.
- **Integration tests could silently wipe the real dev database.**
  `tests/integration/conftest.py`'s default `TEST_DATABASE_URL` and a
  developer's real `DATABASE_URL` could end up pointing at the same
  Postgres database (e.g. both on the default port), and the test
  fixture runs `Base.metadata.drop_all` after every test. Changed the
  default test database name to `openvoice_test` (distinct from the app's
  `openvoice`), documented the risk inline, and re-ran migrations against
  the real dev database after this was caught happening once.

### Changed

- Renamed the `self_hosted` LLM provider to `openai_compatible`
  (`LLM_PROVIDER=openai_compatible`, `OPENAI_COMPATIBLE_BASE_URL`/`_MODEL`/
  `_API_KEY`) and fixed a real bug where its API key was hardcoded to
  `None` — hosted OpenAI-compatible providers requiring auth (DeepSeek,
  Kimi, Qwen, Groq, ...) were previously unreachable through this path.
  See [ADR 0004](docs/adr/0004-generalize-self-hosted-llm-to-openai-compatible.md).

### Known gaps

See [docs/ROADMAP.md](docs/ROADMAP.md)'s "Known gaps" section: the LiveKit
integration hasn't been smoke-tested against a live server/SIP trunk,
voice-driven booking (LLM tool-calling) isn't wired up yet, and Sentry/
Prometheus are declared but not yet instrumented.
