# Architecture

## Goals

OpenVoice answers real phone calls, so the design optimizes for three things
above all: **low end-to-end latency** (STT end → LLM decision → TTS start
under 800ms), **graceful degradation** (every external dependency can fail
without crashing a live call), and **swappable providers** (no business
logic should ever import a specific vendor SDK directly).

## The pluggable-provider pattern

LLM, STT, TTS, calendar, and notification integrations all follow the same
shape:

```
openvoice/<domain>/
  base.py               # Abstract interface (Protocol or ABC) — the only thing
                         # business logic depends on.
  providers/
    anthropic.py         # Concrete implementation
    openai_compatible.py  # Also backs any OpenAI-compatible endpoint
  factory.py             # get_<domain>_provider(settings) -> Base<Domain>Provider
```

Business logic (call handling, appointment booking, CRM) depends only on
`base.py`'s interface, obtained through the factory. The factory reads the
provider choice from `Settings` (environment variables) — switching from
Anthropic to DeepSeek, Kimi, Qwen, or a self-hosted vLLM model is a config
change (`LLM_PROVIDER=openai_compatible` + a base URL/model/API key), never
a code change; no model is ever hardcoded. This is what
`docs/adr/0001-record-architecture-decisions.md` establishes as the
standing convention; provider-specific ADRs reference it instead of
re-justifying the pattern.

## Call lifecycle

```
Incoming call (LiveKit SIP)
  -> Caller resolved by phone number (CRM: get_or_create_client)
  -> AgentSession's Silero VAD segments continuous audio into utterances
  -> stt_node batch-transcribes each utterance via the configured STT provider
  -> ConversationManager: history, LLM provider call (intent + reply +
     tool-calls in one structured response), tool-call loop when the
     model requests a booking action
  -> llm_node routes the reply back into the session
  -> tts_node synthesizes the response via the configured TTS provider
  -> Streamed back over LiveKit; AgentSession owns barge-in/interruption handling
  -> On hangup: CallTranscript rows persisted, summarize_call dispatched via
     Celery (never inline — must not delay call teardown)
```

Booking (find slots / book / cancel / reschedule) is a separate service
(`BookingService`) invoked either from the REST API directly, or from
the conversation itself via native LLM tool-calling
(`openvoice.agent.tools.booking`, wired into `ConversationManager`) --
gated on the caller having a resolved identity and a calendar provider
being configured; see ROADMAP.md.

Every external call in this path (STT, LLM, TTS, calendar, DB, Redis) is
wrapped with an explicit timeout and a `tenacity` exponential-backoff retry.
When a provider is down past its retry budget, the agent's defined fallback
(e.g. "transfer to a human", "apologize and offer a callback") fires —
never a silent hang or an unhandled exception on a live call. See
[docs/adr/0003-livekit-node-override-integration.md](adr/0003-livekit-node-override-integration.md)
for exactly how OpenVoice's providers are bridged into LiveKit's pipeline.

## Data layer

- **PostgreSQL** (SQLAlchemy 2.0 async + Alembic) is the system of record:
  `Client`, `Call`, `Appointment`, `CallTranscript`.
- **pgvector** (extension on the same Postgres instance) backs the Phase 2
  RAG knowledge base — deliberately not a separate vector DB, to keep the
  MVP's operational surface small.
- **Redis** backs Celery.
- **Celery** runs the one thing that must never block call teardown:
  post-call summaries (LLM-generated, from the persisted transcript).
  SMS/email booking confirmations run synchronously inside
  `BookingService` instead — a failed notification is logged, not
  retried via a queue, and never undoes a successful booking.

## API layer

`openvoice/api/` is a thin FastAPI layer over the same services the call
pipeline uses (`CRMService`, `BookingService`) — it doesn't duplicate
business logic. Every route depends on an abstraction via FastAPI's
dependency injection (`openvoice/api/dependencies.py`), never a concrete
provider class. `CalendarError`/`NotificationError`/`BookingError` are
mapped to clean HTTP responses (503/502/409) by app-level exception
handlers in `main.py`, rather than leaking as 500s.

## Observability

Structured JSON logs (`structlog`) carry a `call_id` from the moment a call
is accepted through every STT/LLM/TTS/DB step to hangup, so a single call
can be reconstructed from logs alone. Errors go to Sentry; latency and
volume metrics are exported for Prometheus/Grafana.

## Why these choices (see ADRs for the reasoning behind each)

- LiveKit over Twilio: fully open source, self-hostable, no per-minute
  vendor lock-in for the media path.
- Piper as the default TTS (not Coqui): coqui-ai/TTS is archived and its
  best model (XTTS-v2) is CPML-licensed (non-commercial); Piper is Apache
  2.0, CPU-friendly, and good enough by default, while staying swappable.
- pgvector over a dedicated vector DB: one fewer moving part for the MVP;
  revisit if RAG scale outgrows it (tracked in ROADMAP.md).
- Twilio/Resend via direct REST calls (`httpx` + `tenacity`), not their
  vendor SDKs: the wire protocol is simple enough that a hand-rolled
  client is less code than a dependency, keeps the same retry/timeout
  pattern as every other provider, and needs no extra install to test.
