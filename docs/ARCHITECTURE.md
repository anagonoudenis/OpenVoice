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
  base.py        # Abstract interface (Protocol or ABC) — the only thing
                  # business logic depends on.
  providers/
    anthropic.py  # Concrete implementation
    openai.py
    self_hosted.py
  factory.py      # get_<domain>_provider(settings) -> Base<Domain>Provider
```

Business logic (call handling, appointment booking, CRM) depends only on
`base.py`'s interface, obtained through the factory. The factory reads the
provider choice from `Settings` (environment variables) — switching from
Anthropic to a self-hosted vLLM model is a config change, never a code
change. This is what `docs/adr/0001-record-architecture-decisions.md`
establishes as the standing convention; provider-specific ADRs reference it
instead of re-justifying the pattern.

## Call lifecycle (target, built out across Steps 6-7)

```
Incoming call (LiveKit)
  -> VAD (Silero) detects speech / silence, handles barge-in
  -> STT provider transcribes the utterance
  -> Agent core: intent detection, conversation context, LLM provider call
  -> Action (if needed): calendar lookup/booking, CRM read/write
  -> TTS provider synthesizes the response
  -> Streamed back over LiveKit
  -> CallTranscript persisted; call_id threads through every log line
```

Every external call in this path (STT, LLM, TTS, calendar, DB, Redis) is
wrapped with an explicit timeout and a `tenacity` exponential-backoff retry.
When a provider is down past its retry budget, the agent's defined fallback
(e.g. "transfer to a human", "apologize and offer a callback") fires —
never a silent hang or an unhandled exception on a live call.

## Data layer

- **PostgreSQL** (SQLAlchemy 2.0 async + Alembic) is the system of record:
  `Client`, `Call`, `Appointment`, `CallTranscript`.
- **pgvector** (extension on the same Postgres instance) backs the Phase 2
  RAG knowledge base — deliberately not a separate vector DB, to keep the
  MVP's operational surface small.
- **Redis** holds ephemeral session/turn-taking state and backs Celery.
- **Celery** runs anything that shouldn't block the call: post-call
  summaries, SMS/email confirmations, reminders.

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
