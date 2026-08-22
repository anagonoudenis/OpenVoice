# Roadmap

## Phase 1 — MVP (complete)

Built in this order; each step passed lint + mypy strict + tests before
the next began. Two known gaps carried forward, tracked below and in
their respective ADR/step notes rather than hidden: LiveKit hasn't been
smoke-tested against a live server, and booking isn't yet reachable via
multi-turn voice conversation (only the REST API today).

1. [x] Project foundations: repo structure, `pyproject.toml`, `.gitignore`,
       README, MIT license, pre-commit.
2. [x] Config (Pydantic Settings), Docker Compose (Postgres + Redis), CI.
3. [x] Database models (`Client`, `Call`, `Appointment`, `CallTranscript`) +
       Alembic migrations + tests.
4. [x] Pluggable LLM architecture (interface + Anthropic/OpenAI/self-hosted
       providers + factory + mocked tests).
5. [x] Pluggable STT/TTS interfaces (faster-whisper, Piper, ElevenLabs + mocks).
6. [x] LiveKit call pipeline: inbound/outbound calls, VAD, barge-in,
       STT→LLM→TTS loop, human handoff fallback. Not yet smoke-tested
       against a live LiveKit server/SIP trunk (see ADR 0003) — do this
       before relying on it for real calls.
7. [x] Appointment booking: Google Calendar integration, alternative-slot
       suggestion, SMS/email confirmation, cancel/reschedule. `BookingService`
       is complete and unit tested; it is not yet wired into a multi-turn
       voice conversation (LLM tool-calling/slot-filling for "what day
       works for you?") — exposed via the REST API instead (Step 9).
8. [x] Lightweight CRM: caller recognition by phone number (via the SIP
       `sip.phoneNumber` participant attribute), call history,
       Celery-based post-call summaries.
9. [x] REST API + OpenAPI docs (`/clients`, `/calls`, `/appointments`);
       `/health` with real DB/Redis checks (required) and a best-effort
       LiveKit TCP reachability check (informational, since the API
       process itself doesn't need LiveKit — only the telephony worker
       does).
10. [x] End-to-end integration tests: a full simulated call (intake -> STT ->
        LLM -> TTS -> booking -> transcript persistence -> post-call
        summary) against a real Postgres, every external service faked.
11. [x] Docs finalized: README quickstart, CONTRIBUTING, ARCHITECTURE, this
        roadmap.

### Known gaps to close before a production launch

- **LiveKit integration is unverified end-to-end.** It was built and unit
  tested against the real SDK's types (see ADR 0003), but no LiveKit
  server or SIP trunk was available in the development environment to
  place a real call through it. Do this first.
- **Voice-driven booking isn't wired up.** `BookingService` is complete,
  tested, and reachable via the REST API, but a caller can't yet book an
  appointment purely by talking to the agent — that needs LLM
  tool-calling/function-calling added to `ConversationManager` so it can
  call `BookingService` mid-conversation (propose slots, confirm a time,
  handle "actually, how about Tuesday instead").
- **Docker Compose is unverified in this environment** (the dev machine's
  virtualization was disabled at the OS/BIOS level) but its config was
  validated with `docker compose config`. CI runs real Postgres/Redis
  service containers, so the app code itself is proven against them —
  only the Compose file's exact YAML wasn't run end-to-end locally.
- **Sentry and Prometheus aren't actually wired up.** `Settings.sentry_dsn`
  exists and `prometheus-client` is a dependency (per the original spec's
  observability requirements), but nothing calls `sentry_sdk.init()` and
  there's no `/metrics` endpoint yet — structured logging is the only
  observability channel that's actually live today. Good first issue.

## Phase 2 — Post-MVP

Documented here, not built until Phase 1 is stable and merged:

- **RAG knowledge base** (pgvector) for company-specific FAQ answers.
- **Admin dashboard**: live call listening, statistics, no-code agent
  configuration.
- **Multi-language** support with automatic language detection.
- **Real-time sentiment analysis** during calls.
- **Multi-tenant / white-label** SaaS mode.
- **Webhooks + Zapier/Make/n8n connectors**.
- **Advanced GDPR compliance**: anonymization, automated right-to-erasure.
- **Call queueing, 3-way conferencing, DTMF** support.

## Non-goals for now

- No custom vector DB beyond pgvector unless Phase 2 RAG scale demands it.
- No Twilio dependency for the core telephony path (LiveKit is the
  telephony backbone; Twilio is used only as the pluggable SMS provider).
