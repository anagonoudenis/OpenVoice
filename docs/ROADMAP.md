# Roadmap

## Phase 1 — MVP (complete)

Built in this order; each step passed lint + mypy strict + tests before
the next began. One known gap carried forward, tracked below and in its
ADR rather than hidden: LiveKit hasn't been smoke-tested against a live
server.

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
       is complete and unit tested; reachable both via the REST API and, since
       Phase 1.1, purely by talking to the agent (native LLM tool-calling --
       see `openvoice.agent.tools.booking`).
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
- **Docker Compose is unverified in this environment** (the dev machine's
  virtualization was disabled at the OS/BIOS level) but its config was
  validated with `docker compose config`. CI runs real Postgres/Redis
  service containers, so the app code itself is proven against them —
  only the Compose file's exact YAML wasn't run end-to-end locally.

## Phase 1.1 — Real-call hardening (complete)

Everything here was found and fixed by actually placing calls through
`console` mode and listening to the agent, not by mocked tests alone --
see `CHANGELOG.md` for the full list of bugs this surfaced.

- Native LLM tool-calling (`openvoice.llm.base.ToolDefinition`/`ToolCall`,
  implemented for both the Anthropic and OpenAI-compatible providers) and
  an agentic loop in `ConversationManager`, closing the voice-driven
  booking gap: the agent can check availability, book, cancel, reschedule,
  and list a caller's own appointments purely by talking
  (`openvoice.agent.tools.booking`), gated on an explicit spoken
  confirmation before any mutating action.
- Per-turn latency roughly halved by merging intent classification and
  reply generation into a single LLM call.
- Fixed a silent "agent never replies" bug (a `livekit-agents` turn-taking
  gate on `self.llm is not None`, unrelated to the STT/TTS pipeline it
  looked like), a TTS sample-rate mismatch that made the voice sound slow
  and robotic, and English-only replies regardless of the caller's
  language.
- VAD/turn-handling tuned for phone conversation and migrated off
  deprecated `livekit-agents` APIs.

## Phase 1.2 — Production-reliability hardening (complete)

A second full-project audit, this time before anything went live rather
than after: fixed two real correctness/reliability bugs and closed the
two remaining "known gaps" above.

- **Double-booking guard**: `BookingService.book_appointment` now checks
  for an overlapping appointment for the same client first. An exact
  repeat (same start/end) is idempotent -- returns the existing
  appointment rather than creating a duplicate calendar event -- since
  nothing stops a caller, or a confused tool-calling loop, from asking to
  book the same slot twice. A genuinely different, merely overlapping
  time raises instead, so the caller is told about the conflict rather
  than it being silently resolved either way. Not a database-level
  exclusion constraint, so a race between two truly concurrent bookings
  for the same client (e.g. two simultaneous calls) isn't fully closed.
- **Celery fail-fast**: the default Celery/Kombu broker-connection retry
  budget (up to 100 attempts) meant a Redis outage could stall
  `summarize_call_task.delay()` -- called synchronously from the live
  call-teardown path -- for minutes, blocking the telephony worker
  process from picking up its next call. Reduced to one quick retry with
  a short timeout; the post-call summary was already best-effort
  (failure only logged, never raised), so failing fast costs nothing a
  real outage wasn't already going to cost.
- **Call duration/turn caps**: nothing previously stopped a single call
  (a stuck caller, an abusive one, just a very long conversation) from
  running indefinitely and paying for an LLM/TTS call on every turn.
  `ConversationManager` now hands off to a human once either
  `AGENT_MAX_CONVERSATION_TURNS` or `AGENT_MAX_CALL_DURATION_SECONDS` is
  hit -- checked before calling the LLM, so hitting the cap costs
  nothing further.
- **Sentry and Prometheus wired up.** `openvoice.observability
  .configure_sentry` is called from each of the app's three process
  entrypoints (API server, telephony worker, Celery worker) --
  `Settings.sentry_dsn` actually does something now. `/metrics` is
  mounted on the API app with a small, targeted set of voice-pipeline
  metrics (call volume/duration, LLM errors, tool-call outcomes) -- not a
  general analytics layer, just enough to know something broke without
  waiting for a customer to complain.

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
