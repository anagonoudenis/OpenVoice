# Roadmap

## Phase 1 — MVP (in progress)

Built in this order; each step must pass lint + mypy strict + tests before
the next begins.

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
7. [ ] Appointment booking: Google Calendar integration, alternative-slot
       suggestion, SMS/email confirmation, cancel/reschedule via voice.
8. [ ] Lightweight CRM: caller recognition by phone number, call history,
       Celery-based post-call summaries.
9. [ ] REST API + OpenAPI docs; `/health` with real DB/Redis/LiveKit checks.
10. [ ] End-to-end integration tests (fully mocked simulated call).
11. [ ] Docs finalized: README quickstart, CONTRIBUTING, ARCHITECTURE, this
        roadmap.

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
