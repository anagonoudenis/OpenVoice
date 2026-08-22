# OpenVoice

Open-source AI voice agent that answers phone calls, books appointments, and
handles customer support — with a natural, low-latency synthetic voice.

Every major integration point (LLM, speech-to-text, text-to-speech, calendar,
notifications) is **pluggable**: swap providers via environment variables,
never by touching business logic. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Phase 1 MVP complete: foundations, DB models, pluggable LLM/STT/TTS, the
LiveKit call pipeline, booking + notifications, CRM, the REST API, and an
end-to-end integration test all exist and pass CI. Two things to know before
relying on this in production:

- The LiveKit call pipeline was built and unit-tested against the real SDK,
  but not yet smoke-tested against a live LiveKit server + SIP trunk — see
  [docs/adr/0003-livekit-node-override-integration.md](docs/adr/0003-livekit-node-override-integration.md).
- Appointment booking is fully implemented and API-exposed, but not yet
  wired into multi-turn voice conversation (LLM tool-calling for "what day
  works for you?") — see [docs/ROADMAP.md](docs/ROADMAP.md).

Phase 2 (RAG, admin dashboard, multi-tenant, etc.) is documented but not
built — see [docs/ROADMAP.md](docs/ROADMAP.md).

## Stack

- **Runtime**: Python 3.12+, [uv](https://docs.astral.sh/uv/) for dependency management
- **API**: FastAPI (async), Pydantic v2
- **Telephony**: [LiveKit Agents](https://docs.livekit.io/agents/)
- **STT**: faster-whisper (pluggable)
- **TTS**: Piper (pluggable — ElevenLabs also supported; see ADR 0001 on Coqui)
- **VAD**: Silero
- **LLM**: Anthropic / OpenAI / self-hosted (OpenAI-compatible), pluggable via a Provider Factory
- **Calendar**: Google Calendar (pluggable)
- **Notifications**: Twilio (SMS) / Resend (email), both via direct REST calls
- **Data**: PostgreSQL + pgvector, SQLAlchemy 2.0 (async), Alembic, Redis, Celery
- **Quality**: ruff, mypy (strict), pytest (>85% coverage), pre-commit, GitHub Actions CI

## Quickstart

```bash
git clone <this-repo> && cd OpenVoice
cp .env.example .env          # fill in SECRET_KEY, ANTHROPIC_API_KEY (or OPENAI_API_KEY), etc.
docker compose up -d postgres redis
uv sync --group dev
uv run alembic upgrade head
uv run uvicorn openvoice.main:app --reload
```

Then open http://localhost:8000/health and http://localhost:8000/docs.

### REST API

`/clients`, `/clients/{id}/calls`, `/calls/{id}` (with transcript),
`/appointments` (`available-slots`, book, cancel, reschedule) — see
http://localhost:8000/docs for the full interactive OpenAPI spec.
`/health` reports real DB/Redis status (required) and LiveKit reachability
(informational).

### Running the voice agent (real calls)

The API above is call-independent (clients, appointments, CRM). Taking
actual phone calls is a separate worker process against a LiveKit server:

```bash
uv sync --extra voice --group dev   # faster-whisper, Piper, LiveKit Agents
export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
uv run openvoice-worker              # or: uv run python -m openvoice.telephony.worker
```

Requires a running LiveKit server (self-hosted or LiveKit Cloud) with a
SIP trunk + dispatch rule configured for inbound/outbound telephony — see
[docs/adr/0003-livekit-node-override-integration.md](docs/adr/0003-livekit-node-override-integration.md)
for what is and isn't verified in this integration, and smoke-test a real
call before relying on it in production.

### Background jobs (post-call summaries)

Post-call summaries run via Celery, not inline in the call, so they never
delay hangup or block the worker's event loop:

```bash
uv run celery -A openvoice.tasks.celery_app worker --loglevel=info
```

(Also available as the `celery-worker` service in `docker compose up`.)

## Development

```bash
uv sync --group dev              # install core + dev dependencies
uv sync --extra all --group dev  # + voice/calendar/notifications extras
uv run pre-commit install        # install git hooks

uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src                  # strict type check
uv run pytest -m "not integration"   # unit tests (no DB needed)
uv run pytest                        # + integration tests (needs Postgres: docker compose up -d postgres)
```

## Project layout

```
src/openvoice/
  config.py            Pydantic Settings — all runtime configuration
  logging.py           Structured (structlog) logging setup
  main.py              FastAPI app factory, routers, exception handlers
  api/                 REST API: routers, request/response schemas, DI
  agent/               Conversation core: intent detection, history, fallback
  llm/, stt/, tts/      Pluggable provider interfaces + implementations + factories
  calendar/             Pluggable calendar provider (Google Calendar)
  notifications/        Pluggable SMS/email providers (Twilio, Resend)
  booking/              BookingService: slots, book/cancel/reschedule
  crm/                  Caller recognition, call history
  telephony/             LiveKit call pipeline + worker entrypoint
  tasks/                 Celery app + post-call summary task
  db/                   SQLAlchemy models + session management
migrations/             Alembic migrations
tests/
  unit/                Fast, isolated unit tests (mocked externals)
  integration/          Real-Postgres tests, incl. the end-to-end call test
docs/
  ARCHITECTURE.md       System design + pluggable-provider pattern
  ROADMAP.md            Phase 1 status + Phase 2 (post-MVP) plans
  adr/                  Architecture Decision Records
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
