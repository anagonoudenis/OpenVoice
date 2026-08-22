# OpenVoice

Open-source AI voice agent that answers phone calls, books appointments, and
handles customer support — with a natural, low-latency synthetic voice.

Every major integration point (LLM, speech-to-text, text-to-speech, calendar,
notifications) is **pluggable**: swap providers via environment variables,
never by touching business logic. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Early foundations (Phase 1 MVP in progress). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Stack

- **Runtime**: Python 3.12+, [uv](https://docs.astral.sh/uv/) for dependency management
- **API**: FastAPI (async), Pydantic v2
- **Telephony**: [LiveKit Agents](https://docs.livekit.io/agents/)
- **STT**: faster-whisper (pluggable)
- **TTS**: Piper (pluggable — Coqui/ElevenLabs also supported)
- **VAD**: Silero
- **LLM**: Anthropic / OpenAI / self-hosted (OpenAI-compatible), pluggable via a Provider Factory
- **Data**: PostgreSQL + pgvector, SQLAlchemy 2.0 (async), Alembic, Redis, Celery
- **Quality**: ruff, mypy (strict), pytest (>85% coverage target), pre-commit, GitHub Actions CI

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

## Development

```bash
uv sync --group dev            # install core + dev dependencies
uv sync --extra all --group dev  # + voice/calendar/notifications extras
uv run pre-commit install      # install git hooks

uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy src                # strict type check
uv run pytest                  # tests + coverage report
```

## Project layout

```
src/openvoice/        Application source (installable package)
  config.py            Pydantic Settings — all runtime configuration
  logging.py           Structured (structlog) logging setup
  main.py              FastAPI app factory + /health
tests/
  unit/                Fast, isolated unit tests
  integration/          End-to-end flows (added in later steps)
docs/
  ARCHITECTURE.md       System design + pluggable-provider pattern
  ROADMAP.md            Phase 2 (post-MVP) plans
  adr/                  Architecture Decision Records
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
