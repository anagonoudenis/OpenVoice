# Contributing to OpenVoice

Thanks for considering contributing — this project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md). Found a security issue?
See [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Ground rules

- Every function has type hints; `mypy --strict` must pass with zero errors.
- No bare `except Exception: pass`. Every external call (LLM, STT, TTS,
  calendar, DB, Redis) needs an explicit timeout, a `tenacity` retry policy,
  and a defined, tested fallback behavior.
- No hardcoded secrets. All configuration flows through
  `openvoice.config.Settings`; add new fields there, not `os.environ.get()`
  calls scattered through the codebase.
- New business logic needs unit tests; new cross-module flows (e.g. call →
  STT → LLM → TTS) need an integration test with mocked external services.
- Public modules and classes get a docstring explaining *why*, not a
  restatement of the code.

## Workflow

1. Fork/branch from `main`.
2. `uv sync --group dev && uv run pre-commit install`.
3. Make your change, keeping it scoped — see the codebase's own
   conventions rather than introducing new patterns.
4. Before opening a PR, run locally:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest -m "not integration"        # fast path, no DB needed
   docker compose up -d postgres && uv run pytest --cov-fail-under=85  # full suite
   ```
5. Open a PR against `main`. CI runs the same checks (with real Postgres/Redis
   service containers) and must pass before merge.

## Good first issues

Concrete, self-contained starter tasks — each follows an existing pattern
elsewhere in the codebase, so "read the analogous file, do the same thing"
gets you most of the way:

- **Deepgram STT provider.** `openvoice/stt/factory.py` currently raises
  `NotImplementedError` for `STT_PROVIDER=deepgram`. Deepgram's REST/
  streaming API is HTTP-based, so this can follow the ElevenLabs TTS
  provider's pattern (`openvoice/tts/providers/elevenlabs.py`) — direct
  `httpx` calls, no vendor SDK — rather than faster-whisper's lazy-import
  pattern.
- **A second calendar provider** (Outlook/Microsoft Graph or Cal.com).
  Follow `openvoice/calendar/providers/google_calendar.py` and implement
  `BaseCalendarProvider`.
- **A second email provider** (e.g. SendGrid). Follow
  `openvoice/notifications/providers/resend_email.py`.
- **Wire up Sentry.** `Settings.sentry_dsn` exists but `sentry_sdk.init()`
  is never called anywhere — error tracking isn't actually active yet.
- **Wire up Prometheus metrics.** `prometheus-client` is a dependency but
  there's no `/metrics` endpoint or any instrumentation yet.
- **Voice-driven booking.** The biggest gap: `BookingService` is complete
  and tested but only reachable via the REST API. Wiring it into
  `ConversationManager` via LLM tool-calling (so a caller can book an
  appointment just by talking) needs design work — see
  [docs/ROADMAP.md](docs/ROADMAP.md)'s "Known gaps" section before
  starting, and open an issue to discuss the approach first.

If you pick one of these up, comment on (or open) the corresponding issue
first so effort isn't duplicated — see the repo's Issues tab, or file one
using the "Feature request" template if it doesn't exist yet.

## Architecture decisions

Significant technical choices (a new provider integration, a change to the
data model, a new external dependency) get a short ADR in `docs/adr/`.
Copy the format of `docs/adr/0001-record-architecture-decisions.md`.

## Adding a new provider (LLM / STT / TTS / calendar / notifications)

These are all implemented as an abstract interface + concrete
implementations + a factory selected by config (see
`docs/ARCHITECTURE.md`). To add one:

1. Implement the relevant abstract base class.
2. Register it in the provider factory's mapping.
3. Add an enum value in `openvoice.config` and any provider-specific
   settings fields. Validate required credentials either eagerly (a
   `model_validator` on `Settings`, for providers the app can't function
   without at all — see LLM/TTS) or lazily (a check inside the factory
   function, for an optional sub-feature — see calendar/SMS/email, and
   the rationale comment above `Settings` for why).
4. Add unit tests with the external API mocked — never call a real
   third-party service from the test suite. Prefer dependency injection
   (a fake client/service passed to the constructor) over lazy-importing
   the vendor SDK when the provider needs a heavy or optional dependency
   — see the STT/TTS/calendar providers for the pattern.
