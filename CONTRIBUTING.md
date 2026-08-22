# Contributing to OpenVoice

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
   uv run pytest --cov-fail-under=85
   ```
5. Open a PR against `main`. CI runs the same checks and must pass before
   merge.

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
   settings fields (with a `model_validator` if credentials are required).
4. Add unit tests with the external API mocked — never call a real
   third-party service from the test suite.
