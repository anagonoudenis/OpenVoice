## What does this change?

<!-- One or two sentences. Link the issue it closes, if any: "Closes #123". -->

## Why?

<!-- The motivation, if not obvious from the linked issue. -->

## How was this tested?

<!-- New/updated unit tests? Integration tests? Manual verification (describe it)? -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy src` passes (no new `# type: ignore` without a comment explaining why)
- [ ] `uv run pytest` passes locally (integration tests need `docker compose up -d postgres`)
- [ ] New business logic has unit tests; new cross-module flows have an integration test
- [ ] No hardcoded secrets; new config goes through `openvoice.config.Settings`
- [ ] Docs updated if this changes behavior a user/contributor would rely on
      (README, ARCHITECTURE.md, ROADMAP.md, or a new ADR for a significant
      technical decision — see CONTRIBUTING.md)

## Anything reviewers should focus on?

<!-- Tricky edge cases, deliberate trade-offs, things you're unsure about. -->
