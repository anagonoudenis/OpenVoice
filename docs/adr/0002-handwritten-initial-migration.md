# 2. Hand-written initial Alembic migration

## Status

Accepted

## Context

`alembic revision --autogenerate` needs a live database connection to diff
against. At the time the initial schema (`clients`, `calls`, `appointments`,
`call_transcripts`) was authored, no PostgreSQL instance was reachable in
the dev environment (Docker Desktop was blocked by a host-level
virtualization issue), so autogeneration wasn't an option.

## Decision

Write `migrations/versions/0001_initial.py` by hand, mirroring
`openvoice.db.models` column-for-column. Column types and defaults were
verified against the actual SQLAlchemy 2.0 behavior in this environment
(e.g. confirming `sa.Enum(..., native_enum=False).create_constraint`
defaults to `False`, so enum columns are plain `VARCHAR` with no CHECK
constraint) rather than assumed. The migration was smoke-tested with
`alembic upgrade head --sql` (offline mode, no DB required), which
confirmed it runs without error and renders the expected DDL.

## Consequences

- Once a real PostgreSQL instance is available, run `alembic check` (or
  diff `Base.metadata.create_all` output against this migration) to get a
  second, DB-backed confirmation that they match exactly.
- Future migrations should go back to `alembic revision --autogenerate`
  now that `docker compose up postgres` / CI's Postgres service are
  available — this hand-written approach was a one-time workaround, not
  the standing convention.
