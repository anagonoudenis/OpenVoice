# 1. Record architecture decisions

## Status

Accepted

## Context

OpenVoice makes several technical choices that are not obvious from the
code alone (why LiveKit and not Twilio, why Piper and not Coqui, why
pgvector and not a dedicated vector DB, why the pluggable-provider
pattern). Future contributors — and future us — need the reasoning, not
just the result, to judge whether a decision still holds as the project
evolves.

## Decision

We use lightweight Architecture Decision Records, one per significant
technical choice, stored in `docs/adr/` as `NNNN-title-in-kebab-case.md`
with sections: Status, Context, Decision, Consequences. ADRs are immutable
once accepted — a changed decision gets a new ADR that supersedes the old
one, rather than an edit.

An ADR is warranted for: a new external dependency or vendor integration,
a change to the core data model, a change that affects the pluggable
provider pattern described in `docs/ARCHITECTURE.md`, or any choice a
reviewer is likely to ask "why not X instead?" about.

## Consequences

- Every non-trivial technical decision is discoverable without spelunking
  through git blame or chat history.
- Adds a small amount of process overhead per significant PR.
