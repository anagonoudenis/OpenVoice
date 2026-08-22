# Security Policy

## Supported versions

OpenVoice is pre-1.0 (Phase 1 MVP). Only the `main` branch is supported —
please make sure you're on the latest commit before reporting an issue.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Preferred: use [GitHub's private vulnerability reporting](../../security/advisories/new)
for this repository (Security tab → "Report a vulnerability"). This opens a
private discussion with maintainers before anything is public.

If that isn't available to you, email **denisanagonou259@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal proof of concept if possible)
- Any suggested mitigation, if you have one

You should expect an initial response within 5 business days. We'll keep you
updated as the issue is triaged and fixed, and credit you in the advisory
(unless you'd prefer to stay anonymous) once it's resolved.

## Scope

Particularly interested in reports involving:

- Injection (SQL via raw queries, command injection in provider integrations)
- Auth/authorization bypass in the REST API
- Secrets handling (e.g. logging credentials, leaking them in error responses)
- SSRF or unsafe deserialization in any provider integration
- Dependency vulnerabilities with a demonstrated exploit path in this codebase

Reports about missing security headers, rate limiting, or other hardening
that's explicitly tracked in [docs/ROADMAP.md](docs/ROADMAP.md)'s Phase 2
(e.g. advanced GDPR compliance, multi-tenant isolation) are welcome as
regular feature requests rather than security reports, unless you have a
concrete exploit in the current single-tenant MVP.
