# Security policy

## Reporting a vulnerability

**Please don't report security problems in public issues.**

Report them privately through GitHub instead:
1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.

Only the maintainer can see the report. GitHub gives you a private thread where
the issue can be fixed and disclosed together.

A good report includes:
- what is affected (endpoint, module, or the live demo), with the commit or
  version;
- steps to reproduce, or a proof of concept that uses only data you created;
- the impact you expect, such as reading another session's data, bypassing
  operator auth, or exposing a secret.

This is a single-maintainer project, so responses are best effort. The aim is
to acknowledge a report within a week, agree on a fix and disclosure timeline,
and credit you in the release notes if you'd like.

## Scope

In scope:
- the code in this repository (API, CLI, persistence layer, web UI, Docker
  image, CI workflows);
- the behaviour of the public demo at https://scrag-api.onrender.com and
  https://scrag-web.onrender.com:
  - access to another session's documents, history or jobs;
  - getting around operator-only endpoints;
  - prompt injection that crosses a trust boundary, for example retrieved text
    triggering a sensitive tool;
  - secrets showing up in responses or logs;
  - uploads that escape their limits.

Out of scope:
- denial-of-service or load testing against the demo. It runs on free tiers;
  please test locally with `docker compose up` instead;
- reaching the demo's rate limits or daily budget (they are working as
  intended);
- vulnerabilities in third-party services (Groq, Render, Neon, GitHub). Report
  those to the provider;
- missing hardening headers or best-practice suggestions with no demonstrated
  impact (an ordinary issue is welcome for those).

## Testing the live demo safely

- Create your own sessions, and only access data you created.
- Don't use automated scanners or high request rates against the demo.
- Don't upload personal data. Uploads are deleted within 24 hours, but the
  demo is public.

Good-faith research that follows these rules is welcome. The maintainer won't
pursue it.

## Supported versions

Security fixes go to `main` and are deployed to the demo from there. There are
no maintained release branches.

## How the project handles security

These describe the design that reports are measured against:
- [the threat model](docs/threat-model.md): trust boundaries, threats,
  mitigations with their tests, and residual risks
- [ADR 0011](docs/decisions/0011-security-hardening.md) (agent tools, uploads,
  layered quotas)
- [README: Security](README.md#security)
- [ADR 0008](docs/decisions/0008-postgres-persistence-design.md) (session
  isolation, token hashing, upload limits)
- [the risk register](docs/planning/RISK_REGISTER.md)
