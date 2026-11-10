## What and why

<!-- What does this change, and what problem does it solve? Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran and what you observed. For behaviour changes, name the tests you added. -->

## Checklist

- [ ] `python scripts/verify.py` passes: lint, format, types and tests.
- [ ] Changed behaviour is covered by tests that run offline (fakes, no network).
- [ ] Docs and `.env.example` are updated for new settings or endpoints.
- [ ] `CHANGELOG.md` has an entry under "Unreleased".
- [ ] No secrets, API keys or database URLs anywhere in the diff.
- [ ] Still free-tier friendly: no new paid service, and memory stays within the
      512 MB container budget (the CI `container` job checks this).
