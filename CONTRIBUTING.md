# Contributing to Traigent

This top-level guide provides quick navigation for contributors. The complete
reference remains at
[`docs/contributing/CONTRIBUTING.md`](docs/contributing/CONTRIBUTING.md).

## Getting Started

1. Read the full contributor guide linked above.
2. Install dependencies and run tests locally before opening changes.
3. Review existing issues to avoid duplicate work.

## Code Style

Follow repository linting and formatting rules. Keep changes scoped, typed, and
covered by tests where practical.

## Testing

Run relevant unit/integration tests for your change. For broad changes, run the
full suite in CI-equivalent mode.

## Pull Requests

Open focused pull requests with a clear summary, rationale, and test evidence.
Link related issues and call out any known tradeoffs.

### Protected-target contributor handoff

Pull requests targeting `develop` or `main` must be authored by
`nimrodbusany`, except for the narrow Dependabot lane enforced by
`protected-target-authorization.yml`. Dependabot is authorized only when that
hosted gate verifies the constrained dependency-only change shape; the rule
does not authorize every contributor to open a protected-target pull request.

Before opening the pull request, check the authenticated GitHub identity:

```bash
gh api user --jq .login
```

If the result is not `nimrodbusany`, hand the contributor branch to that
authorized principal. The principal opens the protected-target pull request,
keeps the contributor's commit authorship and credits the contributor in the
pull request body, and validates that the final body contains a real marker
copied from the applicable governance record (`Spine-Trail:`,
`Spine-Session:`, or legacy `Spine:`). Do not impersonate the contributor or
change the authorization allowlist to complete the handoff.

Both `gh pr create` and a direct `gh api` pull-request creation call reach the
same hosted authorization and spine-marker gates. A local shell hook is useful
feedback, but it is not universal enforcement and cannot replace those hosted
checks. A shared pull-request preflight implementation is separate work and is
not added by this guide.

## Contributor Licensing

External contributors must sign Traigent's Contributor License Agreement (CLA)
before non-trivial changes can be merged. See
[`CONTRIBUTOR-LICENSING.md`](CONTRIBUTOR-LICENSING.md) for the policy and
contact path.

## Issue Reporting

When filing an issue, include reproduction steps, expected behavior, actual
behavior, and environment details.
