# Typed Session Smoke Environment

Use a dedicated non-production backend for the typed `/api/v1/sessions`
contract smoke workflows.

## Scope

This environment is for:

- `traigent-js` hybrid backend-guided local execution smoke
- Python SDK typed-session smoke only

It is **not** for:

- local backend debugging from GitHub Actions
- production
- Python remote cloud execution flows

## GitHub Environment

Create a GitHub Environment named `typed-session-smoke` in each repo and add:

- `TRAIGENT_API_URL`
- `TRAIGENT_API_KEY`

Use `TRAIGENT_API_URL` only. Do not configure `TRAIGENT_BACKEND_URL` for the
CI smoke workflows.

## Expected Backend URL

The smoke backend should expose:

- `https://smoke-backend.<your-domain>/api/v1`

The workflow derives `/health` from the API base and validates that before
running the typed session contract smoke.

## Backend Requirements

The smoke backend must support the full typed session lifecycle:

1. create session
2. get next trial
3. submit result
4. finalize
5. status
6. delete

For the recommended AWS deployment shape and operational details, see the
backend runbook:

- [TraigentBackend AWS Runbook](../../TraigentBackend/infrastructure/deployment/aws/README.md)

## Operational Notes

- keep this workflow nightly/manual only
- use separate CI-only API keys for JS and Python
- the smoke script does best-effort stale-session cleanup before create and
  always attempts delete in `finally`
- the Optuna SQLite store in the smoke environment is intentionally ephemeral
