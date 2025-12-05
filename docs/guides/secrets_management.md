---
title: Secure Secret Management
---

# Secure Secret Management

This guide documents the SOC2/ISO-aligned process for handling Traigent
credentials without baking cloud SDK logic into the `traigent` package.

## Goals

- Keep the SDK agnostic: it only reads `os.environ` (or injected config).
- Manage secrets through AWS Secrets Manager (or another vault) using
  dedicated tooling and CI automation.
- Ensure local developers, CI pipelines, and production hosts pull the
  *same* canonical secret blob at runtime.

## Prerequisites

- AWS Secrets Manager secret containing a JSON object
  (default: `traigent/dev/env`).
- AWS CLI configured locally (`aws configure sso|profile`).
- `scripts/secrets/pull_secret.sh` and `scripts/secrets/push_secret.sh`
  from this repo (already tracked under `scripts/secrets/`).

## Local Development Workflow

1. **Pull secrets into a gitignored file**
   ```bash
   scripts/secrets/pull_secret.sh .env.local
   ```
   This writes key/value pairs with quoting/escaping handled for you.

2. **Load them into your shell**
   ```bash
   set -a
   source .env.local
   set +a
   ```
   Tools like `direnv` can automate this step.

3. **Edit safely when needed**
   - Modify `.env.local` (never commit it).
   - Push the change back to AWS:
     ```bash
     scripts/secrets/push_secret.sh .env.local
     ```

4. **Regenerate anytime**
   - Delete `.env.local` if compromised and re-run the pull script.

## CI / Automation Workflow

Each GitHub Actions job (or other CI runner) should:

1. Export AWS credentials via encrypted repo secrets:
   - `CI_AWS_ACCESS_KEY_ID`
   - `CI_AWS_SECRET_ACCESS_KEY`
   - `CI_AWS_REGION`
   - (Optional) `CI_AWS_SESSION_TOKEN`
   - `TRAIGENT_ENV_SECRET_ID` (e.g., `traigent/dev/env`)

2. Run the helper script and hydrate the job environment:
   ```bash
   scripts/secrets/pull_secret.sh ci.env
   while IFS= read -r line; do
     [[ -z "$line" || "$line" == \#* ]] && continue
     echo "$line" >> "$GITHUB_ENV"
   done < ci.env
   ```

3. Subsequent steps automatically inherit the exported variables
   without checking secrets into the repo or Action logs.

4. Jobs that do not require secrets can skip the step; the script is a
   no-op if the AWS credentials are absent.

## Rotation Checklist

- Rotate upstream provider keys (OpenAI/Anthropic/etc.).
- Update `.env.local` with new values.
- `scripts/secrets/push_secret.sh .env.local`
- Trigger CI to ensure new values are picked up.
- Remove local shells or runners that may still have old variables
  exported.

## Best Practices & Notes

- **SDK remains clean**: No boto3/aws-cli imports inside `traigent`.
- **Principle of least privilege**: Use IAM policies that grant read
  access to the secret for CI and read/write for a small ops group.
- **Auditing**: Secrets Manager tracks versions and rotation history.
- **Extensibility**: If customers prefer Vault/Azure, they can adapt
  the scripts or swap them out without touching the SDK.
