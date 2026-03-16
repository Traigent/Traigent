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

- A secrets manager entry containing a JSON object
  (example: `traigent/dev/env` in AWS Secrets Manager).
- AWS CLI configured locally (`aws configure sso|profile`).
- A local `.env.local` file (gitignored by default).

## Local Development Workflow

1. **Pull secrets into a gitignored file**
   Example using AWS Secrets Manager (adjust to your vault/CLI):
   ```bash
   aws secretsmanager get-secret-value \
     --secret-id "${TRAIGENT_ENV_SECRET_ID:-traigent/dev/env}" \
     --query SecretString \
     --output text | \
     python3 - <<'PY' > .env.local
import json
import shlex
import sys

data = json.loads(sys.stdin.read() or "{}")
for key, value in data.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
   ```
   This writes key/value pairs with basic quoting/escaping handled for you.

2. **Load them into your shell**
   ```bash
   set -a
   source .env.local
   set +a
   ```
   Tools like `direnv` can automate this step.

3. **Edit safely when needed**
   - Modify `.env.local` locally (never commit it).
   - Update the canonical secret in your vault using your org's tooling
     (AWS Secrets Manager, Vault, Azure Key Vault, etc.).

4. **Regenerate anytime**
   - Delete `.env.local` if compromised and re-run the pull script.

## Ubuntu 24.04 Desktop (GNOME Keyring / Secret Service)

If your team develops on Ubuntu (GNOME), you can avoid `.env` files entirely for
*local* secrets by storing them in GNOME Keyring and only exporting them at
runtime.

### Prerequisites

```bash
sudo apt install libsecret-tools
```

### Store one secret (one-time per machine)

`libsecret-tools` exposes `secret-tool`, which writes to GNOME Keyring /
Secret Service directly:

```bash
printf '%s' "$OPENAI_API_KEY" | \
  secret-tool store --label='Traigent OPENAI_API_KEY' project traigent name OPENAI_API_KEY
```

Repeat for any other secret you want to keep out of shell history.

### Load secrets into the current shell

```bash
export OPENAI_API_KEY="$(secret-tool lookup project traigent name OPENAI_API_KEY)"
export ANTHROPIC_API_KEY="$(secret-tool lookup project traigent name ANTHROPIC_API_KEY)"
```

### Run commands with secrets loaded

```bash
export GROQ_API_KEY="$(secret-tool lookup project traigent name GROQ_API_KEY)"
python3 examples/tvl/reusable_safety/run_demo.py --real-llm
```

Notes:
- `secret-tool lookup ...` returns an empty string if a key is missing, so add
  your own guardrails in wrapper scripts if you require strict failure.
- For shared rotation, auditing, and revocation, prefer a real secrets manager
  (AWS Secrets Manager / Vault / Azure Key Vault / GCP Secret Manager).

## CI / Automation Workflow

Each GitHub Actions job (or other CI runner) should choose one of the following:

**Option A: Use CI-native secrets**
1. Store secrets in your CI system (for GitHub Actions, use repo/environment secrets).
2. Inject them as environment variables in the job.
3. The SDK reads `os.environ` at runtime.

**Option B: Pull from a secrets manager**
1. Export vault credentials via encrypted repo secrets:
   - `CI_AWS_ACCESS_KEY_ID`
   - `CI_AWS_SECRET_ACCESS_KEY`
   - `CI_AWS_REGION`
   - (Optional) `CI_AWS_SESSION_TOKEN`
   - `TRAIGENT_ENV_SECRET_ID` (e.g., `traigent/dev/env`)
2. Fetch the JSON blob and hydrate the job environment:
   ```bash
   aws secretsmanager get-secret-value \
     --secret-id "${TRAIGENT_ENV_SECRET_ID:-traigent/dev/env}" \
     --query SecretString \
     --output text | \
     python3 - <<'PY' >> "$GITHUB_ENV"
import json
import shlex
import sys

data = json.loads(sys.stdin.read() or "{}")
for key, value in data.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
   ```
3. Subsequent steps inherit the exported variables without checking secrets into logs.

## Rotation Checklist

- Rotate upstream provider keys (OpenAI/Anthropic/etc.).
- Update the canonical secret in your secrets manager.
- Re-pull to refresh `.env.local` as needed.
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
