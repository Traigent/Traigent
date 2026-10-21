# Security Policy

## Supported Versions

Traigent is in a pre-1.0 line (0.x). Security fixes are released for the latest
0.x minor release.

| Version          | Supported |
|------------------|-----------|
| 0.30.x (latest)  | ✅ Yes    |
| < 0.30           | ❌ No     |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

To report a security vulnerability, use either channel:

1. Email: security@traigent.ai
2. [GitHub private vulnerability reporting](https://github.com/Traigent/Traigent/security/advisories/new)

Please include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Measures

Traigent SDK implements:

- JWT-based authentication for cloud operations
- Input validation and sanitization
- No hardcoded credentials (environment variables only)
- Secure defaults for all execution modes

## Credential & data trust model

Traigent separates Traigent backend credentials from model-provider
credentials.

- A Traigent API key authenticates SDK and CLI calls to the Traigent
  portal/backend. It is resolved from `TRAIGENT_API_KEY` or CLI-managed
  credentials and is attached to Traigent backend requests as Traigent auth
  material.
- Model-provider credentials are customer credentials. Examples include
  `OPENAI_API_KEY`, Anthropic/Cohere/Mistral keys, Azure OpenAI credentials,
  and AWS credentials for Bedrock. The supported local and hybrid execution
  paths execute model calls from the customer's process, so provider SDKs use
  the customer's local provider configuration.
- Traigent does not provide or use a shared Traigent-owned model-provider key
  for customer model calls.
- Provider credentials are not part of Traigent backend session, trial, or
  trace payloads. Backend and trace clients use Traigent API credentials for
  Traigent endpoints. See the [telemetry and content logging controls](docs/api-reference/telemetry.md)
  for the metrics, metadata, and optional content logs the SDK can collect.
- The optional Bedrock helper imports `boto3` only when Bedrock is used and
  creates a local `bedrock-runtime` client through the AWS SDK credential
  chain. `AWS_BEARER_TOKEN_BEDROCK`, when used by local AWS credential
  tooling, is also a customer/provider credential and follows the same
  boundary.

Trials run locally in the supported local and hybrid execution modes.

## Best Practices

When using Traigent:

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Use `offline=True` to stop submission to the Traigent backend (the legacy `privacy_enabled` setting is deprecated and does not protect data). While an `offline=True` run is in flight, OpenTelemetry trace export, the langfuse client, workflow-trace ingestion and the standalone API clients are also disabled (fail closed; process-wide, `traigent/utils/env_config.py` `offline_run_scope`). `offline=True` is not a guarantee that nothing leaves the machine: model-provider calls use your own provider credentials and are not gated by it. Traigent observability is not privacy-preserving (issue #2240).
- Keep dependencies updated
