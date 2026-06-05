# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.10.x  | ✅ Yes    |
| 0.9.x   | ✅ Yes    |
| < 0.9   | ❌ No     |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

To report a security vulnerability:

1. Email: security@traigent.ai
2. Include:
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

Current local and hybrid modes run trials locally. The `cloud` remote
execution mode is reserved for future use and fails closed in this SDK.

## Best Practices

When using Traigent:

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Enable privacy mode for sensitive data
- Keep dependencies updated
