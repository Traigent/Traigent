# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.9.x   | ✅ Yes    |
| < 0.9   | ❌ No     |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

To report a security vulnerability:

1. Email: security@traigent.com
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

## Best Practices

When using Traigent:

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Enable privacy mode for sensitive data
- Keep dependencies updated
