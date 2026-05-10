# Traigent SDK Authentication Guide

## Overview

The Traigent SDK provides a modern, secure authentication system that follows industry best practices from tools like GitHub CLI and AWS CLI. This guide covers how to authenticate and manage your credentials.

## Quick Start

### 1. Interactive Login (Recommended)

The easiest way to authenticate is using the interactive CLI:

```bash
traigent auth login
```

This will:
1. Prompt for your email and password
2. Authenticate with the Traigent backend
3. Generate a long-lived API key
4. Store credentials securely on your system

### 2. Environment Variable

Set your API key as an environment variable:

```bash
export TRAIGENT_API_KEY=your_api_key_here
```

> Tip: Credentials saved through `traigent auth login` are resolved by the SDK automatically via the built-in Credential Manager, so you never need to copy keys into code.

### 3. Direct API Key Configuration

If you already have an API key:

```bash
traigent auth configure
# Select option 2 and enter your API key
```

## Custom Backend URL

By default the SDK and CLI target `https://portal.traigent.ai`. To authenticate against a different backend, use `--backend-url`:

```bash
traigent auth login --backend-url https://your-backend.example.com
```

The URL is stored in your credentials so subsequent SDK calls use it automatically.

You can also set the backend via environment variables:

```bash
export TRAIGENT_BACKEND_URL=https://your-backend.example.com
```

Verify which backend you're pointing to with:

```bash
traigent auth status
```

## Authentication Commands

### Login

Authenticate with the Traigent backend:

```bash
# Interactive login
traigent auth login

# Login with email
traigent auth login --email user@example.com

# Non-interactive mode (fails without prompts)
traigent auth login --email user@example.com --non-interactive
```

> **Note:** Non-interactive mode does not prompt for a password. For CI/CD, use `TRAIGENT_API_KEY` instead of `traigent auth login`.

### Check Status

View your current authentication status:

```bash
traigent auth status
```

Output shows:
- Authentication status
- User email and ID
- Backend URL
- API key (masked for security)

### Logout

Clear stored credentials:

```bash
traigent auth logout
```

This removes all locally stored credentials. Your API key remains valid on the backend.

### Refresh Tokens

Refresh expired JWT tokens:

```bash
traigent auth refresh
```

Note: API keys don't expire and don't need refresh.

### Configure

Interactive configuration wizard:

```bash
traigent auth configure
```

Options:
- Change backend URL
- Select authentication method
- Set up API key

### Who Am I

Check API key validity:

```bash
traigent auth whoami tg_your_api_key_here
```

## Automatic Credential Discovery

Unified authentication leverages the shared `CredentialManager` to look for credentials in a secure order. When the CLI stores an API key or refresh token, the SDK reuses it automatically during runtime. This removes the need to duplicate configuration across environments while keeping sensitive material out of source code.

The discovery order is:
- System environment variable (`TRAIGENT_API_KEY`)
- CLI-managed credentials saved via `traigent auth login` (local credentials file)
- Development defaults only when explicit test flags are enabled

If nothing is found, the CLI will prompt you during `traigent auth login`; the SDK expects explicit configuration or environment variables.

## Credential Storage

Credentials are stored using file-based storage with restricted permissions:

### 1. Local File
- Location: `~/.traigent/credentials.json`
- Permissions: 0600 (user read/write only)
- Contents are JSON-encoded

### 2. Environment Variables
- `TRAIGENT_API_KEY`
- Useful for CI/CD environments

## Priority Order

The SDK checks for credentials in this order:
1. Environment variables (highest priority)
2. CLI stored credentials (from `traigent auth login`)
3. Explicit dev-mode credentials — only when **both** of these are set:
   - `TRAIGENT_DEV_MODE=true` (or `TRAIGENT_GENERATE_MOCKS=true`)
   - `TRAIGENT_DEV_API_KEY=<value>`

   Setting only the dev-mode flag (without `TRAIGENT_DEV_API_KEY`) returns no
   credential — the SDK does **not** ship a hard-coded sentinel string fallback.
   This keeps an accidental `TRAIGENT_DEV_MODE=true` in production strictly
   inert rather than handing out a known string the backend may have hardcoded
   for testing.

## Security Features

### Secure Token Management
- Tokens are wrapped in in-memory `SecureToken` containers so raw values are never logged or persisted
- Automatic memory clearing when tokens are no longer needed
- Token values masked in all string representations
- Constant-time string comparison to prevent timing attacks
- Background refresh uses the resilient HTTP client with exponential backoff to renew tokens before expiry

### Rate Limiting Protection
- Exponential backoff after failed authentication attempts
- Maximum 3 failures before rate limiting
- Automatic jitter to prevent thundering herd
- Rate limit resets after successful authentication

### SOC2 Compliance
- Comprehensive audit logging for authentication events
- No sensitive data in logs
- Secure credential storage
- Automatic token refresh before expiry

## Integration with SDK

Once authenticated, the SDK automatically uses your credentials:

```python
import asyncio
import traigent

# The SDK automatically finds credentials from:
# 1. Environment variables
# 2. CLI auth (traigent auth login)
# 3. Development defaults

@traigent.optimize(
    evaluation={"eval_dataset": "data.jsonl"},
    configuration_space={"model": ["gpt-4", "gpt-3.5-turbo"]}
)
def my_function(input_text: str, **config):
    # Your function here
    return result

# Optimization automatically uses authenticated backend
results = asyncio.run(my_function.optimize())
```

## Backend Integration

The authentication system works with the managed Traigent backend:

### Supported Authentication Methods

1. **JWT Tokens**
   - Short-lived (1 hour default)
   - Automatic refresh support
   - Used for initial authentication

2. **API Keys**
   - Long-lived tokens
   - Generated after JWT authentication
   - Preferred for SDK usage

### API Endpoints

Backend endpoints are managed and may vary by deployment. Use the CLI and SDK APIs rather than hardcoding URLs.

## Troubleshooting

### Authentication Failed

1. Check your credentials:
   ```bash
   traigent auth status
   ```

2. Try logging in again:
   ```bash
   traigent auth logout
   traigent auth login
   ```

3. Verify backend URL:
   ```bash
   traigent auth configure
   ```

### Rate Limited

If you see "Rate limit exceeded":
- Wait for the specified retry time
- Check for correct credentials
- Contact support if the issue persists

### Token Expired

JWT tokens are short-lived:
```bash
traigent auth refresh
```

Or switch to API keys (recommended):
```bash
traigent auth login  # Generates API key automatically
```

## CI/CD Integration

For automated environments:

### GitHub Actions

```yaml
- name: Configure Traigent
  env:
    TRAIGENT_API_KEY: ${{ secrets.TRAIGENT_API_KEY }}
  run: |
    # SDK automatically uses environment variable
    python your_optimization_script.py
```

### Docker

```dockerfile
# Set API key at runtime
ENV TRAIGENT_API_KEY=${TRAIGENT_API_KEY}

# Or mount credentials
VOLUME /root/.traigent
```

### Jenkins

```groovy
withCredentials([string(credentialsId: 'traigent-api-key', variable: 'TRAIGENT_API_KEY')]) {
    sh 'python your_optimization_script.py'
}
```

## Best Practices

1. **Use API Keys for Production**
   - More stable than JWT tokens
   - No refresh needed
   - Better for long-running processes

2. **Rotate Keys Regularly**
   - Generate new keys periodically
   - Remove unused keys from backend

3. **Use Environment Variables in CI/CD**
   - Never commit credentials to code
   - Use secure secret management

4. **Monitor Authentication Events**
   - Check audit logs for suspicious activity
   - Review failed authentication attempts

## API Reference

### Python API

```python
from traigent.cloud.credential_manager import CredentialManager

# Check if authenticated
is_auth = CredentialManager.is_authenticated()

# Get API key
api_key = CredentialManager.get_api_key()

# Get auth headers
headers = CredentialManager.get_auth_headers()

# Get full credentials
creds = CredentialManager.get_credentials()

# Clear credentials
CredentialManager.clear_credentials()
```

### CLI Commands

```bash
# Main commands
traigent auth login        # Authenticate
traigent auth logout       # Clear credentials
traigent auth status       # Check status
traigent auth refresh      # Refresh tokens
traigent auth configure    # Configuration wizard
traigent auth whoami KEY   # Validate API key

# Options
--email EMAIL             # Specify email
--non-interactive         # No prompts; use API keys for CI/CD
--backend-url URL         # Target a specific backend URL
--help                   # Show help
```

## Migration Guide

If you're migrating from the old authentication system:

### Old Method
```python
# Manual API key management
os.environ["TRAIGENT_API_KEY"] = "your_key"
```

### New Method
```bash
# Use CLI for secure storage
traigent auth login

# Or continue using environment variables
export TRAIGENT_API_KEY=your_key
```

The new system remains compatible with existing `TRAIGENT_API_KEY` workflows.

## Support

For authentication issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Open a GitHub issue with the CLI output and backend URL
3. For security reports, follow the guidance in `docs/contributing/SECURITY.md`
