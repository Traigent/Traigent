# Environment Configuration Templates

This directory contains environment configuration templates for different deployment scenarios.

## 📁 Templates

### `.env.local.template`
**Purpose**: Local development environment
**Usage**: Copy to project root as `.env`
**Features**:
- Local database configuration (PostgreSQL + Redis)
- Development API keys
- Debug logging enabled
- Mock mode disabled (use real backends)

### `.env.production.template`
**Purpose**: Production deployment
**Usage**: Use with secure secret management systems
**Features**:
- Production API endpoints
- Secret injection placeholders (`${VARIABLE}`)
- Enhanced security features enabled
- Warning-level logging
- **Never** use with actual secrets committed to version control

### `.env.test.template`
**Purpose**: CI/CD testing environments
**Usage**: Copy to project root for automated testing
**Features**:
- Mock mode enabled (no real API calls)
- In-memory SQLite database
- Debug logging for troubleshooting
- Test-safe credentials

## 🚀 Quick Start

### Local Development

```bash
# From project root
cp configs/env-templates/.env.local.template .env

# Edit .env with your actual values
nano .env  # or use your preferred editor

# Important: .env is gitignored - never commit it!
```

### CI/CD Pipeline

```bash
# GitHub Actions, GitLab CI, etc.
cp configs/env-templates/.env.test.template .env

# Or use environment variables directly in CI/CD
```

### Production Deployment

```bash
# Use your secret management system
# AWS Secrets Manager, HashiCorp Vault, etc.

# Example with envsubst (not recommended for production)
export TRAIGENT_API_KEY="your-secret-key"
envsubst < configs/env-templates/.env.production.template > .env
```

## 🔐 Security Best Practices

### ✅ DO
- Use `.env.local.template` as a reference for required variables
- Store production secrets in secure secret management systems
- Use environment variable injection in production
- Keep `.env` gitignored (already configured)
- Rotate API keys regularly
- Use the helper scripts in `scripts/secrets/` to sync `.env.local`
  from AWS Secrets Manager instead of keeping long‑lived keys locally.

### ❌ DON'T
- Commit `.env` files with real credentials
- Share API keys in code or documentation
- Use development keys in production
- Track `.env.production` with real secrets

## 📚 Environment Variables Reference

### Required Variables
```bash
TRAIGENT_BACKEND_URL    # Backend API endpoint
TRAIGENT_API_KEY        # Authentication key
```

### Optional Variables
```bash
# LLM Providers
OPENAI_API_KEY
ANTHROPIC_API_KEY
COHERE_API_KEY

# Database
DATABASE_URL
REDIS_URL

# Security
JWT_SECRET_KEY
ENCRYPTION_KEY

# Logging
LOG_LEVEL
TRAIGENT_LOG_LEVEL

# Features
TRAIGENT_MOCK_MODE
ENABLE_EXPERIMENTAL_FEATURES
```

## 🔒 AWS Secrets Manager Workflow

To keep secrets out of local files and CI logs:

1. Store the canonical JSON blob in AWS Secrets Manager
   (e.g., `traigent/dev/env`).
2. For local development, run:
   ```bash
   scripts/secrets/pull_secret.sh .env.local
   set -a && source .env.local && set +a
   ```
3. After rotating keys, push the updated `.env.local` back to AWS:
   ```bash
   scripts/secrets/push_secret.sh .env.local
   ```
4. CI pipelines should fetch the same secret at runtime and map each
   key into environment variables (see `docs/guides/secrets_management.md`).

## 🛠️ Troubleshooting

### "Missing required environment variable"
**Solution**: Copy the appropriate template and fill in required values

### "API authentication failed"
**Solution**: Verify `TRAIGENT_API_KEY` is set correctly

### "Database connection failed"
**Solution**: Check `DATABASE_URL` and ensure database is running

### "Tests failing with API errors"
**Solution**: Set `TRAIGENT_MOCK_MODE=true` in test environment

## 📖 Related Documentation

- [Project README](../../README.md)
- [Security Documentation](../../reports/3_security/)
- [Development Guide](../../docs/)

## 🔄 Migration from Old Structure

Previous `.env` files in project root have been consolidated:
- `.env.template` → `.env.local.template` (enhanced)
- `.env.example` → `.env.local.template` (merged)
- `.env.production` → `.env.production.template` (sanitized)
- `.env.test` → `.env.test.template` (sanitized)

Old files will be removed in cleanup. Use symlink `.env.example` → `.env.local.template` for compatibility.
