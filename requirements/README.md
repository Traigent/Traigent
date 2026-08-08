# Requirements Files Guide

This directory contains all Traigent SDK dependencies organized by feature category. All files are synchronized with `pyproject.toml`.

## File Structure

```
requirements/
├── requirements.txt              # Core dependencies (required)
├── requirements-analytics.txt    # Analytics and intelligence features
├── requirements-integrations.txt # Framework integrations (LangChain, OpenAI, etc.)
├── requirements-security.txt     # Enterprise security features
├── requirements-test.txt         # Testing dependencies
├── requirements-dev.txt          # Development tools + all features
└── requirements-all.txt          # Safe broad optional features combined
```

## Installation Methods

### Method 1: Using pip (Traditional)

```bash
# Core only
pip install -r requirements/requirements.txt

# Development (all features + dev tools)
pip install -r requirements/requirements-dev.txt
```

### Method 2: Using pyproject.toml extras (Recommended for development)

```bash
pip install -e ".[test,dev,integrations,analytics,bayesian,security]"

# Install broad safe optional features
pip install -e ".[all]"

```

The Traigent-provided Chroma packaging extra is temporarily withdrawn. Do not install
`traigent[chroma]` or add `langchain-chroma` through this SDK while
[GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) has no upstream
patch. Existing manually managed integrations are not changed by this packaging removal.

## Recent Changes (2024-10-14)

### Added Dependencies
1. ✅ `cryptography>=3.4.0` to `requirements.txt`
2. ✅ `rank-bm25` to `requirements.txt`
3. 🆕 `langchain-anthropic>=0.2.0` to `requirements-integrations.txt`
4. 🆕 `anthropic>=0.18.0` to `requirements-integrations.txt`
5. 🆕 `rank_bm25>=0.2.2` to `requirements-integrations.txt`
6. 🆕 `fastapi>=0.95.0` to `requirements-security.txt`
7. 🆕 `uvicorn>=0.18.0` to `requirements-security.txt`
8. 🆕 `redis>=4.0.0` to `requirements-security.txt`

### Removed
- ❌ Duplicate `scripts/test/requirements-test.txt` (consolidated)

### Synchronized
- ✅ All requirements files now match `pyproject.toml` extras exactly

## See Also

- [Installation Guide](../docs/getting-started/installation.md)
- [Testing Guide](../docs/getting-started/testing.md)
