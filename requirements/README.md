# Requirements Files Guide

This directory contains all Traigent SDK dependencies organized by feature category. All files are synchronized with `pyproject.toml`.

## File Structure

```
requirements/
├── requirements.txt              # Core dependencies (required)
├── requirements-analytics.txt    # Analytics and intelligence features
├── requirements-bayesian.txt     # Bayesian optimization (includes Optuna)
├── requirements-integrations.txt # Framework integrations (LangChain, OpenAI, etc.)
├── requirements-security.txt     # Enterprise security features
├── requirements-test.txt         # Testing dependencies
├── requirements-dev.txt          # Development tools + all features
└── requirements-all.txt          # All optional features combined
```

## Installation Methods

### Method 1: Using pip (Traditional)

```bash
# Core only
pip install -r requirements/requirements.txt

# Development (all features + dev tools)
pip install -r requirements/requirements-dev.txt
```

### Method 2: Using uv (Recommended - 10-100x faster)

```bash
# Core only
uv pip install -r requirements/requirements.txt

# Development (all features + dev tools)
uv pip install -r requirements/requirements-dev.txt
```

### Method 3: Using pyproject.toml extras (Recommended for development)

```bash
# Using uv (faster)
uv pip install -e ".[test,dev,integrations,analytics,bayesian,security]"

# Install everything
uv pip install -e ".[all]"
```

## Recent Changes (2024-10-14)

### Added Dependencies
1. ✅ `cryptography>=3.4.0` to `requirements.txt`
2. ✅ `rank-bm25` to `requirements.txt`
3. 🆕 `optuna>=4.5.0` to `requirements-bayesian.txt`
4. 🆕 `langchain-anthropic>=0.2.0` to `requirements-integrations.txt`
5. 🆕 `anthropic>=0.18.0` to `requirements-integrations.txt`
6. 🆕 `rank_bm25>=0.2.2` to `requirements-integrations.txt`
7. 🆕 `fastapi>=0.95.0` to `requirements-security.txt`
8. 🆕 `uvicorn>=0.18.0` to `requirements-security.txt`
9. 🆕 `redis>=4.0.0` to `requirements-security.txt`

### Removed
- ❌ Duplicate `scripts/test/requirements-test.txt` (consolidated)

### Synchronized
- ✅ All requirements files now match `pyproject.toml` extras exactly

## See Also

- [Installation Guide](../docs/getting-started/installation.md)
- [Testing Guide](../docs/getting-started/testing.md)
