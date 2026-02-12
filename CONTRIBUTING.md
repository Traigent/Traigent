# Contributing to Traigent

Thank you for your interest in contributing to Traigent SDK.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Traigent.git`
3. Install development dependencies: `make install-dev`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install with all dev dependencies
make install-dev

# Run tests (uses mock mode - no API costs)
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/

# Format code before committing
make format

# Run linters
make lint
```

## Code Style

- **Formatting**: Black (line length 88) + isort
- **Linting**: Ruff, MyPy, Bandit
- **Docstrings**: Google style with Args/Returns sections

Always run `make format && make lint` before committing.

## Testing

- Use `pytest -m unit` for unit tests
- Use `pytest -m integration` for integration tests
- Always use mock mode: `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true`

## Pull Request Process

1. Update tests for new functionality
2. Ensure all tests pass: `make test`
3. Update documentation if needed
4. Run `make format && make lint`
5. Create PR with clear description

## Issue Reporting

When reporting issues, include:
- Python version
- Traigent version (`traigent --version`)
- Steps to reproduce
- Expected vs actual behavior

## Adding New LLM Integrations

Want to add support for a new LLM provider like Groq, Together AI, or Perplexity?

**Quick Start:**
```bash
# 1. Run the scaffold script
python scripts/scaffold_llm_plugin.py <provider_name>

# 2. Customize the generated files
# 3. Run tests with mock mode
TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/test_<provider>_plugin.py -v

# 4. Format and lint
make format && make lint
```

**Resources:**
- 📘 **Complete Guide**: [docs/contributing/ADDING_NEW_INTEGRATIONS.md](docs/contributing/ADDING_NEW_INTEGRATIONS.md)
- ⚡ **Quick Reference**: [docs/contributing/QUICK_REFERENCE_LLM_INTEGRATION.md](docs/contributing/QUICK_REFERENCE_LLM_INTEGRATION.md)
- 🔧 **Scaffold Tool**: `python scripts/scaffold_llm_plugin.py --help`
- 📋 **Issue Template**: Use [New LLM Integration](.github/ISSUE_TEMPLATE/new_llm_integration.md)
- ✅ **PR Template**: Use [LLM Integration PR](.github/PULL_REQUEST_TEMPLATE/llm_integration.md)

## Questions?

Open a GitHub issue with the `question` label.
