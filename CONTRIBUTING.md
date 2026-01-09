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

## Questions?

Open a GitHub issue with the `question` label.
