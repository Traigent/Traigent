# Contributing to Traigent SDK

Thank you for your interest in contributing to Traigent SDK! We welcome contributions from the community and are excited to work with you to make AI agent discovery more accessible and powerful.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Traigent.git
   cd Traigent
   ```
3. **Install development dependencies**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies (preferred)
   make install-dev

   # Or use pip directly
   pip install -r requirements/requirements-dev.txt
   pip install -e ".[dev,integrations,analytics,security]"
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code follows our style guidelines (Black + isort for formatting, Ruff for linting)
   - All tests pass
   - New features include tests
   - Documentation is updated
   - Run `make format && make lint` before committing

3. **Run tests locally**:
   ```bash
   export TRAIGENT_MOCK_LLM=true

   # Run unit tests
   python -m pytest tests/unit/ -v

   # Run integration tests
   python -m pytest tests/integration/ -v

   # Run with coverage
   python -m pytest tests/ --cov=traigent --cov-report=term-missing
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Test changes
   - `chore:` Build process or auxiliary tool changes

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Code Style

- Formatting: Black (line length 88) + isort (profile "black")
- Linting: Ruff (lint only)
- Use type hints for all public APIs
- Add docstrings to all public functions and classes

Run formatting and linting:
```bash
make format
make lint
```

Or manually:
```bash
isort traigent/ tests/ examples/
black traigent/ tests/ examples/
ruff check traigent/ --fix
black --check traigent/
isort --check-only traigent/
```

## Testing

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest for testing
- Tests should be in the `tests/` directory mirroring the source structure
- Use `pytest -m "unit"` or `pytest -m "integration"` for targeted runs

Example test structure:
```python
def test_feature_behavior():
    """Test that feature behaves correctly."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = feature_function(input_data)

    # Assert
    assert result.is_valid()
```

## Documentation

- Update README.md if adding new features
- Add docstrings following Google style
- Update relevant documentation in `docs/`
- Include examples for new features

## Areas for Contribution

### High Priority
- **Test Coverage**: Help us reach 90%+ coverage
- **Documentation**: Improve guides and API documentation
- **Examples**: Add more real-world examples
- **Bug Fixes**: Check our issue tracker

### Feature Ideas
- New optimization algorithms
- Additional framework integrations
- Performance improvements
- UI/UX enhancements for the playground

## Pull Request Guidelines

1. **PR Title**: Use conventional commit format
2. **Description**: Clearly describe what changes you made and why
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update docs if needed
5. **Size**: Keep PRs focused and reasonably sized

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for features
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Reporting Issues

When reporting issues, please include:
- Python version
- Traigent SDK version
- Minimal reproducible example
- Error messages and stack traces
- Expected vs actual behavior

## Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Pull Requests**: For code contributions

## Project Structure

```
Traigent/
├── traigent/          # Main package
│   ├── api/           # Public API
│   ├── core/          # Core functionality
│   ├── optimizers/    # Optimization algorithms
│   └── ...
├── tests/             # Test suite
├── examples/          # Example scripts
├── playground/        # Interactive UI
└── docs/             # Documentation
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You

Your contributions help make Traigent SDK better for everyone. We appreciate your time and effort!

If you have any questions, feel free to open an issue or start a discussion.
