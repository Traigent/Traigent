# Traigent SDK Copilot Instructions

## 🧠 Project Context
Traigent is a Python SDK for zero-code LLM optimization using decorators (`@traigent.optimize`). It intercepts calls, injects parameters, and optimizes against objectives (accuracy, cost, latency).

**Key Technologies**: Python 3.11+, Optuna, LangChain, FastAPI, asyncio, Pydantic, MyPy

## 🏗️ Architecture & Patterns
- **Entry Point**: `traigent/api/decorators.py` (`@optimize`) wraps user functions in `OptimizedFunction` (`traigent/core/optimized_function.py`).
- **Orchestration**: `OptimizationOrchestrator` (`traigent/core/orchestrator.py`) manages the optimization loop and backend sync.
- **Integrations**: Adapters in `traigent/integrations/` (e.g., `LangChainAdapter`) handle framework-specific logic. **Do not inline framework logic**; use adapters.
- **Execution Modes**: `edge_analytics` (local + analytics), `mock` (testing), `cloud` (production).
- **Security**: `traigent/security/` handles JWT validation and encryption. **Never bypass auth in production code.**

### 📂 Project Structure
```
traigent/
├── api/                # Public API & decorators (@optimize)
├── core/              # Core optimization logic (orchestrator, sampler)
├── integrations/      # LLM & framework integrations (LangChain, OpenAI, etc.)
│   ├── llms/         # LLM-specific clients
│   ├── observability/ # MLflow, Wandb integrations
│   └── vector_stores/ # Vector DB integrations
├── optimizers/        # Optimization algorithms (Optuna, Bayesian)
├── evaluators/        # Evaluation logic & dataset handling
├── metrics/           # Metrics calculation (accuracy, cost, latency)
├── security/          # Auth, JWT, encryption
├── cloud/            # Cloud backend integration (async)
├── hooks/            # Pre/post optimization callbacks
├── storage/          # Local/remote result persistence
├── utils/            # Logging, env config, exceptions
└── tvl/              # Test Validation Language support

examples/
├── core/             # Core feature examples (start with simple-prompt/)
├── advanced/         # Advanced patterns (callbacks, hooks, RAG)
├── integrations/     # Framework integration examples
└── datasets/         # Sample evaluation datasets

tests/
├── unit/             # Fast isolated tests
└── integration/      # Integration & E2E tests
```

## 🛠️ Development Workflow
- **Setup**: `make install-dev` (installs with `[dev,integrations,analytics,security]`).
- **Testing**: `TRAIGENT_MOCK_LLM=true pytest tests/` (Critical: Use mock mode to avoid API costs).
  - **Markers**: `pytest -m "unit"`, `pytest -m "integration"`, `pytest -m "security"`, `pytest -m "slow"`.
- **Linting**: `make lint` (runs Ruff, MyPy, Bandit). `make format` (Black, Isort).
- **Mock Mode**: Always use `TRAIGENT_MOCK_LLM=true` for local development and tests.
- **Quick Fix**: `make quick-fix` applies formatting and safe lint fixes.

## 📐 Code Formatting & Linting (CRITICAL)

**Before committing any Python code changes, ALWAYS run:**

```bash
make format && make lint
```

Or manually:

```bash
# Format code
isort traigent/ tests/ examples/
black traigent/ tests/ examples/

# Lint code (use these exact commands - CI flags differ)
ruff check traigent/ --fix
black --check traigent/
isort --check-only traigent/
```

### Formatting Rules
- **Black**: Line length 88, Python 3.8+ target
- **isort**: Profile "black", line length 88 (must match black)
- **Ruff**: For linting only, not formatting

### ⚠️ CI vs Local Commands
The CI workflow uses different flags than local development:
- **CI**: `ruff check traigent/ --output-format=github` (for GitHub annotations)
- **Local**: `ruff check traigent/ --fix` (to auto-fix issues)

**Never use these flag combinations** (they are incompatible):
- `--output-format=github --statistics` ❌ (ruff doesn't support this)

### Pre-commit Checklist
1. Run `make format` (runs isort + black)
2. Run `make lint` (runs ruff + mypy + bandit)
3. Verify: `black --check traigent/ && isort --check-only traigent/`

## 🚨 Critical Rules
1. **Format Before Commit**: Always run `make format` before committing Python changes.
2. **Cross-Repo Policy**: Do not modify external systems. File issues for backend/upstream changes.
3. **Async-First**: Cloud operations (`traigent/cloud/`) must be async. Avoid blocking sync wrappers.
4. **Secrets**: NEVER hardcode secrets. Use `traigent.utils.env_config` or `os.environ`.
5. **Type Hints**: Required for all public APIs. Run `make lint` to verify.
6. **Dependencies**: Add new deps to `pyproject.toml` under the appropriate optional group (e.g., `integrations`).
7. **Testing Strategy**: Write unit tests first, then integration tests. Mock external APIs in unit tests.
8. **Mock Mode**: All LLM clients must support mock mode via environment variables (see memory: mock mode testing).

## 🔐 Environment Variables
Key environment variables (see `.env.example` for complete list):
- `TRAIGENT_MOCK_LLM=true` - Enable mock mode for testing without API costs
- `TRAIGENT_API_KEY` - Cloud backend authentication
- `TRAIGENT_LOG_LEVEL=DEBUG` - Enable debug logging
- `TRAIGENT_RESULTS_FOLDER=~/.traigent` - Local results storage path
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - LLM provider keys (for real API calls)

## 🧩 Common Tasks

### Adding New LLM Integration
1. Create client in `traigent/integrations/llms/` implementing `BaseLLMClient`
2. Must support: `invoke()`, `ainvoke()`, `invoke_stream()`, `ainvoke_stream()`
3. Add mock mode support (see memory: client consistency patterns)
4. Register in `traigent/integrations/plugin_registry.py`
5. Add unit tests in `tests/unit/integrations/`
6. Add integration test in `tests/integration/`

### Adding New Metric
1. Implement `BaseMetric` in `traigent/metrics/`
2. Register in `metric_registry.py`
3. Add tests in `tests/unit/metrics/`
4. Update documentation

### Adding New Optimizer
1. Implement in `traigent/optimizers/`
2. Follow Optuna patterns (see `traigent/optimizers/optuna/`)
3. Add benchmarks in `tests/integration/optimizers/`

### Debugging Tips
- Set `TRAIGENT_LOG_LEVEL=DEBUG` for verbose logging
- Use `TRAIGENT_MOCK_LLM=true` to avoid API costs during debugging
- Check `~/.traigent/` for local optimization results
- Use `make dev-server` to run the Streamlit UI for interactive debugging
- For async issues, check `asyncio_mode = "auto"` in `pyproject.toml` under `[tool.pytest.ini_options]`

## 📝 Examples Usage
Examples are in `examples/` directory:
- **Core**: `examples/core/` - Core features and getting started (start with `simple-prompt/run.py`)
- **Advanced**: `examples/advanced/` - Callbacks, hooks, RAG patterns
- **Integrations**: `examples/integrations/` - LangChain, OpenAI, Anthropic examples
- **Walkthrough**: `walkthrough/mock/` - Progressive 8-step tutorial

Each example directory contains a README.md with setup instructions and commands.

## 🚀 CI/CD Workflows
Key GitHub Actions workflows (`.github/workflows/`):
- `tests.yml` - Unit & integration tests (runs on all PRs)
- `quality.yml` - Linting, formatting, security checks
- `examples-smoke.yml` - Smoke tests for examples
- `example-auto-tune.yml` - Optional demo workflow for auto-tuning examples
- `architecture-analysis.yml` - Architecture drift detection

## 🔍 Code Review Checklist
Before submitting PRs:
1. ✅ Run `make format && make lint` - No linting errors
2. ✅ Run `pytest tests/unit -v` - All unit tests pass
3. ✅ Run `pytest tests/integration -v -m "not slow"` - Integration tests pass
4. ✅ Test examples with `TRAIGENT_MOCK_LLM=true`
5. ✅ Update docstrings for public APIs
6. ✅ Add/update tests for new features
7. ✅ Check no secrets in code (`make security`)
8. ✅ Verify async functions use `async/await` properly

## 🎯 Best Practices

### Async Code
- Use `async/await` for all I/O operations (API calls, file I/O)
- Cloud operations in `traigent/cloud/` must be async
- Use `asyncio.gather()` for concurrent operations
- Test async code with `pytest.mark.asyncio`

### Error Handling
- Use custom exceptions from `traigent.utils.exceptions`
- Log errors with context: `logger.error("Failed to X", exc_info=True)`
- Provide actionable error messages to users

### Testing
- Mock external APIs in unit tests (use `pytest-mock`)
- Use `TRAIGENT_MOCK_LLM=true` for integration tests
- Test edge cases and error paths
- Use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

### Documentation
- Follow Google docstring style
- Include Examples section in docstrings
- Update README.md when adding user-facing features
- Keep examples/ up to date with API changes
