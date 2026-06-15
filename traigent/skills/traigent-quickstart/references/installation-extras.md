# Traigent Installation Extras

Full reference of optional dependency groups available via `pip install 'traigent[extra_name]'`.

For most users, start with:

```bash
pip install "traigent[recommended]"
```

## Extras Table

| Extra           | Description                                        | Key Packages                                                                 |
| --------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `analytics`     | Analytics and intelligence features                | numpy, pandas, matplotlib                                                    |
| `bayesian`      | Bayesian optimization dependencies (used by the Traigent cloud; not available for local runs) | scikit-learn, scipy                                                          |
| `integrations`  | Framework integrations                             | LangChain (+ community/anthropic/openai/google), OpenAI, Anthropic, Groq, Google GenAI, MLflow, W&B, python-dotenv, boto3, faiss-cpu |
| `dspy`          | DSPy prompt optimization                           | dspy-ai                                                                      |
| `pydanticai`    | PydanticAI agent framework                         | pydantic-ai-slim                                                             |
| `security`      | Enterprise security features                       | passlib, FastAPI, Starlette, uvicorn, redis, defusedxml, pyotp               |
| `visualization` | Visualization and plotting                         | matplotlib, plotly                                                           |
| `hybrid`        | Portal-tracked local execution and external hybrid API integrations | httpx with HTTP/2, claude-code-sdk, mcp                                      |
| `tracing`       | OpenTelemetry tracing                              | opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp            |
| `test`          | Testing dependencies                               | pytest, pytest-asyncio, pytest-cov, pytest-mock, pytest-timeout, pytest-xdist, coverage, ragas, rapidfuzz, hypothesis |
| `dev`           | Development tools (linting + testing)              | pytest suite, black, isort, flake8, mypy, pre-commit, ruff, bandit           |
| `docs`          | Documentation generation                           | mkdocs, mkdocs-material, mkdocstrings                                        |
| `ml`            | Machine learning bundle                            | bayesian + analytics + numpy + scipy                                         |
| `cloud`         | Reserved dependencies for future remote execution; not needed for portal-tracked `hybrid` runs | security + boto3                                                             |
| `recommended`   | Recommended end-user bundle for running optimizations | integrations, analytics, bayesian, visualization, hybrid, pydanticai         |
| `all`           | All optional features combined                     | analytics, bayesian, integrations, pydanticai, security, visualization, test, tracing, hybrid |

## Install Commands

### pip

```bash
# Recommended install
pip install "traigent[recommended]"

# Individual extras
pip install 'traigent[analytics]'
pip install 'traigent[bayesian]'
pip install 'traigent[integrations]'
pip install 'traigent[dspy]'
pip install 'traigent[pydanticai]'
pip install 'traigent[security]'
pip install 'traigent[visualization]'
pip install 'traigent[hybrid]'
pip install 'traigent[tracing]'
pip install 'traigent[test]'
pip install 'traigent[dev]'
pip install 'traigent[docs]'
pip install 'traigent[ml]'
pip install 'traigent[cloud]'

# Combined bundles
pip install 'traigent[recommended]'
pip install 'traigent[all]'

# Multiple extras at once
pip install 'traigent[integrations,analytics,visualization]'
```

## Core Dependencies (Always Installed)

These packages are installed with the base `pip install traigent`:

- click (CLI framework)
- rich (terminal output formatting)
- aiohttp (async HTTP client)
- requests (sync HTTP fallback)
- jsonschema (schema validation)
- cryptography (encryption)
- PyJWT[crypto] (JWT signing and verification)
- backoff (retry with backoff)
- litellm (LLM provider abstraction)
- rank-bm25 (BM25 retrieval)
- optuna (optimization engine — used by the Traigent cloud; smart algorithms are not available in local runs)
- psutil (system monitoring)

## Notes

- Requires Python >= 3.11.
- `faiss-cpu` (in `integrations`) is not available on Windows.
- The `recommended` bundle installs `integrations`, `analytics`, `bayesian`, `visualization`, `hybrid`, and `pydanticai`. The `bayesian` extra installs dependencies used by the Traigent cloud; smart optimization algorithms are not available locally.
- The `all` bundle includes most extras for broad local development and testing.
