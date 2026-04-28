# Traigent Installation Extras

Full reference of optional dependency groups available via `pip install 'traigent[extra_name]'`.

## Extras Table

| Extra           | Description                                        | Key Packages                                                                 |
| --------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `analytics`     | Analytics and intelligence features                | numpy, pandas, matplotlib                                                    |
| `bayesian`      | Bayesian optimization algorithms                   | scikit-learn, scipy                                                          |
| `integrations`  | Framework integrations                             | LangChain (+ community/anthropic/openai/google), OpenAI, Anthropic, Groq, Google GenAI, MLflow, W&B, python-dotenv, boto3, faiss-cpu |
| `dspy`          | DSPy prompt optimization                           | dspy-ai                                                                      |
| `pydanticai`    | PydanticAI agent framework                         | pydantic-ai                                                                  |
| `security`      | Enterprise security features                       | PyJWT, passlib, FastAPI, Starlette, uvicorn, redis, defusedxml, pyotp        |
| `visualization` | Visualization and plotting                         | matplotlib, plotly                                                           |
| `hybrid`        | Hybrid mode (external agentic service optimization)| httpx with HTTP/2                                                            |
| `tracing`       | OpenTelemetry tracing                              | opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp            |
| `test`          | Testing dependencies                               | pytest, pytest-asyncio, pytest-cov, pytest-mock, pytest-timeout, pytest-xdist, coverage, ragas, rapidfuzz, hypothesis |
| `dev`           | Development tools (linting + testing)              | pytest suite, black, isort, flake8, mypy, pre-commit, ruff, bandit           |
| `docs`          | Documentation generation                           | mkdocs, mkdocs-material, mkdocstrings                                        |
| `ml`            | Machine learning bundle                            | bayesian + analytics + numpy + scipy                                         |
| `cloud`         | Cloud execution mode                               | security + boto3                                                             |
| `all`           | All optional features combined                     | analytics, bayesian, integrations, pydanticai, security, visualization, test, tracing, hybrid |
| `enterprise`    | Enterprise bundle with all production features     | analytics, bayesian, integrations, security, visualization, test, tracing, ml, cloud, hybrid  |

## Install Commands

### pip

```bash
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
pip install 'traigent[all]'
pip install 'traigent[enterprise]'

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
- claude-code-sdk (Claude Code integration)
- mcp (Model Context Protocol)
- backoff (retry with backoff)
- litellm (LLM provider abstraction)
- rank-bm25 (BM25 retrieval)
- optuna (optimization engine)
- psutil (system monitoring)

## Notes

- Requires Python >= 3.11.
- `faiss-cpu` (in `integrations`) is not available on Windows.
- The `all` and `enterprise` bundles include most extras. The difference is that `enterprise` also includes `ml` and `cloud` bundles.
