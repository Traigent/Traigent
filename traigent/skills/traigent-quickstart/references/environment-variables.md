# Traigent Environment Variables

Complete reference of environment variables recognized by the Traigent SDK.

## Environment Variables Table

| Variable                           | Default         | Description                                                                                         |
| ---------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
| `TRAIGENT_MOCK_LLM`               | `false`         | When `true`, mocks supported LLM calls made through Traigent's integration/interceptor path. No provider API keys needed for the documented local dry-run path. |
| `TRAIGENT_OFFLINE_MODE`            | `false`         | When `true`, skips all backend communication. Use for local-only development.                       |
| `TRAIGENT_RUN_COST_LIMIT`         | `2.0`           | Maximum cost budget (in USD) per optimization run. Optimization stops when this limit is reached.    |
| `TRAIGENT_COST_APPROVED`          | `false`         | Exact value `true` pre-approves both the cost-limit prompt and unpriced-model preflight. `1`, `yes`, and `on` do not approve. |
| `TRAIGENT_VENDOR_MAX_RETRIES`     | `0`             | Bounded auto-retries on a transient vendor error (`429` rate-limit / `503` service-unavailable) before the run stops. `0` (default) preserves the immediate graceful stop; set e.g. `2` so a single transient blip does not abort an unattended/CI run. |
| `TRAIGENT_VENDOR_RETRY_BACKOFF`   | `1.0`           | Base backoff (seconds) between vendor auto-retries; grows exponentially and honors a vendor `Retry-After` when present. Only used when `TRAIGENT_VENDOR_MAX_RETRIES` > 0. |
| `TRAIGENT_SKIP_PROVIDER_VALIDATION`| `false`        | When `true`, skips API key validation at decoration time. Useful in CI environments.                |
| `TRAIGENT_VALIDATION_TIMEOUT`     | `5.0`           | Timeout in seconds for provider API key validation checks.                                          |
| `TRAIGENT_STRICT_COST_ACCOUNTING` | `false`         | Exact value `true` fails fast before trial 1 on unpriced models and when runtime cost extraction is missing or unknown. |
| `TRAIGENT_LOG_LEVEL`              | `INFO`          | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                    |
| `TRAIGENT_DEBUG`                  | (unset)         | When set to `1`, shows full tracebacks for `ConfigurationError` instead of user-friendly messages.  |
| `TRAIGENT_STRICT_VALIDATION`      | `true`          | When `true`, DTO schema validation raises exceptions. When `false`, logs warnings only.             |
| `ENVIRONMENT`                      | `development`   | Execution environment. Set to `production` for production deployments.                              |
| `JWT_SECRET_KEY`                   | (none)          | Secret key for JWT token validation. Required for production security features.                     |
| `TRAIGENT_API_KEY`                | (none)          | API key for authenticated backend/portal tracking. Use with `execution_mode="hybrid"` for portal-visible runs. `execution_mode="cloud"` is reserved for future remote execution. |

## LLM Provider API Keys

These are standard provider API keys consumed by the respective LLM SDKs. Traigent passes them through.

| Variable              | Provider                |
| --------------------- | ----------------------- |
| `OPENAI_API_KEY`      | OpenAI (GPT models)     |
| `ANTHROPIC_API_KEY`   | Anthropic (Claude models)|
| `GROQ_API_KEY`        | Groq (fast inference)   |
| `GOOGLE_API_KEY`      | Google (Gemini models)  |
| `WANDB_API_KEY`       | Weights & Biases        |
| `MLFLOW_TRACKING_URI` | MLflow tracking server  |

## Usage Examples

### Local Development (No API Costs)

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
export TRAIGENT_LOG_LEVEL=DEBUG
python my_optimization.py
```

Mock mode skips the optimized-function pricing preflight because supported
provider calls are intercepted and return canned responses. It is not a global
network sandbox: direct provider calls made before Traigent installs its
interceptors, or calls through unsupported clients, should be stubbed explicitly
or protected with provider-side spend controls.

### CI/CD Pipeline

```bash
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true
export TRAIGENT_COST_APPROVED=true
export TRAIGENT_SKIP_PROVIDER_VALIDATION=true
pytest tests/
```

`TRAIGENT_COST_APPROVED=true` must be the exact value `true`. It acknowledges
cost-sensitive execution for both cost-limit confirmation and unpriced-model
coverage warnings; it does not supply pricing. Runtime `cost_approved=True` must
be a real boolean, because string values are ignored and logged as warnings.

### Production with Cost Controls

```bash
export OPENAI_API_KEY=sk-...
export TRAIGENT_RUN_COST_LIMIT=5.0
export TRAIGENT_STRICT_COST_ACCOUNTING=true
export TRAIGENT_COST_APPROVED=true
export TRAIGENT_LOG_LEVEL=WARNING
python optimize_production.py
```

Unpriced models block before trial 1 unless cost-sensitive execution is
pre-approved: an interactive terminal prompts `y/N`, and non-interactive runs
fail closed. Mock LLM mode skips the optimized-function pricing preflight, but
it does not disable cost permits or accounting globally. Budget overruns are
controlled by `TRAIGENT_RUN_COST_LIMIT` / `CostLimitExceeded`, not strict cost
accounting.

### Portal-Tracked Hybrid Runs

```bash
export TRAIGENT_API_KEY=sk-... # pragma: allowlist secret
export TRAIGENT_BACKEND_URL=https://portal.traigent.ai
# In code: ExecutionOptions(execution_mode="hybrid")
python optimize_with_portal_tracking.py
```

### Debug Mode

```bash
export TRAIGENT_LOG_LEVEL=DEBUG
export TRAIGENT_DEBUG=1
python my_optimization.py
```

## .env File Support

When `python-dotenv` is installed (included in the `integrations` extra), Traigent automatically loads variables from a `.env` file in the current working directory.

Example `.env` file:

```
TRAIGENT_MOCK_LLM=true
TRAIGENT_OFFLINE_MODE=true
TRAIGENT_LOG_LEVEL=DEBUG
OPENAI_API_KEY=sk-...
```
