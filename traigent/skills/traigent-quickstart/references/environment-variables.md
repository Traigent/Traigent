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
| `TRAIGENT_STRICT_COST_ACCOUNTING` | unset (strict at runtime when `cost` is an objective) | Unset: runs whose objectives include `cost` fail on an unpriced call instead of recording `$0`; other runs warn. `true`: also fails before trial 1 on unpriced models. `false`: never strict, unpriced calls are recorded as `$0` with a warning. |
| `LITELLM_LOCAL_MODEL_COST_MAP`     | set to `True` by Traigent, unless `TRAIGENT_LITELLM_LIVE_PRICES` is set | `True` makes LiteLLM use its bundled price table instead of downloading one from GitHub on import. Traigent sets this (via `setdefault`, so your own value always wins) as soon as `traigent` is imported, provided `traigent` is imported before `litellm` — see below. |
| `TRAIGENT_LITELLM_LIVE_PRICES`     | `false`         | `1`/`true`/`yes`/`on` opts out of Traigent's default LiteLLM local-price-table pin (`LITELLM_LOCAL_MODEL_COST_MAP` / `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS`), restoring LiteLLM's own network fetch on import. Use this to pick up new-model prices before Traigent's next release. |
| `TRAIGENT_HF_TOKENIZER_DOWNLOAD`   | `false`         | Token counting for models whose exact LiteLLM tokenizer lives on Hugging Face (Llama, Cohere `command-r`, older non-`claude-3` Anthropic ids) never downloads it by default — Traigent sets LiteLLM's own `disable_hf_tokenizer_download` flag, so counting falls back to LiteLLM's tiktoken-based approximate count for those models instead. `1`/`true`/`yes`/`on` opts back into LiteLLM's original download-capable (exact) tokenizer selection. Setting `HF_HUB_OFFLINE=1` yourself also works for this path (and for anything else in your process that honors it) but Traigent never sets it for you, since customers may legitimately download real Hugging Face models through Traigent elsewhere. |
| `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS` | set to `True` by Traigent, unless `TRAIGENT_LITELLM_LIVE_PRICES` is set | `True` stops LiteLLM fetching its Anthropic beta-header config from GitHub. Only matters for Anthropic models. Same default and opt-out as `LITELLM_LOCAL_MODEL_COST_MAP`. |
| `TRAIGENT_LOG_LEVEL`              | `INFO`          | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                    |
| `TRAIGENT_DEBUG`                  | (unset)         | When set to `1`, shows full tracebacks for `ConfigurationError` instead of user-friendly messages.  |
| `TRAIGENT_STRICT_VALIDATION`      | `true`          | When `true`, DTO schema validation raises exceptions. When `false`, logs warnings only.             |
| `ENVIRONMENT`                      | `development`   | Execution environment. Set to `production` for production deployments.                              |
| `JWT_SECRET_KEY`                   | (none)          | Secret key for JWT token validation. Required for production security features.                     |
| `TRAIGENT_API_KEY`                | (none)          | API key for authenticated backend/portal tracking. Use with `execution_mode="hybrid"` for portal-visible runs. `execution_mode="cloud"` is reserved for future remote execution. |
| `TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS` | `false`      | `injection_mode="seamless"` raises `ConfigurationError` before running the function when a non-empty `configuration_space` has no injectable target (no local assignment or parameter named after a config key), instead of silently running the unvaried code. Set `true` to restore the previous warning-only behavior for the rare intentional case. |

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

## Restricted Networks and Reproducible Cost

LiteLLM, which Traigent uses for pricing, can reach the network on its own:

- **On import** it downloads its model price table from `raw.githubusercontent.com`,
  and (for Anthropic models) a separate beta-header config from the same host, unless
  `LITELLM_LOCAL_MODEL_COST_MAP=True` / `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True` are
  already set. **Traigent sets both by default** as soon as `traigent` is imported
  (`os.environ.setdefault`, so an explicit value of yours is never overridden) —
  `import traigent` no longer makes this outbound call. Set
  `TRAIGENT_LITELLM_LIVE_PRICES=1` to opt back into LiteLLM's live fetch (e.g. to
  pick up prices for a model newer than your installed LiteLLM, ahead of Traigent's
  next release).
  The pin only works if `traigent` is imported before `litellm` in your process —
  if something else imports `litellm` first (an import sorter such as Ruff can
  reorder `import litellm` ahead of `import traigent`, or you may import litellm
  directly), that import already ran before Traigent's pin could apply, and Traigent
  logs a debug-level notice rather than silently doing nothing. Either import
  `traigent` first, or set the two `LITELLM_LOCAL_*` variables yourself before
  anything imports `litellm`. Blocked networks do not break the unpinned import:
  LiteLLM waits up to 5 seconds, prints a warning, and falls back.
- **When counting tokens** for Llama-family models (also Cohere `command-r` and older
  non-`claude-3` Anthropic ids), LiteLLM's *exact* tokenizer lives on Hugging Face and
  it used to fetch it lazily, per model, on first use. **Traigent turns this off by
  default** (LiteLLM's own `disable_hf_tokenizer_download` flag) — `import traigent`
  plus a token-counting call for one of these models no longer makes this outbound
  call; counting falls back to LiteLLM's tiktoken-based count instead, which is
  approximate (not the model's own tokenizer) for exactly these models. Set
  `TRAIGENT_HF_TOKENIZER_DOWNLOAD=1` to opt back into LiteLLM's original,
  download-capable exact tokenizer selection. Traigent does **not** set
  `HF_HUB_OFFLINE` for you — that would also block a customer's own, legitimate
  Hugging Face model downloads elsewhere in the SDK.

The bundled and live price tables differ, so the same run can report different costs
depending on which one loaded. A model newer than your installed LiteLLM may then have no price — this now applies
by default, not only when you opted in. When `cost` is an
objective the run stops at the first such call and names the fix, rather than scoring
the model as free. Provide a price with `TRAIGENT_CUSTOM_MODEL_PRICING_JSON` or
`TRAIGENT_CUSTOM_MODEL_PRICING_FILE`. Providers that report cost on each response,
such as OpenRouter, are priced from that report and are not affected.

Every run records which table it used in `result.metadata["pricing"]`
(`price_table_source` is `local` or `remote`), together with whether strict cost
accounting was on and why, and whether the run measured any LLM usage at all
(`usage_captured`). A run that declares a cost objective and captures no usage on any
trial has an unmeasured `$0` cost column, not a cheap one: it fails under strict
accounting and carries the `COST_OBJECTIVE_NO_USAGE_CAPTURED` warning otherwise.
Mock-LLM runs (`TRAIGENT_MOCK_LLM=true`) always warn rather than fail here — there is
no spend to measure in a simulated run.

When any trial's cost cannot be measured, the cost limit cannot bound spend, so
the whole run stops after `max_unmeasured_trials` trials (default `10`) with
`stop_reason == "cost_limit"` and the `COST_UNMEASURED_TRIAL_LIMIT_REACHED` warning
code. One unmeasured trial is enough, and `max_trials` alone does not lift the limit.
To go further, capture usage (see `docs/user-guide/cost_capture.md`; the run is then
bounded by `cost_limit` / `TRAIGENT_RUN_COST_LIMIT`) or raise the limit with
`max_unmeasured_trials=` on `@traigent.optimize` or `.optimize()` (an int >= 1). The
`TRAIGENT_FALLBACK_TRIAL_LIMIT` environment variable sets it when the parameter is not
given. A cost-objective run where only some trials captured
usage carries `COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED`; the unmeasured trials cannot
win on cost.

The run-scoped default does **not** reach the LangChain and Pydantic AI callback
handlers. Each reads `TRAIGENT_STRICT_COST_ACCOUNTING` once, when the handler object is
constructed — usually at import time, before any run starts — so a run that becomes
strict because `cost` is an objective does not make an already-built handler strict. If
you optimize through either integration, set `TRAIGENT_STRICT_COST_ACCOUNTING=true`
explicitly.

## .env File Support

When `python-dotenv` is installed (included in the `integrations` extra), Traigent automatically loads variables from a `.env` file in the current working directory.

Example `.env` file:

```
TRAIGENT_MOCK_LLM=true
TRAIGENT_OFFLINE_MODE=true
TRAIGENT_LOG_LEVEL=DEBUG
OPENAI_API_KEY=sk-...
```
