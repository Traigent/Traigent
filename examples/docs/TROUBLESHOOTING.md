# Traigent Troubleshooting Guide

Fast fixes for the most common issues. Start here, then dive deeper if needed.

## One-minute fixes
- Missing API keys: set `TRAIGENT_MOCK_LLM=true`.
- Import errors: run `pip install -e ".[examples]"` from repo root.
- High cost or latency: lower `max_trials` and pick cheaper models.
- Slow runs: shrink the configuration space; enable caching; reduce concurrency.
- Empty/poor results: verify dataset paths and that objectives match your goal.

## Setup and keys
```bash
# Install
pip install -e ".[examples]"

# Mock mode (no keys)
export TRAIGENT_MOCK_LLM=true

# Real keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Common run problems
- `ModuleNotFoundError: traigent`: not installed or wrong cwd -> install as above and run from repo root.
- `RateLimitError`: reduce concurrency or add `retry_on_rate_limit=True` with a small delay.
- `No feasible configuration found`: loosen constraints or switch to multi-objective (`objectives=["accuracy","cost"]`).
- `Config path not found`: use absolute/validated paths for `eval_dataset` and configs.

## Cost and performance
- Start with mock mode; then constrain spend via `max_trials`, cheaper models, or `constraints={"cost_per_call": "<0.01"}`.
- Reduce search space to discrete lists; add `early_stopping=True`.
- Cache and parallelize carefully: `parallel_config=ParallelConfig(trial_concurrency=2)` and avoid oversubscription.

## Debugging signals
- Enable logs: `TRAIGENT_LOG_LEVEL=DEBUG` or `logging.basicConfig(level=logging.DEBUG)`.
- Capture failures: add an `error_callback` to surface trial errors.
- Reproduce issues: run a single trial with a tiny dataset to isolate bad configs.

## Integration hiccups
- LangChain/OpenAI/Anthropic: install integrations via `pip install -e ".[integrations]"` (or include with `.[examples]`/`.[all]`).
- Network: increase `request_timeout`, set proxies if required.

## Prevention checklist
- Use a virtualenv; keep configs in version control.
- Always test in mock mode first.
- Start with small `max_trials`; watch cost/latency metrics.
- Confirm datasets exist under `examples/datasets/<example>/`.

Still stuck? See `EXAMPLES_GUIDE.md` or open a GitHub issue.
