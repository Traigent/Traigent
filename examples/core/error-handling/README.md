# Error Handling Example

Demonstrates graceful failure modes and fallback patterns with Traigent.

## Scenarios

1. **Invalid Configuration Space** — Bad types in `configuration_space` raise `ValueError`
2. **Sample-Budget Exceeded** — `budget_limit`/`budget_metric` stops optimization when budget is exhausted
3. **Optimization Timeout** — Short timeout results in `stop_reason == "timeout"`
4. **Preflight Validation** — Check environment variables before running optimization
5. **Graceful Fallback** — Fall back to a default configuration on any failure

## Run

```bash
# Mock mode (no API key needed)
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python run.py

# Real mode (requires API keys)
python run.py
```

## Key Patterns

- Validate `configuration_space` types before decoration
- Use `budget_limit`/`budget_metric` to cap resource usage
- Check `result.stop_reason` for timeout detection (not an exception)
- Always have a fallback configuration for production resilience
