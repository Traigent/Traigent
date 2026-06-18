# Mock Mode Reference

## Overview

Traigent provides two environment variables for running without external dependencies:

| Variable | Purpose |
|---|---|
| `TRAIGENT_MOCK_LLM=true` | Mock supported LLM calls made through Traigent's integration/interceptor path. No API keys required for the documented local dry-run path. |
| `TRAIGENT_OFFLINE_MODE=true` | Skip backend/cloud connection. No Traigent service needed. |

These are typically used together for testing, CI/CD, and development.

## Enabling Mock Mode

### Environment Variables

```bash
# Both together (most common)
export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true

# Then run your code
python my_optimization.py
```

### In Python

```python
import os
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ["TRAIGENT_OFFLINE_MODE"] = "true"

# Must be set BEFORE importing traigent
import traigent
```

### In pytest

```python
import pytest

@pytest.fixture(autouse=True)
def mock_traigent_env(monkeypatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
```

Or via pytest CLI:

```bash
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/
```

## What TRAIGENT_MOCK_LLM Does

When `TRAIGENT_MOCK_LLM=true`:

- supported LLM API calls return synthetic/mock responses instead of calling real providers
- No API keys are required (OpenAI, Anthropic, etc.)
- No network calls are made to LLM providers for supported calls made while Traigent's interceptors are active
- Cost tracking reports zero or minimal cost
- Response times are near-instant

### What Gets Mocked

- LiteLLM completion calls (`litellm.completion(...)`) made inside optimized/evaluated Traigent runs after the SDK installs its interceptor
- LangChain model invocations made via Traigent's framework integration layer (e.g. `ChatOpenAI`, `ChatAnthropic` when `traigent[integrations]` is installed)

**Important:** Raw `openai.chat.completions.create(...)` and `anthropic.messages.create(...)` calls made directly (outside the LiteLLM or LangChain integration layer) are **not** intercepted and will attempt to reach real provider APIs. Stub those calls explicitly (e.g. with `unittest.mock`) for a guaranteed $0 rehearsal.

### What Is NOT Mocked

- Your own function logic (runs normally)
- Evaluator functions (run normally)
- Dataset loading and validation
- Configuration space sampling
- The optimization loop itself
- Provider calls made before the Traigent optimization/evaluation path installs interceptors
- Provider clients that Traigent does not support yet; stub those calls explicitly for a guaranteed $0 rehearsal

This means mock mode is useful for testing:
- Configuration space setup
- Evaluator logic
- Dataset format
- End-to-end optimization flow
- CI/CD pipeline integration

## What TRAIGENT_OFFLINE_MODE Does

When `TRAIGENT_OFFLINE_MODE=true`:

- No connection to the Traigent cloud backend
- Results are stored locally only
- No experiment syncing or cloud dashboard updates
- No authentication required

Use this when:
- Running on a machine without internet access
- Running in CI/CD where the backend is not available
- Developing locally without a backend service
- Testing the SDK in isolation

## Using Both Together

The most common pattern is to enable both:

```bash
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python my_script.py
```

This gives you a fully self-contained environment:
- No API keys needed
- No backend connection needed
- No provider network calls for supported calls made through the documented Traigent dry-run path
- Fast execution (no real LLM latency)

## Limitations of Mock Mode

Mock mode has important limitations to be aware of:

1. **Mock responses are not realistic**: The mock LLM returns synthetic text, not real model outputs. Do not use mock mode to evaluate actual model quality.

2. **Cost is not accurate**: Mock mode reports zero or minimal cost. Use real API calls for cost estimation.

3. **Latency is not representative**: Mock calls return instantly. Real optimization takes longer due to API latency.

4. **Provider-specific behavior is not simulated**: Rate limits, token limits, and provider-specific formatting are not mocked.

5. **Evaluator scores may differ**: If your evaluator scores based on output quality, mock responses will produce different (usually lower) scores than real LLM responses.

## Example: CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Optimization Setup
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      TRAIGENT_MOCK_LLM: "true"
      TRAIGENT_OFFLINE_MODE: "true"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install traigent[dev]
      - run: pytest tests/
```

## Example: Quick Validation Script

Validate your optimization setup before spending API budget:

```python
import os
os.environ["TRAIGENT_MOCK_LLM"] = "true"
os.environ["TRAIGENT_OFFLINE_MODE"] = "true"

import traigent

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
    },
    objectives=["accuracy"],
    eval_dataset="data.jsonl",
    max_trials=3,
)
def my_func(text):
    config = traigent.get_config()
    return f"Response using {config['model']}"

# Quick validation run
results = my_func.optimize()

print(f"Ran {len(results.trials)} trials")
print(f"Stop reason: {results.stop_reason}")
print(f"Best config: {results.best_config}")
print("Setup is valid - ready for real optimization")
```

## Disabling Mock Mode

Remove or unset the environment variables:

```bash
unset TRAIGENT_MOCK_LLM
unset TRAIGENT_OFFLINE_MODE
```

Or set to any value other than `true`:

```bash
export TRAIGENT_MOCK_LLM=false
```
