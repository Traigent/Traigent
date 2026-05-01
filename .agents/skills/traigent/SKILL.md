---
name: traigent
description: "Guide users through Traigent optimization: setup, dry-run validation, and real execution. Use when a user asks to optimize a function with @traigent.optimize, run an optimization, or set up Traigent. ALWAYS start with dry-run (mock mode) to validate the full pipeline, then switch to real execution only when the user explicitly requests it."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent: Dry-Run First, Real When Ready

## Your Role

When a user asks you to optimize a function with Traigent, **always start with a dry run**. Real optimization costs real tokens and money. Never run real optimization until the user explicitly asks.

**Workflow:**

1. Set up the decorated function
2. Validate dataset, config space, and providers
3. Dry-run in mock mode — verify the full pipeline end-to-end at zero cost
4. Report what the dry run found, estimate real costs
5. **Wait** for the user to say "run it for real"

## Step 1: Set Up the Decorator

The user's function needs four things:

```python
import traigent
from traigent import Choices, Range

@traigent.optimize(
    eval_dataset="eval_data.jsonl",                    # 1. Dataset
    objectives=["accuracy"],                           # 2. What to optimize
    model=Choices(["gpt-4o-mini", "gpt-4o"]),          # 3. Config space (inline)
    temperature=Range(0.0, 1.0),
)
def my_function(query: str) -> str:                    # 4. The function
    config = traigent.get_config()
    return call_llm(model=config["model"], temperature=config["temperature"], prompt=query)
```

### Config Space: Inline vs Dict

Parameters can be defined inline on the decorator or in `configuration_space=`:

```python
# Inline (cleaner for simple spaces)
@traigent.optimize(
    model=Choices(["gpt-4o-mini", "gpt-4o"]),
    temperature=Range(0.0, 1.0),
)

# Dict (better for dynamic or large spaces)
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7, 1.0],
        "max_tokens": [256, 512, 1024],
    },
)
```

### Setup Mistakes to Catch

| Mistake | Fix |
|---|---|
| Config values as bare strings (`model="gpt-4"`) | Must be list or Range/Choices (`model=Choices(["gpt-4"])`) |
| `get_config()` called outside the function | Must be inside the decorated function body |
| Function doesn't return a value | Must return the prediction to score |
| Empty objectives list | At least one objective required |
| Dataset file doesn't exist | Create it or fix the path |

## Step 2: Validate Before Running

Use the SDK's built-in validation tools before any optimization.

### Validate the Dataset

```bash
traigent validate eval_data.jsonl
# With options: traigent validate eval_data.jsonl --objectives accuracy -v
```

Each line must be valid JSON with an `input` field. Include `output` when optimizing accuracy-like metrics:

```jsonl
{"input": "I was charged twice", "output": "billing"}
{"input": "API returns 500 error", "output": "technical"}
{"input": "What are your hours?", "output": "general"}
```

Minimum 5 examples for any signal. 10-20+ recommended.

### Check System and Algorithms

```bash
traigent info        # SDK version, Python version, enabled features
traigent algorithms  # Available algorithms with descriptions and best-use cases
```

### CLI Dry-Run Check

The `traigent check` command validates decorated functions without running real optimization:

```bash
# Dry-run: show what would be checked without running optimization
traigent check my_script.py --dry-run

# Check specific functions
traigent check my_script.py --functions="my_function" --dry-run
```

This discovers `@traigent.optimize` decorated functions and validates that dataset, objectives, and config space are defined in the decorator. It does not validate dataset file contents — use `traigent validate` for that.

## Step 3: Run Mock Optimization

Enable mock mode in code, then run the full optimization pipeline at zero cost. This tests everything — decorator wiring, config sampling, dataset loading, trial execution, scoring — end to end. Mock mode is hard-blocked when `ENVIRONMENT=production`, so this cannot accidentally swap real LLM calls for canned text in a deployed system.

> **Scope note:** `enable_mock_mode_for_quickstart()` flips a *runtime* flag — the LLM interceptors honor it from the moment it's called. It runs AFTER `import traigent`, so any *import-time* behavior of the SDK's optional dependencies (e.g., LiteLLM's model-cost-map fetch) has already executed. For fully hermetic startup (CI, air-gapped runs), set the equivalent env vars BEFORE Python imports anything: `TRAIGENT_MOCK_LLM=true`, `TRAIGENT_OFFLINE_MODE=true`, `LITELLM_LOCAL_MODEL_COST_MAP=True`. The bundled `traigent quickstart` command does this for you.

```python
import os
os.environ["TRAIGENT_OFFLINE_MODE"] = "true"   # Skip Traigent backend calls

import traigent
from traigent import Choices, Range
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()              # Mock LLM responses (dev-only)

@traigent.optimize(
    eval_dataset="eval_data.jsonl",
    objectives=["accuracy"],
    model=Choices(["gpt-4o-mini", "gpt-4o"]),
    temperature=Range(0.0, 1.0),
)
def my_function(query: str) -> str:
    config = traigent.get_config()
    return call_llm(model=config["model"], temperature=config["temperature"], prompt=query)

# Mock optimization — zero cost, validates the full pipeline
results = my_function.optimize_sync(max_trials=4, algorithm="random")

print(f"Trials ran:    {len(results.trials)}")
print(f"Failed trials: {len(results.failed_trials)}")
print(f"Stop reason:   {results.stop_reason}")
print(f"Best config:   {results.best_config}")
print(f"Best score:    {results.best_score}")
```

### Interpret Mock Results

| Check | Pass | Fail |
|---|---|---|
| Trials ran | `len(results.trials) > 0` | No trials = config space or dataset error |
| No failures | `len(results.failed_trials) == 0` | Failures = function or evaluator bug |
| Stop reason | `"max_trials_reached"` or `"optimizer"` | `"error"` = something broke |
| Config keys | Expected keys in `best_config` | Missing keys = config space mismatch |

Mock scores are random — ignore score values. Focus on whether the pipeline **runs without errors**.

### If Mock Fails

Enable debug logging:

```bash
export TRAIGENT_LOG_LEVEL=DEBUG
export TRAIGENT_DEBUG=1
```

Common failures:
- `ConfigurationError` — Fix decorator arguments (see setup mistakes table)
- `EvaluationError` — Fix scoring function or dataset format
- `OptimizationStateError` — `get_config()` called outside optimization context
- `ModuleNotFoundError` — `pip install traigent[integrations]`
- All trials failed — Test the function standalone with a hardcoded config first

**Do not proceed to real mode until mock passes cleanly.**

## Step 4: Report and Estimate Costs

After a successful mock run, tell the user:

1. **Pipeline validated** — trials, config space, dataset all working
2. **Config space size** — how many unique configurations
3. **Estimated LLM calls** — `max_trials x dataset_size` (upper bound)
4. **Cost limit** — default $2.00 USD per run (`TRAIGENT_RUN_COST_LIMIT`)
5. **Ask for go/no-go**

Example:

> Mock run passed: 4/4 trials, 0 failures, pipeline is valid.
>
> Config space: 2 models x continuous temperature. With `max_trials=10` and 15 dataset examples, that's up to 150 LLM calls.
>
> Default cost limit is $2.00 USD. Want me to run it for real? This will use your API keys and cost real tokens.

## Step 5: Run Real Optimization (Only When Asked)

When the user explicitly says to proceed:

### 1. Verify API Keys

Models in the config space need corresponding provider keys. Traigent auto-validates keys before starting and raises `ProviderValidationError` with details if validation fails.

```bash
export OPENAI_API_KEY="sk-..."         # For gpt-* models
export ANTHROPIC_API_KEY="sk-ant-..."  # For claude-* models
export GEMINI_API_KEY="..."            # For gemini-* models
```

### 2. Skip the Mock-Mode Activation and Set Cost Controls

```python
import os

# Just don't call enable_mock_mode_for_quickstart() this run.
# Mock mode is process-local — start a fresh interpreter for the
# real run if the previous one had it on.
os.environ.pop("TRAIGENT_OFFLINE_MODE", None)

# Cost limit — default $2.00 USD per run
os.environ["TRAIGENT_RUN_COST_LIMIT"] = "2.00"
```

### 3. Run Real Optimization

```python
from traigent.utils.exceptions import CostLimitExceeded

try:
    results = my_function.optimize_sync(max_trials=10, algorithm="bayesian")
except CostLimitExceeded as e:
    print(f"Budget hit: ${e.accumulated:.2f} / ${e.limit:.2f}")
    print("Increase TRAIGENT_RUN_COST_LIMIT to allow more spending.")

print(f"Best config:  {results.best_config}")
print(f"Best score:   {results.best_score}")
print(f"Total cost:   ${results.total_cost:.2f}" if results.total_cost else "")
print(f"Duration:     {results.duration:.1f}s")
print(f"Stop reason:  {results.stop_reason}")
```

### 4. Apply and Export

```python
# Lock in the best config for production
my_function.apply_best_config(results)

# Function now uses the winning config automatically
answer = my_function("What is Python?")

# Export config to file for deployment
my_function.export_config("best_config.json")
```

## Quick Reference

| | Mock (Dry Run) | Real |
|---|---|---|
| Activation | `traigent.testing.enable_mock_mode_for_quickstart()` | (don't call it) |
| `TRAIGENT_OFFLINE_MODE` | `true` | unset |
| API keys needed | No | Yes |
| LLM calls | Mocked | Real |
| Cost | $0 | Real tokens |
| Scores meaningful | Custom scorer recommended (built-in mock returns generic text) | Yes |
| Production-safe | Hard-blocked when `ENVIRONMENT=production` | — |
| Use when | Always first | After mock passes |

## See Also

- `traigent-quickstart` — Installation and first-time setup
- `traigent-configuration-space` — Range, Choices, IntRange, LogRange, constraints
- `traigent-decorator-setup` — EvaluationOptions, InjectionOptions, ExecutionOptions
- `traigent-run-optimization` — Algorithms, cost limits, parallel execution
- `traigent-analyze-results` — Interpret results, compare trials, apply best config
- `traigent-debugging` — Error diagnosis and troubleshooting
