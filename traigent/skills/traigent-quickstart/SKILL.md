---
name: traigent-quickstart
description: "Install and set up the Traigent SDK for LLM optimization. Use when the user wants to install traigent, set up their first optimization, create an eval dataset, or get started with @traigent.optimize. Covers pip install, environment variables, mock mode, and running a first optimization."
license: Apache-2.0
metadata:
  author: Traigent
  version: "1.0"
---

# Traigent Quickstart

## When to Use

Use this skill when:

- Setting up Traigent for the first time in a new project
- Installing the SDK and configuring the environment
- Creating a first `@traigent.optimize` decorated function
- Building an evaluation dataset in JSONL format
- Verifying that the installation works correctly
- Running optimization in mock/offline mode for development

## Installation

### Basic Install

```bash
pip install traigent
```

### With Optional Extras

```bash
# Framework integrations (LangChain, OpenAI, Anthropic, MLflow, W&B)
pip install 'traigent[integrations]'

# Analytics (numpy, pandas, matplotlib)
pip install 'traigent[analytics]'

# All optional features
pip install 'traigent[all]'

# Enterprise bundle (all production features)
pip install 'traigent[enterprise]'
```

See `references/installation-extras.md` for the full table of extras and their contents.

### Requirements

- Python >= 3.11

## Environment Setup

### Development Mode (Recommended for Getting Started)

Mock mode is the keyless dev path. Activate it in code:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

Skip backend communication via env:

```bash
export TRAIGENT_OFFLINE_MODE=true
```

- `enable_mock_mode_for_quickstart()` is hard-blocked when `ENVIRONMENT=production`.
- `TRAIGENT_OFFLINE_MODE=true` skips backend communication so you do not need a running Traigent backend.
- `TRAIGENT_MOCK_LLM=true` remains a non-production legacy fallback for older scripts; prefer the in-code API for new code.

### Using a .env File

Traigent supports `.env` files via `python-dotenv` (included in the `integrations` extra). Create a `.env` file in your project root:

```
TRAIGENT_OFFLINE_MODE=true
TRAIGENT_LOG_LEVEL=DEBUG
```

### Production Mode

For production, set your provider API keys and do not call `enable_mock_mode_for_quickstart()`:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

See `references/environment-variables.md` for all available environment variables.

## Your First Optimization

Here is a complete working example. This function classifies customer queries using an LLM, and Traigent will find the best model and temperature combination.

```python
import asyncio
import traigent
from traigent import Range, Choices

@traigent.optimize(
    eval_dataset="eval_queries.jsonl",
    objectives=["accuracy"],
    model=Choices(["gpt-4o-mini", "gpt-4o"]),
    temperature=Range(0.0, 1.0),
)
def classify_query(query: str) -> str:
    config = traigent.get_config()
    # Use config values in your LLM call
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=config["model"],
        temperature=config["temperature"],
        messages=[
            {"role": "system", "content": "Classify the query as: billing, technical, or general."},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


async def main():
    # Run optimization (async)
    results = await classify_query.optimize(max_trials=6, algorithm="grid")

    # Inspect results
    print(f"Best config: {results.best_config}")
    print(f"Best score:  {results.best_score}")
    print(f"Trials run:  {len(results.trials)}")

    # Apply the best configuration for production use
    classify_query.apply_best_config(results)

    # Now calling the function uses the best config
    answer = classify_query("I can't log in to my account")
    print(f"Classification: {answer}")


asyncio.run(main())
```

### Synchronous Alternative

If you prefer synchronous execution:

```python
results = classify_query.optimize_sync(max_trials=6, algorithm="grid")
```

### Key Concepts

1. **`@traigent.optimize(...)`** -- Decorator that wraps your function for optimization. Define what parameters to tune in the decorator arguments.
2. **`traigent.get_config()`** -- Call inside your function to retrieve the current trial's configuration. Works during optimization trials and after `apply_best_config()`.
3. **`func.optimize(max_trials=N)`** -- Run the optimization loop asynchronously. Returns an `OptimizationResult`.
4. **`func.apply_best_config(results)`** -- Lock in the best configuration found so that subsequent calls use it.

## Dataset Format

Traigent uses JSONL (JSON Lines) files for evaluation datasets. Each line must have an `input` field and an `output` field.

### Example: `eval_queries.jsonl`

```jsonl
{"input": "I was charged twice for my subscription", "output": "billing"}
{"input": "The API returns a 500 error on POST requests", "output": "technical"}
{"input": "What are your business hours?", "output": "general"}
```

- **`input`** -- The value passed to your function during evaluation.
- **`output`** -- The expected/ground-truth result used for scoring.

You can include additional fields for metadata, but `input` and `output` are required.

### Tips for Good Datasets

- Include at least 10-20 examples for meaningful optimization.
- Cover edge cases and diverse inputs.
- Ensure ground-truth `output` values are consistent and well-defined.

## Verify Installation

### Check SDK info

```bash
traigent info
```

This prints the installed version, Python version, available integrations, and execution mode.

### Verify from Python

```python
import traigent
print(traigent.get_version_info())
```

### Validate an eval dataset

```bash
traigent validate --dataset eval_queries.jsonl
```

## CLI Quick Reference

| Command               | Description                                    |
| --------------------- | ---------------------------------------------- |
| `traigent info`       | Show SDK version, environment, and integrations |
| `traigent algorithms` | List available optimization algorithms          |
| `traigent validate`   | Validate dataset files and configuration        |

## Next Steps

- **Define parameter search spaces** -- See the `traigent-configuration-space` skill for `Range`, `IntRange`, `Choices`, `LogRange`, factory presets, and constraints.
- **Choose an optimization algorithm** -- Run `traigent algorithms` to see available options (grid, random, bayesian, optuna, etc.).
- **Add multiple objectives** -- Use `objectives=["accuracy", "cost", "latency"]` for multi-objective optimization.
- **Use framework integrations** -- Install `traigent[integrations]` for LangChain, OpenAI, and Anthropic adapters.
