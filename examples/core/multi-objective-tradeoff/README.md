# Multi-Objective Tradeoff

Balance accuracy vs cost when optimizing math expression evaluation.

## Quick Start

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/multi-objective-tradeoff/run_openai.py --max-trials 5
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `model` | gpt-4o-mini, gpt-4.1-nano, gpt-5-nano | Model comparison |
| `temperature` | 0, 0.1, 0.3 | Response variance |
| `max_tokens` | 64, 128, 256 | Output length limit |

## What It Optimizes

- Model selection across providers (OpenAI)
- Token budget vs accuracy tradeoff
- Cost optimization with accuracy constraints

## Expected Output

```
Best config: {'model': 'openai/gpt-4o-mini', 'temperature': 0.1, 'max_tokens': 128}
Best score: 0.95

Optimization summary:
  total_cost: $0.000234
  total_duration: 12.5s
```

## Key Concepts

- **Bayesian optimization**: Smart parameter search
- **ObjectiveSchema**: Define weighted objectives
- **Parallel execution**: Configurable concurrency

## Variants

- `run_openai.py` - OpenAI models with Bayesian search
- `run_openai_optuna.py` - Optuna integration
- `run_antropic.py` - Anthropic models
- `run_many_providers.py` - Cross-provider comparison

## Next Steps

- [token-budget-summarization](../token-budget-summarization/) - Strict token budgets
- [prompt-style-optimization](../prompt-style-optimization/) - Style optimization
