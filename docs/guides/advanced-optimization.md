# Advanced Optimization Guide

This guide covers advanced optimization strategies including budget constraints, exploration strategies, and smart sampling.

## Optimization Strategy

The `optimization_strategy` parameter provides fine-grained control over how TraiGent explores the configuration space.

```python
from traigent.api.decorators import EvaluationOptions

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-3-sonnet"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 500, 2000]
    },
    evaluation=EvaluationOptions(eval_dataset="complex_tasks.jsonl"),
    objectives=["accuracy", "cost"],
    optimization_strategy={
        "max_cost_budget": 100.0,      # Stop when $100 spent
        "exploration_ratio": 0.3,       # 30% exploration, 70% exploitation
        "adaptive_sample_size": True    # Smart dataset subset selection
    }
)
def complex_reasoning_task(query: str) -> str:
    # Your implementation
    ...
```

## Strategy Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_cost_budget` | `float` | Maximum total cost in USD before stopping optimization |
| `exploration_ratio` | `float` | Balance between exploring new configs (high) vs exploiting known good ones (low). Range: 0.0-1.0 |
| `adaptive_sample_size` | `bool` | Enable smart dataset subset selection to reduce costs |

## Budget Control

### Cost-Based Stopping

Set a hard budget limit to prevent runaway costs:

```python
optimization_strategy={
    "max_cost_budget": 50.0,  # Stop at $50
}
```

### Trial-Based Stopping

Control the number of trials via the `.optimize()` method:

```python
results = await my_agent.optimize(
    max_trials=20,  # Maximum configurations to test
    algorithm="bayesian",  # Use Bayesian optimization for efficiency
)
```

## Exploration vs Exploitation

The `exploration_ratio` controls the trade-off:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0` | Pure exploitation | When you have strong priors about good configs |
| `0.3` | Balanced (recommended) | General optimization |
| `0.5` | Equal exploration/exploitation | Large configuration spaces |
| `1.0` | Pure exploration | Initial discovery phase |

## Adaptive Sampling

When `adaptive_sample_size=True`, TraiGent intelligently selects dataset subsets:

1. **Early trials**: Use smaller subsets for quick filtering
2. **Promising configs**: Test with larger subsets for accuracy
3. **Final validation**: Full dataset evaluation for top candidates

This can reduce optimization costs by 60-80% while maintaining result quality.

## Algorithm Selection

Choose the right algorithm for your scenario:

```python
results = await my_agent.optimize(
    algorithm="random",    # Fast, good for small spaces
    # algorithm="grid",    # Exhaustive, for tiny spaces
    # algorithm="bayesian", # Smart, for large spaces
    max_trials=50,
)
```

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| `random` | Quick exploration | Fast, simple | May miss optima |
| `grid` | Small spaces (< 20 configs) | Exhaustive | Exponential scaling |
| `bayesian` | Large spaces | Sample-efficient | Higher overhead |

## See Also

- [Advanced Objectives](advanced-objectives.md) - Weighted multi-objective optimization
- [Performance Guide](performance.md) - Parallel execution
- [Execution Modes](execution-modes.md) - Local vs cloud modes
