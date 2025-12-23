# Advanced Objectives Configuration

This guide covers advanced objective configuration for fine-grained control over optimization priorities.

## Overview

For most use cases, simple objective lists work well:

```python
objectives=["accuracy", "cost"]  # TraiGent infers directions and equal weights
```

When you need explicit control over weights or orientations, use `ObjectiveSchema`.

## Custom Weighted Objectives

Use `ObjectiveSchema` when you want to:

- Assign different weights to objectives
- Explicitly specify maximize vs minimize
- Define custom metric names

```python
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.api.decorators import EvaluationOptions

custom_objectives = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.7),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ]
)

@traigent.optimize(
    objectives=custom_objectives,
    configuration_space={
        # Use tuples for continuous ranges, lists for categorical values
        "temperature": (0.0, 1.0),  # Continuous range
        "top_p": (0.1, 1.0),        # Continuous range
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],  # Categorical
    },
    evaluation=EvaluationOptions(eval_dataset="qa_samples.jsonl"),
)
def weighted_agent(question: str) -> str:
    # Your implementation
    ...
```

## Objective Orientations

| Orientation | Description | Use For |
|-------------|-------------|---------|
| `maximize` | Higher is better | Accuracy, F1 score, success rate |
| `minimize` | Lower is better | Cost, latency, error rate |

## Weight Guidelines

- Weights are relative (0.7 and 0.3 is the same as 70 and 30)
- All weights should sum to 1.0 for clarity
- Higher weight = more importance in optimization

### Example Weight Scenarios

**Accuracy-focused (research):**
```python
ObjectiveDefinition("accuracy", orientation="maximize", weight=0.9),
ObjectiveDefinition("cost", orientation="minimize", weight=0.1),
```

**Cost-focused (production at scale):**
```python
ObjectiveDefinition("accuracy", orientation="maximize", weight=0.4),
ObjectiveDefinition("cost", orientation="minimize", weight=0.6),
```

**Balanced:**
```python
ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
ObjectiveDefinition("cost", orientation="minimize", weight=0.5),
```

## Working Example

See `examples/quickstart/03_custom_objectives.py` for a complete runnable example.

## See Also

- [Evaluation Guide](evaluation.md) - Dataset formats and custom evaluators
- [Execution Modes](execution-modes.md) - Local vs cloud optimization
