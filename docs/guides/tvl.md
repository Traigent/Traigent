# TVL (TraiGent Validation Language) Guide

TVL is a YAML-based specification language for defining optimization parameters, objectives, constraints, and budgets in a declarative format.

## Overview

TVL defines the _what_—constraints, objectives, and boundaries—while leaving the _how_ to any compatible optimizer. The power is in the specification, not the implementation.

## Basic Usage

```python
@traigent.optimize(tvl_spec="path/to/spec.tvl.yml")
def rag_agent(query: str) -> str:
    ...
```

TVL sections control the configuration space, objectives, constraints, and budgets—no extra arguments required.

## CLI Usage

```bash
traigent optimize path/to/module.py --tvl-spec path/to/spec.tvl.yml
traigent optimize path/to/module.py --tvl-spec path/to/spec.tvl.yml --tvl-environment staging
```

## TVL Spec Structure

A TVL spec file typically includes:

```yaml
# Example TVL spec
version: "1.0"

configuration_space:
  model:
    type: categorical
    values: ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
  temperature:
    type: continuous
    range: [0.0, 1.0]
  k:
    type: categorical
    values: [3, 5, 10]

objectives:
  - name: accuracy
    orientation: maximize
    weight: 0.7
  - name: cost
    orientation: minimize
    weight: 0.3

constraints:
  max_cost_per_call: 0.05
  min_accuracy: 0.8

budget:
  max_trials: 50
  max_cost: 10.0
```

## TVLOptions for Fine-Grained Control

```python
from traigent.tvl.options import TVLOptions

@traigent.optimize(
    tvl=TVLOptions(
        spec_path="path/to/spec.tvl.yml",
        environment="production",
        validate_constraints=True,
        apply_configuration_space=True,
        apply_objectives=True,
        apply_constraints=True,
        apply_budget=True,
    )
)
def my_agent(query: str) -> str:
    ...
```

## Why Use TVL?

1. **Portability**: A TVL spec can be validated by any conformant tool
2. **Version Control**: Track optimization parameters alongside code
3. **Environment-Specific**: Use different specs for dev/staging/production
4. **Separation of Concerns**: Keep optimization config separate from code

## See Also

- [Advanced Objectives Guide](advanced-objectives.md) - Weighted objectives
- [Execution Modes](execution-modes.md) - Local vs cloud optimization
