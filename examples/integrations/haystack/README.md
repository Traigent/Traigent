# Haystack Integration Examples

This directory contains examples demonstrating how to use Traigent with Haystack pipelines for automated parameter optimization.

## Prerequisites

```bash
pip install traigent[integrations]
# For real Haystack pipelines:
pip install haystack-ai>=2.0.0
```

## Examples Overview

| Example | Description | Epic Coverage |
|---------|-------------|---------------|
| [01_pipeline_introspection.py](01_pipeline_introspection.py) | Discover tunable parameters in a Haystack pipeline | Epic 1 |
| [02_exploration_space.py](02_exploration_space.py) | Define and manipulate configuration spaces | Epic 2 |
| [03_evaluation_dataset.py](03_evaluation_dataset.py) | Create and manage evaluation datasets | Epic 3.1 |
| [04_execute_with_config.py](04_execute_with_config.py) | Run pipelines with different configurations | Epic 3.2-3.3 |
| [05_haystack_evaluator.py](05_haystack_evaluator.py) | Use HaystackEvaluator with optimizers | Epic 3.4-3.5 |
| [06_tvl_roundtrip.py](06_tvl_roundtrip.py) | Export and import TVL specifications | Epic 3.6 |

## Quick Start

### 1. Pipeline Introspection

```python
from traigent.integrations.haystack import ExplorationSpace

# Discover parameters in your pipeline
space = ExplorationSpace.from_pipeline(my_pipeline)

print(f"Found {len(space.tvars)} tunable parameters:")
for name, tvar in space.tvars.items():
    print(f"  {name}: {tvar.constraint}")
```

### 2. Define Search Space

```python
from traigent.integrations.haystack import (
    TVAR,
    CategoricalConstraint,
    ExplorationSpace,
    NumericalConstraint,
)

space = ExplorationSpace(
    tvars={
        "generator.model": TVAR(
            name="model",
            scope="generator",
            python_type="str",
            default_value="gpt-4o",
            constraint=CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"]),
        ),
        "generator.temperature": TVAR(
            name="temperature",
            scope="generator",
            python_type="float",
            default_value=0.7,
            constraint=NumericalConstraint(min=0.0, max=2.0),
        ),
    }
)
```

### 3. Create Evaluation Dataset

```python
from traigent.integrations.haystack import EvaluationDataset

dataset = EvaluationDataset.from_dicts([
    {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence..."},
    {"input": {"query": "What is ML?"}, "expected": "Machine Learning..."},
])
```

### 4. Run Optimization

```python
from traigent.integrations.haystack import HaystackEvaluator, EvaluationDataset
from traigent.optimizers import GridSearchOptimizer

# Create evaluator
evaluator = HaystackEvaluator(
    pipeline=my_pipeline,
    haystack_dataset=dataset,
    metrics=["accuracy"],
    output_key="llm.replies",
)

# Create optimizer
optimizer = GridSearchOptimizer(
    config_space={
        "generator.temperature": [0.0, 0.5, 1.0],
        "retriever.top_k": [5, 10, 15],
    },
    objectives=["accuracy"],
)

# Run trials
history = []
while not optimizer.should_stop(history):
    config = optimizer.suggest_next_trial(history)
    result = await evaluator.evaluate(
        func=pipeline.run,
        config=config,
        dataset=dataset.to_core_dataset(),
    )
    history.append({"config": config, "score": result.metrics["accuracy"]})
```

### 5. Export Configuration

```python
# Export best configuration to TVL
space.to_tvl("best_config.yaml", description="Optimized RAG pipeline")

# Load later
reloaded = ExplorationSpace.from_tvl_spec("best_config.yaml")
```

## Running Examples

Each example can be run directly:

```bash
# Run all examples
python examples/integrations/haystack/01_pipeline_introspection.py
python examples/integrations/haystack/02_exploration_space.py
python examples/integrations/haystack/03_evaluation_dataset.py
python examples/integrations/haystack/04_execute_with_config.py
python examples/integrations/haystack/05_haystack_evaluator.py
python examples/integrations/haystack/06_tvl_roundtrip.py
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `ExplorationSpace` | `traigent.integrations.haystack` | Configuration space for optimization |
| `TVAR` | `traigent.integrations.haystack` | Tunable Variable representation |
| `EvaluationDataset` | `traigent.integrations.haystack` | Dataset for pipeline evaluation |
| `HaystackEvaluator` | `traigent.integrations.haystack` | Evaluator for optimization loops |
| `execute_with_config` | `traigent.integrations.haystack` | Run pipeline with config |
| `apply_config` | `traigent.integrations.haystack` | Apply config to pipeline |

## Constraint Types

| Type | Use Case | Example |
|------|----------|---------|
| `CategoricalConstraint` | Discrete choices | Model selection |
| `NumericalConstraint` | Continuous/integer ranges | Temperature, top_k |
| `ConditionalConstraint` | Parent-dependent values | Max tokens per model |

## Advanced Features

### Log Scale Sampling

For parameters spanning multiple orders of magnitude:

```python
NumericalConstraint(min=1e-5, max=1e-1, log_scale=True)
```

### Stepped Values

For discrete integer values with specific increments:

```python
NumericalConstraint(min=16, max=256, step=16)  # 16, 32, 48, ...
```

### Conditional Parameters

For parameters that depend on other parameter values:

```python
ConditionalConstraint(
    parent_qualified_name="generator.model",
    conditions={
        "gpt-4o": NumericalConstraint(min=100, max=8192),
        "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
    },
    default_constraint=NumericalConstraint(min=100, max=2048),
)
```

## Test Coverage

These examples correspond to extensively tested functionality:

- **413 integration tests** for Haystack integration
- **93% code coverage** for `traigent/integrations/haystack/`
- **100% coverage** for `evaluation.py`
- **99% coverage** for `evaluator.py`
- **98% coverage** for `execution.py`

Run tests:

```bash
TRAIGENT_MOCK_MODE=true pytest tests/integrations/ --cov=traigent/integrations/haystack
```
