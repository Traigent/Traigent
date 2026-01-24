# Reusable Safety Constraints with TVL Inheritance

This example demonstrates how to use **TVL spec inheritance** to create reusable safety constraints that multiple agents can share. This follows the DRY (Don't Repeat Yourself) principle for evaluation and safety standards.

## Overview

```
reusable_safety/
├── base_safety.tvl.yml      # Base spec with shared safety constraints
├── qa_agent.tvl.yml         # Q&A agent (extends base, accuracy-focused)
├── support_agent.tvl.yml    # Support agent (extends base, latency/cost-focused)
├── qa_agent.py              # Q&A agent implementation
├── support_agent.py         # Support agent implementation
├── datasets/
│   ├── safety_hallucination.jsonl  # 50 examples for hallucination detection
│   ├── safety_toxicity.jsonl       # 50 examples for toxicity detection
│   ├── safety_bias.jsonl           # 50 examples for bias detection
│   ├── qa_accuracy.jsonl           # 50 examples for Q&A accuracy
│   └── support_resolution.jsonl    # 50 examples for support resolution
└── README.md
```

## Key Concepts

### 1. TVL Inheritance with `extends`

Child specs inherit from a base spec using the `extends` keyword:

```yaml
# qa_agent.tvl.yml
extends: ./base_safety.tvl.yml

tvl:
  module: "corp.agents.qa"

# Agent-specific configuration...
```

### 2. Shared Safety Constraints

The base spec defines 3 safety constraints that all agents must pass:

| Constraint | Metric | Threshold | Description |
|------------|--------|-----------|-------------|
| Hallucination Check | `hallucination_rate` | ≤ 10% | Factual consistency with context |
| Toxicity Check | `toxicity_score` | ≤ 5% | Absence of harmful content |
| Bias Check | `bias_score` | ≤ 10% | Fairness and lack of bias |

Each constraint is validated against 50 examples using LLM-as-a-judge evaluators.

### 3. Agent-Specific Objectives

While both agents share safety requirements, they have different optimization objectives:

**Q&A Agent (accuracy-focused):**
- Accuracy: weight 3.0 (primary)
- Latency P95: weight 1.0 (secondary)

**Support Agent (balanced efficiency):**
- Latency P50: weight 2.0
- Cost per Request: weight 2.0
- Resolution Accuracy: weight 1.0
- Customer Satisfaction: weight 1.5

### 4. Reusable Datasets and Evaluators

The base spec defines reusable datasets and evaluators that child specs inherit:

```yaml
# base_safety.tvl.yml
datasets:
  safety_hallucination:
    path: "./datasets/safety_hallucination.jsonl"
    size: 50

evaluators:
  ragas_faithfulness:
    type: ragas
    metric: faithfulness
```

## Running the Examples

### With Mock LLM (for testing)

```bash
cd examples/tvl/reusable_safety

# Run Q&A agent
TRAIGENT_MOCK_LLM=true python qa_agent.py

# Run Support agent
TRAIGENT_MOCK_LLM=true python support_agent.py
```

### With Real LLM

```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Run agents
python qa_agent.py
python support_agent.py
```

## How Inheritance Works

When a child spec uses `extends`, the following happens:

1. **Base spec is loaded** first
2. **Sections are merged**:
   - `safety_constraints`: Concatenated (child adds to parent's)
   - `objectives`: Concatenated
   - `tvars`: Concatenated
   - `constraints`: Concatenated
   - Other sections: Deep merged (child overrides parent)
3. **Cycle detection** prevents circular extends
4. **Path resolution** is relative to the child spec

Example of what the resolved Q&A agent spec contains:

```yaml
# Inherited from base_safety.tvl.yml
safety_constraints:
  - id: hallucination-check  # From base
  - id: toxicity-check       # From base
  - id: bias-check           # From base

# From qa_agent.tvl.yml
tvars:
  - name: model
  - name: temperature
  - name: retrieval_k
  # ...

objectives:
  - name: accuracy      # Weight: 3.0
  - name: latency_p95   # Weight: 1.0
```

## Benefits of This Pattern

1. **DRY Principle**: Safety constraints defined once, used everywhere
2. **Consistency**: All agents meet the same safety standards
3. **Maintainability**: Update base spec to change all agents
4. **Flexibility**: Agents can add their own constraints and objectives
5. **Compliance**: Easier to audit and certify safety requirements

## Integration with Safety Presets

The Python code uses Traigent's safety presets:

```python
from traigent.api.safety import (
    faithfulness,        # RAGAS metric
    hallucination_rate,  # Custom metric
    toxicity_score,      # Custom metric
    bias_score,          # Custom metric
    safety_score,        # Custom metric
)

@optimize(
    spec="qa_agent.tvl.yml",
    safety_constraints=[
        faithfulness.above(0.85, confidence=0.95),
        hallucination_rate().below(0.1),
        toxicity_score().below(0.05),
    ],
)
def qa_agent(question: str, **kwargs):
    ...
```

## Extending the Example

### Add a New Agent

1. Create `new_agent.tvl.yml`:
   ```yaml
   extends: ./base_safety.tvl.yml

   tvl:
     module: "corp.agents.new_agent"

   tvars:
     - name: model
       domain: ["gpt-4o-mini"]

   objectives:
     - name: your_metric
       direction: maximize
   ```

2. Create `new_agent.py` with `@optimize(spec="new_agent.tvl.yml")`

### Add a New Safety Constraint

1. Add to `base_safety.tvl.yml`:
   ```yaml
   safety_constraints:
     - id: new-check
       metric: new_metric
       threshold:
         operator: "<="
         value: 0.1
   ```

2. All child specs automatically inherit the new constraint

## See Also

- [TVL 0.9 Specification](../../../docs/tvl_spec.md)
- [Safety Presets Documentation](../../../docs/safety_presets.md)
- [RAGAS Integration](../../../docs/ragas_integration.md)
