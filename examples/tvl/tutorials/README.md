# TVL 0.9 Tutorials

Welcome to the TVL (Tuned Variables Language) 0.9 tutorials! These hands-on examples will teach you how to use TVL with TraiGent for LLM optimization.

## Prerequisites

- Python 3.10+
- TraiGent SDK installed: `pip install traigent`
- Set `TRAIGENT_MOCK_MODE=true` for tutorials (no API costs)

## Tutorial Overview

### 01. Getting Started
**Location:** `01_getting_started/`

Your first TVL spec! Learn the basics:
- Creating a TVL 0.9 spec file
- Defining TVARs (tuned variables)
- Setting objectives and constraints
- Running optimization with the `@traigent.optimize` decorator

```bash
cd 01_getting_started
python run_optimization.py
```

### 02. Typed TVARs
**Location:** `02_typed_tvars/`

Deep dive into the TVAR type system:
- All TVAR types: `bool`, `int`, `float`, `enum[T]`, `tuple[...]`, `callable[Proto]`
- Domain specifications: ranges, enums, resolution
- Registry domains (for dynamic value sets)
- Structural constraints with `when/then` implications

```bash
cd 02_typed_tvars
python explore_tvars.py
```

### 03. Multi-Objective Optimization
**Location:** `03_multi_objective/`

Real-world LLM systems have competing objectives:
- Weighted multi-objective optimization
- Pareto fronts and trade-offs
- Epsilon-Pareto dominance
- Hypervolume analysis

```bash
cd 03_multi_objective
python analyze_tradeoffs.py
```

### 04. Promotion Policy
**Location:** `04_promotion_policy/`

Production-grade deployment decisions:
- Statistical significance testing
- Benjamini-Hochberg correction for multiple comparisons
- Chance constraints for safety guarantees
- PromotionGate for candidate evaluation

```bash
cd 04_promotion_policy
python test_promotion.py
```

### 05. Statistical Testing
**Location:** `05_statistical_testing/`

Advanced statistical methods:
- TOST (Two One-Sided Tests) for equivalence
- Banded objectives for SLO compliance
- BandedObjectiveSpec for programmatic use
- Confidence intervals and p-values

```bash
cd 05_statistical_testing
python tost_demo.py
```

## Quick Reference

### TVL 0.9 Spec Structure

```yaml
spec:
  id: my-optimization
  version: "1.0"

metadata:
  owner: team@company.com
  description: My LLM optimization

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o", "gpt-4o-mini"]
    default: "gpt-4o-mini"

  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
    default: 0.7

objectives:
  - name: accuracy
    direction: maximize
  - name: latency
    direction: minimize

constraints:
  structural:
    - expr: "params.temperature <= 0.9"

promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    accuracy: 0.02
    latency: 50.0
```

### Loading a TVL Spec in Python

```python
from traigent.tvl import load_tvl_spec

spec = load_tvl_spec(spec_path="my_config.tvl.yml")

# Access TVARs
for tvar in spec.tvars:
    print(f"{tvar.name}: {tvar.type}")

# Access promotion policy
if spec.promotion_policy:
    print(f"Alpha: {spec.promotion_policy.alpha}")
```

### Using the Decorator

```python
import traigent

@traigent.optimize(tvl_spec="my_config.tvl.yml")
def my_llm_function(query: str, *, model: str, temperature: float):
    # TraiGent injects optimized parameters
    return call_llm(model, temperature, query)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **TVAR** | Tuned Variable - a parameter to optimize |
| **Domain** | The valid values for a TVAR (range, enum, registry) |
| **Objective** | A metric to optimize (maximize, minimize, or band) |
| **Epsilon-Pareto** | Dominance requiring improvement by at least epsilon |
| **TOST** | Two One-Sided Tests for equivalence testing |
| **Chance Constraint** | Probabilistic safety guarantee: P(metric >= threshold) >= confidence |
| **BH Correction** | Benjamini-Hochberg multiple testing correction |

## Next Steps

1. Complete these tutorials in order
2. Read the [TVL 0.9 Specification](../../docs/tvl/)
3. Explore the [Advanced Examples](../advanced/)
4. Join the TraiGent community for support

Happy optimizing! 🚀
