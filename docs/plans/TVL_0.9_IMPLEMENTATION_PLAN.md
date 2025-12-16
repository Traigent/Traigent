# TVL 0.9 Complete Implementation Plan

**Document Version:** 1.0
**Date:** 2025-12-15
**Branch:** `feature/tvl-language-complete`
**Author:** Claude (AI Assistant)
**Status:** Draft - Pending Review

---

## Executive Summary

This document outlines the plan to complete TVL (Tuned Variables Language) 0.9 support in the TraiGent SDK. TVL is a declarative language for specifying LLM optimization experiments, including typed variables, structural constraints, multi-objective optimization, and statistical promotion policies.

**Key Insight:** The TVL ecosystem is split across two repositories:
- **TraigentPaper/tvl/** - Standalone TVL specification, validation tools (`tvl-lint`, `tvl-check-structural`, etc.)
- **Traigent/** (SDK) - Runtime consumption of TVL specs for optimization

This plan focuses on **TraiGent SDK enhancements** only. Validation tooling already exists in the paper repository.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Current State Analysis](#2-current-state-analysis)
3. [Gap Analysis](#3-gap-analysis)
4. [Implementation Plan](#4-implementation-plan)
5. [Data Models](#5-data-models)
6. [API Changes](#6-api-changes)
7. [Test Plan](#7-test-plan)
8. [Migration Guide](#8-migration-guide)
9. [Open Questions](#9-open-questions)
10. [Appendices](#appendices)

---

## 1. Architecture Overview

### 1.1 Repository Split

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TraigentPaper/tvl/                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   tvl-lint      │  │ tvl-check-      │  │  tvl-ci-gate    │     │
│  │   (Phase 1A/B)  │  │ structural      │  │  (Promotion)    │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                │                                    │
│                    ┌───────────▼───────────┐                        │
│                    │   tvl/python/tvl/     │                        │
│                    │   - lints.py          │                        │
│                    │   - structural_parser │                        │
│                    │   - structural_sat    │                        │
│                    │   - model.py          │                        │
│                    └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ TVL Spec Files (.tvl.yml)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Traigent SDK                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    traigent/tvl/                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ spec_loader │  │   models    │  │     options         │  │   │
│  │  │   .py       │  │    .py      │  │       .py           │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │   │
│  │         │                │                    │              │   │
│  │         └────────────────┼────────────────────┘              │   │
│  │                          ▼                                   │   │
│  │              ┌───────────────────────┐                       │   │
│  │              │   TVLSpecArtifact     │                       │   │
│  │              │   (Runtime Artifact)  │                       │   │
│  │              └───────────┬───────────┘                       │   │
│  └──────────────────────────┼───────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  traigent/api/decorators.py                  │   │
│  │                       @optimize                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 TVL 0.9 Spec Structure (from Paper)

```yaml
# Required sections (per tvl.schema.json)
tvl:
  module: "corp.product.spec"
  validation:                    # Optional
    skip_budget_checks: false
    skip_cost_estimation: false
tvl_version: "0.9"

environment:
  snapshot_id: "2024-02-15T00:00:00Z"  # RFC3339 timestamp
  components: {}                       # Model catalog, prices, quotas

evaluation_set:
  dataset: "s3://datasets/eval.jsonl"
  seed: 2024                           # Optional

tvars:                                 # Typed variables
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o"]
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
      resolution: 0.05

constraints:
  structural:                          # Over TVARs only
    - when: zero_shot = true
      then: retriever.k = 0
    - expr: temperature <= 0.8
  derived:                             # Over environment symbols
    - require: latency_p95_ms <= 1500

objectives:
  - name: quality
    direction: maximize
  - name: latency_p95_ms
    direction: minimize
  - name: response_length                # Banded objective
    band:
      target: [100, 200]                 # Or {center, tol}
      test: TOST
      alpha: 0.05

promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    quality: 0.01
    latency_p95_ms: 50
  adjust: BH                            # Benjamini-Hochberg
  chance_constraints:
    - name: latency_slo
      threshold: 1500
      confidence: 0.95
  tie_breakers:
    response_length: min_abs_deviation

exploration:
  strategy:
    type: nsga2
  initial_sampling: latin_hypercube
  parallelism:
    max_parallel_trials: 4
  convergence:
    metric: hypervolume_improvement
    window: 10
    threshold: 0.001
  budgets:
    max_trials: 100
    max_spend_usd: 50.0
    max_wallclock_s: 3600
```

---

## 2. Current State Analysis

### 2.1 Existing Implementation in TraiGent SDK

| File | Purpose | Status |
|------|---------|--------|
| `traigent/tvl/spec_loader.py` | Load YAML specs → `TVLSpecArtifact` | Partial |
| `traigent/tvl/models.py` | TVL 0.9 data models | Comprehensive |
| `traigent/tvl/options.py` | `TVLOptions` Pydantic model | Complete |
| `traigent/tvl/__init__.py` | Public exports | Complete |

### 2.2 What `spec_loader.py` Currently Handles

```python
# Current TVLSpecArtifact fields
@dataclass(slots=True)
class TVLSpecArtifact:
    path: Path
    environment: str | None              # Environment NAME only (for overlays)
    configuration_space: dict[str, Any]  # Parsed from configuration_space
    objective_schema: ObjectiveSchema | None
    constraints: list[Callable]          # Compiled constraint callables
    default_config: dict[str, Any]
    metadata: dict[str, Any]
    budget: TVLBudget
    algorithm: str | None
```

**Current parsing functions:**
- `_parse_configuration_space()` - Handles legacy `configuration_space` section
- `_parse_objectives()` - Standard objectives only (no bands)
- `_parse_budget()` - From `optimization.budget`
- `_compile_constraints()` - CEL-like expressions, conditional, forbidden
- `_apply_environment()` - Environment overlays (inheritance)

### 2.3 What `models.py` Provides (Already Implemented)

```python
# Domain specifications
@dataclass
class DomainSpec:
    kind: Literal["enum", "range", "set", "registry"]
    values: list[Any] | None
    range: tuple[float, float] | None
    resolution: float | None
    registry: str | None
    filter: str | None
    version: str | None

# TVAR declarations
@dataclass
class TVarDecl:
    name: str
    type: TVarType  # bool, int, float, enum, tuple, callable
    raw_type: str
    domain: DomainSpec
    default: Any | None
    unit: str | None

# Band targets for banded objectives
@dataclass
class BandTarget:
    low: float | None
    high: float | None
    center: float | None
    tol: float | None

# Chance constraints
@dataclass
class ChanceConstraint:
    name: str
    threshold: float
    confidence: float

# Promotion policy
@dataclass
class PromotionPolicy:
    dominance: Literal["epsilon_pareto"]
    alpha: float
    min_effect: dict[str, float]
    adjust: Literal["none", "BH"]
    chance_constraints: list[ChanceConstraint]
    tie_breakers: dict[str, TieBreaker]

# Structural constraints
@dataclass
class StructuralConstraint:
    expr: str | None
    when: str | None
    then: str | None
    index: int

# Derived constraints
@dataclass
class DerivedConstraint:
    require: str
    index: int
    description: str | None
```

---

## 3. Gap Analysis

### 3.1 Missing in `spec_loader.py`

| Feature | Paper Schema | Current State | Priority |
|---------|--------------|---------------|----------|
| **`tvl.module`** | Required namespace | Not parsed | HIGH |
| **`tvl.validation`** | Optional flags | Not parsed | MEDIUM |
| **`environment.snapshot_id`** | Required RFC3339 | Not parsed | HIGH |
| **`environment.components`** | Optional dict | Not parsed | LOW |
| **`evaluation_set`** | Required section | Not parsed | HIGH |
| **`tvars`** (array form) | Required | Not parsed (uses legacy `configuration_space`) | HIGH |
| **Banded objectives** | `band.target`, `band.test`, `band.alpha` | Not parsed | HIGH |
| **`constraints.structural`** | New format | Partially (different format) | MEDIUM |
| **`constraints.derived`** | Linear arithmetic | Not parsed | MEDIUM |
| **`promotion_policy`** | Full policy | Not integrated | HIGH |
| **`exploration` section** | Full exploration config | Partial (`budget` only) | MEDIUM |
| **Convergence criteria** | `hypervolume_improvement`, etc. | Not parsed | MEDIUM |

### 3.2 Schema Differences

**Current SDK expects:**
```yaml
configuration_space:
  model:
    type: categorical
    values: [...]
```

**TVL 0.9 spec defines:**
```yaml
tvars:
  - name: model
    type: enum[str]
    domain: [...]
```

### 3.3 Constraint Format Differences

**Current SDK:**
```yaml
constraints:
  - id: my-constraint
    type: expression
    rule: 'params.temperature <= 0.8'
```

**TVL 0.9:**
```yaml
constraints:
  structural:
    - expr: temperature <= 0.8
    - when: zero_shot = true
      then: retriever.k = 0
  derived:
    - require: latency_p95_ms <= budget_limit
```

---

## 4. Implementation Plan

### Phase 1: Core TVL 0.9 Parsing (Priority: HIGH)

#### 4.1.1 Add TVL Header Parsing

```python
# New dataclass
@dataclass(slots=True)
class TVLHeader:
    module: str
    version: str | None
    skip_budget_checks: bool
    skip_cost_estimation: bool

def _parse_header(raw_data: dict[str, Any]) -> TVLHeader:
    """Parse tvl block and tvl_version."""
    tvl_block = raw_data.get("tvl", {})
    if not isinstance(tvl_block, dict):
        raise TVLValidationError("tvl block must be a mapping")

    module = tvl_block.get("module")
    if not isinstance(module, str) or not module:
        raise TVLValidationError("tvl.module is required")

    validation = tvl_block.get("validation", {})
    return TVLHeader(
        module=module,
        version=str(raw_data.get("tvl_version", "0.9")),
        skip_budget_checks=bool(validation.get("skip_budget_checks", False)),
        skip_cost_estimation=bool(validation.get("skip_cost_estimation", False)),
    )
```

#### 4.1.2 Add Environment Section Parsing

```python
@dataclass(slots=True)
class TVLEnvironment:
    snapshot_id: str  # RFC3339 timestamp
    components: dict[str, Any]

def _parse_environment_section(raw_data: dict[str, Any]) -> TVLEnvironment:
    """Parse environment block with snapshot_id."""
    env = raw_data.get("environment", {})
    if not isinstance(env, dict):
        raise TVLValidationError("environment must be a mapping")

    snapshot_id = env.get("snapshot_id")
    if not isinstance(snapshot_id, str):
        raise TVLValidationError("environment.snapshot_id is required (RFC3339)")

    # Optional: Validate RFC3339 format
    # datetime.fromisoformat(snapshot_id.replace('Z', '+00:00'))

    return TVLEnvironment(
        snapshot_id=snapshot_id,
        components=env.get("components", {}),
    )
```

#### 4.1.3 Add Evaluation Set Parsing

```python
@dataclass(slots=True)
class TVLEvaluationSet:
    dataset: str
    seed: int | None

def _parse_evaluation_set(raw_data: dict[str, Any]) -> TVLEvaluationSet:
    """Parse evaluation_set block."""
    eval_set = raw_data.get("evaluation_set", {})
    if not isinstance(eval_set, dict):
        raise TVLValidationError("evaluation_set must be a mapping")

    dataset = eval_set.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise TVLValidationError("evaluation_set.dataset is required")

    seed = eval_set.get("seed")
    return TVLEvaluationSet(
        dataset=dataset,
        seed=int(seed) if seed is not None else None,
    )
```

#### 4.1.4 Add TVars Parsing (Array Form)

```python
def _parse_tvars(raw_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[TVarDecl]]:
    """Parse tvars array into configuration space.

    Returns:
        (configuration_space, defaults, tvar_declarations)
    """
    tvars = raw_data.get("tvars", [])

    # Support both legacy configuration_space and new tvars
    if not tvars and "configuration_space" in raw_data:
        # Fall back to legacy format
        return _parse_configuration_space(raw_data)

    if not isinstance(tvars, list) or not tvars:
        raise TVLValidationError("tvars must be a non-empty list")

    config_space: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    declarations: list[TVarDecl] = []

    for idx, decl in enumerate(tvars):
        if not isinstance(decl, dict):
            raise TVLValidationError(f"tvars[{idx}] must be a mapping")

        name = decl.get("name")
        if not isinstance(name, str):
            raise TVLValidationError(f"tvars[{idx}].name is required")

        raw_type = decl.get("type", "")
        normalized_type = normalize_tvar_type(raw_type)
        if normalized_type is None:
            raise TVLValidationError(f"Unsupported type '{raw_type}' for TVAR '{name}'")

        domain_data = decl.get("domain")
        domain_spec = parse_domain_spec(name, normalized_type, domain_data)

        tvar = TVarDecl(
            name=name,
            type=normalized_type,
            raw_type=raw_type,
            domain=domain_spec,
            default=decl.get("default"),
            unit=decl.get("unit"),
        )
        declarations.append(tvar)

        # Convert to configuration space format
        config_space[name] = domain_spec.to_configuration_space_entry()

        if tvar.default is not None:
            defaults[name] = tvar.default

    return config_space, defaults, declarations
```

### Phase 2: Banded Objectives (Priority: HIGH)

#### 4.2.1 Extend Objective Parsing

```python
@dataclass(slots=True)
class BandedObjective:
    name: str
    target: BandTarget
    test: str  # "TOST"
    alpha: float

def _parse_objectives_v09(raw_data: dict[str, Any]) -> tuple[ObjectiveSchema | None, list[BandedObjective]]:
    """Parse objectives including banded objectives."""
    objectives = raw_data.get("objectives", [])
    if not isinstance(objectives, list) or not objectives:
        raise TVLValidationError("objectives must be a non-empty list")

    standard_objectives: list[ObjectiveDefinition] = []
    banded_objectives: list[BandedObjective] = []

    for idx, obj in enumerate(objectives):
        if not isinstance(obj, dict):
            raise TVLValidationError(f"objectives[{idx}] must be a mapping")

        name = obj.get("name")
        if not isinstance(name, str):
            raise TVLValidationError(f"objectives[{idx}].name is required")

        # Check for banded objective
        band = obj.get("band")
        if band is not None:
            if not isinstance(band, dict):
                raise TVLValidationError(f"objectives[{idx}].band must be a mapping")

            target = band.get("target")
            band_target = BandTarget.from_dict(target)

            test = band.get("test", "TOST")
            if test != "TOST":
                raise TVLValidationError(f"Only TOST test supported, got '{test}'")

            alpha = float(band.get("alpha", 0.05))
            if not 0 < alpha <= 1:
                raise TVLValidationError(f"Band alpha must be in (0, 1], got {alpha}")

            banded_objectives.append(BandedObjective(
                name=name,
                target=band_target,
                test=test,
                alpha=alpha,
            ))
        else:
            # Standard objective
            direction = (obj.get("direction") or "maximize").lower()
            if direction not in {"maximize", "minimize"}:
                raise TVLValidationError(f"Invalid direction '{direction}'")

            standard_objectives.append(ObjectiveDefinition(
                name=name,
                orientation=cast(Literal["maximize", "minimize"], direction),
                weight=float(obj.get("weight", 1.0)),
                unit=obj.get("unit"),
            ))

    schema = ObjectiveSchema.from_objectives(standard_objectives) if standard_objectives else None
    return schema, banded_objectives
```

### Phase 3: Constraints Refactoring (Priority: MEDIUM)

#### 4.3.1 Support Both Constraint Formats

```python
def _parse_constraints_v09(
    raw_data: dict[str, Any],
    validate: bool,
    path: Path,
) -> tuple[list[CompiledConstraint], list[StructuralConstraint], list[DerivedConstraint]]:
    """Parse TVL 0.9 constraint format.

    Supports both:
    - Legacy: constraints: [{id, type, rule}, ...]
    - TVL 0.9: constraints: {structural: [...], derived: [...]}
    """
    constraints_section = raw_data.get("constraints", {})

    # Check if legacy format (list)
    if isinstance(constraints_section, list):
        # Use existing _compile_constraints
        compiled = _compile_constraints(constraints_section, validate, path)
        return compiled, [], []

    if not isinstance(constraints_section, dict):
        return [], [], []

    # TVL 0.9 format
    structural_defs: list[StructuralConstraint] = []
    derived_defs: list[DerivedConstraint] = []
    compiled: list[CompiledConstraint] = []

    # Parse structural constraints
    structural = constraints_section.get("structural", [])
    if isinstance(structural, list):
        for idx, clause in enumerate(structural):
            if not isinstance(clause, dict):
                continue

            sc = StructuralConstraint.from_dict(clause, index=idx)
            structural_defs.append(sc)

            # Compile to runtime callable
            rule_expr = sc.to_rule_expression()
            compiled_rule = compile_constraint_expression(rule_expr, label=f"{path}:structural[{idx}]")

            compiled.append(CompiledConstraint(
                identifier=f"structural_{idx}",
                description=f"Structural constraint {idx}",
                requires_metrics=False,
                evaluator=compiled_rule,
                constraint_type="structural",
            ))

    # Parse derived constraints (stored as metadata, not compiled)
    derived = constraints_section.get("derived", [])
    if isinstance(derived, list):
        for idx, clause in enumerate(derived):
            if not isinstance(clause, dict):
                continue
            derived_defs.append(DerivedConstraint.from_dict(clause, index=idx))

    return compiled, structural_defs, derived_defs
```

### Phase 4: Promotion Policy Integration (Priority: HIGH)

#### 4.4.1 Parse Promotion Policy

```python
def _parse_promotion_policy(raw_data: dict[str, Any]) -> PromotionPolicy | None:
    """Parse promotion_policy section."""
    policy = raw_data.get("promotion_policy")
    if policy is None:
        return None

    if not isinstance(policy, dict):
        raise TVLValidationError("promotion_policy must be a mapping")

    return PromotionPolicy.from_dict(policy)
```

### Phase 5: Exploration Section (Priority: MEDIUM)

#### 4.5.1 Full Exploration Parsing

```python
@dataclass(slots=True)
class TVLExploration:
    strategy_type: str
    strategy_config: dict[str, Any]
    initial_sampling: str | dict | None
    max_parallel_trials: int | None
    convergence_metric: str | None
    convergence_window: int | None
    convergence_threshold: float | None
    max_trials: int | None
    max_spend_usd: float | None
    max_wallclock_s: int | None

def _parse_exploration(raw_data: dict[str, Any]) -> TVLExploration | None:
    """Parse full exploration section."""
    exploration = raw_data.get("exploration")
    if exploration is None:
        return None

    if not isinstance(exploration, dict):
        return None

    strategy = exploration.get("strategy", {})
    parallelism = exploration.get("parallelism", {})
    convergence = exploration.get("convergence", {})
    budgets = exploration.get("budgets", {})

    return TVLExploration(
        strategy_type=strategy.get("type", "tpe"),
        strategy_config={k: v for k, v in strategy.items() if k != "type"},
        initial_sampling=exploration.get("initial_sampling"),
        max_parallel_trials=parallelism.get("max_parallel_trials"),
        convergence_metric=convergence.get("metric"),
        convergence_window=convergence.get("window"),
        convergence_threshold=convergence.get("threshold"),
        max_trials=budgets.get("max_trials"),
        max_spend_usd=budgets.get("max_spend_usd"),
        max_wallclock_s=budgets.get("max_wallclock_s"),
    )
```

### Phase 6: Update TVLSpecArtifact (Priority: HIGH)

```python
@dataclass(slots=True)
class TVLSpecArtifact:
    """Normalized view of a TVL 0.9 spec ready for the decorator/runtime."""

    # Existing fields
    path: Path
    environment: str | None  # Legacy: environment overlay name
    configuration_space: dict[str, Any]
    objective_schema: ObjectiveSchema | None
    constraints: list[Callable[[dict[str, Any], dict[str, Any] | None], bool]]
    default_config: dict[str, Any]
    metadata: dict[str, Any]
    budget: TVLBudget
    algorithm: str | None

    # NEW: TVL 0.9 specific fields
    header: TVLHeader | None = None
    environment_snapshot: TVLEnvironment | None = None
    evaluation_set: TVLEvaluationSet | None = None
    tvars: list[TVarDecl] | None = None
    banded_objectives: list[BandedObjective] | None = None
    structural_constraints: list[StructuralConstraint] | None = None
    derived_constraints: list[DerivedConstraint] | None = None
    promotion_policy: PromotionPolicy | None = None
    exploration: TVLExploration | None = None
```

---

## 5. Data Models

### 5.1 New Dataclasses to Add

```python
# File: traigent/tvl/spec_loader.py (or new file: traigent/tvl/sections.py)

@dataclass(slots=True)
class TVLHeader:
    """TVL module header with namespace and validation options."""
    module: str
    version: str | None = "0.9"
    skip_budget_checks: bool = False
    skip_cost_estimation: bool = False

@dataclass(slots=True)
class TVLEnvironment:
    """Environment snapshot from TVL spec."""
    snapshot_id: str  # RFC3339
    components: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class TVLEvaluationSet:
    """Evaluation dataset specification."""
    dataset: str
    seed: int | None = None

@dataclass(slots=True)
class BandedObjective:
    """Banded (TOST) objective specification."""
    name: str
    target: BandTarget
    test: str = "TOST"
    alpha: float = 0.05

@dataclass(slots=True)
class TVLExploration:
    """Full exploration configuration."""
    strategy_type: str
    strategy_config: dict[str, Any] = field(default_factory=dict)
    initial_sampling: str | dict | None = None
    max_parallel_trials: int | None = None
    convergence_metric: str | None = None
    convergence_window: int | None = None
    convergence_threshold: float | None = None
    max_trials: int | None = None
    max_spend_usd: float | None = None
    max_wallclock_s: int | None = None
```

### 5.2 Type Hierarchy

```
TVLSpecArtifact (runtime artifact)
├── TVLHeader
│   └── module, version, validation flags
├── TVLEnvironment
│   └── snapshot_id, components
├── TVLEvaluationSet
│   └── dataset, seed
├── list[TVarDecl]
│   └── name, type, domain (DomainSpec)
├── ObjectiveSchema (standard objectives)
├── list[BandedObjective]
│   └── name, target (BandTarget), test, alpha
├── list[CompiledConstraint] (runtime callables)
├── list[StructuralConstraint] (metadata)
├── list[DerivedConstraint] (metadata)
├── PromotionPolicy
│   ├── dominance, alpha, min_effect, adjust
│   ├── list[ChanceConstraint]
│   └── tie_breakers
├── TVLExploration
│   └── strategy, convergence, budgets
└── TVLBudget (legacy, derived from exploration)
```

---

## 6. API Changes

### 6.1 Public API (No Breaking Changes)

The `load_tvl_spec()` function signature remains unchanged:

```python
def load_tvl_spec(
    *,
    spec_path: str | Path,
    environment: str | None = None,  # Still supports legacy overlays
    validate_constraints: bool = True,
) -> TVLSpecArtifact:
```

### 6.2 TVLSpecArtifact Changes (Additive Only)

All new fields have default values of `None`, maintaining backward compatibility:

```python
# Existing code continues to work
artifact = load_tvl_spec(spec_path="spec.tvl.yml")
space = artifact.configuration_space  # Still works

# New code can access TVL 0.9 features
if artifact.header:
    print(f"Module: {artifact.header.module}")

if artifact.banded_objectives:
    for band in artifact.banded_objectives:
        print(f"Band target: {band.target.low} - {band.target.high}")
```

### 6.3 Decorator Integration

The `@optimize` decorator should be updated to utilize new fields:

```python
# In traigent/api/decorators.py

def _apply_tvl_artifact(self, artifact: TVLSpecArtifact) -> None:
    """Apply TVL spec to the optimized function."""

    # Existing logic
    self.configuration_space = artifact.configuration_space
    self.objective_schema = artifact.objective_schema

    # NEW: Apply promotion policy
    if artifact.promotion_policy:
        self._promotion_policy = artifact.promotion_policy

    # NEW: Apply exploration settings
    if artifact.exploration:
        if artifact.exploration.max_trials:
            self._max_trials = artifact.exploration.max_trials
        if artifact.exploration.strategy_type:
            self._algorithm = artifact.exploration.strategy_type
```

---

## 7. Test Plan

### 7.1 Unit Tests

```python
# tests/unit/tvl/test_spec_loader_v09.py

class TestTVL09HeaderParsing:
    def test_parses_module_namespace(self):
        spec = {"tvl": {"module": "corp.product.spec"}}
        header = _parse_header(spec)
        assert header.module == "corp.product.spec"

    def test_requires_module(self):
        spec = {"tvl": {}}
        with pytest.raises(TVLValidationError, match="tvl.module is required"):
            _parse_header(spec)

    def test_parses_validation_options(self):
        spec = {
            "tvl": {
                "module": "test",
                "validation": {"skip_budget_checks": True}
            }
        }
        header = _parse_header(spec)
        assert header.skip_budget_checks is True


class TestTVL09EnvironmentParsing:
    def test_parses_snapshot_id(self):
        spec = {"environment": {"snapshot_id": "2024-02-15T00:00:00Z"}}
        env = _parse_environment_section(spec)
        assert env.snapshot_id == "2024-02-15T00:00:00Z"

    def test_requires_snapshot_id(self):
        spec = {"environment": {}}
        with pytest.raises(TVLValidationError, match="snapshot_id"):
            _parse_environment_section(spec)


class TestTVL09TVarsParsing:
    def test_parses_enum_tvar(self):
        spec = {
            "tvars": [
                {"name": "model", "type": "enum[str]", "domain": ["a", "b"]}
            ]
        }
        config, defaults, tvars = _parse_tvars(spec)
        assert config["model"] == ["a", "b"]
        assert tvars[0].type == "enum"

    def test_parses_range_tvar(self):
        spec = {
            "tvars": [
                {"name": "temp", "type": "float", "domain": {"range": [0, 1]}}
            ]
        }
        config, defaults, tvars = _parse_tvars(spec)
        assert config["temp"] == (0.0, 1.0)

    def test_falls_back_to_legacy_configuration_space(self):
        spec = {
            "configuration_space": {
                "model": {"type": "categorical", "values": ["a", "b"]}
            }
        }
        config, defaults, tvars = _parse_tvars(spec)
        assert "model" in config


class TestTVL09BandedObjectives:
    def test_parses_interval_band(self):
        spec = {
            "objectives": [
                {"name": "length", "band": {"target": [100, 200], "test": "TOST", "alpha": 0.05}}
            ]
        }
        schema, banded = _parse_objectives_v09(spec)
        assert len(banded) == 1
        assert banded[0].target.low == 100
        assert banded[0].target.high == 200

    def test_parses_center_tol_band(self):
        spec = {
            "objectives": [
                {"name": "cost", "band": {"target": {"center": 0.01, "tol": 0.005}, "test": "TOST", "alpha": 0.05}}
            ]
        }
        schema, banded = _parse_objectives_v09(spec)
        assert banded[0].target.low == 0.005
        assert banded[0].target.high == 0.015


class TestTVL09PromotionPolicy:
    def test_parses_full_policy(self):
        spec = {
            "promotion_policy": {
                "dominance": "epsilon_pareto",
                "alpha": 0.05,
                "min_effect": {"quality": 0.01},
                "adjust": "BH",
                "chance_constraints": [
                    {"name": "latency", "threshold": 1500, "confidence": 0.95}
                ]
            }
        }
        policy = _parse_promotion_policy(spec)
        assert policy.dominance == "epsilon_pareto"
        assert policy.adjust == "BH"
        assert len(policy.chance_constraints) == 1


class TestTVL09Exploration:
    def test_parses_convergence_criteria(self):
        spec = {
            "exploration": {
                "strategy": {"type": "nsga2"},
                "convergence": {
                    "metric": "hypervolume_improvement",
                    "window": 10,
                    "threshold": 0.001
                },
                "budgets": {"max_trials": 100}
            }
        }
        expl = _parse_exploration(spec)
        assert expl.convergence_metric == "hypervolume_improvement"
        assert expl.convergence_window == 10
        assert expl.max_trials == 100
```

### 7.2 Integration Tests

```python
# tests/integration/tvl/test_full_spec_loading.py

def test_loads_complete_tvl09_spec():
    """Test loading a complete TVL 0.9 spec file."""
    artifact = load_tvl_spec(spec_path="fixtures/complete_v09.tvl.yml")

    # Header
    assert artifact.header.module == "corp.test.spec"

    # Environment
    assert artifact.environment_snapshot.snapshot_id is not None

    # Evaluation set
    assert artifact.evaluation_set.dataset.startswith("s3://")

    # TVars
    assert len(artifact.tvars) >= 2

    # Objectives (standard + banded)
    assert artifact.objective_schema is not None
    assert len(artifact.banded_objectives) >= 1

    # Promotion policy
    assert artifact.promotion_policy.dominance == "epsilon_pareto"

    # Exploration
    assert artifact.exploration.strategy_type == "nsga2"


def test_backward_compatible_with_legacy_specs():
    """Ensure legacy specs still work."""
    artifact = load_tvl_spec(spec_path="fixtures/legacy_spec.tvl.yml")

    # Should still produce valid configuration_space
    assert "model" in artifact.configuration_space

    # New fields should be None
    assert artifact.header is None
    assert artifact.tvars is None
```

### 7.3 Test Fixtures

Create fixture files:
- `tests/fixtures/tvl/complete_v09.tvl.yml` - Full TVL 0.9 spec
- `tests/fixtures/tvl/legacy_spec.tvl.yml` - Legacy format spec
- `tests/fixtures/tvl/banded_objectives.tvl.yml` - Banded objectives only
- `tests/fixtures/tvl/minimal_v09.tvl.yml` - Minimal valid TVL 0.9

---

## 8. Migration Guide

### 8.1 For SDK Users

No migration required. Existing specs continue to work. New TVL 0.9 features are opt-in:

```python
# Before (still works)
@traigent.optimize(tvl_spec="old_spec.yml")
def my_function(): ...

# After (new features available)
@traigent.optimize(tvl_spec="new_v09_spec.tvl.yml")
def my_function():
    # Access new features via artifact
    ...
```

### 8.2 Converting Legacy Specs to TVL 0.9

```yaml
# Before (legacy)
configuration_space:
  model:
    type: categorical
    values: ["gpt-4o-mini", "gpt-4o"]
  temperature:
    type: continuous
    range: {min: 0, max: 1}

objectives:
  - name: quality
    direction: maximize

optimization:
  strategy: pareto_optimal
  budget:
    max_trials: 100

# After (TVL 0.9)
tvl:
  module: my.namespace.spec
tvl_version: "0.9"

environment:
  snapshot_id: "2024-01-01T00:00:00Z"

evaluation_set:
  dataset: "my-dataset"

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o"]
  - name: temperature
    type: float
    domain:
      range: [0, 1]

objectives:
  - name: quality
    direction: maximize

promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    quality: 0.01

exploration:
  strategy:
    type: nsga2
  budgets:
    max_trials: 100
```

---

## 9. Open Questions

### 9.1 Design Questions (Need Review)

1. **Should `tvars` completely replace `configuration_space`?**
   - Option A: Keep both, auto-detect format
   - Option B: Deprecate `configuration_space` in favor of `tvars`
   - **Current choice:** Option A (backward compatibility)

2. **How to handle banded objectives in `ObjectiveSchema`?**
   - Option A: Separate `banded_objectives` list
   - Option B: Extend `ObjectiveDefinition` with optional band info
   - **Current choice:** Option A (cleaner separation)

3. **Should we validate RFC3339 timestamp format?**
   - Option A: Strict validation
   - Option B: Accept any string (validation in tvl-lint)
   - **Current choice:** Option B (validation is tvl-lint's job)

4. **How to integrate `promotion_policy` with runtime?**
   - The SDK doesn't implement the statistical gate yet
   - Should we store it for future use or ignore it?
   - **Current choice:** Store in artifact for future integration

5. **Convergence criteria integration?**
   - Optuna/NSGA-II don't natively support hypervolume convergence
   - Custom callback needed?
   - **Current choice:** Store in artifact, implement callback later

### 9.2 Implementation Questions

6. **Registry domain resolution:**
   - Paper defines `registry` domains resolved at runtime
   - SDK has `RegistryResolver` protocol but no implementation
   - **Question:** Should we implement a default resolver?

7. **Derived constraint evaluation:**
   - Derived constraints reference environment symbols, not TVARs
   - SDK doesn't have access to live environment data
   - **Question:** Store as metadata only, or attempt evaluation?

8. **Multiple constraint formats:**
   - Legacy: `[{id, type, rule}]`
   - TVL 0.9: `{structural: [...], derived: [...]}`
   - **Question:** Support both indefinitely or deprecate legacy?

---

## Appendices

### A. TVL 0.9 EBNF Grammar (Key Excerpts)

```ebnf
module        = header, environment, evaluation_set, tvars, constraints,
                objectives, promotion_policy, [exploration] ;

header        = "tvl", ":", "{", "module", ":", ident,
                [",", "validation", ":", validation_opts], "}",
                [ "tvl_version", ":", string ] ;

tvars         = "tvars", ":", "[", { tvar_decl { "," } }, "]" ;
tvar_decl     = "{", "name", ":", ident, ",", "type", ":", type,
                ",", "domain", ":", domain_spec, "}" ;

type          = "bool" | "int" | "float" | "enum[str]" |
                "tuple[" , { type { "," } }, "]" |
                "callable[" , ProtoId , "]" ;

constraints   = "constraints", ":", "{", [ structural ], [ ",", derived ], "}" ;
structural    = "structural", ":", "[", { struct_clause { "," } }, "]" ;
struct_clause = "{", ( ("when", ":", formula, ",", "then", ":", formula) |
                       ("expr", ":", formula) ), "}" ;

objectives    = "objectives", ":", "[", { objective_decl { "," } }, "]" ;
objective_decl= std_objective | band_objective ;
band_objective= "{", "name", ":", ident, ",", "band", ":", "{",
                "target", ":", ( "[" , number, ",", number, "]" |
                                 "{", "center", ":", number, ",", "tol", ":", number, "}" ),
                [ ",", "test", ":", "TOST" ], [ ",", "alpha", ":", number ], "}", "}" ;

promotion_policy = "promotion_policy", ":", "{",
                   "dominance", ":", "epsilon_pareto",
                   [ ",", "alpha", ":", number ],
                   [ ",", "min_effect", ":", "{", ... "}" ],
                   [ ",", "adjust", ":", ("none" | "BH") ],
                   [ ",", "chance_constraints", ":", "[", ... "]" ],
                   [ ",", "tie_breakers", ":", "{", ... "}" ],
                   "}" ;

exploration   = "exploration", ":", "{",
                "strategy", ":", "{", "type", ":", strategy_type, ... "}",
                [ ",", "convergence", ":", "{", ... "}" ],
                [ ",", "budgets", ":", "{", ... "}" ],
                "}" ;
```

### B. Paper Formalization Reference

Key definitions from §III of the paper:

1. **Environment Snapshot (E_τ):** `⟨M_τ, P_τ, Q_τ, R_τ⟩` - model catalog, prices, quotas, policies
2. **Configuration Space:** `X(E_τ) = ∏_i D_{t_i}(E_τ)` - product of TVAR domains
3. **Structural Feasibility:** `F^str(E_τ) = {x ∈ X(E_τ) | C^str_{E_τ}(x)}`
4. **Operational Feasibility:** `F^op(E_τ, B_τ) ⊆ F^str(E_τ)`
5. **Stochastic ε-Pareto Dominance:** Definition 3 in paper
6. **TOST (Two One-Sided Tests):** For banded equivalence testing

### C. File Locations

```
Traigent SDK:
  traigent/tvl/
  ├── __init__.py
  ├── spec_loader.py    ← Main changes here
  ├── models.py         ← Already has TVL 0.9 types
  └── options.py

Paper Repository (reference only):
  TraigentPaper-master/tvl/
  ├── spec/grammar/tvl.ebnf
  ├── spec/grammar/tvl.schema.json
  ├── python/tvl/lints.py
  ├── python/tvl/structural_parser.py
  └── tvl_tools/tvl_lint/cli.py
```

### D. Example Complete TVL 0.9 Spec

```yaml
tvl:
  module: corp.support.rag_bot
  validation:
    skip_budget_checks: false
tvl_version: "0.9"

environment:
  snapshot_id: "2024-02-15T00:00:00Z"
  components:
    retriever: bm25-v3
    llm_gateway: us-east-1

evaluation_set:
  dataset: s3://datasets/support-tickets/dev.jsonl
  seed: 2024

tvars:
  - name: model
    type: enum[str]
    domain: ["gpt-4o-mini", "gpt-4o", "llama3.1"]
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
      resolution: 0.05
  - name: retriever.k
    type: int
    domain:
      range: [0, 20]
  - name: zero_shot
    type: bool
    domain: [true, false]

constraints:
  structural:
    - when: zero_shot = true
      then: retriever.k = 0
  derived:
    - require: latency_p95_ms <= 1500

objectives:
  - name: quality
    direction: maximize
  - name: latency_p95_ms
    direction: minimize
  - name: response_length
    band:
      target: [100, 200]
      test: TOST
      alpha: 0.05

promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  min_effect:
    quality: 0.5
    latency_p95_ms: 50
  adjust: BH
  chance_constraints:
    - name: latency_slo
      threshold: 1500
      confidence: 0.95

exploration:
  strategy:
    type: nsga2
  convergence:
    metric: hypervolume_improvement
    window: 5
    threshold: 0.01
  budgets:
    max_trials: 48
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-15 | Claude | Initial draft |

---

*End of Document*
