# Traigent Plugin Architecture Refactoring Plan

## Phase 1: Comprehensive Feature Matrix

This document maps all Traigent features to help plan the plugin/package architecture.

---

## CRITICAL ISSUES TO ADDRESS

### Import Dependency Issues (Must Fix Before Extraction)

| Issue | Location | Impact | Resolution |
|-------|----------|--------|------------|
| Orchestrator imports cloud client | `core/orchestrator.py:22` | Breaks if cloud moves to plugin | Lazy import inside method |
| Orchestrator imports ParallelExecutionManager | `core/orchestrator.py:48` | Breaks if parallel moves to plugin | Feature flag + lazy import |
| LocalAnalytics imports cloud auth | `analytics/local_analytics.py` | Edge analytics needs cloud | Keep minimal auth interface in base OR shim |
| Agent config depends on cloud models | `api/config_mapper.py:17` | Agents tied to cloud | Move agent config to cloud plugin |
| ObjectiveSchema used in stop conditions | `core/stop_conditions.py:13` | Plateau detection needs it | **Keep ObjectiveSchema in base** |
| Range builder methods import constraints | `api/parameter_ranges.py:157+` | `.equals()`, `.gt()` etc need Condition | **Keep entire constraints module in base** (see note below) |
| Seamless provider always registered | `config/providers.py:982` | `_PROVIDERS["seamless"]` always present | Gate registration with try/except or feature flag |

**Constraints Module Note**: `Condition` class inherits from `BoolExpr` (`api/constraints.py:207`), so they cannot be split. The entire constraints module (`BoolExpr`, `Condition`, `implies()`, `when().then()`) must stay in base package. The `traigent-constraints` plugin concept is removed.

### Algorithm Registry Issues

| Issue | Location | Resolution |
|-------|----------|------------|
| Default algorithm is "bayesian" | `api/functions.py:564` | Change default to "tpe" (Optuna TPE) |
| Optuna registry auto-registers CMA-ES/NSGA-II | `optimizers/registry.py:159-177` | Gate advanced algorithms with feature flag |
| random/grid are separate modules | `optimizers/registry.py:129-133` | Keep in base - no changes needed |
| Bayesian optimizer requires scikit-learn | `optimizers/registry.py:136-142` | Already uses try/except - move to plugin |

### Mock Mode Clarification

- **MOCK is NOT an execution mode** - it's `MockModeOptions` (separate config bundle)
- Location: `config/types.py:16`, `api/decorators.py:148`
- Resolution: Don't list in "execution modes" table; keep as separate config option in base

### Registry Architecture (TWO registries exist)

| Registry | Location | Purpose | Plugin Integration |
|----------|----------|---------|-------------------|
| Optimizer registry | `optimizers/registry.py:17` | Algorithm registration | Extend for plugin algorithms via entry points |
| Integration plugin registry | `integrations/plugin_registry.py:25` | LLM/vector store plugins | Keep for integrations; don't conflate with optimizer registry |

**Resolution**: Use a UNIFIED plugin discovery mechanism:
1. **Single entry point**: All plugins register via `[project.entry-points."traigent.plugins"]` in pyproject.toml
2. **PluginRegistry class**: New `traigent/plugins/registry.py` discovers and loads all plugins at import time
3. **Optimizer registry extends**: `optimizers/registry.py` remains for algorithm-specific registration, but plugins register algorithms through it
4. **Integration registry extends**: `integrations/plugin_registry.py` remains for LLM/vector store integrations
5. **Feature flags**: `PluginRegistry.has_feature()` queries loaded plugins for capabilities (parallel, multi_objective, etc.)

This provides a single source of truth (entry points) with specialized sub-registries for different plugin types.

### File Path Corrections

| Original Plan Path | Correct Path | Notes |
|-------------------|--------------|-------|
| `core/invokers.py` | `invokers/local.py` | Invokers are in `traigent/invokers/` |
| `evaluators/local.py:22` | `invokers/local.py` | LocalInvoker |
| `api/agents.py` | `api/config_mapper.py` | Agent config |
| `TimeoutStopCondition` | N/A | Timeout is a parameter to orchestrator, not a class |
| `BoolChoice` | N/A | Doesn't exist - use `Choices([True, False])` |

### Seamless Mode Extraction Issue

- Seamless provider is **always registered** in `config/providers.py:982`
- Uses `config/seamless_injection.py`, `config/ast_transformer.py`, `config/seamless_optuna_adapter.py`
- **Resolution**: Gate provider registration with try/except:
  ```python
  try:
      from traigent_seamless import SeamlessParameterProvider
      _PROVIDERS["seamless"] = SeamlessParameterProvider
  except ImportError:
      pass  # Seamless not available
  ```

### Feature Reporting Hardcoded

- `api/functions.py:736-746` hardcodes all features as `True`
- **Resolution**: Query plugin registries and feature flags instead:
  ```python
  "features": {
      "parallel_evaluation": PluginRegistry.has_feature("parallel"),
      "multi_objective": PluginRegistry.has_feature("multi_objective"),
      ...
  }
  ```

### auto_override_frameworks Behavior

- `decorators.py:94` - When `auto_override_frameworks=True` (default), tries to load integrations
- **Resolution**: If integrations plugin not installed, either:
  - Set default to `False` in base
  - Gracefully handle missing integrations with warning

---

## DECISIONS MADE

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base package features | Proposed set (random/grid/TPE, context/param/attribute injection, edge/mock, basic stops) | Good balance of value and simplicity |
| Plugin dependencies | Feature flags in plugins | Plugins detect available features and enable/disable |
| Monetization | Tiered bundles | Free base, paid bundles (ML tier, Enterprise tier) |
| Framework integrations | Separate plugin | Keep base minimal; integrations are optional |
| Repository structure | **Monorepo** | All plugins in same repo, easier atomic releases |
| Seamless injection mode | **Move to plugin** | AST manipulation is complex, make opt-in via traigent-seamless |
| Cost/budget enforcement | **Keep in base** | Basic budget_limit and cost tracking are essential guardrails |
| Eval dataset support | **Keep in base** | Dataset evaluation is core to optimization workflow |
| Constraints surface | **Keep in base** | `Condition` inherits from `BoolExpr`, so they cannot be split. All constraints stay in base. |
| Optuna algorithms | **Split** | Keep TPE/random/grid in base; move CMA-ES/NSGA-II to plugins |
| Registry architecture | **Unified discovery** | Single entry point (`traigent.plugins`) for all plugins; specialized sub-registries for algorithms and integrations |
| auto_override_frameworks | **Default to False** | Prevents errors when integrations not installed; enable only when plugin present |
| Default algorithm | **Change to "tpe"** | Current default "bayesian" requires scikit-learn; TPE is always available with Optuna |
| ObjectiveSchema | **Keep in base** | Required for single-objective optimization and stop conditions |
| MockModeOptions | **Keep in base** | Testing mode is essential; NOT an execution mode |

---

## FEATURE MATRIX

### 1. CORE OPTIMIZATION FEATURES (Base Package)

| Feature | Description | Dependencies | Files |
|---------|-------------|--------------|-------|
| `@traigent.optimize` decorator | Main API entry point | optuna | `api/decorators.py` |
| OptimizedFunction | Wrapped function with optimization | optuna | `core/optimized_function.py` |
| OptimizationOrchestrator | Core optimization loop | optuna | `core/orchestrator.py` |
| Random Search | Basic optimizer | optuna | `optimizers/random.py` |
| Grid Search | Exhaustive search | optuna | `optimizers/grid.py` |
| TPE (Tree Parzen Estimator) | Default Bayesian sampler | optuna | `optimizers/optuna_optimizer.py` |
| Configuration Space | Parameter ranges (Range, IntRange, Choices) | - | `api/parameter_ranges.py` |
| Basic Stop Conditions | max_trials, plateau (timeout is param) | - | `core/stop_conditions.py` |
| Result Selection | Best config selection | - | `core/result_selection.py` |
| Trial Lifecycle | Trial state management | - | `core/trial_lifecycle.py` |
| LocalInvoker | Basic function invocation | - | `invokers/local.py` |
| Cache Policy | allow_repeats, prefer_new dedup | - | `core/cache_policy.py` |
| Local Storage | Results persistence, locking, resume | - | `storage/` |
| CLI Core | optimize, results, info commands | click, rich | `cli/` |
| Cost Tracking | budget_limit, cost_limit enforcement | - | `core/cost_enforcement.py` |
| ObjectiveSchema | Single-objective definition (required for base) | - | `core/objectives.py` |
| Condition (minimal) | For Range builder methods (.equals, .gt, etc.) | - | `api/constraints.py` (partial) |
| MockModeOptions | Testing mode (NOT an execution mode) | - | `config/types.py` |

**Note**: Agent config (`api/config_mapper.py`) depends on cloud models and will move to `traigent-cloud` plugin.

### 2. INJECTION MODES

| Mode | Description | Parallel-Safe | Base Package |
|------|-------------|---------------|--------------|
| **CONTEXT** | Uses `contextvars` (thread-safe) | Yes | Base |
| **PARAMETER** | Explicit `config` parameter | Yes | Base |
| **ATTRIBUTE** | Function object attribute | No | Base |
| **SEAMLESS** | AST source code modification | Depends | `traigent-seamless` |

### 3. EXECUTION MODES

| Mode | Description | Cloud Required | Base Package |
|------|-------------|----------------|--------------|
| **EDGE_ANALYTICS** | Local + optional telemetry | No | Yes |
| **HYBRID** | Local + cloud | Yes | No |
| **STANDARD** | Cloud orchestration | Yes | No |
| **CLOUD** | Full SaaS | Yes | No |

**Note**: `MockModeOptions` is NOT an execution mode - it's a separate configuration bundle for testing (`api/decorators.py:148`). Mock mode can be combined with any execution mode.

---

## PROPOSED PACKAGE STRUCTURE

### `traigent` (Base Package) - ~$0/free tier

**Core Dependencies:**
- `optuna>=4.5.0` (required)
- `click>=8.0.0`, `rich>=12.0.0` (CLI)
- `jsonschema>=4.0.0` (validation)

**Included Features:**
| Category | Features |
|----------|----------|
| Optimization | Random, Grid, TPE algorithms |
| Injection | context, parameter, attribute modes |
| Execution | edge_analytics, mock modes |
| Stop Conditions | max_trials, timeout |
| Objectives | Single objective (minimize/maximize) |
| Config Space | Range, IntRange, Choices (use `Choices([True, False])` for booleans) |
| Results | Basic result selection, save/load JSON |

---

### `traigent-parallel` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- Pure asyncio (no aiohttp required - uses stdlib)

**Features:**
| Feature | Description | Source File |
|---------|-------------|-------------|
| ParallelExecutionManager | Async trial execution (pure asyncio) | `core/parallel_execution_manager.py` |
| ParallelBatchOptimizer | Concurrent trials | `optimizers/batch_optimizers.py` |
| AdaptiveBatchOptimizer | Dynamic batch sizing | `optimizers/batch_optimizers.py` |
| trial_concurrency | Multiple parallel trials | Config option |
| example_concurrency | Concurrent examples per trial | Config option |
| Cost permit system | Budget-aware parallelism | `core/parallel_execution_manager.py` |
| BatchInvoker | Batch function invocation | `invokers/batch.py` |
| StreamingInvoker | Streaming response handling | `invokers/streaming.py` |

**Invoker Migration Strategy**: To avoid breaking `from traigent.invokers import BatchInvoker`:
1. Keep stub modules in `traigent/invokers/batch.py` and `traigent/invokers/streaming.py`
2. Stubs import from plugin if available, otherwise raise `FeatureNotAvailableError`
3. Example stub pattern:
   ```python
   # traigent/invokers/batch.py (stub in base)
   try:
       from traigent_parallel.invokers import BatchInvoker
   except ImportError:
       def BatchInvoker(*args, **kwargs):
           raise FeatureNotAvailableError("Install traigent-parallel for BatchInvoker")
   ```

---

### `traigent-multi-objective` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `numpy>=1.21.0` (calculations)

**Features:**
| Feature | Description | Source File |
|---------|-------------|-------------|
| Multi-objective support | Multiple objectives in single optimization | Extends base ObjectiveSchema |
| NSGA-II algorithm | Pareto optimization | `optimizers/optuna_optimizer.py:653` |
| ParetoFrontCalculator | Pareto analysis | `utils/multi_objective.py` |
| Hypervolume calculation | Solution quality metric | `utils/multi_objective.py` |

**Ownership Clarification**:
- **Base package keeps**: `ObjectiveSchema` class, basic `AggregationMode` enum (WEIGHTED_SUM), single-objective plateau detection
- **Plugin adds**: Multiple objectives (`len(objectives) > 1`), NSGA-II algorithm, Pareto front calculation, hypervolume, advanced aggregation modes (HARMONIC, CHEBYSHEV), banded objectives

---

### `traigent-seamless` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`

**Features:**
| Feature | Description | Source File |
|---------|-------------|-------------|
| SeamlessParameterProvider | AST-based source code modification | `config/providers.py:977` → plugin |
| AST Transformer | Code transformation engine | `config/ast_transformer.py` |
| Seamless Injection | Variable assignment override | `config/seamless_injection.py` |
| Optuna Adapter | Seamless-mode Optuna integration | `config/seamless_optuna_adapter.py` |

**Extraction Note**: Must gate registration in `config/providers.py:982` - currently always registers `"seamless"` in `_PROVIDERS`.

---

### ~~`traigent-constraints`~~ (REMOVED - Stays in Base)

**Reason**: `Condition` class inherits from `BoolExpr` (`api/constraints.py:207`), making them inseparable. The entire constraints module stays in the base package:
- `BoolExpr` abstract base class
- `Condition` class (used by Range builder methods)
- `implies()`, `when().then()` fluent API
- Constraint validation and conflict detection

---

### `traigent-advanced-algorithms` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `scikit-learn>=1.0.0`, `scipy>=1.7.0` (for Bayesian GP)

**Features:**
| Feature | Description | Source File |
|---------|-------------|-------------|
| CMA-ES optimizer | Evolution strategy (Optuna) | `optimizers/optuna_optimizer.py` |
| Bayesian (GP) optimizer | Gaussian process (scikit-learn) | `optimizers/bayesian.py` |
| Custom pruners | Ceiling pruner | `optimizers/optuna_optimizer.py` |
| Adaptive sampling | Performance-based | Various |

**Extraction Note**: CMA-ES is auto-registered at `optimizers/registry.py:175`. Must gate with feature flag. Bayesian already uses try/except at `registry.py:136-142`.

---

### `traigent-tvl` (Plugin Package - TVL/CI-CD Hooks)

**Additional Dependencies:**
- Base `traigent`
- `pyyaml>=6.0` (spec parsing)

**Features:**
| Feature | Description |
|---------|-------------|
| TVL spec loader | Parse .tvl files |
| Environment overlays | Dev/staging/prod configs |
| PromotionGate | Epsilon-Pareto evaluation |
| TOST testing | Statistical significance |
| CI/CD integration | Approval workflows |

---

### `traigent-cloud` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `aiohttp>=3.8.0`
- `cryptography>=44.0.1`
- `mcp>=1.23.0`

**Features:**
| Feature | Description |
|---------|-------------|
| Cloud execution modes | hybrid, standard, cloud |
| BackendSynchronizer | State sync |
| TraigentCloudClient | API client |
| Session management | Remote sessions |
| Cost tracking | Cloud billing |

---

### `traigent-analytics` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `numpy>=1.21.0`, `pandas>=1.3.0`
- `matplotlib>=3.5.0`

**Features:**
| Feature | Description |
|---------|-------------|
| Meta-learning | Algorithm selection |
| Cost forecasting | Predictive analytics |
| Anomaly detection | Performance monitoring |
| Visualization | Progress plots |

---

### `traigent-integrations` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- Framework-specific (optional): `openai`, `anthropic`, `langchain`, etc.

**Features:**
| Feature | Description |
|---------|-------------|
| LLM plugins | OpenAI, Anthropic, etc. |
| Vector store plugins | ChromaDB, Pinecone, etc. |
| Framework override | Auto-parameter injection |
| Model discovery | Dynamic model listing |

---

### `traigent-experiment-tracking` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `mlflow>=3.8.1` OR `wandb>=0.15.0`

**Features:**
| Feature | Description |
|---------|-------------|
| MLflow integration | Experiment logging |
| W&B integration | Run tracking |
| Artifact management | Result storage |

---

### `traigent-security` (Plugin Package - Enterprise)

**Additional Dependencies:**
- Base `traigent`
- `pyjwt>=2.4.0`, `passlib>=1.7.4`
- `fastapi>=0.95.0`, `redis>=4.0.0`

**Features:**
| Feature | Description |
|---------|-------------|
| MFA support | TOTP, SMS |
| SAML/OIDC | SSO integration |
| Encryption | Field-level encryption |
| Audit logging | Compliance reporting |
| Multi-tenancy | Tenant isolation |

---

### `traigent-tracing` (Plugin Package)

**Additional Dependencies:**
- Base `traigent`
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-exporter-otlp>=1.20.0`

**Features:**
| Feature | Description |
|---------|-------------|
| OpenTelemetry spans | Distributed tracing |
| Jaeger export | Trace visualization |
| Performance profiling | Latency analysis |

---

### `traigent-hooks` (Plugin Package - CI/CD Policy)

**Additional Dependencies:**
- Base `traigent`

**Features:**
| Feature | Description |
|---------|-------------|
| Hook installer | Install/validate hooks |
| Cost constraints | Block runs exceeding cost |
| Performance constraints | Enforce quality gates |
| Model allow/block lists | Restrict model usage |
| traigent.yml config | Declarative policy config |

---

### `traigent-evaluation` (Plugin Package - Advanced Eval)

**Additional Dependencies:**
- Base `traigent`
- `ragas>=0.3.6` (RAG evaluation)
- `pandas>=1.3.0`

**Features:**
| Feature | Description |
|---------|-------------|
| RAGAS metrics | RAG-specific evaluation |
| Parallel evaluation | Concurrent example eval |
| reps_per_trial | Statistical repetitions |
| reps_aggregation | mean, median, min, max |
| Custom metrics | Advanced metric functions |

---

### `traigent-ui` (Plugin Package - Playground/Control Center)

**Additional Dependencies:**
- Base `traigent`
- `streamlit>=1.0.0`

**Features:**
| Feature | Description |
|---------|-------------|
| Control center | Interactive dashboard |
| Optimization demos | Example workflows |
| Real-time monitoring | Live optimization view |
| Configuration editor | Visual config builder |

---

## FEATURE DEPENDENCY GRAPH

```
traigent (base)
├── traigent-parallel
│   └── (requires base)
├── traigent-multi-objective
│   └── (requires base)
├── traigent-seamless
│   └── (requires base)
├── traigent-constraints
│   └── (requires base)
├── traigent-advanced-algorithms
│   └── (requires base)
├── traigent-tvl
│   ├── (requires base)
│   └── (optional: traigent-multi-objective for Pareto gates)
├── traigent-cloud
│   ├── (requires base)
│   └── (optional: traigent-parallel for cloud parallelism)
├── traigent-analytics
│   └── (requires base)
├── traigent-integrations
│   └── (requires base)
├── traigent-experiment-tracking
│   └── (requires base)
├── traigent-security
│   ├── (requires base)
│   └── (requires traigent-cloud)
├── traigent-tracing
│   └── (requires base)
├── traigent-hooks
│   └── (requires base)
├── traigent-evaluation
│   ├── (requires base)
│   └── (optional: traigent-parallel for parallel eval)
└── traigent-ui
    └── (requires base)
```

---

## CONFIGURATION OPTIONS MATRIX

### Parameters by Package

| Parameter | Base | Parallel | Multi-Obj | Constraints | TVL | Cloud |
|-----------|------|----------|-----------|-------------|-----|-------|
| `objectives` (single) | Yes | Yes | Yes | Yes | Yes | Yes |
| `objectives` (multi) | No | No | Yes | No | Yes | Yes |
| `configuration_space` | Yes | Yes | Yes | Yes | Yes | Yes |
| `constraints` | No | No | No | Yes | Yes | Yes |
| `injection_mode` | Yes | Yes | Yes | Yes | Yes | Yes |
| `execution_mode` (local) | Yes | Yes | Yes | Yes | Yes | Yes |
| `execution_mode` (cloud) | No | No | No | No | No | Yes |
| `parallel_config` | No | Yes | No | No | No | Yes |
| `algorithm` (basic) | Yes | Yes | Yes | Yes | Yes | Yes |
| `algorithm` (advanced) | No | No | Yes | No | No | Yes |
| `max_trials` | Yes | Yes | Yes | Yes | Yes | Yes |
| `timeout` | Yes | Yes | Yes | Yes | Yes | Yes |
| `plateau_window` | Yes | Yes | Yes | Yes | Yes | Yes |
| `budget_limit` | Yes | Yes | Yes | Yes | Yes | Yes |
| `tvl_spec` | No | No | No | No | Yes | Yes |
| `privacy_enabled` | No | No | No | No | No | Yes |

---

## STOP CONDITIONS BY PACKAGE

| Stop Condition | Base | Parallel | Multi-Obj | Cloud |
|----------------|------|----------|-----------|-------|
| MaxTrialsStopCondition | Yes | Yes | Yes | Yes |
| PlateauStopCondition | Yes | Yes | Yes | Yes |
| BudgetStopCondition | Yes | Yes | Yes | Yes |
| MaxSamplesStopCondition | Yes | Yes | Yes | Yes |
| CostLimitStopCondition | Yes | Yes | Yes | Yes |

**Note**: Timeout is a parameter to the orchestrator (`timeout` kwarg), not a separate stop condition class.

---

## ALGORITHMS BY PACKAGE

| Algorithm | Base | Advanced | Multi-Obj |
|-----------|------|----------|-----------|
| random | Yes | Yes | Yes |
| grid | Yes | Yes | Yes |
| tpe | Yes | Yes | Yes |
| cmaes | No | Yes | No |
| bayesian (GP) | No | Yes | No |
| nsgaii | No | No | Yes |

---

## BUNDLE PACKAGES (Convenience)

| Bundle | Includes | Target User |
|--------|----------|-------------|
| `traigent[full]` | All plugins | Enterprise |
| `traigent[ml]` | base + parallel + multi-obj + analytics | ML Engineers |
| `traigent[llm]` | base + integrations + experiment-tracking | LLM Developers |
| `traigent[enterprise]` | base + cloud + security + tracing | Enterprise |

---

## TIERED BUNDLE STRATEGY

### Free Tier: `traigent` (Base)
- Random, Grid, TPE algorithms
- Injection modes: context, parameter, attribute (seamless is plugin)
- edge_analytics, mock execution modes
- Basic stop conditions (max_trials, timeout, plateau, budget)
- Single-objective optimization
- Basic result selection and JSON save/load
- LocalInvoker, cache policy, local storage
- CLI core commands (optimize, results, info)

### ML Bundle: `traigent[ml]` - Paid
- Base package
- `traigent-parallel` (concurrent trials)
- `traigent-multi-objective` (Pareto optimization)
- `traigent-advanced-algorithms` (CMA-ES, Bayesian GP)
- `traigent-analytics` (meta-learning, forecasting)
- `traigent-evaluation` (RAGAS, parallel eval, reps)

### LLM Bundle: `traigent[llm]` - Paid
- Base package
- `traigent-integrations` (OpenAI, Anthropic, LangChain, etc.)
- `traigent-experiment-tracking` (MLflow, W&B)
- `traigent-parallel` (parallel trials)
- `traigent-seamless` (AST-based injection)

### Enterprise Bundle: `traigent[enterprise]` - Paid
- All plugins included
- `traigent-cloud` (cloud execution modes)
- `traigent-security` (MFA, SAML, audit)
- `traigent-tvl` (CI/CD promotion gates)
- `traigent-hooks` (policy enforcement)
- `traigent-tracing` (OpenTelemetry)
- `traigent-ui` (control center, dashboards)
- Priority support

---

## PLUGIN REGISTRATION & FEATURE FLAGS

### Plugin Discovery Pattern
```python
# In base package: traigent/plugins/registry.py
class PluginRegistry:
    _plugins: dict[str, "TraigentPlugin"] = {}

    @classmethod
    def register(cls, name: str, plugin: "TraigentPlugin"):
        cls._plugins[name] = plugin

    @classmethod
    def has_feature(cls, feature: str) -> bool:
        """Check if a feature is available via any installed plugin."""
        return any(p.provides_feature(feature) for p in cls._plugins.values())

    @classmethod
    def get_feature(cls, feature: str) -> Optional[Any]:
        """Get the implementation of a feature if available."""
        for plugin in cls._plugins.values():
            if impl := plugin.get_feature(feature):
                return impl
        return None
```

### Plugin Auto-Registration (entry points)
```toml
# In traigent-parallel/pyproject.toml
[project.entry-points."traigent.plugins"]
parallel = "traigent_parallel:ParallelPlugin"
```

### Feature Flag Usage in Base
```python
# In base package: traigent/core/orchestrator.py
from traigent.plugins import PluginRegistry

class OptimizationOrchestrator:
    def __init__(self, config):
        # Check for parallel feature
        if config.parallel_config and not PluginRegistry.has_feature("parallel"):
            raise FeatureNotAvailableError(
                "Parallel execution requires 'traigent-parallel' plugin. "
                "Install with: pip install traigent[ml] or pip install traigent-parallel"
            )

        # Check for multi-objective
        if len(config.objectives) > 1 and not PluginRegistry.has_feature("multi_objective"):
            raise FeatureNotAvailableError(
                "Multi-objective optimization requires 'traigent-multi-objective'. "
                "Install with: pip install traigent[ml]"
            )
```

### Cross-Plugin Feature Detection
```python
# In traigent-tvl plugin
from traigent.plugins import PluginRegistry

class TVLPlugin:
    def provides_feature(self, feature: str) -> bool:
        return feature in ["tvl_spec", "promotion_gate", "ci_cd_hooks"]

    def get_feature(self, feature: str):
        if feature == "promotion_gate":
            # Check if multi-objective is available for Pareto gates
            if PluginRegistry.has_feature("pareto_front"):
                from traigent_tvl.pareto_promotion import ParetoPromotionGate
                return ParetoPromotionGate
            else:
                # Fallback to single-objective promotion
                from traigent_tvl.simple_promotion import SimplePromotionGate
                return SimplePromotionGate
```

---

## PROPOSED DIRECTORY STRUCTURE

```
traigent/                          # Base package (monorepo root)
├── traigent/                      # Core source
│   ├── __init__.py
│   ├── api/
│   │   ├── decorators.py
│   │   ├── parameter_ranges.py
│   │   └── functions.py
│   ├── core/
│   │   ├── optimized_function.py
│   │   ├── orchestrator.py
│   │   ├── stop_conditions.py
│   │   └── result_selection.py
│   ├── optimizers/
│   │   ├── base.py
│   │   └── optuna_optimizer.py   # Random, Grid, TPE only
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py               # TraigentPlugin base class
│   │   └── errors.py             # FeatureNotAvailableError
│   └── utils/
├── plugins/                       # Plugin packages (separate repos or monorepo)
│   ├── traigent-parallel/
│   │   ├── pyproject.toml
│   │   └── traigent_parallel/
│   │       ├── __init__.py
│   │       ├── plugin.py
│   │       ├── execution_manager.py
│   │       └── batch_optimizers.py
│   ├── traigent-multi-objective/
│   │   ├── pyproject.toml
│   │   └── traigent_multi_objective/
│   │       ├── __init__.py
│   │       ├── plugin.py
│   │       ├── objectives.py
│   │       ├── pareto.py
│   │       └── nsga2.py
│   ├── traigent-seamless/
│   ├── traigent-constraints/
│   ├── traigent-advanced-algorithms/
│   ├── traigent-tvl/
│   ├── traigent-cloud/
│   ├── traigent-integrations/
│   ├── traigent-analytics/
│   ├── traigent-security/
│   ├── traigent-tracing/
│   ├── traigent-experiment-tracking/
│   ├── traigent-hooks/
│   ├── traigent-evaluation/
│   └── traigent-ui/
└── pyproject.toml                 # Base package config
```

---

## IMPLEMENTATION PHASES (Minimal-Risk Extraction Order)

The extraction order is designed to minimize breakage by:
1. Starting with leaf modules (no dependents)
2. Addressing import dependencies BEFORE extraction
3. Using lazy imports for conditional features
4. Testing each extraction in isolation

---

### Phase 0: Pre-Extraction Preparation (CRITICAL)

**Fix import dependencies before any extraction:**

1. **Orchestrator lazy imports** (`orchestrator.py:22,48`)
   ```python
   # Before: from traigent.cloud.client import TraigentCloudClient
   # After: Lazy import inside method that uses it
   def _get_cloud_client(self):
       from traigent.cloud.client import TraigentCloudClient
       return TraigentCloudClient(...)
   ```

2. **Change default algorithm** (`functions.py:560`)
   ```python
   # Before: algorithm: str = "bayesian"
   # After: algorithm: str = "tpe"
   ```

3. **Create base plugin infrastructure**
   - Create new `traigent/plugins/registry.py` for unified plugin discovery
   - Extend existing `optimizers/registry.py` with plugin hooks
   - Extend existing `integrations/plugin_registry.py` for LLM/vector store plugins
   - Add `FeatureNotAvailableError` to existing `traigent/utils/exceptions.py`

4. **Shim LocalAnalytics** (`local_analytics.py:19`)
   - Keep minimal auth interface in base
   - Make cloud-specific analytics optional

---

### Phase 1: Extract Integrations Plugin (LOWEST RISK)

**Why first**: No core code depends on integrations; they're already isolated.

**Files to move:**
- `integrations/` → `traigent-integrations/`
- Keep `integrations/plugin_registry.py` in base (extended for all plugins)

**Changes:**
1. Update `pyproject.toml` optional deps
2. Add feature flag for `auto_override_frameworks`
3. Test: `pip install traigent` works without integrations

---

### Phase 2: Extract Tracing Plugin (LOW RISK)

**Why second**: Tracing is observability-only, no core logic depends on it.

**Files to move:**
- `tracing/` → `traigent-tracing/`
- OpenTelemetry setup code

**Changes:**
1. Guard tracing initialization with feature flag
2. Test: Optimization works without tracing installed

---

### Phase 3: Extract Analytics Plugin (LOW RISK)

**Why third**: Analytics is read-only analysis, doesn't affect optimization.

**Files to move:**
- `analytics/` → `traigent-analytics/`
- Visualization code

**Changes:**
1. Guard analytics CLI commands
2. Test: `traigent optimize` works without analytics

---

### Phase 4: Extract UI Plugin (LOW RISK)

**Why fourth**: Playground is completely separate.

**Files to move:**
- `playground/` → `traigent-ui/`

**Changes:**
1. Move streamlit deps to plugin
2. Test: CLI works without UI installed

---

### Phase 5: Extract Security Plugin (MEDIUM RISK)

**Why fifth**: Security is enterprise-only, core doesn't depend on it.

**Files to move:**
- `security/` → `traigent-security/`

**Changes:**
1. Keep basic JWT validation in base (cloud needs it)
2. Move MFA, SAML, audit to plugin
3. Test: Edge analytics works without security plugin

---

### Phase 6: Extract Hooks Plugin (MEDIUM RISK)

**Files to move:**
- `hooks/` → `traigent-hooks/`
- `traigent.yml` schema

**Changes:**
1. Guard hook installation CLI
2. Test: Optimization works without hooks

---

### Phase 7: Extract TVL Plugin (MEDIUM RISK)

**Files to move:**
- `tvl/` → `traigent-tvl/`

**Changes:**
1. Guard `--tvl-spec` CLI arg
2. Implement fallback for Pareto gates (use single-objective if multi-obj plugin missing)
3. Test: Optimization works without TVL

---

### Phase 8: Extract Evaluation Plugin (MEDIUM RISK)

**Files to move:**
- RAGAS metrics from `evaluators/`
- `reps_per_trial` advanced logic

**Keep in base:**
- Basic `LocalEvaluator` with simple scoring
- Dataset evaluation support

**Changes:**
1. Guard RAGAS import
2. Test: Basic evaluation works without RAGAS

---

### Phase 9: Extract Parallel Plugin (HIGHER RISK)

**Why later**: Orchestrator imports ParallelExecutionManager.

**Prerequisites:**
- Lazy import in orchestrator completed (Phase 0)

**Files to move:**
- `core/parallel_execution_manager.py` → `traigent-parallel/`
- `optimizers/batch_optimizers.py` → `traigent-parallel/`

**Changes:**
1. Feature flag for `parallel_config`
2. `FeatureNotAvailableError` with install hint
3. Test: Sequential optimization works without parallel plugin

---

### Phase 10: Extract Multi-Objective Plugin (HIGHER RISK)

**Why later**: ObjectiveSchema used in decorators and stop conditions.

**Key insight**: Keep ObjectiveSchema in base, only move multi-obj EXTENSIONS.

**Files to move:**
- `utils/multi_objective.py` → `traigent-multi-objective/`
- NSGA-II sampler from `optimizers/optuna_optimizer.py`

**Keep in base:**
- `ObjectiveSchema` class (single-obj mode)
- Basic objective handling in decorators

**Changes:**
1. Feature flag for `len(objectives) > 1`
2. `FeatureNotAvailableError` with install hint
3. Test: Single-objective optimization works without multi-obj plugin

---

### Phase 11: Extract Advanced Algorithms Plugin (HIGHER RISK)

**Files to move:**
- CMA-ES sampler from `optimizers/optuna_optimizer.py`
- Bayesian GP optimizer

**Keep in base:**
- Random, Grid, TPE (Optuna defaults)

**Changes:**
1. Extend optimizer registry for plugin algorithms
2. `FeatureNotAvailableError` if algorithm not found
3. Test: TPE optimization works without advanced algorithms

---

### Phase 12: Extract Constraints Plugin (HIGHER RISK)

**Key insight**: `parameter_ranges.py:157` imports constraints. Keep BASIC constraints in base.

**Files to move:**
- `api/constraints.py` BoolExpr system → `traigent-constraints/`
- Advanced constraint validation

**Keep in base:**
- Basic Range/IntRange/Choices constraints (no BoolExpr)

**Changes:**
1. Feature flag for `implies()`, `when().then()`
2. Test: Basic config space works without constraint plugin

---

### Phase 13: Extract Cloud Plugin (HIGHEST RISK)

**Why last**: Most import dependencies, affects multiple modules.

**Prerequisites:**
- Orchestrator lazy imports (Phase 0)
- LocalAnalytics shimmed (Phase 0)
- Agent config moved (this phase)

**Files to move:**
- `cloud/` → `traigent-cloud/`
- `api/config_mapper.py` (agent config)

**Keep in base:**
- Minimal auth interface for edge analytics shim
- Local storage (no cloud sync)

**Changes:**
1. Feature flag for `execution_mode` in [HYBRID, STANDARD, CLOUD]
2. `FeatureNotAvailableError` with install hint
3. Test: `edge_analytics` mode works without cloud plugin

---

### Phase 14: Extract Seamless Plugin (LOWEST PRIORITY)

**Files to move:**
- AST injection code → `traigent-seamless/`

**Changes:**
1. Feature flag for `injection_mode=SEAMLESS`
2. Test: Other injection modes work

---

### Phase 15: Documentation & Testing

1. **Migration guide**: Document breaking changes, install hints
2. **Bundle tests**: Test all `traigent[ml]`, `traigent[llm]`, `traigent[enterprise]` combinations
3. **Base-only tests**: Verify `pip install traigent` works with graceful degradation
4. **Update examples**: Show plugin installation patterns

---

## REQUIRED TESTS

### Base-Only Install Tests

- `pip install traigent` (base only) should work without plugin deps
- Attempting `parallel_config` should raise `FeatureNotAvailableError("Install traigent-parallel")`
- Attempting `len(objectives) > 1` should raise `FeatureNotAvailableError("Install traigent-multi-objective")`
- Attempting `tvl_spec` should raise `FeatureNotAvailableError("Install traigent-tvl")`
- Attempting cloud execution modes should raise `FeatureNotAvailableError("Install traigent-cloud")`
- Attempting `injection_mode="seamless"` should raise `FeatureNotAvailableError("Install traigent-seamless")`
- Attempting `auto_override_frameworks=True` without integrations should log warning and continue

### Algorithm Registry Tests

- `algorithm="tpe"` works in base (not "bayesian" which requires scikit-learn)
- `algorithm="cmaes"` raises `FeatureNotAvailableError("Install traigent-advanced-algorithms")`
- `algorithm="nsga2"` raises `FeatureNotAvailableError("Install traigent-multi-objective")`
- Default algorithm is "tpe" after refactor

### Constraint Builder Tests

- `Range(0, 1).equals(0.5)` works in base (returns `Condition`)
- `implies(cond1, cond2)` works in base (constraints stay in base package)
- `when(cond).then(effect)` works in base (fluent API stays in base)

### Feature Reporting Tests

- `get_version_info()["features"]` reflects actual plugin availability
- CLI `traigent info` shows correct feature availability

### Plugin Discovery Tests

- Plugins auto-register via entry points on install
- Optimizer plugins extend `optimizers/registry.py`
- Integration plugins extend `integrations/plugin_registry.py`
- Feature plugins set flags in new feature flag system

### Graceful Degradation Tests

- Edge analytics works without cloud plugin (shimmed auth)
- Single-objective optimization works without multi-objective plugin
- Constraints (`implies()`, `when().then()`) work in base (no plugin needed)
- Local invoker works without batch/streaming invoker plugins

---

## BREAKING CHANGES & MIGRATION PLAN

This refactor introduces breaking changes. Plan for a **major version bump** (e.g., v2.0.0).

### Default Algorithm Change

**Before**: `algorithm="bayesian"` (requires scikit-learn)
**After**: `algorithm="tpe"` (Optuna built-in, always available)

**Migration**:
- Users explicitly using `algorithm="bayesian"` must install `traigent-advanced-algorithms`
- Add deprecation warning in v1.x: "Default algorithm will change to 'tpe' in v2.0"

### auto_override_frameworks Default Change

**Before**: `auto_override_frameworks=True` (tries to load integrations)
**After**: `auto_override_frameworks=False` (explicit opt-in)

**Migration**:
- Users relying on auto-override must add `auto_override_frameworks=True` explicitly
- Or install `traigent-integrations` which sets the default back to `True`
- Add deprecation warning in v1.x

### Import Path Changes

Some imports will require plugin installation:

| Old Import | New Requirement |
|------------|-----------------|
| `from traigent.invokers import BatchInvoker` | Install `traigent-parallel` |
| `from traigent.invokers import StreamingInvoker` | Install `traigent-parallel` |
| `from traigent.cloud import TraigentCloudClient` | Install `traigent-cloud` |
| `from traigent.integrations.llms import OpenAIPlugin` | Install `traigent-integrations` |

**Migration**: Stub modules in base provide helpful error messages pointing to the correct plugin.

### Compatibility Shims

For gradual migration, v2.0 will include optional compatibility shims:

```python
# Users can install traigent[compat] for backwards compatibility
# This meta-package includes all plugins that were previously in base
pip install traigent[compat]
```

### Version Timeline

1. **v1.x (current)**: Add deprecation warnings for upcoming changes
2. **v2.0**: Plugin architecture with breaking changes
3. **v2.x**: Remove compatibility shims after migration period
