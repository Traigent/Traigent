# Traigent SDK Architecture

This page summarizes how the open-source Traigent SDK is structured and how the local (`edge_analytics`) execution flow works. Cloud/hybrid orchestration remains roadmap-only in the OSS build; run locally unless you have a managed backend.

## High-Level Architecture (OSS)

```
┌──────────────────────────────────────────┐
│ Public API & CLI                         │
│  @optimize decorator · configure · CLI   │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│ Core Orchestration                       │
│  orchestrator.py · optimized_function.py │
│  parallel_execution_manager.py           │
│  stop_condition_manager.py · trial_*     │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│ Component Layer                          │
│  Optimizers: grid, random, optuna adapter│
│  Invokers: local, batch, streaming       │
│  Evaluators: local + metrics helpers     │
│  Samplers/pruners/cache policies         │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│ Integration Layer                        │
│  Framework overrides (OpenAI/Anthropic/  │
│   LangChain) · vector stores · observability│
│  CLI commands: optimize, validate, results│
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│ Infrastructure                           │
│  Storage (JSON/local) · logging · utils  │
└──────────────────────────────────────────┘
```

## Module Structure (short map)

```
traigent/
├── api/          # Public API surface (decorators, helpers, types)
├── core/         # Orchestrator, optimized_function, lifecycle, stop conditions
├── optimizers/   # grid, random, optuna adapter/coordinator, interactive, registry
├── evaluators/   # base + local evaluator and metrics helpers
├── invokers/     # local, batch, streaming invokers
├── config/       # context-based configuration and type definitions
├── storage/      # JSON/local storage helpers
├── analytics/    # experimental analytics helpers (anomaly, cost_optimization, etc.)
├── integrations/ # framework overrides and provider plugins (LLMs, observability, vector stores)
├── security/     # auth/encryption/rate limiting utilities
├── plugins/      # plugin interfaces and registry
├── cloud/        # experimental/back-compat cloud client models
├── agents/       # agent execution utilities
├── cli/          # `traigent` CLI entrypoint and subcommands
└── utils/, visualization/, tvl/ # shared utilities, plotting, TVL helpers
```

## Execution Flow (edge_analytics)

1. Decorate a function with `@traigent.optimize(...)` to create an `OptimizedFunction` wrapper.
2. Call `.optimize()` (async) to start the run. The orchestrator wires the optimizer, invoker, evaluator, stop conditions, and progress tracking.
3. The optimizer proposes configs; the invoker executes the function; the evaluator scores and aggregates metrics; stop conditions decide early exits.
4. Results are stored locally and surfaced via `results.best_config`, `current_config`, and the CLI (`traigent results`, `traigent plot`). Use `traigent validate` / `validate-config` for datasets/configs up front.

## Module Overview & Sequence Flow

### Main Modules
- **Core API Layer**: `traigent/__init__.py` (public exports), `api/decorators.py` (`@optimize`), `api/functions.py` (helpers like `get_config` / `get_trial_config` (deprecated)), `api/types.py` (public types).
- **Core Orchestration**: `core/optimized_function.py` (lifecycle + state), `core/orchestrator.py` (optimization loop), `core/objectives.py`, `core/evaluator_wrapper.py`.
- **Optimization Algorithms**: `optimizers/` (grid.py, random.py, optuna_adapter.py/coordinator.py, interactive_optimizer.py; base.py + registry.py).
- **Evaluation**: `evaluators/` (base.py, local.py, metrics.py) plus stop conditions under `core/stop_condition_manager.py` and `core/stop_conditions.py`.
- **Configuration**: `config/` (contextvars-based injection, providers, types + types_ext).
- **Integrations**: `integrations/framework_override.py` and provider plugins under `integrations/llms`, `integrations/observability`, `integrations/vector_stores`.

### Sequence Flow (local)
1. **Decoration**: `@optimize` wraps the function into `OptimizedFunction`, validates configuration space/objectives/dataset, and sets state to `UNOPTIMIZED`.
2. **Optimize Call (async)**: state → `OPTIMIZING`; orchestrator selects optimizer (grid/random/optuna adapter), evaluator (local), invoker (local/batch), and stop conditions.
3. **Trials**: optimizer suggests configs → invoker runs function with injected config (context/parameter/attribute/seamless) → evaluator scores → progress/stop conditions checked.
4. **Results**: best configs recorded; state → `OPTIMIZED`; access via `results.best_config`, `func.current_config`, or `traigent.get_config()` inside trials/functions (avoid `get_trial_config()`, which is deprecated).
5. **Later Calls**: calling the function uses the applied best config automatically; framework overrides can apply to LangChain/OpenAI/Anthropic if enabled.

> Execution modes: `edge_analytics` is the supported mode in OSS. Cloud/hybrid remain roadmap-only.

## Notes on Roadmap Features

- Cloud/hybrid orchestration and managed smart-trial services are not available in the OSS build. Keep `execution_mode="edge_analytics"` unless you have a provisioned backend.
- Advanced analytics flags (`enable_meta_learning`, `enable_cost_optimization`, etc.) are not part of the current API surface. The `analytics/` modules are experimental helpers only.
