# Native JS Parity Matrix

This matrix tracks the phase-1 native JS optimization surface against the Python SDK reference.

| Area | Python SDK | Native JS | Status | Notes |
| --- | --- | --- | --- | --- |
| Grid search | Yes | Yes | `matched` | JS uses deterministic ordered enumeration. |
| Random search | Yes | Yes | `matched` | JS uses Python-compatible seeded sampling for oracle fixtures. |
| Bayesian optimizer | Yes | Yes | `partial` | JS supports sequential in-process Bayesian search only. |
| Log-scale params | Yes | Yes | `matched` | JS supports log-scale sampling and multiplicative grid steps. |
| Budget stop | Yes | Yes | `matched` | JS relies only on numeric `metrics.cost`. |
| Timeout stop | Yes | Yes | `matched` | JS exposes `timeoutMs` and returns `stopReason: 'timeout'`. |
| Error stop | Yes | Yes | `matched` | Runtime trial errors return `stopReason: 'error'`. |
| Trial concurrency | Yes | No | `deferred` | Phase 1 does not expose public trial concurrency. |
| Example concurrency | Yes | No | `deferred` | Current JS trial contract does not let the SDK own per-example execution. |
| Checkpoint/resume | Yes | No | `deferred` | Explicitly out of scope for the phase-1 branch. |
| `apply_best_config` ergonomics | Yes | No | `deferred` | Wrapper-local config application is deferred. |
| Global `get_config()` | Yes | No | `deferred` | Intentionally not added to preserve JS runtime semantics. |
| Hybrid authoring | Yes | Yes | `matched` | `toHybridConfigSpace()` preserves the hybrid wire format. |
| Cloud/hybrid orchestration | Yes | No | `deferred` | JS native remains local Node execution only. |

## Normalized Stop Reason Mapping

These mappings are used by parity tests and release reports:

| Python | Native JS |
| --- | --- |
| `max_trials_reached` | `maxTrials` |
| `cost_limit` | `budget` |
| `timeout` | `timeout` |
| `error` | `error` |
| `optimizer` | `completed` when the search space is exhausted |

## Explicit Deferrals

- Optuna-family algorithms
- Pareto multi-objective search
- Conditionals and constraints
- Trial concurrency in the public SDK API
- Example-level concurrency
- Checkpoint and resume
- Wrapper-local `applyBestConfig()` / `currentConfig()`
- Cloud and hybrid orchestration
- Provider-side cost estimation and budget reservation
