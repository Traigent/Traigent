# Native JS Parity Matrix

This matrix tracks the current native JS optimization surface against the Python SDK reference.

| Area | Python SDK | Native JS | Status | Notes |
| --- | --- | --- | --- | --- |
| Grid search | Yes | Yes | `matched` | JS uses deterministic ordered enumeration. |
| Random search | Yes | Yes | `matched` | JS now uses Python-style seeded sampling for parity fixtures. |
| Bayesian optimizer | Yes | Yes | `partial` | JS supports sequential in-process Bayesian search; Optuna/TPE parity is deferred. |
| Optuna family | Yes | No | `deferred` | Includes TPE, CMA-ES, NSGA-II, Optuna grid/random. |
| Log-scale params | Yes | Yes | `matched` | JS supports log-scale sampling and multiplicative grid steps. |
| Budget stop | Yes | Yes | `matched` | JS still relies only on numeric `metrics.cost`. |
| Timeout stop | Yes | Yes | `matched` | JS exposes `timeoutMs` and returns `stopReason: 'timeout'`. |
| Error stop | Yes | Yes | `matched` | Runtime trial errors return `stopReason: 'error'`. |
| Plateau stop | Yes | Yes | `matched` | JS exposes `plateau: { window, minImprovement }`. |
| Cancellation | Yes | Yes | `matched` | JS uses `AbortSignal` and `stopReason: 'cancelled'`. |
| Trial concurrency | Yes | Yes | `partial` | JS supports `trialConcurrency` for `grid` and `random` only. |
| Example concurrency | Yes | No | `deferred` | Current JS trial contract does not let the SDK own per-example execution. |
| Checkpoint/resume | Yes | Yes | `partial` | Trial-boundary checkpoints for native JS; no Optuna/stateful remote parity. |
| `apply_best_config` ergonomics | Yes | Yes | `partial` | JS uses wrapper-local `applyBestConfig()` and `currentConfig()` only. |
| Global `get_config()` | Yes | No | `deferred` | Intentionally not added to preserve JS runtime semantics. |
| Hybrid authoring | Yes | Yes | `matched` | `toHybridConfigSpace()` keeps the existing hybrid wire format. |
| Cloud/hybrid orchestration | Yes | No | `deferred` | JS native remains local Node execution only. |

## Normalized Stop Reason Mapping

These mappings are used by parity tests and release reports:

| Python | Native JS |
| --- | --- |
| `max_trials_reached` | `maxTrials` |
| `cost_limit` | `budget` |
| `timeout` | `timeout` |
| `error` | `error` |
| `user_cancelled` | `cancelled` |
| `plateau` | `plateau` |
| `optimizer` | `completed` when the search space is exhausted |

## Explicit Deferrals

- Optuna-family algorithms
- Pareto multi-objective search
- Conditionals and constraints
- Example-level concurrency
- Pruners and early-stop policies beyond plateau
- Cloud and hybrid orchestration
- Provider-side cost estimation and budget reservation
