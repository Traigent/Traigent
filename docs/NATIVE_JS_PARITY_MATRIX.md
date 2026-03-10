# Python SDK vs JS SDK Feature Matrix

This matrix compares the current Python SDK surface with this specific JS checkout.

Important: this repo is the native-first safety branch. It supports high-level
agent optimization and hybrid spec authoring, but it does **not** implement the
backend-guided hybrid execution work from the separate hybrid branch/worktree.

## Optimize / Decorator Surface

| Capability | Python SDK | JS SDK | Status | Notes |
| --- | --- | --- | --- | --- |
| Optimize plain agent functions | Yes | Yes | `matched` | JS now defaults to a high-level agent contract with SDK-owned evaluation. |
| SDK-owned evaluation loop | Yes | Yes | `matched` | JS iterates `evaluation.data` / `evaluation.loadData` and aggregates metrics. |
| `scoring_function` / `scoringFunction` | Yes | Yes | `matched` | JS uses `evaluation.scoringFunction`. |
| `metric_functions` / `metricFunctions` | Yes | Yes | `matched` | JS supports additive metric functions. |
| `custom_evaluator` / `customEvaluator` | Yes | Yes | `matched` | JS supports both the high-level context callback and the Python-style `(agentFn, config, row)` callback shape. |
| Advanced manual trial contract | No public primary path | Yes | `js-only` | JS keeps `execution.contract = 'trial'` as a deprecated compatibility path. |
| `default_config` baseline | Yes | Yes | `matched` | JS uses `defaultConfig` as the baseline injected config before optimization and when merging trial configs. |
| Constraints | Yes | Yes | `partial` | JS supports callback-based pre-trial config constraints and post-trial metric constraints, but not the Python TVL/builder DSL. |
| Safety constraints | Yes | Yes | `partial` | JS supports callback-based post-trial safety constraints, but not Python's preset/statistical safety layer. |

## Injection and Runtime Access

| Capability | Python SDK | JS SDK | Status | Notes |
| --- | --- | --- | --- | --- |
| Context injection | Yes | Yes | `partial` | JS uses `getTrialParam()` / `getTrialConfig()` instead of Python `get_config()`. |
| Parameter injection | Yes | Yes | `partial` | Python supports named `config_param`; JS uses `agentFn(input, config?)` second-argument injection. |
| Seamless AST/source transformation | Yes | Yes | `partial` | JS supports codemod/build-time rewriting and a narrow runtime fallback, not full Python-style runtime AST parity. |
| Framework interception / override | Yes | Yes | `partial` | JS supports explicit wrappers for OpenAI, LangChain, and Vercel AI. |
| `auto_override_frameworks` | Yes | Yes | `partial` | JS seamless defaults to active wrapped targets and now exposes `autoWrapFrameworkTarget(s)` helpers, but it still does not implicitly discover and wrap framework clients on its own. |
| `framework_targets` selection | Yes | Yes | `partial` | JS validates `frameworkTargets`, defaults seamless auto-override to all active wrapped targets, but still requires users to wrap framework clients/models explicitly. |
| Global config access after optimization | Yes | No | `deferred` | JS intentionally avoids a global `get_config()` API. |
| `apply_best_config` ergonomics | Yes | Yes | `partial` | JS uses wrapper-local `applyBestConfig()` / `currentConfig()`. |
## Native Optimization Runtime

| Capability | Python SDK | JS SDK | Status | Notes |
| --- | --- | --- | --- | --- |
| Grid search | Yes | Yes | `matched` | JS uses deterministic ordered enumeration. |
| Random search | Yes | Yes | `matched` | JS uses Python-style seeded sampling for parity fixtures. |
| Bayesian optimizer | Yes | Yes | `partial` | JS supports sequential in-process Bayesian search only. |
| Optuna-family algorithms | Yes | No | `deferred` | Includes TPE, CMA-ES, NSGA-II, Optuna grid/random. |
| Log-scale params | Yes | Yes | `matched` | JS supports log-scale sampling and multiplicative grid steps. |
| Budget stop | Yes | Yes | `matched` | JS relies on numeric `metrics.total_cost` or `metrics.cost`. |
| Timeout stop | Yes | Yes | `matched` | JS exposes `timeoutMs`. |
| Error stop | Yes | Yes | `matched` | Runtime errors produce `stopReason: 'error'`. |
| Plateau stop | Yes | Yes | `matched` | JS exposes `plateau`. |
| Cancellation | Yes | Yes | `matched` | JS uses `AbortSignal`. |
| Max total examples / sample budget | Yes | Yes | `matched` | JS exposes `execution.maxTotalExamples` and stops with `maxExamples`. |
| Repetition stability controls | Yes | Yes | `matched` | JS exposes `execution.repsPerTrial` and `execution.repsAggregation`. |
| Trial concurrency | Yes | Yes | `partial` | JS supports `trialConcurrency` for `grid` and `random` only. |
| Example concurrency | Yes | No | `deferred` | JS does not parallelize per-example evaluation in this checkout. |
| Checkpoint/resume | Yes | Yes | `partial` | Trial-boundary checkpoints only. |
| Runtime cost accounting detail | Yes | Yes | `partial` | JS records `input_cost`, `output_cost`, `total_cost`, and `cost`, but still uses a lightweight local pricing table instead of Python's canonical calculator stack. |
| Pre-trial constraint rejection | Yes | Yes | `matched` | Invalid configs are skipped before execution and do not consume native trial slots. |
| Post-trial constraint rejection | Yes | Yes | `partial` | Rejected trials are recorded and excluded from best-trial selection, but JS does not yet expose Python's richer pruned/failed trial taxonomy. |

## Hybrid and Execution Modes

| Capability | Python SDK | JS SDK | Status | Notes |
| --- | --- | --- | --- | --- |
| Native/local execution | Yes | Yes | `matched` | JS local/native path is the primary implementation here. |
| Cloud execution | Yes | No | `deferred` | Not implemented in this checkout. |
| Hybrid execution orchestration | Yes | No | `deferred` | `execution.mode = 'hybrid'` throws in this branch. |
| Hybrid spec authoring to legacy wire format | Yes | Yes | `matched` | JS supports `toHybridConfigSpace()` for current hybrid HTTP routes. |
| Legacy Python-to-Node bridge | Yes | Yes | `partial` | JS still exposes the CLI runner for bridge use, but it is legacy-only. |

## TVL / Advanced Specification Features

| Capability | Python SDK | JS SDK | Status | Notes |
| --- | --- | --- | --- | --- |
| TVL spec loading | Yes | Yes | `partial` | JS supports a focused native subset through `parseTvlSpec()` / `loadTvlSpec()` and now exposes a `nativeCompatibility` report describing supported vs reduced-semantics features. |
| TVL banded objectives | Yes | Yes | `partial` | JS parses banded objectives and the native scorer honors them, including sample-based TOST-style band comparison. |
| TVL promotion policy | Yes | Yes | `partial` | JS applies `minEffect`, `tieBreakers`, behavioral `chanceConstraints`, sample-based paired/TOST promotion, and now emits bounded `promotionDecision` reports. Full Python promotion-gate lifecycle parity is still deferred. |
| Constraints | Yes | Yes | `partial` | Native callback constraints work, and TVL structural/derived constraints compile into those callbacks through a parsed safe-expression subset. |
| Safety constraints | Yes | Yes | `partial` | Native callback safety predicates work; Python's safety preset library is still deferred. |
| Multi-agent execution options | Yes | No | `deferred` | Not implemented here. |

## Normalized Stop Reason Mapping

These mappings are used by parity tests and reports when comparing Python and JS output:

| Python | JS |
| --- | --- |
| `max_trials_reached` | `maxTrials` |
| `max_samples_reached` | `maxExamples` |
| `cost_limit` | `budget` |
| `timeout` | `timeout` |
| `error` | `error` |
| `user_cancelled` | `cancelled` |
| `plateau` | `plateau` |
| `optimizer` | `completed` when the search space is exhausted |

## Explicit Deferrals in This Checkout

- Full Python-style runtime AST seamless injection
- Full TVL statistical/promotion behavior beyond native sample-based promotion
  best-trial selection
- Python safety preset/statistical validation layer
- Cloud/hybrid execution orchestration
- Global `get_config()` semantics
