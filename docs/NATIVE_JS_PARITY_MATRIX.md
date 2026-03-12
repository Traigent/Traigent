# Native JS Parity Matrix

This matrix describes the verified state of the native-first JS checkout in
this repository root.

Use it together with:

- [Python SDK Module Catalog and Gap Analysis](./PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md)
- [JS Parity Roadmap](./JS_PARITY_ROADMAP.md)
- [Hybrid JS Parity Matrix](../../traigent-js-hybrid-optuna/docs/HYBRID_JS_PARITY_MATRIX.md)

Important scope note:

- this file is about the native-first checkout only
- if a capability exists only in the hybrid-enabled worktree, the native status
  here is not upgraded to `matched`

## Label Semantics

- `matched`: implemented in this checkout and covered by passing tests or a
  verified public example
- `partial`: implemented with bounded semantics and covered, but still behind
  Python
- `gap`: no backend change is required; JS could implement it in this project
  today
- `deferred-backend`: blocked on hybrid/backend protocol or control-plane work
- `out-of-scope`: not a target for the native JS SDK

## Optimize / Decorator Surface

| Capability | Python | Native JS | Overall JS | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Optimize plain agent functions | Yes | `matched` | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts), [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts) | Native JS defaults to the high-level agent contract. |
| SDK-owned evaluation loop | Yes | `matched` | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | The SDK owns dataset iteration, scoring, aggregation, and runtime-metric capture. |
| `scoringFunction` | Yes | `matched` | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | Supports raw output plus expected-output scoring. |
| `metricFunctions` | Yes | `matched` | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | Additive metric computation is supported. |
| `customEvaluator` | Yes | `matched` | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | Supports both the JS context form and the Python-style `(agentFn, config, row)` form. |
| Deprecated trial contract | Legacy/advanced | `partial` | `partial` | [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts) | Still supported via `execution.contract = "trial"` but explicitly deprecated. |
| `default_config` baseline | Yes | `matched` | `matched` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Merged into trial configs and `applyBestConfig()` flows. |
| Context injection | Yes | `partial` | `partial` | [`tests/unit/core/context.test.ts`](../tests/unit/core/context.test.ts), [`examples/quickstart/00_agent_injection_map.mjs`](../examples/quickstart/00_agent_injection_map.mjs) | JS uses `getTrialParam()` / `getTrialConfig()` instead of Python’s global `get_config()`. |
| Parameter injection | Yes | `partial` | `partial` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | JS uses `agentFn(input, config?)` second-argument injection. |
| Seamless injection | Yes | `partial` | `partial` | [`tests/unit/seamless/transform.test.ts`](../tests/unit/seamless/transform.test.ts), [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Codemod, build-time transform, and framework-interception paths exist; full Python runtime AST parity is intentionally not claimed. |

## Native Runtime

| Capability | Python | Native JS | Overall JS | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Grid search | Yes | `matched` | `matched` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Deterministic ordered enumeration. |
| Random search | Yes | `matched` | `matched` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts), [`tests/unit/optimization/python-random.test.ts`](../tests/unit/optimization/python-random.test.ts) | Python-style seeded sampling for parity. |
| Local Bayesian search | Yes | `partial` | `partial` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts), [`tests/unit/optimization/native-bayesian.test.ts`](../tests/unit/optimization/native-bayesian.test.ts) | Sequential in-process Bayesian only; no Optuna-family parity here. |
| Stop reasons: timeout/error/cancel/plateau/budget/maxExamples | Yes | `matched` | `matched` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Native JS now covers the practical local stop-condition set. |
| Repetition stability controls | Yes | `matched` | `matched` | [`tests/unit/optimization/native-reps.test.ts`](../tests/unit/optimization/native-reps.test.ts), [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | `execution.repsPerTrial` and `execution.repsAggregation` are implemented. |
| Max total examples | Yes | `matched` | `matched` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Stops with `maxExamples`. |
| Trial concurrency | Yes | `partial` | `partial` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Supported for `grid` and `random`, not Bayesian. |
| Example concurrency | Yes | `partial` | `partial` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | Opt-in `execution.exampleConcurrency` parallelizes per-example evaluation while preserving stable aggregation and sample order. |
| Checkpoint/resume | Yes | `partial` | `partial` | [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Trial-boundary checkpoints only. |
| Runtime cost accounting | Yes | `partial` | `partial` | [`tests/unit/optimization/native-cost.test.ts`](../tests/unit/optimization/native-cost.test.ts), [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts) | Records `input_cost`, `output_cost`, `total_cost`, and `cost`, but still uses a lightweight local pricing table. |
| Constraints and safety constraints | Yes | `partial` | `partial` | [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts), [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | Callback-based constraints are implemented; Python’s richer preset/statistical safety layer is still missing. |

## TVL / Advanced Specification

| Capability | Python | Native JS | Overall JS | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| TVL loading | Yes | `partial` | `partial` | [`tests/unit/optimization/tvl.test.ts`](../tests/unit/optimization/tvl.test.ts), [`examples/core/tvl-loading/run.mjs`](../examples/core/tvl-loading/run.mjs) | Focused native subset with explicit `nativeCompatibility` reporting. |
| TVL constraints | Yes | `partial` | `partial` | [`tests/unit/optimization/tvl-expression.test.ts`](../tests/unit/optimization/tvl-expression.test.ts), [`tests/unit/optimization/tvl.test.ts`](../tests/unit/optimization/tvl.test.ts) | Safe-expression subset compiles into native callbacks. |
| TVL banded objectives | Yes | `partial` | `partial` | [`tests/unit/optimization/native-scoring.test.ts`](../tests/unit/optimization/native-scoring.test.ts), [`tests/unit/optimization/native-promotion.test.ts`](../tests/unit/optimization/native-promotion.test.ts) | Native scorer and promotion logic honor banded objectives. |
| TVL promotion policy | Yes | `partial` | `partial` | [`tests/unit/optimization/native-promotion.test.ts`](../tests/unit/optimization/native-promotion.test.ts), [`tests/unit/optimization/native-scoring.test.ts`](../tests/unit/optimization/native-scoring.test.ts), [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts) | `minEffect`, `tieBreakers`, sample-based paired/TOST promotion, and behavioral `chanceConstraints` are implemented. Native results also expose bounded `promotionDecision` and `reporting` summaries; full Python promotion-gate lifecycle parity is still not present. |
| Full TVL CLI/runtime breadth | Yes | `gap` | `gap` | n/a | The native subset is deliberate; broader TVL authoring/runtime parity remains open. |

## Framework Ergonomics

| Capability | Python | Native JS | Overall JS | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI interception | Yes | `matched` | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Real override injection and provider-usage capture. |
| LangChain interception | Yes | `matched` | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Proxy-based wrapping plus runtime metrics extraction. |
| Vercel AI interception | Yes | `matched` | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Generate/stream wrappers and cost recording are implemented. |
| Auto-wrap helpers | Yes | `matched` | `matched` | [`tests/unit/integrations/auto-wrap.test.ts`](../tests/unit/integrations/auto-wrap.test.ts), [`examples/core/seamless-autowrap/run.mjs`](../examples/core/seamless-autowrap/run.mjs) | `autoWrapFrameworkTarget(...)` and `autoWrapFrameworkTargets(...)` are available. |
| Framework auto-override diagnostics | Yes | `matched` | `matched` | [`tests/unit/integrations/registry.test.ts`](../tests/unit/integrations/registry.test.ts), [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts) | `frameworkAutoOverrideStatus()` and `seamlessResolution()` expose active targets, selected targets, and the resolved seamless path. |
| Implicit framework discovery | Yes | `gap` | `gap` | n/a | JS still requires explicit wrapping; it does not auto-discover arbitrary framework clients inside user code. |
| Tuned-variable discovery | Yes | `partial` | `partial` | [`tests/unit/tuned-variables/discovery.test.ts`](../tests/unit/tuned-variables/discovery.test.ts), [`tests/unit/cli/detect.test.ts`](../tests/unit/cli/detect.test.ts) | Native JS now has bounded heuristic tuned-variable discovery plus a `traigent detect tuned-variables` CLI, but it is still well behind Python’s fuller dataflow/range-generation pipeline. |

## Hybrid / Cloud Boundaries

| Capability | Python | Native JS | Overall JS | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `execution.mode = "hybrid"` | Yes | `out-of-scope` | `partial` | Native rejection: [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts); hybrid implementation: [Hybrid JS Parity Matrix](../../traigent-js-hybrid-optuna/docs/HYBRID_JS_PARITY_MATRIX.md) | Native checkout intentionally rejects hybrid execution; the hybrid worktree carries the backend-guided local-execution path. |
| Typed `/sessions` control-plane helpers | Yes | `out-of-scope` | `matched` | [Hybrid JS Parity Matrix](../../traigent-js-hybrid-optuna/docs/HYBRID_JS_PARITY_MATRIX.md) | Create/list/status/finalize/delete on the typed session surface exist only in the hybrid-enabled worktree. |
| Python remote cloud-execution control plane | Yes | `out-of-scope` | `out-of-scope` | [Hybrid JS Parity Matrix](../../traigent-js-hybrid-optuna/docs/HYBRID_JS_PARITY_MATRIX.md) | JS does not target Python's server-side `/agent/optimize`-style remote execution model; the product boundary is backend-guided local execution. |

## Explicit Non-Goals for the Native Checkout

- full cloud/session control-plane parity
- Python-style remote cloud execution and server-side agent reconstruction
- Python’s platform/security/analytics families
- global `get_config()` semantics
- full Optuna-family parity in local/native JS
