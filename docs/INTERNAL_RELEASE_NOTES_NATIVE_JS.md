# Internal Release Notes: Native JS Optimization

## Scope

This release hardens the current native JS optimization surface in `@traigent/sdk` for this native-first checkout.

## SDK Changes

- Root exports now include `optimize`, `param`, `getOptimizationSpec`, and `toHybridConfigSpace`.
- Wrapped functions now support a high-level agent contract by default:
  - plain agent function returns output
  - SDK-owned `evaluation` block computes metrics
  - injection modes: `context`, `parameter`, `seamless`
- Wrapped functions still expose native `.optimize({ algorithm, maxTrials, randomSeed, timeoutMs, trialConcurrency, plateau, checkpoint })`.
- Framework interception is available through explicit wrappers for OpenAI, LangChain, and Vercel AI.
- Seamless support for hardcoded local tuned variables is now available through:
  - `traigent migrate seamless`
  - `@traigent/sdk/babel-plugin-seamless`
  - an experimental runtime rewrite fallback for self-contained plain Node functions when explicitly opted in for trusted local code
- The old low-level manual trial contract still exists behind `execution.contract = 'trial'`, but it is deprecated.

## Demo / Migration State

- `child-age-agent-a` is code-defined and serialized with `toHybridConfigSpace()`.
- `child-age-agent-b` remains JSON-backed for one migration cycle.
- Startup fails when spec-backed and JSON-backed config spaces disagree.
- Runtime config is bound only inside capability execution through `TrialContext.run(...)`.

## Intentional v1 Limits

- Log-scale grid search requires a multiplicative `step > 1`.
- `trialConcurrency` is currently limited to `grid` and `random`.
- `execution.mode = 'hybrid'` is not implemented in this checkout.
- `seamless` is no longer framework-only, but full Python-style runtime AST parity is still not implemented.
- No worker pools, Optuna-family algorithms, example-level concurrency, or hybrid API orchestration in the native runner.
- `budget.maxCostUsd` requires numeric `metrics.total_cost` or `metrics.cost` on every trial.
- Native JS now supports `defaultConfig`, callback-based `constraints`, and callback-based `safetyConstraints`.
- TVL loading is implemented for a focused native subset:
  - typed `tvars`
  - banded objectives
  - structural and derived constraints via a parsed safe-expression subset
  - exploration strategy and budget mapping
  - promotion-policy parsing
- Promotion policy is now partially behavioral in this branch:
  `minEffect` and `tieBreakers` influence native best-trial selection,
  per-objective metric samples enable statistical promotion for standard and
  banded objectives, and `chanceConstraints` reject trials when they expose
  explicit `{successes, trials}` counts or binary metric samples. Full Python
  promotion-gate parity remains deferred.
- Python's safety preset/statistical layer and hybrid API orchestration are not implemented in this branch.

## Release Validation

Run in `traigent-js`:

```bash
npm test
npm run smoke:example
npm run build
npm pack --dry-run
```

Run in `JS-Mastra-APIs-Validation`:

```bash
npm test
DEMO_MOCK_GENERATION=1 npm run api:dev
npm run api:test
```

## Branch Isolation

Cut the release from an isolated branch or PR that excludes unrelated local SDK work already present in the `traigent-js` worktree.
