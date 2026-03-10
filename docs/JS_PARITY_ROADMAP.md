# JS Parity Roadmap

This roadmap turns the current Python-vs-JS gap analysis into an execution
order.

Use this together with:

- [NATIVE_JS_PARITY_MATRIX.md](./NATIVE_JS_PARITY_MATRIX.md)
- [PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md](./PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md)

Important branch note:

- this checkout is the native-first JS branch
- backend-guided hybrid/cloud parity belongs to the hybrid-enabled JS branch

## Prioritization Rules

Gaps are ranked by:

1. user-visible parity impact
2. whether the gap belongs in this branch
3. whether the divergence is justified
4. implementation leverage across multiple Python module families

## P0: Highest Product Gap, But Not For This Branch

These are the largest remaining Python-vs-JS gaps overall, but they should be
implemented in the hybrid/cloud JS branch, not in this native-first checkout.

### Backend-guided hybrid execution

- Python status:
  - full cloud/hybrid/session stack
  - remote execution and optimizer orchestration
- JS status:
  - native-first only in this branch
  - `execution.mode = 'hybrid'` is intentionally rejected here
- Why it matters:
  - this is the biggest remaining product-level parity gap
  - it is the path for backend-owned Optuna-like functionality

### Cloud/session/control-plane parity

- Python status:
  - remote clients
  - session lifecycle
  - subset selection
  - billing/privacy/auth DTOs
- JS status:
  - not active in this checkout
- Why it matters:
  - required for full remote parity
  - should be solved once in the hybrid branch, not partially copied here

## P1: Highest-Value Native Gaps In This Branch

These are the most important justified parity gaps that still belong in this
native-first checkout.

### TVL promotion and statistical semantics

- Current JS status:
  - focused TVL loader
  - explicit `nativeCompatibility` reporting on loaded TVL artifacts
  - banded objectives
  - safe-expression constraint compilation
  - `minEffect`, `tieBreakers`
  - sample-based paired promotion
  - TOST-style band promotion
  - behavioral `chanceConstraints`
- Remaining gap:
  - no full promotion-gate lifecycle semantics beyond best-trial selection and
    post-trial rejection
  - no full TVL CLI/models/runtime parity beyond the supported native subset
- Priority reason:
  - this is the biggest native-local parity gap still left
  - it affects spec compatibility, promotion semantics, and cross-SDK behavior

### Auto framework override ergonomics

- Current JS status:
  - explicit wrappers exist for OpenAI, LangChain, and Vercel AI
  - seamless mode defaults `autoOverrideFrameworks` to all active wrapped targets
  - `frameworkTargets` can narrow active targets and `autoOverrideFrameworks: false`
    disables framework interception entirely for a seamless function
  - `autoWrapFrameworkTarget(...)` and `autoWrapFrameworkTargets({...})` now
    batch-wrap supported framework objects in one call
- Remaining gap:
  - JS still does not auto-discover and wrap framework clients implicitly inside arbitrary user code
  - framework auto-override diagnostics could be surfaced more prominently in user-facing tooling
- Priority reason:
  - high user impact
  - closes a real adoption gap without requiring cloud work

### Broader TVL authoring/runtime parity

- Current JS status:
  - native subset is implemented and tested
- Remaining gap:
  - full TVL language/runtime parity
  - tighter alignment with Python authoring expectations
- Priority reason:
  - important if cross-SDK spec portability is a product requirement

## P2: Valuable, But Secondary

These are useful parity improvements, but they are lower priority than P1.

### Example concurrency

- Python status:
  - supports broader evaluation parallelism
- JS status:
  - no per-example concurrency in this checkout
- Why it is secondary:
  - performance-oriented rather than core contract parity

### Broader optimizer family parity

- Python status:
  - richer optimizer set, including Optuna-family modes
- JS status:
  - native grid/random/Bayesian only
- Why it is secondary:
  - if the long-term strategy is backend-guided hybrid for advanced optimizers,
    native-local JS does not need to mirror the full Python optimizer family

### Richer trial taxonomy and pruning semantics

- Python status:
  - more detailed failed/pruned/rejected distinctions
- JS status:
  - simpler native trial outcome model
- Why it is secondary:
  - useful for introspection, but not the main user-facing parity blocker

### Global config access parity

- Python status:
  - global-style `get_config()` semantics
- JS status:
  - intentional wrapper-local / context-local model instead
- Why it is secondary:
  - parity exists as a mental-model difference, but the JS divergence is partly
    intentional and arguably better aligned with JS runtime behavior

## P3: Real Python Capabilities, But Not Yet Justified To Copy

These appear in the Python audit, but they should not drive near-term JS SDK
work in this repo.

### Platform / enterprise families

- security
- auth
- tenancy
- session hardening
- analytics/meta-learning

These are platform-side concerns, not core local-SDK parity targets.

### Assisted generation and discovery families

- tuned-variable discovery
- config generation pipelines
- benchmark/objective recommendation

These are real Python capabilities, but they are a separate product layer from
the local native JS optimizer/runtime.

### Server / wrapper / adapter families

- wrapper service/server
- hooks
- execution adapters
- bridge/server infrastructure

These should be treated as separate surfaces, not copied blindly into the
native-first JS SDK package.

## Recommended Execution Order

### Native branch

1. finish TVL promotion/statistical parity
2. improve auto framework override ergonomics
3. decide whether full TVL parity is a product requirement or keep a documented
   supported subset
4. optionally add example concurrency if runtime throughput becomes a priority

### Hybrid branch

1. backend-guided hybrid execution parity
2. cloud/session/control-plane parity
3. align remote DTO/session contracts with Python
4. only then revisit advanced optimizer breadth through the backend path

## Non-Goals For This Branch

- implementing cloud/session clients directly in this checkout
- copying platform/security/analytics modules into the JS SDK package
- forcing Python-style global config semantics into JS if the current wrapper-
  local/context-local design remains clearer
