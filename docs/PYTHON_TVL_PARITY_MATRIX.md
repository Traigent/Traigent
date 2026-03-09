# Python / TVL Parity Matrix

This matrix tracks optimization features exposed by the Python `@optimize(...)`
decorator or the TVL language, and how they map into the JS SDK.

## Planner / Executor

"Planner / executor" refers to a multi-step agent pattern where one model or
prompt produces a plan and a second step executes that plan, often with tools
or retrieval in the loop. It is an agent capability lane, not a distinct
optimizer algorithm.

## Optimization Feature Matrix

| Feature | Python / TVL | JS Status |
| --- | --- | --- |
| Built-in objectives (`accuracy`, `cost`, `latency`) | Supported | Supported |
| Explicit objective objects with direction | Supported | Supported |
| Weighted objective objects | Supported | Supported in hybrid |
| Enum / int / float parameters | Supported | Supported |
| Bool parameters | Supported | Supported via `param.bool()` |
| Conditional parameters with fallback defaults | Supported | Supported in hybrid |
| Cost budget | Supported | Supported |
| Trial-count budget | Supported | Supported in hybrid |
| Wall-clock budget | Supported | Supported in hybrid |
| Structural constraints | Supported | Supported in hybrid |
| Derived constraints | Supported | Supported in hybrid |
| TVL subset loading | Supported | Supported for `tvars`, objectives, constraints, budgets, defaults, and promotion policy |
| TVL banded objectives | Supported | Supported |
| TVL promotion policy | Supported | Partial: parsed/transported policy metadata, not behaviorally enforced yet |
| TVL tuple / callable domains | Supported | Supported |
| Default config baseline | Supported | Supported |
| Execution defaults for optimization runs | Supported | Supported |
| Auto-load best config from local persistence | Supported | Supported |
| TVL exploration strategy hints | Supported | Partial: parsed as metadata, execution still configured via `.optimize(...)` options |
| Native local execution | Supported | Supported |
| Backend-guided Optuna execution | Supported | Supported |

## Reclassified Runtime Features

These are broader Python decorator/runtime concerns, not the core optimization /
TVL transport layer:

- `safety_constraints`
- `injection`
- `mock`
- multi-agent execution options

## Current Recommendation

- Use JS spec authoring for the common case.
- Use hybrid mode as the default optimizer path when you need weighted
  objectives, conditionals, structural constraints, derived constraints, or
  TVL-style budgets, banded objectives, tuple/callable domains, or promotion-policy metadata.
- Use `mode: "native"` only for intentionally local/offline optimization flows.
