# Hybrid JS Parity Matrix

This matrix describes the verified state of the hybrid-enabled JS worktree at
`traigent-js-hybrid-optuna`.

Use it together with:

- [Native JS Parity Matrix](../../traigent-js/docs/NATIVE_JS_PARITY_MATRIX.md)
- [Python SDK Module Catalog and Gap Analysis](../../traigent-js/docs/PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md)

## Label Semantics

- `matched`: implemented and covered by passing tests or a verified public example
- `partial`: implemented with bounded semantics and covered, but still behind Python
- `gap`: the backend route is reachable today and the JS side could implement it now
- `deferred-backend`: blocked on missing or insufficiently specified backend/protocol support
- `out-of-scope`: not a current JS SDK target

## High-level Agent and Execution Surface

| Capability | Python | Hybrid JS | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Plain agent optimization | Yes | `matched` | [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts), [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts) | Hybrid JS now supports the same high-level agent contract as the native checkout. |
| Contract inference (`agent` vs `trial`) | Yes | `partial` | [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts) | The legacy trial contract remains as a deprecated compatibility path. |
| Hybrid session lifecycle | Yes | `matched` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Create → next-trial → submit-result → finalize is implemented and tested. |
| Stop-reason normalization | Yes | `partial` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Exhaustive normalization is implemented, but exact parity still depends on backend message/field conventions. |
| Hybrid cost/budget handling | Yes | `partial` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Budget enforcement works, but still expects the current backend/client cost metric conventions. |

## Session / Cloud Control Surface

| Capability | Python | Hybrid JS | Evidence | Notes |
| --- | --- | --- | --- | --- |
| `getOptimizationSessionStatus(...)` | Yes | `matched` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts), [`examples/core/hybrid-session-control/run.mjs`](../examples/core/hybrid-session-control/run.mjs) | Raw and wrapped envelopes are normalized, including `progress`. |
| `finalizeOptimizationSession(...)` | Yes | `matched` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts), [`examples/core/hybrid-session-control/run.mjs`](../examples/core/hybrid-session-control/run.mjs) | Finalization reporting is normalized into the same shape as `.optimize()` results. |
| `deleteOptimizationSession(...)` | Yes | `matched` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts), [`examples/core/hybrid-session-control/run.mjs`](../examples/core/hybrid-session-control/run.mjs) | Safe default is `cascade: false`. |
| Reporting summary on `.optimize()` | Yes | `matched` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts), [`examples/core/hybrid-session-control/run.mjs`](../examples/core/hybrid-session-control/run.mjs) | `result.reporting` surfaces backend finalization summary directly. |
| Full-history finalization reporting | Yes | `partial` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Supported via `includeFullHistory`, but typed DTO breadth is intentionally bounded. |
| Broader cloud/session DTO breadth | Yes | `partial` | [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Status/delete/finalize are normalized; billing/privacy/subsets and wider cloud clients still trail Python. |
| Billing/privacy/subset selection APIs | Yes | `deferred-backend` | n/a | No equivalent JS control-plane surface is implemented yet. |

## Framework Interception and Seamless Ergonomics

| Capability | Python | Hybrid JS | Evidence | Notes |
| --- | --- | --- | --- | --- |
| OpenAI seamless interception | Yes | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Backend-suggested config is applied to local OpenAI calls in hybrid mode. |
| LangChain seamless interception | Yes | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Proxy wrapping plus usage extraction works in hybrid mode. |
| Vercel AI seamless interception | Yes | `matched` | [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts) | Generate/stream overrides and provider usage capture are implemented. |
| Auto-wrap helpers | Yes | `matched` | [`tests/unit/integrations/auto-wrap.test.ts`](../tests/unit/integrations/auto-wrap.test.ts) | `autoWrapFrameworkTarget(...)` and `autoWrapFrameworkTargets(...)` are available in the hybrid worktree too. |
| Framework auto-override diagnostics | Yes | `matched` | [`tests/unit/integrations/registry.test.ts`](../tests/unit/integrations/registry.test.ts), [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts), [`examples/core/hybrid-session-control/run.mjs`](../examples/core/hybrid-session-control/run.mjs) | `frameworkAutoOverrideStatus()` and `seamlessResolution()` expose active targets, selected targets, and the resolved seamless path. |
| Implicit framework discovery | Yes | `gap` | n/a | Users still need to wrap framework targets explicitly, even though auto-wrap helpers and diagnostics now reduce the manual setup burden. |

## TVL / Spec Compatibility

| Capability | Python | Hybrid JS | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Typed objective objects | Yes | `matched` | [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts), [`tests/unit/optimization/hybrid.test.ts`](../tests/unit/optimization/hybrid.test.ts) | Objective objects, weights, and hybrid serialization are implemented. |
| Conditional parameters | Yes | `partial` | [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts), [`tests/unit/optimization/tvl.test.ts`](../tests/unit/optimization/tvl.test.ts) | Parsed and validated in the JS surface, but still bounded by the current backend contract. |
| Bounded TVL support | Yes | `partial` | [`tests/unit/optimization/tvl.test.ts`](../tests/unit/optimization/tvl.test.ts) | TVL support exists, but the hybrid worktree still inherits the same bounded subset strategy as the native checkout. |
| Full TVL/runtime/control-plane parity | Yes | `deferred-backend` | n/a | Requires broader backend/runtime alignment than the current session surface provides. |

## Biggest Remaining Gaps

| Capability Family | Current Status | Why It Is Still Open |
| --- | --- | --- |
| Broader cloud/session control plane | `partial` | Status/delete/finalize are in place, but the wider Python cloud family is larger than the current JS helper surface. |
| Implicit framework discovery | `gap` | No backend work is required, but the hybrid worktree still expects explicit wrapping even though diagnostics and auto-wrap helpers are now in place. |
| Full TVL/runtime breadth | `deferred-backend` | The backend/session contract does not yet expose the whole Python TVL lifecycle. |
