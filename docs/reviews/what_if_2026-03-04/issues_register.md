# What-If Issue Register (Traigent SDK)

Legend:
- Priority: `P0` critical, `P1` high, `P2` medium, `P3` low
- Possibility: whether scenario can occur (`Yes`/`No`)
- Probability: `Low` / `Medium` / `High`

| ID | Priority | Issue | Evidence | Possibility | Probability | Impact (What If...) |
|---|---|---|---|---|---|---|
| WI-001 | P0 | TVL constraint sandbox escape via `eval` object-graph traversal | `traigent/tvl/spec_loader.py:805`, `traigent/tvl/spec_loader.py:1918` | Yes | Medium | If TVL expressions come from untrusted input, attacker can access `__globals__` and import arbitrary modules, breaking code-exec isolation. |
| WI-002 | P1 | SaaS optimization polling can hang indefinitely | `traigent/traigent_client.py:362` | Yes | Medium | If backend stays in non-terminal status (or returns unexpected status like CANCELLED), client loop never exits and workflow stalls forever. |
| WI-003 | P1 | Cloud execution silently falls back to local on unexpected exceptions | `traigent/core/optimized_function.py:1480` | Yes | Medium | If cloud path fails unexpectedly, local execution starts automatically; this can violate user expectations/governance around execution mode and data routing. |
| WI-004 | P2 | `progress` property is inconsistent when `max_trials=None` | `traigent/core/orchestrator.py:517` | Yes | Medium | Progress reports `1.0` before first trial and `0.0` afterwards for unlimited runs, breaking observability and UI/automation logic that depends on progress monotonicity. |
| WI-005 | P2 | Local session IDs can collide within the same second | `traigent/storage/local_storage.py:203` | Yes | Medium | If two sessions for the same function start in the same second, they get identical IDs and can overwrite session files. |
| WI-006 | P2 | File-lock implementation has no stale-lock recovery path | `traigent/storage/local_storage.py:163`, `traigent/storage/local_storage.py:170`, `traigent/storage/local_storage.py:177` | Yes | Medium | If process crashes while holding lock file, future attempts can repeatedly timeout until manual cleanup. |
| WI-007 | P2 | `max_trials` semantics are inconsistent across public APIs | `traigent/api/functions.py:558`, `traigent/core/optimization_pipeline.py:73` | Yes | Medium | Some entry points reject `max_trials=0` while core resolver allows it, causing confusing no-op/stop behaviors depending on API surface used. |
| WI-008 | P3 | Monitoring stop can block indefinitely on thread join | `traigent/security/enterprise.py:384` | Yes | Low | If monitoring thread gets stuck in collector/IO path, `stop_monitoring()` can hang without timeout and block shutdown flows. |
| WI-009 | P3 | Optimization history grows unbounded in long-lived process | `traigent/core/optimized_function.py:1403`, `traigent/core/optimized_function.py:1776` | Yes | Medium | Repeated optimizations append to in-memory history without cap; long-running services may see avoidable memory growth. |

## Validation Note for WI-001
A local proof-of-concept expression successfully bypassed the intended restricted eval context by traversing:
`params.__class__.__init__.__globals__['__builtins__']['__import__'](...)`

This confirms the what-if scenario is not theoretical under current AST/eval guarding.

## Remediation Status (2026-03-05)
- Fixed in code and regression tests: `WI-001`, `WI-002`, `WI-003`, `WI-004`, `WI-005`, `WI-006`, `WI-007`, `WI-008`, `WI-009`
- Still open (not addressed in this patch): none
