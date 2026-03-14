# n8n Optimization Execution Playbook

This playbook defines how to execute n8n agent optimization through Traigent once access and candidate selection are ready.

Depends on:

1. `#203` access + local runbook completion.
2. `#205` filled agent evaluation matrix.

## Phase 0: Preconditions

All must be true:

1. `n8n-access-and-local-runbook.md` completed.
2. At least one agent marked `Readiness=Y` in `n8n-agent-evaluation-matrix.md`.
3. Objective weights agreed by owner.
4. Baseline metrics recorded.

## Phase 1: Baseline Capture

For selected agent:

1. Run fixed baseline configuration on representative dataset.
2. Capture:
   - accuracy
   - cost
   - latency
   - failure rate
3. Store baseline run metadata:
   - tunable ID
   - dataset snapshot ID
   - config snapshot
   - timestamp

## Phase 2: Optimization Loop

1. Use Traigent Hybrid Mode evaluator against the selected tunable.
2. Run bounded optimization (trial and budget caps).
3. Track per-trial metrics and stop reason.
4. Select best config by objective-weighted score.

## Phase 3: Validation Gate

Before rollout:

1. Re-run best config on holdout or replay set.
2. Compare vs baseline for:
   - quality delta
   - cost delta
   - latency delta
3. Confirm no regression beyond agreed tolerance.

## Phase 4: Front/MCP Reporting (if enabled)

1. Publish run summary to chosen frontend view.
2. Include:
   - baseline vs optimized metrics
   - selected config parameters
   - trial history link or artifact

## Definition of Done for #206

Close `#206` only when all are true:

1. One n8n agent completed end-to-end optimization cycle.
2. Improvement is measurable against baseline.
3. Final config and evidence are documented.
4. Front/MCP visibility path is demonstrated (or explicitly deferred with owner sign-off).
