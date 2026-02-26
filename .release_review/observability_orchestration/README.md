# Observability Orchestration

## Current State
- Phase 1 (`phase-1-dashboard-foundation`) is completed and synced.
- Phase 2 (`phase-2-correlation-intelligence`) is active for SDK, Backend, and Frontend.
- Phase 2 branches currently do **not** include the Phase 1 hardening fixes in code.
- A direct cherry-pick from Phase 1 to Phase 2 was attempted and then intentionally aborted due major codepath drift.

## What To Run Next
- Use the Phase 2 prompts in:
  - `.release_review/observability_orchestration/prompts/phase-2-correlation-intelligence/sdk_codex_prompt.md`
  - `.release_review/observability_orchestration/prompts/phase-2-correlation-intelligence/backend_codex_prompt.md`
  - `.release_review/observability_orchestration/prompts/phase-2-correlation-intelligence/frontend_codex_prompt.md`
- Each prompt now includes:
  - Hardening port requirements (from review findings)
  - Phase 2 correlation deliverables
  - File-level implementation targets
  - Required test gates

## Canonical Port Plan
- Detailed cross-repo mapping lives in:
  - `.release_review/observability_orchestration/PHASE_2_PORT_PLAN.md`

## Operator Notes
- Keep each team on its phase branch:
  - `observability-sdk-phase-2-correlation-intelligence`
  - `observability-backend-phase-2-correlation-intelligence`
  - `observability-frontend-phase-2-correlation-intelligence`
- Sync after each team reports green tests and updates its Phase 2 status file.
