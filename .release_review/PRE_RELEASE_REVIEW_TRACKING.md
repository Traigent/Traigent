# Pre-Release Review Tracking (Slim Board v3)

**Protocol Version**: 3
**Status Source of Truth**: this board + `.release_review/runs/<release_id>/gate_results/verdict.json`

## Active Run

- Release ID: `<release_id>`
- Version: `<version>`
- Baseline SHA: `<baseline_sha>`
- Captain: `<captain>`
- Release owner: `<release_owner>`
- Review mode: `<strict|quick>`
- Run root: `.release_review/runs/<release_id>/`

## Gate Status

| Check | Status | Evidence |
|---|---|---|
| release-gate/lint-type | Not started | |
| release-gate/tests-unit | Not started | |
| release-gate/tests-integration | Not started | |
| release-gate/security | Not started | |
| release-gate/dependency-review | Not started | |
| release-gate/codeql | Not started | |
| release-gate/release-review-consistency | Not started | |
| release-verdict/peer-review-completeness | Not started | `.release_review/runs/<release_id>/gate_results/verdict.json` |

## Component Board

| Component | Priority | Owner | Secondary | Tertiary | Status | Gate | Primary Evidence | Secondary Evidence | Tertiary Evidence | Reconciliation Evidence | Approved At |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| Public API + Safety | P0 | - | - | - | Not started | Pending | | | | | |
| Core Orchestration + Config | P0 | - | - | - | Not started | Pending | | | | | |
| Integrations + Invokers | P1 | - | - | - | Not started | Pending | | | | | |
| Optimizers + Evaluators | P1 | - | - | - | Not started | Pending | | | | | |
| Packaging + CI | P1 | - | - | - | Not started | Pending | | | | | |
| Docs + Release Ops | P2 | - | - | - | Not started | Pending | | | | | |

## Decision Summary

- Verdict: `TBD`
- Unresolved P0: 0
- Unresolved P1: 0
- Failed required checks: 0
- Failed required reviews: 0
- Waivers: 0

## Review Notes Log

- `YYYY-MM-DDTHH:MM:SSZ`: initialized.
