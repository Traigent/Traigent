# Pre-Release Review Plan v3 (Gate-First + Source Waves)

This plan defines fail-closed release review behavior for Traigent SDK.

## Scope

In scope:

- SDK code shipped from `traigent/` and `traigent_validation/`
- Release engineering and CI workflows
- Security, dependency, and test gates
- Docs required for release operation (`README.md`, release docs)

Out of scope for the initial hard gate:

- Prompt quality scoring as a blocker
- SLSA L3 controls (post-launch milestone)

## Source Review Contract

- This repository does not use a repo-root `src/`; the canonical source roots are
  `traigent/**/*.py` and `traigent_validation/**/*.py`.
- The exact staged source inventory lives in `.release_review/inventories/source_files.txt`
  and currently contains 362 Python source files as of March 6, 2026.
- The canonical source-wave contract lives in `.release_review/components.yml`.
- The ordered wave plan and rationale live in `.release_review/PRIORITY_REVIEW_WAVES.md`.
- Wave execution is a cost-control mechanism only. `READY` still requires:
  - full source coverage across every wave
  - the broader release scope defined in `.release_review/scope.yml`

## Major-Release Gate Stack

Run bundle:

```bash
python3 .release_review/automation/release_gate_runner.py --release-id <RELEASE_ID> --strict
```

Required gates:

- Lint/type
- Unit tests
- Integration tests
- Security checks
- Dependency review
- CodeQL
- Release-review consistency checks

## Severity Model

Use `.release_review/SEVERITY_POLICY.md`.

- P0: release-blocking
- P1: release-blocking unless valid waiver
- P2/P3: non-blocking unless explicitly promoted

## Broad Component Matrix

These are the high-level readiness buckets already enforced by the current release-review flow.

| Component | Priority | Scope | 4-Angle Review | Blocking |
|---|---:|---|---|---|
| Public API + Safety | P0 | `traigent/api/`, `traigent/security/` | Yes | Yes |
| Core Orchestration + Config | P0 | `traigent/core/`, `traigent/config/`, `traigent/utils/`, `traigent/cloud/` | Yes | Yes |
| Integrations + Invokers | P1 | `traigent/integrations/`, `traigent/invokers/`, `traigent/hybrid/`, `traigent/wrapper/` | Yes | Yes |
| Optimizers + Evaluators | P1 | `traigent/optimizers/`, `traigent/evaluators/`, `traigent/metrics/`, `traigent/tvl/` | Yes | Yes |
| Packaging + CI | P1 | `.github/workflows/`, `pyproject.toml`, `requirements/`, `Makefile` | Yes | Yes |
| Docs + Release Ops | P2 | `README.md`, `docs/`, `.release_review/` | Yes | Yes |

## Priority Source Waves

The source waves are the resumable execution unit for the expensive part of the review.
The exact file lists live under `.release_review/inventories/priority_wave*.txt`.

| Wave | Priority | Graph | Files | Approx lines |
|---|---:|---|---:|---:|
| 01 | P0 | Public API contract | 20 | 11,782 |
| 02 | P0 | Security/auth boundary | 22 | 9,554 |
| 03 | P0 | Config/runtime foundation | 17 | 5,813 |
| 04 | P0 | Utils foundation | 31 | 13,924 |
| 05 | P0 | Core orchestration/runtime | 23 | 13,410 |
| 06 | P1 | Core metrics/selection | 21 | 6,839 |
| 07 | P1 | Cloud auth/client surface | 16 | 11,054 |
| 08 | P1 | Cloud operations/data graph | 18 | 10,419 |
| 09 | P1 | Integrations provider graph | 44 | 12,276 |
| 10 | P1 | Execution/transport graph | 45 | 14,320 |
| 11 | P1 | Config generation/tuning | 34 | 11,015 |
| 12 | P1 | Optimizers | 20 | 8,901 |
| 13 | P1 | Evaluators + metrics | 13 | 8,744 |
| 14 | P2 | CLI/agents/experimental | 26 | 11,114 |
| 15 | P2 | Analytics/observability | 12 | 8,506 |

## Non-Negotiable Readiness Rule

- `READY` and `READY_WITH_ACCEPTED_RISKS` are forbidden unless every blocking component
  has approved `primary`, `secondary`, `tertiary`, and `reconciliation` evidence.
- `READY` and `READY_WITH_ACCEPTED_RISKS` are forbidden unless every path in
  `.release_review/inventories/source_files.txt` appears in `files_reviewed[]` for approved
  `primary`, `secondary`, `tertiary`, and `reconciliation` evidence.
- `READY` and `READY_WITH_ACCEPTED_RISKS` are forbidden unless every source wave inventory
  has been fully exhausted. A partial wave review is useful for triage, but it is not a
  readiness state.
- `READY` and `READY_WITH_ACCEPTED_RISKS` are forbidden unless every in-scope changed file
  also has per-file artifacts in `.release_review/runs/<release_id>/file_reviews/` for:
  - `primary` + `codex_cli`
  - `secondary` + `claude_cli`
  - `tertiary` + (`codex_cli` or `copilot_cli`)
  - `reconciliation` + `codex_cli`
- `READY` and `READY_WITH_ACCEPTED_RISKS` are forbidden unless required artifacts include:
  - `schema_version >= 2`
  - non-empty `review_summary` for each component role artifact
  - non-empty `checks_performed[]` for each component/file role artifact
  - non-empty `strengths[]` for each component/file role artifact
  - explanatory `notes` when a per-file artifact is approved with zero defects
- For P0/P1 components and waves, primary and secondary reviewers must be from different
  model families.
- Primary vs tertiary and secondary vs tertiary must not use the same exact model string.
- `gate_results/verdict.json` must have `failed_required_reviews: []` before any ready status.

## Traigent-Specific Hard Checks

Contract and runtime checks:

- Public API exports and type contracts remain backward-compatible.
- `@traigent.optimize(...)` config validation rejects invalid values at ingress.
- Core trial lifecycle preserves stop reasons and budget guardrails.
- Mock/offline mode is stable for release-critical test paths.

Async and integration checks:

- No blocking I/O in async-heavy execution, transport, or invoker paths.
- Integration adapters fail safely when optional deps are absent.
- Provider parameter normalization remains deterministic.
- Cloud auth/session helpers never silently widen privilege or tenancy boundaries.

## External Benchmarks

Use these for traceable rationale:

- NIST SSDF SP 800-218
- OWASP ASVS + OWASP API Security Top 10
- SLSA level progression (L2 target in post-launch hardening)
- GitHub code scanning/dependency review controls

## Acceptance Criteria

1. Gate runtime stays stable in strict mode for normal release-sized changes.
2. Two consecutive dry-run releases produce stable verdicts without manual artifact edits.
3. The full 15-wave source sweep is resumable without re-splitting files.
4. `python3 .release_review/automation/verify_source_wave_coverage.py` passes with no gaps
   or duplicates before a wave plan is considered current.
5. False-positive rate for blocking findings stays at or below 5% over the first 3 release cycles.
6. No new CI secrets are required.
7. Required checks are visible in branch or ruleset configuration.
8. Run artifacts are retained at least 90 days.
9. No release run reports `READY*` with non-empty `failed_required_reviews`.
