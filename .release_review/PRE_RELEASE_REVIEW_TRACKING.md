# Pre-Release Review Tracking (TraiGent SDK v0.8.0)

Update this file as reviews complete. Link evidence (PRs, issues, CI runs) so release readiness is auditable.

This tracking file is intended to be committed on a dedicated `release-review/<version>` branch so the whole team can see status and history. Agent notes/reports belong under `.release_review/artifacts/` (git-ignored).

## Roles

- Release captain: Claude Code (Opus 4.5)
- Human release owner (final sign-off): TBD
- Target release date: TBD
- Branch/tag: `release-review/v0.8.0` (baseline: `v0.8.0-rc1` @ 52baff0)

## Session Handoff Protocol

Before ending a session, the captain must:
1. Ensure all code changes are committed on their component branches (or merged).
2. Update this tracking file statuses for all in-flight components.
3. Append a timestamped entry to "Review Notes Log" summarizing progress and blockers.
4. Push the `release-review/<version>` branch so other reviewers can sync.

## Status Legend

- **Not started**
- **In progress**
- **Blocked**
- **Approved**

## Priority Scoring (Risk-Based Review Order)

Goal: review **high-impact + high-likelihood** areas first.

Each component gets a score based on:
- **L (Likelihood)**: how likely we are to find issues quickly (size, complexity, churn, weak tests, cross-cutting behavior).
- **S (Severity)**: damage if an issue ships (security, data loss, cost blowups, wrong optimization results, broken installs).
- **C (Centrality)**: how many users/flows are affected (how early and how often it's hit).

Scale: `1` (low) -> `5` (very high).

**Priority score (0-100)**:
`round((0.40*C + 0.35*S + 0.25*L) * 20)`

### Scoring Rubric (1-5)

| Score | Likelihood (L) | Severity (S) | Centrality (C) |
|-------|----------------|--------------|----------------|
| **5** | Active churn, complex logic, weak/spotty tests, cross-cutting behavior, recent major changes | Security/privacy breach, data loss/corruption, major cost blowup, broken install, silent wrong results | Core critical path hit by ~all users on every run |
| **4** | Moderate complexity with some test gaps, touched recently, has known edge cases | Significant incorrect behavior, noticeable perf degradation, confusing errors that block users | Common feature used by most users regularly |
| **3** | Moderate complexity, reasonable test coverage, occasional regressions in history | Wrong/unclear behavior in some scenarios, degraded results, recoverable errors | Standard feature used by many users |
| **2** | Low complexity, good test coverage, stable history | Minor UX issues, cosmetic bugs, edge case failures with workarounds | Less common feature or optional path |
| **1** | Simple, well-tested, rarely changed | Trivial cosmetic issues, internal-only impact, no user-visible effect | Niche/rare path, internal tooling only |

### Priority Tiers

- **P0 (90-100)**: Review immediately, release blocker potential
- **P1 (75-89)**: Review early, high-value fixes
- **P2 (60-74)**: Standard review priority
- **P3 (<60)**: Review if time permits, lower risk

## SDK Runtime (`traigent/`)

| Component | Priority | L/S/C | Scope | Owner | Approver | Status | Review | Tests | Docs | Evidence |
|-----------|----------|-------|-------|-------|----------|--------|--------|-------|------|----------|
| Integrations (LLMs/frameworks/vector stores/observability) | 100 | 5/5/5 | `traigent/integrations/` | Agent (thorough) | Claude (Captain) | **Approved** | [x] | [x] | [x] | Tests: 1197/1197 passed | Linting: clean | Types: clean | No blocking issues. Minor P2/P3 DRY suggestions for post-release. SHIP AS-IS. |
| Configuration & injection | 95 | 4/5/5 | `traigent/config/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Core orchestration | 95 | 4/5/5 | `traigent/core/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Optimizers | 95 | 4/5/5 | `traigent/optimizers/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Invokers | 95 | 4/5/5 | `traigent/invokers/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Storage & persistence | 88 | 4/4/5 | `traigent/storage/`, `traigent/utils/persistence.py` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Utilities | 88 | 4/4/5 | `traigent/utils/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Evaluators | 88 | 4/4/5 | `traigent/evaluators/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| `traigent/__init__.py` + public exports | 83 | 3/4/5 | `traigent/__init__.py`, `traigent/api/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Security & privacy | 79 | 4/5/3 | `traigent/security/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Metrics | 75 | 3/4/4 | `traigent/metrics/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| CLI surface | 68 | 3/3/4 | `traigent/cli/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Cloud/hybrid client code | 65 | 4/3/3 | `traigent/cloud/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Agents | 60 | 3/3/3 | `traigent/agents/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| OptiGen integration entry | 60 | 3/3/3 | `traigent/optigen_integration.py` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Execution adapters | 55 | 2/3/3 | `traigent/adapters/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Hooks | 52 | 3/3/2 | `traigent/hooks/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Analytics (experimental helpers) | 45 | 3/2/2 | `traigent/analytics/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Telemetry | 40 | 2/2/2 | `traigent/telemetry/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Visualization | 40 | 2/2/2 | `traigent/visualization/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| TVL helpers | 40 | 2/2/2 | `traigent/tvl/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Plugin registry (non-integrations) | 40 | 2/2/2 | `traigent/plugins/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Experimental sandbox | 37 | 3/2/1 | `traigent/experimental/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |

## User-Facing Surfaces

| Component | Priority | L/S/C | Scope | Owner | Approver | Status | Review | Tests | Docs | Evidence |
|-----------|----------|-------|-------|-------|----------|--------|--------|-------|------|----------|
| Main docs | 83 | 3/4/5 | `README.md`, `docs/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Quickstart examples | 83 | 3/4/5 | `examples/quickstart/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Examples (all) | 68 | 3/3/4 | `examples/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Walkthrough | 55 | 2/3/3 | `walkthrough/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Use-cases | 55 | 2/3/3 | `use-cases/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Playground (Streamlit) | 45 | 3/2/2 | `playground/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |

## Release Engineering / Tooling

| Component | Priority | L/S/C | Scope | Owner | Approver | Status | Review | Tests | Docs | Evidence |
|-----------|----------|-------|-------|-------|----------|--------|--------|-------|------|----------|
| Release blockers | 100 | 5/5/5 | `RELEASE_BLOCKERS_TODO.md` | Claude (Opus 4.5) | Codex (GPT-5.2) | **Review done** | [x] | [x] | [x] | Commits: 17ac9ea | Tests: `TRAIGENT_MOCK_MODE=true pytest tests/security/ tests/unit/test_security_fixes_simple.py` -> PASS (97/97) | Model: Claude/Opus4.5 | Timestamp: 2025-12-13T19:45:00Z | All Critical+High items verified fixed. Accepted risk: in-memory token revocation (SDK use case) |
| Packaging + deps | 90 | 3/5/5 | `pyproject.toml`, `requirements/`, `uv.lock`, `MANIFEST.in` | Agent | Claude (Captain) | **Approved** | [x] | [x] | [x] | Commits: b06b2bc | Fixed: typing-extensions marker, missing security deps, missing NOTICE in MANIFEST.in | CLI verified: `traigent --version` -> 0.8.0 |
| CI workflows | 75 | 3/4/4 | `.github/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Test suite health | 75 | 3/4/4 | `tests/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Scripts (setup/test/analysis) | 45 | 3/2/2 | `scripts/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |
| Tools (code review/traceability) | 40 | 2/2/2 | `tools/` | TBD | TBD | Not started | [ ] | [ ] | [ ] | |

## Review Notes Log (append-only)

- 2025-12-13T19:20:00Z: **Session start** — Captain (Claude Code/Opus 4.5) initialized release-review/v0.8.0 branch from main@52baff0. Tagged baseline as v0.8.0-rc1. Phase 0 hardening in progress.
- 2025-12-13T19:35:00Z: **Phase 1 start** — Dispatching first 3 parallel reviews: (1) Release blockers [Claude lead], (2) Integrations [Codex lead], (3) Packaging [Copilot lead]. Cross-model review policy active.
- 2025-12-13T19:50:00Z: **Phase 1 batch 1 complete** — All 3 P0 components reviewed:
  - Release blockers: APPROVED (97/97 security tests pass, all Critical/High fixed)
  - Integrations: APPROVED (1197/1197 tests pass, no blocking issues)
  - Packaging: APPROVED (4 fixes committed in b06b2bc, CLI verified working)
  - Next: Continue with P1 components (Config, Core, Optimizers, Invokers)
