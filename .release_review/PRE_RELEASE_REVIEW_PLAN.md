# Pre-Release Review Plan v2 (Gate-First)

This plan defines what is reviewed for Traigent SDK and what blocks release.

## Scope

In scope:

- SDK code shipped from `traigent/` and `traigent_validation/`
- Release engineering and CI workflows
- Security, dependency, and test gates
- Docs required for release operation (`README.md`, release docs)

Out of scope for initial hard gate:

- Prompt quality scoring as a blocker
- SLSA L3 controls (post-launch milestone)

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

## Component Review Matrix

| Component | Priority | Scope | Dual Review | Blocking |
|---|---:|---|---|---|
| Public API + Safety | P0 | `traigent/api/`, `traigent/security/` | Yes | Yes |
| Core Orchestration + Config | P0 | `traigent/core/`, `traigent/config/` | Yes | Yes |
| Integrations + Invokers | P1 | `traigent/integrations/`, `traigent/invokers/` | Yes | Yes |
| Optimizers + Evaluators | P1 | `traigent/optimizers/`, `traigent/evaluators/`, `traigent/metrics/` | Yes | Yes |
| Packaging + CI | P1 | `.github/workflows/`, `pyproject.toml`, `requirements/`, `Makefile` | Recommended | Yes |
| Docs + Release Ops | P2 | `README.md`, `docs/`, `.release_review/` | No | No |

## Traigent-Specific Hard Checks

Contract and runtime checks:

- Public API exports and type contracts remain backward-compatible.
- `@traigent.optimize(...)` config validation rejects invalid values at ingress.
- Core trial lifecycle preserves stop reasons and budget guardrails.
- Mock/offline mode is stable for release-critical test paths.

Async and integration checks:

- No blocking I/O in async-heavy execution/invoker paths.
- Integration adapters fail safely when optional deps are absent.
- Provider parameter normalization remains deterministic.

## External Benchmarks

Use these for traceable rationale:

- NIST SSDF SP 800-218
- OWASP ASVS + OWASP API Security Top 10
- SLSA level progression (L2 target in post-launch hardening)
- GitHub code scanning/dependency review controls

## Acceptance Criteria

1. Gate runtime <= 15 minutes on normal release-sized changes.
2. Two consecutive dry-run releases produce stable verdicts without manual artifact edits.
3. False-positive rate for blocking findings <= 5% over first 3 release cycles.
4. No new CI secrets required.
5. Required checks visible in branch/ruleset configuration.
6. Run artifacts retained at least 90 days.
