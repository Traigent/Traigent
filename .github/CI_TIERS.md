# CI tiers

Canonical policy: `traigent-validation-spine/ops/_validation/policies/ci_tiers.md`.

This SDK already keeps most deep checks manual or scheduled. Keep that shape:

- T0/T1: release PRs to `main` and package/install-sensitive changes.
- T2: cross-SDK parity and Sonar on release PRs or manual dispatch.
- T3: weekly parity and any long-running smoke coverage.

When adding a required check, use a stable aggregator job name and put change detection inside
the workflow. Do not use top-level `paths:` filters on a branch-protection-required workflow.
