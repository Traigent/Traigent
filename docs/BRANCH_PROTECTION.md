# Branch Protection

Repository settings are not managed from code, but `main` should be protected with these required checks:

- `CI / lint-and-typecheck`
- `CI / test (node 18)`
- `CI / test (node 20)`
- `CI / test (node 22)`
- `CI / build (node 18)`
- `CI / build (node 20)`
- `CI / build (node 22)`
- `CI / package-smoke (node 18)`
- `CI / package-smoke (node 20)`
- `CI / package-smoke (node 22)`
- `Changeset Required / changeset-required`
- `CodeQL / Analyze`

The nightly/manual `Hybrid Backend Smoke` workflow is valuable operational coverage, but it is not a required branch-protection check because it depends on external backend secrets and infrastructure.

Recommended additional settings:

- require pull request before merging
- require at least 1 approving review
- dismiss stale approvals on new commits
- require branches to be up to date before merging
- restrict direct pushes to `main`
- require CODEOWNERS review if a `CODEOWNERS` file is later added
