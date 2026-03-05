# Captain Protocol v2: Release Review Gate-First Orchestration

**Protocol Version**: 2
**Runtime**: `python3` only
**Canonical Root**: `.release_review/runs/<release_id>/`

This protocol defines the release captain workflow for major release review.

## Canonical Inputs

- Plan: `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- Severity + waiver policy: `.release_review/SEVERITY_POLICY.md`
- Scope policy: `.release_review/scope.yml`
- Tracking board: `.release_review/PRE_RELEASE_REVIEW_TRACKING.md`
- Automation toolkit: `.release_review/automation/`

## Run Contract (Required)

Every run MUST use this structure:

```text
.release_review/runs/<release_id>/
├── run_manifest.json
├── REVIEW_LOG.md
├── gate_results/
│   ├── check_results.json
│   └── verdict.json
├── inventories/
│   ├── src_files.txt
│   └── tests_files.txt
└── waivers/
```

## Preconditions

Before starting:

1. Confirm tooling: `git`, `python3`, `pytest`, `ruff`, `mypy`, `bandit`, `pip-audit`.
2. Confirm branch target and release ID.
3. Confirm run scope from `.release_review/scope.yml`.
4. Confirm no unknown local changes outside intended scope.

## Trigger Model

- Blocking release gate trigger: push tag `v*`.
- Dry run trigger: `workflow_dispatch`.
- Optional PR dry run to `main`: non-blocking until hardened.

## Model Runtime Policy (CLI)

Use this assignment unless explicitly overridden by release owner:

- Captain/orchestrator: `Codex CLI 5.3` with effort `xhigh`
- Primary reviewers: `Codex CLI 5.3` with effort `high`
- Secondary/adversarial reviewers: `Claude CLI Opus 4.6` in `extended` mode
- Optional tertiary spot-check: `Copilot CLI` configured to `Gemini 3.1 Pro`
- If Copilot/Gemini is unavailable, run dual-review with Codex 5.3 + Opus 4.6 only.

Evidence for each component must include the actual model string used.

## Captain Loop

### Step 1: Initialize run workspace

```bash
python3 .release_review/automation/generate_tracking.py \
  --version <VERSION> \
  --release-id <RELEASE_ID> \
  --base-branch main
```

### Step 2: Execute baseline release gate bundle

```bash
python3 .release_review/automation/release_gate_runner.py \
  --release-id <RELEASE_ID> \
  --strict
```

### Step 3: Create or refresh verdict

```bash
python3 .release_review/automation/build_release_verdict.py \
  --release-id <RELEASE_ID>
```

### Step 4: Component review orchestration

- Max 3 concurrent component reviewers.
- Non-overlapping file scopes.
- P0/P1 requires dual review (different model families).
- Captain is the only writer for tracking board and final verdict.
- Reviewer assignment follows the Model Runtime Policy.

### Step 5: Reconcile and approve

- Validate evidence file schema (`evidence_validator.py`).
- Validate scope (`scope_guard.py`).
- Validate test claims (`verify_tests.sh`) on sampled components.
- Update tracking board rows.

## Required CI Check Names

The repository must expose these checks (ruleset/branch protection configuration is admin-owned):

- `release-gate/lint-type`
- `release-gate/tests-unit`
- `release-gate/tests-integration`
- `release-gate/security`
- `release-gate/dependency-review`
- `release-gate/codeql`
- `release-gate/release-review-consistency`

## Compatibility Matrix (Existing CI)

| Existing workflow | v2 treatment | Notes |
|---|---|---|
| `.github/workflows/release-review.yml` | **Authoritative gate** | Tag-triggered blocking path; dry-run on PR/dispatch |
| `.github/workflows/quality.yml` | **Supplemental quality signal** | Keep broad quality checks; release gating is in `release-review.yml` |
| `.github/workflows/tests.yml` | **Supplemental coverage signal** | Keep matrix breadth; release gate uses deterministic unit/integration set |
| `.github/workflows/auto-tune-secure.yml` | **Supplemental hardening** | Not authoritative for release verdict |
| `.github/workflows/traigent-ci-gates.yml` | **Product regression signal** | Non-authoritative for release verdict |

## Evidence Requirements

Evidence must be JSON files validated against:

- `.release_review/automation/schemas/evidence.schema.json`

Minimum fields:
- `component`
- `review_type`
- `reviewer_model`
- `commit_sha`
- `findings[]`
- `tests[]`
- `decision`
- `timestamp_utc`

## Escalation and Waivers

- P0 unresolved findings: block release unless approved waiver within 24h.
- P1 unresolved findings: block release unless approved waiver within 72h.
- Waiver requires 2 maintainers + explicit expiry.
- Waivers are stored under `.release_review/runs/<release_id>/waivers/`.

## Emergency Override (Break-Glass)

Allowed only through CI workflow path:

- Requires protected environment approvals by 2 CODEOWNERS.
- Requires `incident_id`, `reason`, `expires_at`.
- Automatically writes waiver artifact.
- Automatically creates remediation issue due in 48h.
- Never bypasses unresolved P0 security findings.

## Stop Conditions

Captain stops only when:

1. Verdict status is `READY` or `READY_WITH_ACCEPTED_RISKS`, and
2. Tracking board is complete for required components, and
3. Required CI checks are green (or explicitly waived with valid artifact).
