# Captain Protocol v3: Release Review Gate-First Orchestration

**Protocol Version**: 3
**Runtime**: `python3` only
**Canonical Root**: `.release_review/runs/<release_id>/`

This protocol defines the release captain workflow for major release review.

## Canonical Inputs

- Plan: `.release_review/PRE_RELEASE_REVIEW_PLAN.md`
- Source-wave plan: `.release_review/PRIORITY_REVIEW_WAVES.md`
- Source-wave contract: `.release_review/components.yml`
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
│   ├── tests_files.txt
│   └── review_scope_files.txt
├── components/
│   └── <component evidence JSON files>
├── file_reviews/
│   └── <component>/<review_type>/<agent_type>/<repo_file>.json
└── waivers/
```

The run-local `src_files.txt` naming is legacy. In this repository, the shipped source tree being
reviewed lives under `traigent/` and `traigent_validation/`, and the static staged source-wave
inventories live under `.release_review/inventories/`.

## Review Modes

- `strict`
  - full release-readiness review
  - requires component evidence and the full per-file peer matrix
  - requires all four angles: `security_authz`, `correctness_regression`,
    `async_concurrency_performance`, `dto_api_contract`
- `quick`
  - incremental fast-pass mode
  - single-model lane only, configured in `.release_review/scope.yml`
  - reduced angle set from `.release_review/scope.yml`
  - may reuse prior per-file artifacts when the file is unchanged since that artifact's commit
  - does not replace a final strict sign-off

## Hard Readiness Rule (Non-Negotiable)

- `verdict.json` must remain `NOT_READY` until peer-review completeness is satisfied.
- In `strict` mode, required completeness for every component in the component matrix:
  - `primary` evidence exists and latest decision is `approved`
  - `secondary` evidence exists and latest decision is `approved`
  - `tertiary` evidence exists and latest decision is `approved`
  - `reconciliation` evidence exists and latest decision is `approved`
- In `strict` mode, required completeness for every in-scope changed file:
  - file appears in `files_reviewed[]` for latest `primary` evidence of its component
  - file appears in `files_reviewed[]` for latest `secondary` evidence of its component
  - file appears in `files_reviewed[]` for latest `tertiary` evidence of its component
  - file appears in `files_reviewed[]` for latest `reconciliation` evidence of its component
- Required per-file artifact matrix for every in-scope changed file is mode-specific and defined in `.release_review/scope.yml`.
- Prior per-file artifacts may satisfy a requirement when:
  - the required `review_type + angle + file` coverage exists
  - the file is unchanged since that artifact's `commit_sha`
  - the file was not forced back into review with `--force-rereview`
- Substance requirement for every required role and file artifact:
  - component evidence includes non-empty `review_summary` (>=50 chars)
  - component evidence includes `checks_performed[]` (>=1)
  - component evidence includes `strengths[]` (>=1 positive finding)
  - approved per-file artifact includes `checks_performed[]` + `strengths[]`
  - approved per-file artifact with no defects must include explanatory `notes`
- For P0/P1 components, primary and secondary reviewers must be from different model families.
- Primary and tertiary reviewers must not use the same exact model string.
- Secondary and tertiary reviewers must not use the same exact model string.
- `failed_required_reviews` in `verdict.json` must be empty before `READY` or `READY_WITH_ACCEPTED_RISKS`.

## Preconditions

Before starting:

1. Confirm tooling: `git`, `python3`, `pytest`, `ruff`, `mypy`, `bandit`, `pip-audit`.
2. Confirm branch target and release ID.
3. Confirm run scope from `.release_review/scope.yml`.
4. Confirm the staged source inventory from `.release_review/inventories/source_files.txt`.
5. Confirm no unknown local changes outside intended scope.

## Trigger Model

- Blocking release gate trigger: push tag `v*`.
- Dry run trigger: `workflow_dispatch`.
- Continuous dry run trigger: push to `develop`.
- Optional PR dry run to `main`: non-blocking until hardened.

## Model Runtime Policy (CLI)

Use this assignment unless explicitly overridden by release owner:

- Captain/orchestrator: `Codex CLI 5.3` with effort `xhigh`
- Primary reviewers: `Codex CLI 5.3` with effort `high`
- Secondary/adversarial reviewers: `Claude CLI Opus 4.6` in `extended` mode
- Tertiary independent reviewers: `Codex CLI 5.3` (independent pass, separate prompt/angle)
- Optional fourth model family: `Copilot CLI` configured to `Gemini 3.1 Pro` for tertiary role when available
- If Copilot/Gemini is unavailable, use Codex 5.3 for tertiary while keeping secondary on Opus 4.6.

Evidence for each component must include the actual model string used.

## Captain Loop

### Step 1: Initialize run workspace

```bash
python3 .release_review/automation/generate_tracking.py \
  --version <VERSION> \
  --release-id <RELEASE_ID> \
  --base-branch main \
  --review-mode <strict|quick>
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
- In `strict` mode, every matrix component requires primary + secondary + tertiary + reconciliation evidence.
- In `strict` mode, every in-scope changed file requires primary + secondary + tertiary + reconciliation coverage.
- In `strict` mode, P0/P1 primary and secondary must use different model families.
- Captain is the only writer for tracking board and final verdict.
- Reviewer assignment follows the Model Runtime Policy.
- Use `inventories/review_pending_files.txt` as the work queue and treat `inventories/review_skipped_files.txt` as reused coverage unless the owner explicitly forces a re-review.

## Cost-Controlled Source-Wave Mode

- For the expensive source review pass, use `.release_review/inventories/source_files.txt` as the
  canonical inventory and `.release_review/inventories/priority_wave*.txt` as the resumable
  execution units.
- This repository has no repo-root `src/`; the source roots are `traigent/` and
  `traigent_validation/`.
- Wave order and rationale are defined in `.release_review/PRIORITY_REVIEW_WAVES.md`.
- `READY` is forbidden while any source wave is incomplete, even if the highest-priority waves
  are already reviewed.
- `quick` mode may reduce churn between strict runs, but it does not waive the requirement for a full strict source-wave pass before release sign-off.
- If a run stops mid-review, resume from the first incomplete wave rather than inventing a new
  batch split.
- Before locking a new wave plan, run:

```bash
python3 .release_review/automation/verify_source_wave_coverage.py
```

### Step 5: Reconcile and approve

- Validate evidence file schema (`evidence_validator.py`).
- Validate scope (`scope_guard.py`).
- Validate test claims (`verify_tests.sh`) on sampled components.
- Update tracking board rows.
- Rebuild verdict and confirm `failed_required_reviews` is empty.

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
- `.release_review/automation/schemas/file_review_artifact.schema.json`

Minimum fields:
- `schema_version` (>=2)
- `component`
- `review_type`
- `agent_type`
- `reviewer_model`
- `commit_sha`
- `files_reviewed[]`
- `findings[]`
- `strengths[]` (positive findings)
- `checks_performed[]`
- `tests[]`
- `review_summary`
- `decision`
- `timestamp_utc`

Per-file review artifact minimum fields:
- `schema_version` (>=2)
- `component`
- `review_type`
- `agent_type`
- `reviewer_model`
- `file` (repository-relative path)
- `angles_reviewed[]`
- `commit_sha`
- `notes`
- `findings[]`
- `strengths[]` (positive findings)
- `checks_performed[]`
- `decision`
- `timestamp_utc`

Template:
- `.release_review/templates/FILE_REVIEW_ARTIFACT.json`

## Prompt Matrix (Required)

Prompt templates are mandatory per `agent_type + review_type` lane:

- `.release_review/prompts/codex_cli__captain.md`
- `.release_review/prompts/codex_cli__primary.md`
- `.release_review/prompts/claude_cli__secondary.md`
- `.release_review/prompts/codex_cli__tertiary.md`
- `.release_review/prompts/copilot_cli__tertiary.md`
- `.release_review/prompts/codex_cli__reconciliation.md`

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
2. Tracking board is complete for required components and per-file coverage, and
3. Required CI checks are green (or explicitly waived with valid artifact), and
4. `failed_required_reviews` in `verdict.json` is empty.
