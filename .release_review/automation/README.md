# Release Review Automation Toolkit (v2)

All commands use `python3`.

## Review Modes

- `strict`: full release-readiness review with all peer lanes and all four angles.
- `quick`: incremental single-model sweep with a reduced angle set for fast follow-up passes.

Artifact reuse applies in both modes when enabled by `.release_review/scope.yml`:
- previous `file_reviews/` artifacts are searched automatically
- unchanged files with matching `(review_type, angle, file)` coverage are skipped by default
- use `--force-rereview <glob>` to override reuse

## Primary Commands

1. Initialize run + tracking

```bash
python3 .release_review/automation/generate_tracking.py \
  --version v1.0.0 \
  --release-id v1.0.0-20260305_120000 \
  --base-branch main \
  --review-mode strict
```

2. Run release gate checks

```bash
python3 .release_review/automation/release_gate_runner.py \
  --release-id v1.0.0-20260305_120000 \
  --strict
```

3. Build canonical verdict

```bash
python3 .release_review/automation/build_release_verdict.py \
  --release-id v1.0.0-20260305_120000
```

`build_release_verdict.py` is the hard readiness gate:
- blocks READY when required checks fail
- blocks READY when `failed_required_reviews` is non-empty
- in `strict` mode, blocks READY when any in-scope changed file lacks primary/secondary/tertiary/reconciliation `files_reviewed[]` coverage
- blocks when any in-scope changed file lacks the selected mode's required per-file role/angle coverage
- accepts prior per-file artifacts when the file is unchanged since that artifact's commit and reuse is enabled
- blocks READY when artifacts are low-substance (missing `review_summary`, `checks_performed[]`, `strengths[]`, or clean-approval `notes`)

Wave preflight:

```bash
python3 .release_review/automation/verify_source_wave_coverage.py
```

This verifies that `.release_review/inventories/source_files.txt` matches the current
`traigent/**/*.py` and `traigent_validation/**/*.py` tree and that the priority wave inventories
cover every source file exactly once.

## Script Catalog

| Script | Purpose |
|---|---|
| `generate_tracking.py` | Initialize run workspace + slim tracking board |
| `release_gate_runner.py` | Execute gate checks + persist check results |
| `build_release_verdict.py` | Emit canonical `verdict.json` and enforce peer-review completeness |
| `verify_source_wave_coverage.py` | Verify source-wave inventories cover every source file exactly once |
| `evidence_validator.py` | Validate evidence JSON against schema and git state |
| `scope_guard.py` | Enforce in-scope file modifications |
| `verify_tests.sh` | Re-run tests and verify claimed results via pytest JSON report |
| `rotation_scheduler.py` | Deterministic model-role scheduling |
| `validate_release_review_consistency.py` | Detect deprecated paths/commands in release-review docs |

## Contracts

- Scope config: `.release_review/scope.yml`
- Evidence schema: `.release_review/automation/schemas/evidence.schema.json`
- File-review artifact schema: `.release_review/automation/schemas/file_review_artifact.schema.json`
- Run root: `.release_review/runs/<release_id>/`
- Per-file scope inventory: `.release_review/runs/<release_id>/inventories/review_scope_files.txt`
- Pending review inventory: `.release_review/runs/<release_id>/inventories/review_pending_files.txt`
- Skipped/reused inventory: `.release_review/runs/<release_id>/inventories/review_skipped_files.txt`
- Per-file artifacts root: `.release_review/runs/<release_id>/file_reviews/`
- Per-file artifact template: `.release_review/templates/FILE_REVIEW_ARTIFACT.json`
- Static source inventory: `.release_review/inventories/source_files.txt`
- Static wave inventories: `.release_review/inventories/priority_wave*.txt`

## Model Assignment Policy

- Captain/orchestrator: Codex CLI 5.3 xhigh.
- Primary reviewer tasks: Codex CLI 5.3 high.
- Secondary/adversarial tasks: Claude CLI Opus 4.6 extended.
- Tertiary independent tasks: Codex CLI 5.3 high/medium.
- Optional tertiary model family: Copilot CLI Gemini 3.1 Pro.

## Prompt Matrix

Use one prompt file per `agent_type + review_type` lane:

- `.release_review/prompts/codex_cli__captain.md`
- `.release_review/prompts/codex_cli__primary.md`
- `.release_review/prompts/claude_cli__secondary.md`
- `.release_review/prompts/codex_cli__tertiary.md`
- `.release_review/prompts/copilot_cli__tertiary.md`
- `.release_review/prompts/codex_cli__reconciliation.md`
