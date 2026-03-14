# Review Log

## Run Metadata

- Release ID: `<release_id>`
- Version: `<version>`
- Baseline SHA: `<sha>`
- Review mode:
- Started At (UTC): `<timestamp>`

## Gate Execution Notes

- lint-type:
- tests-unit:
- tests-integration:
- security:
- dependency-review:
- codeql:
- release-review-consistency:

## Component Notes

### <component>
- Primary reviewer (model):
- Secondary reviewer (model):
- Tertiary reviewer (model):
- Reconciliation reviewer (model):
- Findings:
- Evidence files (primary/secondary/tertiary/reconciliation):
- Decision:

#### Completion Checklist
- [ ] primary decision is approved
- [ ] secondary decision is approved
- [ ] tertiary decision is approved
- [ ] reconciliation decision is approved
- [ ] every in-scope changed file is listed in `files_reviewed[]` for primary/secondary/tertiary/reconciliation
- [ ] every in-scope changed file has per-file artifacts in `file_reviews/` for primary/codex, secondary/claude, tertiary/(codex or copilot), reconciliation/codex
- [ ] `review_pending_files.txt` was fully reviewed, or explicit forced re-review instructions were followed
- [ ] reused artifacts are documented in `inventories/reused_file_review_artifacts.json`
- [ ] each role artifact includes `review_summary` + `checks_performed[]` + `strengths[]` (positive findings)
- [ ] each approved per-file artifact with zero defects includes explanatory `notes`
- [ ] for P0/P1: primary/secondary model families are different
- [ ] evidence commit SHA matches run baseline SHA

## Waivers

- None

## Final Recommendation

- `NOT_READY` / `READY_WITH_ACCEPTED_RISKS` / `READY`
- `failed_required_reviews` count: `<n>`
