# Start Release Review (v3)

Copy into your reviewer session:

```text
You are the release-review captain.

Read and follow:
- .release_review/CAPTAIN_PROTOCOL.md
- .release_review/PRE_RELEASE_REVIEW_PLAN.md
- .release_review/PRIORITY_REVIEW_WAVES.md
- .release_review/components.yml
- .release_review/SEVERITY_POLICY.md

Task:
1) Initialize run workspace
2) Execute release gates
3) Execute the staged source-wave review for `traigent/**/*.py` and `traigent_validation/**/*.py`
   using `.release_review/inventories/source_files.txt` and `.release_review/inventories/priority_wave*.txt`
4) Complete the remaining non-source review scope from `.release_review/scope.yml`
5) Rebuild verdict until `failed_required_reviews` is empty and verdict is READY or READY_WITH_ACCEPTED_RISKS

Inputs:
- VERSION: <VERSION>
- RELEASE_ID: <VERSION>-<YYYYMMDD_HHMMSS>

Commands:
python3 .release_review/automation/generate_tracking.py --version <VERSION> --release-id <RELEASE_ID> --base-branch main --review-mode strict
python3 .release_review/automation/release_gate_runner.py --release-id <RELEASE_ID> --strict
python3 .release_review/automation/build_release_verdict.py --release-id <RELEASE_ID>

Rules:
- Continue autonomously; do not pause between components or source waves
- Use `strict` for final release sign-off. Use `quick` only for incremental fast passes.
- Never assume a repo-root `src/`; the source review roots in this repo are `traigent/` and `traigent_validation/`
- Never stop early while any required component lacks peer-review evidence
- Never stop early while any file in `.release_review/inventories/source_files.txt` lacks primary+secondary+tertiary+reconciliation coverage in `files_reviewed[]`
- Never stop early while any source wave inventory under `.release_review/inventories/priority_wave*.txt` remains partially reviewed
- Never stop early while any in-scope changed file lacks required per-file artifacts in `file_reviews/` for primary/codex, secondary/claude, tertiary/(codex or copilot), and reconciliation/codex
- Never stop early while required artifacts are missing `review_summary`, `checks_performed[]`, `strengths[]`, or clean-approval explanatory `notes`
- Use 4-angle review for all components; enforce cross-family primary/secondary for P0/P1
- When `review_pending_files.txt` and `review_skipped_files.txt` exist, review only the pending set unless the owner explicitly forces a re-review.
- Use model policy: Codex CLI 5.3 xhigh (captain), Codex CLI 5.3 high (primary), Claude Opus 4.6 extended (secondary), Codex CLI 5.3 high/medium (tertiary), optional Copilot Gemini 3.1 Pro for tertiary
- Use prompt matrix files under `.release_review/prompts/` (`<agent_type>__<review_type>.md`)
- Keep all artifacts under .release_review/runs/<release_id>/ including `file_reviews/`
- Only captain updates PRE_RELEASE_REVIEW_TRACKING.md
```

## Monitor Progress

```bash
cat .release_review/PRE_RELEASE_REVIEW_TRACKING.md
cat .release_review/runs/<release_id>/gate_results/verdict.json
cat .release_review/runs/<release_id>/REVIEW_LOG.md
jq '.failed_required_reviews' .release_review/runs/<release_id>/gate_results/verdict.json
```

## Strict CI Mode

For release branch or tag CI:

```bash
python3 .release_review/automation/release_gate_runner.py --release-id <release_id> --strict
```

## Quick Local Mode

For a fast incremental sweep between strict runs:

```bash
python3 .release_review/automation/generate_tracking.py --version <VERSION> --release-id <RELEASE_ID> --base-branch main --review-mode quick
python3 .release_review/automation/build_release_verdict.py --release-id <RELEASE_ID>
```
