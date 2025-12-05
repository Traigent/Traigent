# LLM-Driven Module Review (sonnet4.5)

This document is the concise protocol for an LLM to review a given module or folder using the four tracks provided in this toolkit. Use together with INSTRUCTIONS.md for step-by-step execution and self-validation.

## Review Tracks
- Code Quality: style, readability, maintainability
- Soundness/Correctness: invariants, error handling, validation, tests
- Performance: hot paths, memory, I/O, batching, concurrency
- Security: sanitization, secrets, crypto, logging, path/SSRF

## Output Contract
- For each track and module, write a JSON report to `reports/1_quality/automated_reviews/<track>/<module>.review.json` (from the repo root).
- Conform to the template in `<track>/templates/review_template.json`.
- Cover every top-level function and class method in `functions[]`.
- Include all checks listed in `<track>/required_checks.json` in `checks[]`.

## Self-Validation
1. Build function inventory (single module):
   ```bash
   python tools/code_review/automation/_shared/function_inventory.py --module <module> --json
   ```
2. Ensure all listed functions appear in your `functions[]` section.
3. Ensure each required check appears in your `checks[]` section.
4. Save your JSON under `reports/1_quality/automated_reviews/<track>/...` and run the validator:
   ```bash
   python tools/code_review/automation/<track>/validate_module.py \
     --module <module> \
     --report reports/1_quality/automated_reviews/<track>/<module>.review.json
   ```
5. For folders, after producing all per-module reports, run:
   ```bash
   python tools/code_review/automation/run_all_validations.py --folder <folder>
   ```

## Suggested Workflow for a Module
1. Read the source file.
2. Extract function inventory (shared script).
3. Perform per-track analysis following `<track>/instructions.md`.
4. Emit JSON report and validate. Iterate until validation passes.
5. Repeat for all tracks.

## Passing Criteria
- Validators return 0 for all tracks.
- No missing functions.
- No missing required checks.

## Notes
- Keep evidence concise but sufficient to trace observations.
- If uncertain, use status `needs_followup` with rationale.
- Do not change code; this is a review pass. Create remediation tickets in `issues[]` with priorities when appropriate.
