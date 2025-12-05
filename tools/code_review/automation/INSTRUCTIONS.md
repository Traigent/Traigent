Master Instructions for LLM Reviewer (sonnet4.5)

Goal
Conduct a comprehensive module review across four tracks: code-quality, soundness/correctness, performance, and security. Produce one JSON report per track per module. Use the per-track instructions to guide analyses. When asked to review a folder, iterate over all Python files, reusing the same module procedure.

Scope
- Input will specify either:
  - A single module path, e.g., `traigent/core/cache_policy.py`, or
  - A folder path, e.g., `traigent/core` (review all `*.py` files inside, excluding test files and __init__.py by default)

Output Requirements
- For each module and track, write a JSON report file under: `reports/1_quality/automated_reviews/<track>/<relative-module-path>.review.json` (from the repo root).
- The JSON must conform to the schema in `templates/review_template.json` inside each track folder.
- The `classes` array must list every class present in the module.
- The `functions` array must list every top-level function and every class method (fully qualified as `ClassName.method`) present in the module.
- Each `functions[]` entry requires: `name`, `status` (ok|issue|needs_followup), and `notes` (>= 5 chars).
- Each `checks[]` entry requires: `name`, `result` (pass|fail|needs_followup), and `evidence` (>= 10 chars, ideally with `path:line`).
- Include `issues` (list with `id`, `title`, optional `severity` = low|medium|high|critical, optional `scope`) and `recommendations` (list).
- The `checks` array must include at least the checks listed in each track’s `required_checks.json`.

Self-Validation Flow
1) Use the shared function inventory script to gather the function list:
   - `python tools/code_review/automation/_shared/function_inventory.py --module <module.py> --json`
2) Ensure your report includes all discovered functions. If any are missing, add an entry with at least a short status and notes.
3) Ensure your report contains all required checks listed in the track’s `required_checks.json`.
4) Save the report JSON to the `reports/1_quality/automated_reviews/<track>/...` path.
5) Trigger the validator for the module/track:
   - `python tools/code_review/automation/<track>/validate_module.py --module <module.py> --report reports/1_quality/automated_reviews/<track>/<module>.review.json`
6) When reviewing a folder, first enumerate modules:
   - `python tools/code_review/automation/_shared/list_modules.py --folder <folder>`
   Maintain a checklist and only finish when each module has a report and passes validation.

Anchored evidence & syntax sanity
- When claiming a syntax error, confirm by parsing the module (we already parse for inventory). Do not report syntax errors if the module parses successfully.
- Include `path:line` anchors in evidence when practical; the validator will check that referenced line numbers exist in the target module.
- Before calling out a runtime bug (e.g., division by zero, None dereference), trace the relevant guards or conditionals in the code. Only escalate severity if the failing path is reachable.

Evidence expectations
- Every `issues[]` entry must include an `evidence` field (>= 40 chars) that cites concrete anchors such as `module.py:line`.
- High/critical severities must also explain observed or theoretical magnitude (e.g., `O(n^2)` on a loop where `n` can exceed 10k, or blocking I/O executed once per request). If you cannot demonstrate impact, downgrade the severity or mark the function as `needs_followup`.
- When referencing performance or resource claims, double-check the module for existing caches, guards, or reuse mechanisms. Document the code you inspected before concluding that a gap exists.
- If the analysis depends on workload size, state the assumptions explicitly in the evidence.

Comprehensive Review Procedure
For each track, follow its `instructions.md` exactly. At minimum:
- Produce a clear summary of findings.
- Fill `checks` with pass/fail results and short evidence.
- Fill `functions` with one entry per discovered function/method, including status and notes.
- If you cannot fully verify something, mark it as “needs_followup” with rationale.

Fast‑Path Loop (recommended)
- Step 0: Load the module source and generate the AST inventory (classes/functions). Respect a time budget (e.g., 90s).
- Step 1: Red‑flag triage (blockers first). Record high/critical items immediately.
- Step 2: Checklist pass. Provide one anchored evidence line per check (preferably with `path:line`).
- Step 3: Per‑function table. One‑line notes each (≥ 20 chars).
- Step 4: Summary. 5–10 bullets ordered by impact.

Severity guardrails
- Critical: Demonstrably catastrophic performance/scalability/security bug without mitigations (e.g., unbounded quadratic loop on hot path, credential leak). Must include path:line evidence and quantified impact.
- High: Material issue that meaningfully degrades behaviour under realistic workloads (e.g., synchronous network call inside per-request loop). Provide concrete scenario and why it cannot be amortized.
- Medium: Noticeable but bounded inefficiency or risk. Explain assumptions and whether simple mitigation exists.
- Low: Minor clean-up, stylistic improvements, or speculative enhancements. If unsure, prefer “needs_followup” with explicit open questions.

Final Sanity Check
- After completing all tracks for the given scope, run:
  - Single module: `python tools/code_review/automation/run_all_validations.py --module <module.py>`
  - Folder (all tracks): `python tools/code_review/automation/run_all_validations.py --folder <folder>`
  - Folder (single track example): `python tools/code_review/automation/soundness_correctness/validate_folder.py --folder <folder> --reports-root reports/1_quality/automated_reviews/soundness_correctness`
  This must complete without errors, indicating all required checks exist and function coverage is complete for each track.
