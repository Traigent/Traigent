Code Quality Review — Instructions for LLM

Category: code_quality
Scope: Single module file (Python). When asked to review a folder, repeat this per module.

Deliverables
- Write a JSON report to: `../../reports/code_review_results/code_quality/<module-path>.review.json`
- Must conform to `templates/review_template.json` in this folder.

Schema expectations (validated):
- `classes`: list of class names present in the module.
- `functions`: array of objects with fields: `name` (top-level or `Class.method`), `status` (ok|issue|needs_followup), `notes` (>= 20 chars).
- `checks`: array of objects with fields: `name`, `result` (pass|fail|needs_followup), `evidence` (>= 30 chars; include brief rationale and, when helpful, file:line references like `traigent/core/file.py:123`). Optional `confidence` (low|medium|high).
- `issues`: list of objects with `id`, `title`, required `severity` (low|medium|high|critical), optional `confidence` (low|medium|high), and optional `scope` (functions/classes from this module).
- `recommendations`: list of actionable next steps.
- Optional: `metadata` object and `skip_reasons: string[]`.

Checklist (required checks)
- docstrings_present: functions and classes have meaningful docstrings
- type_hints_coverage: function signatures and returns use type hints
- line_length_compliance: lines respect configured limit (default 120)
- complexity_hotspots: identify high cyclomatic/branching complexity areas
- naming_conventions: identifiers follow PEP8-like standards
- imports_hygiene: no unused imports; absolute vs relative is justified
- logging_practices: consistent, non-noisy, non-sensitive logging
- dead_code_smells: unreachable code, unused params, anti-patterns

Per-Function/Class Coverage
- Include every class name in `classes[]`.
- For each top-level function and class method (ClassName.method), include an entry in `functions[]` with:
  - name, status (ok|issue|needs_followup), notes (one line; longer details in summary)

Evidence & Summary
- In `checks[]`, for each required check, set result pass/fail/needs_followup and include specific evidence (≥ 30 chars). Evidence should be traceable (quote a symbol name or include a `path:line`).
- In `summary`, provide 5–10 bullets with prioritized findings and recommended actions.

Reasoning reminders
- Before reporting a style or correctness defect, scan nearby guards/branches to confirm the issue is reachable.
- Downgrade to `needs_followup` when unsure and capture the uncertainty explicitly.
- Use line references to anchor claims about long lines, complex functions, or redundant code.

Self-Validation Steps
1) Build inventory: `python tools/code_review/automation/_shared/function_inventory.py --module <module> --json`
2) Ensure all classes are present in `classes[]` and all functions/methods from the inventory are present in `functions[]`.
3) Ensure all checks above exist in `checks[]`.
4) Save JSON to `../../reports/code_review_results/code_quality/...` and run:
   `python tools/code_review/automation/code_quality/validate_module.py --module <module> --report reports/1_quality/automated_reviews/code_quality/<module>.review.json`
Coverage clarifications
- Include properties and dunder methods (e.g., `__repr__`) in `functions[]`.
- Exclude nested inner functions from `functions[]` (may be referenced in evidence).
- For `@overload`, include only the concrete implementation function in `functions[]`.
- Imports under `typing.TYPE_CHECKING` do not count as unused for `imports_hygiene`.
