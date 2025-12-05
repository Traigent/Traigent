Soundness/Correctness Review — Instructions for LLM

Category: soundness_correctness
Scope: Single module file. For folders, iterate modules.

Deliverables
- JSON to: `../../reports/code_review_results/soundness_correctness/<module>.review.json`
- Must match `templates/review_template.json` here.

Schema expectations (validated):
- `classes`: list of class names.
- `functions`: objects with `name`, `status` (ok|issue|needs_followup), `notes` (>= 20 chars focused on correctness risks).
- `checks`: objects with `name`, `result` (pass|fail|needs_followup), `evidence` (>= 30 chars, preferably with `path:line` refs). Optional `confidence` (low|medium|high).
- `issues` / `recommendations`: as in code quality; severity required, optional confidence.
- Optional: `metadata` and `skip_reasons` fields.

Checklist (required checks)
- input_validation_coverage: inputs validated and typed where feasible
- error_handling_paths: exceptions explicit, meaningful messages, no silent failures
- invariants_contracts: key pre/post conditions stated and enforced
- edge_cases_listed: enumerated edge cases and handling assessment
- state_mutation_safety: shared state changes are safe and predictable
- testability_plan: unit tests recommended, cases enumerated or present

Per-Function/Class Coverage
- Include every class name in `classes[]`.
- Include an entry in `functions[]` for every function/method with status and notes focused on correctness risks.

Summary & Evidence
- Populate `checks[]` with pass/fail/needs_followup and concise evidence (>= 30 chars, e.g., references to code sections). Include at least one concrete example per failing/needs_followup check.
- Provide prioritized issues and remediation steps in `summary` and `recommendations`.

Reasoning reminders
- When flagging potential runtime failures (division by zero, None access, index errors), inspect nearby guards/conditionals to confirm the path is reachable.
- For data validation findings, look for existing defensive code (isinstance checks, defaults) before escalating severity.
- If uncertain whether a path is reachable, downgrade to `needs_followup` and pose a clarifying question.

Self-Validation
1) Inventory functions and classes with the shared script.
2) Ensure all classes are present in `classes[]` and all functions/methods are in `functions[]`.
3) Ensure all checks exist as listed above.
4) Save JSON and run: `python tools/code_review/automation/soundness_correctness/validate_module.py --module <module> --report reports/1_quality/automated_reviews/soundness_correctness/<module>.review.json`
