Security Review — Instructions for LLM

Category: security
Deliverables
- JSON: `../../reports/code_review_results/security/<module>.review.json`
- Match `templates/review_template.json`.

Schema expectations (validated):
- `classes`: list of class names.
- `functions`: objects with `name`, `status` (ok|issue|needs_followup), `notes` (>= 20 chars) on security posture.
- `checks`: objects with `name`, `result` (pass|fail|needs_followup), `evidence` (>= 30 chars; reference specific APIs or lines for risky areas). Optional `confidence` (low|medium|high).
- `issues` / `recommendations`: included with severity (required) and scoped symbols when applicable; optional confidence.
- Optional: `metadata` and `skip_reasons`.

Checklist (required checks)
- input_sanitization: inputs normalized/validated to prevent injection and traversal
- secret_handling: keys/tokens not logged; proper loading/storage patterns
- crypto_hashing: approved primitives (avoid md5/sha1); salts/IVs where needed
- logging_pii_guardrails: sensitive data redacted or omitted
- dependency_risks: known dangerous calls/APIs flagged (exec, eval, pickle)
- network_filesystem_safety: SSRF, path traversal, sandbox escapes mitigated

Per-Function/Class Coverage
- Include every class name in `classes[]`.
- Include each function/method with a security risk assessment.

Reasoning reminders
- Verify claims about sensitive logging, authentication bypass, or path traversal by examining surrounding safeguards.
- When flagging crypto or secret-handling issues, identify the exact API usage or missing safeguard with a line reference.
- If uncertain whether an attack path is reachable, mark the check as `needs_followup` and document the uncertainty.

Self-Validation
1) Build inventory: `python tools/code_review/automation/_shared/function_inventory.py --module <module> --json`
2) Ensure all functions from the inventory are present in `functions[]`.
3) Ensure all checks listed above exist in `checks[]` (see `required_checks.json`).
4) Save JSON to `../../reports/code_review_results/security/<module>.review.json` and run:
   `python tools/code_review/automation/security/validate_module.py --module <module> --report reports/1_quality/automated_reviews/security/<module>.review.json`
