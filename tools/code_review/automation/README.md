Code Review Automation with LLMs

This toolkit defines clear, machine-checkable instructions for an LLM (e.g., sonnet4.5) to conduct comprehensive reviews on a single module (file) or an entire folder by iterating modules.

Highlights
- Four review tracks: code-quality, soundness/correctness, performance, security.
- Unified output format (JSON) with a per-track checklist so we can verify results programmatically.
- Validation scripts to ensure basic sanity checks (e.g., every top-level function and class method in the module is covered in the LLM’s report, required checks are present).

Quick Start
1) Choose scope:
   - Single module: e.g., `traigent/core/cache_policy.py`
   - Folder: e.g., `traigent/core`
2) Open the relevant instructions under each track and ask the LLM to review the target. Instruct it to save JSON reports under `reports/1_quality/automated_reviews/<track>/<relative-module-path>.review.json` from the repo root.
3) Run validations:
   - All tracks: `python tools/code_review/automation/run_all_validations.py --module traigent/core/cache_policy.py`
   - Or per track: `python tools/code_review/automation/code_quality/validate_module.py --module traigent/core/cache_policy.py --report reports/1_quality/automated_reviews/code_quality/traigent/core/cache_policy.py.review.json`

Output Contract
- The LLM must produce a JSON file matching the template in each track’s `templates/review_template.json`.
- Scripts will fail if:
  - Any required check is missing for the track
  - Not all discovered functions (top-level + class methods) are covered in the `functions` section

Tracks
- code_quality/: style, clarity, maintainability, type hints, docstrings
- soundness_correctness/: invariants, error handling, input validation, testability
- performance/: complexity, memory, I/O hotspots, batching, concurrency
- security/: sanitization, secret handling, crypto, logging PII, path/SSRF risks

See INSTRUCTIONS.md for the master prompt that references all tracks.
