Performance Review — Instructions for LLM

Category: performance
Deliverables
- JSON: `../../reports/code_review_results/performance/<module>.review.json`
- Match `templates/review_template.json`.

- `issues` / `recommendations`: included; severity required, optional confidence.
- Optional: `metadata` and `skip_reasons`.

Checklist (required checks)
- algorithmic_complexity: big-O assessment for hot paths
- memory_allocation: large structures, copies, and retention
- io_hotspots: file/network/db access patterns
- caching_opportunities: memoization, batching, reuse
- parallelism_concurrency: async/threads/processes suitability and hazards
- data_vectorization: list/vectorized operations where applicable

Per-Function/Class Coverage
- Include every class name in `classes[]`.
- For each function/method, include an entry with performance observations (status + notes).

Reasoning reminders
- Before flagging hotspots, estimate branching/nesting depth or big-O behaviour to confirm impact.
- Cite concrete loops, allocations, or I/O calls with line references when raising issues.
- If evidence is speculative, mark as `needs_followup` and pose a measurement question.
- Check for existing mitigations before raising an issue. Examples: a dataset that is loaded once and memoized, a cached auth token, or a semaphore limiting concurrency. If a mitigation exists, describe why it is insufficient rather than assuming it is missing.
- Only classify an inefficiency as `high` or `critical` when it is demonstrably on a hot path and either (a) scales worse than O(n log n) for large n, or (b) blocks request/response flow with synchronous I/O or CPU-bound work. When unsure, downgrade severity or use `needs_followup`.
- Quantify the potential impact where possible (e.g., “loop runs once per example; dataset has 15 items by default”, “network call inside every trial submission with expected latency ~50 ms”).
- Distinguish between per-run costs (acceptable) and per-iteration costs (often problematic). Mention the frequency explicitly.
- For concurrency recommendations, confirm the code is actually async-capable and identify shared state that would make parallelism unsafe.

Self-Validation
1) Build inventory: `python tools/code_review/automation/_shared/function_inventory.py --module <module> --json`
2) Ensure all functions from the inventory are present in `functions[]`.
3) Ensure all checks listed above exist in `checks[]` (see `required_checks.json`).
4) Save JSON to `../../reports/code_review_results/performance/<module>.review.json` and run:
   `python tools/code_review/automation/performance/validate_module.py --module <module> --report reports/1_quality/automated_reviews/performance/<module>.review.json`
