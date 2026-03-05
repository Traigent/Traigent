# Team Review Checklist (Internal)

Purpose: collect technical feedback before external sharing.

## Scope
- Confirm client report is accurate, non-ambiguous, and aligned with runtime code paths.
- Confirm visuals/tables are readable and interpretations are correct.

## Checklist
- [ ] Accuracy formula is clear and mathematically correct:
      Accuracy = (1/N) * sum(accuracy_i), N=3 for this run.
- [ ] The report clearly states that Accuracy is NOT computed from (tool+param+text)/3 in this run.
- [ ] Score=Accuracy behavior is described correctly for this run.
- [ ] Cost explanation is correct: FE cost is per-example display; SDK total cost is per-trial total.
- [ ] Objective explanation is correct: optimizer objective was tool_accuracy.
- [ ] Tie handling explanation is acceptable and does not overclaim.
- [ ] Raw example-level table values match trials_v2.jsonl example_results.
- [ ] All model-level metrics match across results.json, CSV export, and trial files.
- [ ] Color scale semantics are consistent (green=better; for latency/cost lower is better).
- [ ] Table abbreviations/legends are sufficient and not confusing.
- [ ] No sensitive internal-only details appear in the client-facing report.

## Evidence to inspect
- Runtime logic: traigent/evaluators/hybrid_api.py
- Bazak run config: examples/bazak/run_optimization.py
- Tests: tests/unit/evaluators/test_hybrid_api_evaluator.py
- Internal evidence summary: internal/LOGIC_EVIDENCE.md

## Feedback template
- Finding:
- Severity (high/medium/low):
- Evidence (file:line):
- Suggested change:

Generated on: 2026-03-03 12:33:54 IST
