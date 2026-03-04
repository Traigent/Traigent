Bazak Optimization - Share Package

Contents:
- report/CLIENT_VALIDATION_REPORT.pdf : Final validation report
- report/CLIENT_VALIDATION_REPORT.md  : Markdown source for traceability
- report/latex/*                      : LaTeX source and assets used for report generation
- data/results.json                   : SDK optimization results summary
- data/configuration_runs_...csv      : Frontend/exported run metrics table
- data/m1_ex_measure_accuracy_nulls.csv
                                     : Dedicated M1-Ex-Measure null-audit table (model x example)
- data/m1_ex_measure_accuracy_nulls_summary.json
                                     : Summary of missing/null per-example accuracy counts
- trials/trials_v2.jsonl              : Trial-level log stream
- trials/trial_trial_*_v2.json        : Per-trial result artifacts

Notes:
- Experiment artifacts were reused from existing runs; no experiments were rerun.
- `timestamp` fields in `trials/*.json` and `trials/trials_v2.jsonl` are intentionally redacted and set to the sentinel value `REDACTED`.
- `example_results[].query`, `example_results[].response`, and `example_results[].expected` are intentionally omitted (`null`) in this package. As a result, independent re-scoring of per-example outputs is not possible from this share alone.
- `example_results[].cost_usd` is an allocated display value in this package (`trial total_cost / examples_attempted`), not a true per-example provider-billed measurement.
