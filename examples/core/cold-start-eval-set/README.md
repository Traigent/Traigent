# Cold-start evaluation dataset

This fully offline example starts with one decorated square agent, then
constructs a small, tune-only evaluation dataset for that same agent. It never
executes the target while creating the dataset: static inspection unwraps the
decorated function to obtain typed input constraints, and a separate local
oracle supplies expected outputs.

Run it from the SDK repository root:

```bash
python examples/core/cold-start-eval-set/run.py --output-dir ./coldstart-eval-output
```

`--output-dir` is caller-selected, but it must be under the current working
directory (or under `TRAIGENT_DATASET_ROOT` when that is configured). This is
the same trusted-path rule used by `Dataset.from_jsonl`.

The command writes three local artifacts:

- `coldstart_tuning.jsonl` contains only admissible `tune` rows and is accepted
  by `Dataset.from_jsonl`.
- `coldstart_audit.jsonl` records candidate outcomes without raw inputs or gold;
  it is intentionally not an evaluation dataset.
- `coldstart_manifest.json` records the real dataset SHA-256, construction
  provenance, and `holdout_prohibited: true`.

The script defines its one real `@traigent.optimize` agent up front with
`eval_dataset=None`, offline execution, and two local arms: `correct` and
`predictably-wrong`. After generation, it verifies the tuning path, loads it
with `Dataset.from_jsonl(...)`, and calls the decorated function's public
`set_eval_dataset_override(...)`. Use `--run-optimize` to invoke the existing
offline SDK optimizer on that same decorated function. The default command
stops at the canonical handoff so it neither executes the agent nor creates a
second runner for configuration trials, scoring, or cost tracking.

The example uses the default `accuracy` scorer, which is exact-match and so
matches `ScoringContract.EXACT_MATCH`. Other scoring contracts need a matching
existing metric function; the generated row contract does not automatically
configure or enforce one. The generated tuning rows are not holdout evidence.

No cloud credential or model call is needed. In a real project, replace the
local deterministic oracle with a separately maintained, deterministic source
of ground truth; do not wrap the target callable as its own oracle.
