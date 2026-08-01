# Cold-start evaluation dataset

This fully offline example starts with one decorated square agent, then
constructs a small, tune-only evaluation dataset for that same agent. It never
executes the target while creating the dataset: static inspection unwraps the
decorated function to obtain typed input constraints, and a separate local
oracle supplies exact-match expected outputs.

Run it from the SDK repository root with a fresh output directory:

```bash
python examples/core/cold-start-eval-set/run.py --output-dir ./coldstart-eval-attempt-001
```

`--output-dir` must be under the current working directory (or under
`TRAIGENT_DATASET_ROOT` when configured). Keep it fresh for each construction
attempt: an attempt can truthfully return discovery-only, and a discovery-only
write is rejected beside an earlier tuning artifact rather than silently
reusing it.

The command writes three local artifacts:

- `coldstart_tuning.jsonl` contains only admissible `tune` rows and is accepted
  by `Dataset.from_jsonl`.
- `coldstart_audit.jsonl` records candidate outcomes without raw inputs or gold;
  it is intentionally not an evaluation dataset.
- `coldstart_manifest.json` records the real dataset SHA-256, construction
  provenance, bounded-inspection status, and `holdout_prohibited: true`.

The script defines its one real `@traigent.optimize` agent up front with
`eval_dataset=None`, offline execution, and two local arms: `correct` and
`predictably-wrong`. After generation, it verifies the tuning path, loads it
with `Dataset.from_jsonl(...)`, and calls the decorated function's public
`set_eval_dataset_override(...)`. Use `--run-optimize` to invoke the existing
offline SDK optimizer on that same decorated function. The default command
stops at the canonical handoff so it neither executes the agent nor creates a
second runner for configuration trials, scoring, or cost tracking.

This example intentionally pins its inspection to its one source file so the
fixture remains small. That option is not required for regular projects:
default options recursively select a bounded, source-first subset of normal
repository Python files, prune known vendor/build directories, and record any
truncation instead of failing at the file cap. A static source/root/read/size
failure produces `STATIC_INSPECTION_FAILED`. A missing parameter annotation
produces `UNTYPED_INPUT_CONTRACT`; a variadic, zero-input, or otherwise
unsupported signature produces `UNSUPPORTED_INPUT_CONTRACT`. None produces a
partial tuning dataset.

The example uses the default `accuracy` scorer behind
`ScoringContract.EXACT_MATCH`. This is the SDK accuracy comparator, not
byte-for-byte equality: it trims and lowercases strings and applies the SDK's
small existing float tolerance. Its generated rows are tuning-only and are not
holdout evidence. No cloud credential or model call is needed. In a real
project, replace the local deterministic oracle with a separately maintained,
deterministic source of ground truth; do not wrap the target callable as its
own oracle.
