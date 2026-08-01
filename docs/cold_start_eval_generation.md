# Cold-start evaluation dataset generation

`traigent.generation.coldstart` builds a local, tuning-only evaluation dataset
before an optimization run. Construction statically inspects the callable and
repository, proposes inputs, and requires independent ground truth. It does
not execute the target function, run a baseline, compare configurations, or
select examples based on current performance.

```python
from pathlib import Path

import traigent
from traigent.generation.coldstart import (
    CallableOracle,
    ColdStartOutcome,
    assert_optimizer_eligible,
    generate_eval_set,
)


@traigent.optimize(
    eval_dataset=None,
    objectives=["accuracy"],
    configuration_space={"mode": ["correct", "predictably-wrong"]},
    offline=True,
)
def classify(text: str) -> str:
    # This target is not called during dataset construction. In a real agent,
    # make each configuration arm affect the output deterministically.
    mode = str(traigent.get_config().get("mode", "correct"))
    return "positive" if mode == "correct" and text == "good" else "negative"


def labels(inputs: dict[str, object]) -> str:
    # A separate deterministic, independently maintained truth source.
    return "positive" if inputs["text"] == "good" else "negative"


result = generate_eval_set(
    func=classify,
    repo_root=Path.cwd(),
    oracle=CallableOracle(labels, oracle_id="local_labels.v1"),
    output_dir=Path.cwd() / "coldstart-output",
)
if result.outcome is ColdStartOutcome.EVAL_SET:
    assert result.tuning_path is not None
    assert_optimizer_eligible(result.tuning_path)
    dataset = traigent.Dataset.from_jsonl(str(result.tuning_path))
    classify.set_eval_dataset_override(dataset)

    # This same decorated function owns execution and comparison. Do not run
    # this in a construction-only smoke command.
    best = await classify.optimize()
```

`output_dir` is caller-selected but must be within the trusted dataset root:
the current working directory by default, or `TRAIGENT_DATASET_ROOT` when that
environment variable is configured. This preserves the existing
`Dataset.from_jsonl` path-security contract.

## Admission policy

A tuning row is written only when all of these are true:

- input is a non-empty JSON mapping, within the fixed local size limit, free of
  concrete instruction-injection markers, and unique;
- one expected output exists;
- its ground-truth source is exactly `spec_derived` or `oracle_computed`;
- its scoring contract is one of the supported deterministic contracts; and
- its literal split is `tune`.

There is no admission override. Labels made by `ExampleSynthesizer` are always
discarded; a `SynthesizedInputGenerator` contributes inputs only, and the
oracle must ground them independently. A missing oracle, untyped/insufficient
callable contract, unavailable gold, or zero eligible rows returns
`ColdStartOutcome.DISCOVERY_ONLY`, `tuning_path=None`, and only audit/manifest
artifacts.

## Artifacts and integrity

Eligible output contains these fixed files:

- `coldstart_tuning.jsonl` is compatible with `Dataset.from_jsonl`. Each row
  has a unique `example_id`, `expected_output`, and `traigent_coldstart`
  provenance (source, scoring contract, oracle/generator ids, seed, system
  fingerprint, row digest, schema version, and `tune` split).
- `coldstart_audit.jsonl` contains candidate digests, states, and quarantine
  reasons only. It intentionally has no `input` field and cannot be loaded as
  an evaluation dataset.
- `coldstart_manifest.json` carries the real dataset SHA-256, descriptors,
  counts, gaps, SDK/schema versions, and `holdout_prohibited: true`.

`assert_optimizer_eligible(path)` re-checks the manifest SHA and re-derives row
eligibility before the path is passed to `@traigent.optimize`. It detects
accidental edits and malformed provenance; it is an integrity check, **not
authentication**. It cannot prove an oracle's semantic correctness or stop a
party that can alter both dataset and manifest from recomputing hashes.

## Handoff to optimization

Define the one decorated agent before generation, with `eval_dataset=None`, as
shown above. Pass that decorated function to `generate_eval_set(...)`; static
inspection unwraps its stored callable without executing either wrapper or
target. After a successful result, load the generated path with
`Dataset.from_jsonl(...)` and call the existing public
`set_eval_dataset_override(...)` on that same decorated function. The existing
optimizer then owns all target execution and configuration comparison; do not
create a parallel runner for configurations, scoring, or cost tracking.

The default `accuracy` scorer is exact-match, matching
`ScoringContract.EXACT_MATCH`. If generated rows use another scoring contract,
provide an existing matching metric function to the decorator: the row contract
does not automatically configure or enforce the metric. Cold-start rows are
strictly `tune` rows and are not holdout evidence.

Pass an `ExecutionBudget` to `generate_eval_set` when construction shares an
execution budget with later work. Calls to injected generators/oracles have no
portable cost receipt, so their cost is recorded with
`ExecutionBudget.record_external(cost_usd=None, ...)`: cost tracking remains
unknown/incomplete rather than being reported as `$0`.
