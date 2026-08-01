# Cold-start evaluation dataset generation

`traigent.generation.coldstart` builds a local, tuning-only evaluation dataset
before an optimization run. Construction statically inspects the callable and
repository, proposes inputs from its typed contract, and obtains every expected
output from a separate local oracle. It does not execute the target function,
run a baseline, compare configurations, or select examples based on current
performance.

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
    output_dir=Path.cwd() / "coldstart-attempt-001",
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

`output_dir` must be within the trusted dataset root: the current working
directory by default, or `TRAIGENT_DATASET_ROOT` when configured. Use a fresh,
per-attempt output directory. A construction attempt can honestly degrade to
discovery-only, and the writer rejects discovery-only output beside a prior
tuning artifact rather than risking an ambiguous handoff.

## Static inspection and truthful discovery

The default `ColdStartOptions()` recursively considers Python files in a normal
repository. It prunes `.git`, virtual environments and `site-packages`,
`node_modules`, `build`, `dist`, and `__pycache__`; never follows symlinks;
and never imports or executes repository code. The callable's source is always
selected first. Remaining eligible files are selected in deterministic path
order up to `max_files`; oversized or unreadable non-source files are skipped.
Every file read, including the callable source, is bounded to
`max_file_bytes + 1` after its size check. A non-source file that grows between
those operations is skipped; growth beyond the cap in the required callable
source fails static inspection. Neither path performs an unbounded read.

This bounded selection is deliberately not fatal. `SystemSpec` reports
`inspection_truncated` and `skipped_file_count` whenever matching files were
omitted, so a caller can see that the static view was bounded. The manifest
records only structural metadata such as file count and fingerprint; it never
persists source paths or file content.

Discovery-only is a valid, explicit outcome. `UNTYPED_INPUT_CONTRACT` means a
callable parameter has no annotation. `UNSUPPORTED_INPUT_CONTRACT` means the
signature is outside the v1 scalar boundary: it is variadic, has no input
parameters, or uses an unsupported annotation. Repository root,
source-location, symlink, read, size, parse, and other static-inspection
failures instead produce `STATIC_INSPECTION_FAILED`; they are not relabeled as
an input-contract diagnosis. Missing oracle, unavailable oracle ground truth,
or zero eligible rows likewise return their own typed `DiscoveryGap` values.
Every discovery-only result has `tuning_path=None` and writes only audit and
manifest artifacts.

## Admission policy

A tuning row is written only when all of these are true:

- input is a non-empty JSON mapping, within the fixed local size limit, free of
  concrete instruction-injection markers, and unique;
- a local oracle returns one expected output that is neither `None` nor an
  empty/whitespace-only string; blank gold is `NO_GOLD` and is never
  optimizer-eligible;
- expected output is a scalar or list, not a mapping. V1 rejects mapping gold
  because the existing accuracy evaluator unwraps mapping-valued actual output
  through its `text` field, which cannot score a mapping gold value reliably;
- its ground-truth source is exactly `oracle_computed`;
- its scoring contract is exactly `exact_match`; and
- its literal split is `tune`.

There is no admission override, no model-label adapter, and no seed-requiring
synthesis path in v1. `ContractGroundedGenerator` is the built-in zero-seed
input proposer: it can propose only from the inspected scalar parameter
contract, and the supplied oracle must independently ground each proposal.

The built-in generator and local `CallableOracle` example above perform no
network I/O. Custom `ScenarioGenerator` and `Oracle` implementations (including
the callable wrapped by `CallableOracle`) are caller-supplied Python code; the
SDK cannot prevent that code from using the network. Privacy-sensitive users
must keep those implementations local and audit them before use.

`ScoringContract.EXACT_MATCH` binds these rows to the SDK's existing `accuracy`
comparison path. It is a scoring-contract name, not byte-for-byte equality:
string outputs are trimmed and compared case-insensitively, while floating
outputs use the SDK's small existing relative and absolute tolerances.

## Artifacts and integrity

Eligible output contains these fixed files:

- `coldstart_tuning.jsonl` is compatible with `Dataset.from_jsonl`. Each row
  has a unique `example_id`, `expected_output`, and `traigent_coldstart`
  provenance (oracle/generator ids, exact-match contract, seed, system
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

Cold-start rows are strictly tuning rows: they are not holdout data and cannot
support a holdout or generalization claim. The default `accuracy` scorer is
exact-match, which is the only cold-start scoring contract in v1.

Pass an `ExecutionBudget` to `generate_eval_set` when construction shares an
execution budget with later work. Proposal generation and oracle grounding are
separate external units: each proposal is counted once for generation and once
for its oracle-grounding attempt. Neither injected seam has a portable cost
receipt, so both are recorded with
`ExecutionBudget.record_external(cost_usd=None, ...)`; their cost remains
untracked/incomplete rather than being reported as `$0`.
This integration records proposal/oracle units and unknown cost but does not
stop construction when the budget is exhausted; `ColdStartOptions.num_candidates`
is the enforced construction cap.
