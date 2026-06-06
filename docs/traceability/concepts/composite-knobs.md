# Composite Knobs: Patterns, Execution, and Telemetry

Composite knobs let you declare grouped control-flow shapes — cascades,
ensembles, and refinement loops — over the RFC 0001 one-knob binding model
(`Tuned | Fixed | Calibrated`). The declaration layer (the
[Composite-Knob IR](composite-knob-ir.md)) is the algebra; this page covers the
**runnable** surface the SDK ships today:

- the **pattern catalog** (the six named factories and what each expands to);
- **executing a composite** in-trial with `execute_composite` + `StageRunner`;
- **certified selection** with a composite (the `binary_cascade` strict recipe);
- **telemetry → the measures channel** (`composite_measures`).

A runnable, offline example accompanies this page:
`examples/advanced/composite-knobs/composite_telemetry.py`.

> Scope note: this page documents what the SDK *does* procedurally. It does not
> make guarantees about model behavior or future inputs. Where a "certified"
> decision is mentioned, it is a procedural decision over the evidence and
> policy you supply.

## Pattern catalog

`traigent.knobs.patterns` provides six named factories over the composite
algebra. Each returns a `CompositeKnob` declaration bundle (`.structure` is the
IR root, `.members` are member knob declarations, `.provenance` stamps the
pattern name + a hash of its validated params, `.telemetry_names` are the
§3.10 standard measure names). Factories return **declarations only**; they do
not bind values and do not trigger cloud effectuation.

| Pattern | Kind | Expands to |
|---|---|---|
| `binary_cascade` | cascade | `Cascade(arms=[stage(base), stage(expert)], gates=[margin_below θ], post)` — the two-arm escalate-or-stop policy. |
| `n_cascade` | cascade | `Cascade(arms=[stage(a₁)..stage(a_m)], gates=[θ₁..θ_{m-1}], post)` — ordered escalation; `|thresholds| == |stages| - 1`. |
| `self_consistency` | ensemble | `Ensemble(arms=[stage(a)], cardinality=k, majority_vote, accept?)` — sample `k` times, majority-vote; optional `vote_margin` acceptance. |
| `best_of_n` | ensemble | `Ensemble(arms=[stage(a)], cardinality=k, judge_max(stage(judge)))` — sample `k`, pick the judge's highest-scoring candidate. |
| `self_refine` | loop | `Loop(body=stage(a), state_keys, stop=signal_accept(σ, θ), max_iters=K)` — refine until a calibrated signal accepts (unroll-eligible). |
| `self_debug` | loop | `Loop(body=stage(a), state_keys, stop=external_accept(tests), max_iters=K)` — retry until an opaque predicate (e.g. tests) passes. |

```python
from traigent.knobs.patterns import binary_cascade

answerer = binary_cascade(
    "answerer",
    base_stage="cheap",
    expert_stage="strong",
    threshold="router_margin_threshold",  # a calibrated CVAR name
)
```

## Executing a composite

The SDK ships execution for the **post-cascade** kind today (ensemble and loop
execution are deferred and raise an explicit `NotImplementedError` naming the
gap — they are never silently skipped).

You execute a composite for one item with `execute_composite`, supplying a
`StageRunner` (or a bare callable) for each stage name and the live calibrated
gate thresholds:

```python
from traigent.knobs.runtime import StageRunner, execute_composite


def stage(outputs):
    # A voting stage: returns its sample multiset; key_fn maps an output to a
    # content-free equivalence key (required for a stage that feeds a gate).
    return StageRunner(run=lambda _item: list(outputs), key_fn=lambda x: x,
                       samples=len(outputs))


run = execute_composite(
    answerer.structure,
    stages={"cheap": stage(["A", "A", "B"]), "strong": stage(["STRONG"])},
    config={"variant": "strong"},
    calibrated_values={"router_margin_threshold": 0.9},
)
print(run.result_kind)  # output | no_accept | error
print(run.output)       # the selected arm's output (when result_kind is output)
print(run.measures)     # the §3.10 content-free telemetry dict
```

`execute_composite` returns a frozen `CompositeRunResult`:

- `result_kind` is the result-algebra tag: `output`, `no_accept` (ran but
  nothing met acceptance — an honest no-output outcome, not an error), or
  `error`;
- `output` carries the selected arm's output only when `result_kind` is
  `output`;
- `measures` is the §3.10 telemetry dict (see below);
- `error` carries a fail-closed diagnostic string when `result_kind` is `error`.

Execution **fails closed**: a stage exception, a missing stage callable, or a
missing / non-finite calibrated threshold yields an `error` result — never a
silent fallback to some stage's output.

## Certified selection with a composite

A composite's gate threshold is a **calibrated variable** (a CVAR), so a
`binary_cascade` slots into the strict certified-selection path. The shape
(mirrored by the offline example and the integration suite):

1. declare the cascade with a calibrated gate threshold;
2. build a `ConfigSpace` from `binary_cascade(...).members` with the gate bound
   `Calibrated` (the searched space is the tuned variant; the calibrated
   threshold stays client-side);
3. execute the composite in-trial so the gate creates real, observable
   dominance between candidate configs;
4. let the promotion gate decide over the collected metrics.

```python
from traigent.tvl.models import PromotionPolicy
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionGate

gate = PromotionGate(
    PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.0}),
    [ObjectiveSpec("accuracy", "maximize")],
)
decision = gate.evaluate(
    incumbent_metrics={"accuracy": incumbent_accuracy_samples},
    candidate_metrics={"accuracy": candidate_accuracy_samples},
)
# decision.decision is "promote", "reject", or "no_decision".
```

`decision.decision` is one of `promote`, `reject`, or `no_decision`. A
`no_decision` is the honest outcome when the evidence does not establish
dominance under the policy — it is not a failure and is never reported as a
winner. The offline example prints a CERTIFIED WINNER or an honest "no winner
yet" depending on the samples it collects.

In a hybrid or cloud run you do not call the promotion gate yourself; the
orchestrator runs it over the per-trial metrics your decorated function
submits.

## Telemetry → the measures channel

Every composite run produces the RFC 0002 §3.10 **content-free** telemetry dict
(counts, rates, enums, finite numbers only — never anything output-derived).
For a cascade:

- `escalation_rate` — the per-run 0/1 escalated indicator;
- `stage_selected` — the selected arm index;
- `gate_margin_pass_rate` — a per-gate map (keyed by gate index) over the gates
  actually evaluated: `1.0` where the gate did not escalate (the margin
  passed), `0.0` where it did.

`run.measures` is structured (it nests the per-gate map). The Traigent
**measures wire channel** — the per-trial numeric metrics that ride a trial
submission — is flat: numeric values only, with Python-identifier keys, and a
backend cap of 50 keys per trial (`traigent.cloud.dtos.MeasuresDict`).

`traigent.knobs.telemetry.composite_measures` is the adapter between the two.
It flattens the run's measures into identifier-safe, numeric-only keys:

```python
from traigent.knobs.telemetry import composite_measures, merge_composite_measures

composite_measures(run)
# {
#   "composite_escalation_rate": 1.0,
#   "composite_stage_selected": 1,
#   "composite_gate_0_margin_pass_rate": 0.0,
# }
```

Flattening rules:

- a scalar measure `m` becomes `{prefix}_{m}` (default prefix `composite`);
- the per-gate map is flattened per gate into
  `{prefix}_gate_{index}_margin_pass_rate`;
- non-finite or non-numeric values are dropped (the channel carries finite
  numbers only);
- the total key count is capped with headroom below the 50-key ceiling so your
  own metrics co-exist; over the cap the lowest-priority keys are truncated
  deterministically and a warning is logged — `composite_measures` never raises
  mid-trial.

The output is content-free by construction: the adapter reads `run.measures`
only (never `run.output`) and emits finite numbers and integer gate indices.

### Integration recipe (riding the existing channel)

To make composite telemetry visible on the wire, merge `composite_measures(run)`
into the metrics your decorated function returns. No new wire surface is
introduced — the `composite_*` keys ride the existing per-trial measures
channel as ordinary numeric metrics:

```python
import traigent
from traigent.knobs.runtime import execute_composite
from traigent.knobs.telemetry import composite_measures, merge_composite_measures


@traigent.optimize(
    eval_dataset=...,
    objectives=["accuracy"],
    configuration_space={"variant": ["cheap", "strong"]},
    default_config={"variant": "cheap"},
    execution_mode="hybrid",
)
def answer(text: str) -> tuple[str, dict[str, float]]:
    cfg = traigent.get_config()
    params = dict(cfg)
    run = execute_composite(
        answerer.structure,
        stages={"cheap": stage([...]), "strong": stage(["STRONG"])},
        config=params,
        calibrated_values={"router_margin_threshold": params["router_margin_threshold"]},
    )
    metrics = {"accuracy": 1.0 if str(run.output) == "STRONG" else 0.0}
    merge_composite_measures(metrics, run)  # composite_* keys ride the wire
    return str(run.output), metrics
```

On submission, the SDK splits off the reserved `measures` / `summary_stats`
keys and validates the remaining numeric metrics (your `accuracy` plus the
`composite_*` keys) through `MeasuresDict` before posting them in the trial
result body. The integration test
`tests/unit/cloud/test_composite_measures_submission.py` asserts the
`composite_*` keys appear in the posted body.

## See also

- [Composite-Knob IR](composite-knob-ir.md) — the declaration algebra.
- `examples/advanced/composite-knobs/composite_telemetry.py` — the runnable,
  offline example mirrored from this page.
