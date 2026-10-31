# Resuming a local run (`resume_from`)

`optimize(resume_from=...)` lets a local run reuse trials a previous run already
finished, so an interrupted or extended study does not pay for them twice.

```python
first = agent.optimize_sync(max_trials=20)           # interrupted or short
more = agent.optimize_sync(max_trials=40, resume_from=first)
```

`resume_from` takes either an `OptimizationResult`, or the name or directory path
of a result saved with `PersistenceManager.save_result` (the path it returns
works). A saved result must carry a sha256 integrity manifest that verifies; a
corrupted or edited result, or one with no manifest (legacy, or the manifest was
deleted), raises `ValueError` and nothing is reused. The manifest detects
corruption and casual edits only, not someone who can rewrite both files.
`allow_unverified_resume=True` is an explicit, unsafe opt-in for legacy results:
their trials are then trusted as given. An in-memory `OptimizationResult` you
pass is trusted as given.

## What is reused

A prior trial is reused only when all of these hold:

- the dataset, scorer (evaluator), agent function and configuration space of the
  prior run have the same `fp1` artifact fingerprints as this run (the same
  fingerprints the SDK already computes; they are stored in
  `result.metadata["artifact_fingerprints"]`);
- the trial status is `completed`;
- its cost is measured, or flagged unmeasured. A completed trial with no cost
  information at all is not reused. A measured cost is normalized to the
  `total_cost` metric so study totals include it.

If any identity component differs, or a fingerprint is missing, nothing is
reused and one warning names the components that differ (never their values).

Trials that were running, pending or in an unknown state are not reused and are
not retried: they are not counted, and their configurations are excluded from
this run's suggestions. Failed and cancelled trials are not reused and may be
proposed again.

## Partial trials: example-level reuse

A prior trial that finished only some of its examples (interrupted and saved,
cut short by `max_total_examples`, or with example-level failures) is not counted
as a trial. When the optimizer proposes that same configuration again, the
examples it completed are reused and only the missing ones run; the result is
aggregated as one trial over the whole dataset.

A stored example is reused only when all of these hold:

- the run identity (dataset, scorer, agent, space fingerprints) matches, as above;
- the example completed: success, no error, and an output;
- its example id equals the current dataset row's id, and its stored input and
  expected output equal the row's (a changed row is re-run);
- the trial's cost for it is not charged again: it stays in the trial and study
  totals, but is not charged to this run's `cost_limit` or shared budget, and a
  row whose cost was unknown stays unknown.

Failed or unfinished examples are never reused. Reused examples are not
re-executed, re-scored or re-judged: their stored scores are used. They use none
of the `max_total_examples` budget.

Fallbacks: a trial whose result carries no per-example results (or only
redacted ones) is handled at trial level, as above. Prior trials still marked
running or pending are never retried, even if they stored some examples. Example
reuse needs the detailed per-example evaluator (the local default); otherwise the
trial is re-run in full.

## Accounting

- Reused trials count toward `max_trials`, appear in `result.trials` with
  `trial.metadata["resumed_from_prior_run"] = True`, and the optimizer does not
  propose their configurations again.
- Their cost is part of the result totals, but is not charged again against this
  run's `cost_limit`: this run did not spend it. A trial whose cost was unknown
  stays unknown.
- `result.metadata["resume"]` summarises what was reused and what was not.

## Limits

`max_total_examples` counts only examples this run newly attempts. Reused trials
and reused examples execute nothing, so they use none of that budget.

If the run has a `default_config` baseline that a reused or in-flight prior trial already covers, the
baseline is not run again. Batch optimizers forward the exclusions to their inner
optimizer.

Local runs only. A run that would use the managed cloud or a hybrid API
evaluator raises `ValueError("resume_from supports local runs only for now")`.
