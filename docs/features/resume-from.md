# Resuming a local run (`resume_from`)

`optimize(resume_from=...)` lets a local run reuse trials a previous run already
finished, so an interrupted or extended study does not pay for them twice.

```python
first = agent.optimize_sync(max_trials=20)           # interrupted or short
more = agent.optimize_sync(max_trials=40, resume_from=first)
```

`resume_from` takes either an `OptimizationResult`, or the name or directory path
of a result saved with `PersistenceManager.save_result` (the path it returns
works). A saved result is checked against its sha256 integrity manifest; a
corrupted or edited result raises `ValueError` and nothing is reused.

## What is reused

A prior trial is reused only when all of these hold:

- the dataset, scorer (evaluator), agent function and configuration space of the
  prior run have the same `fp1` artifact fingerprints as this run (the same
  fingerprints the SDK already computes; they are stored in
  `result.metadata["artifact_fingerprints"]`);
- the trial status is `completed`;
- its cost is measured, or flagged unmeasured. A completed trial with no cost
  information at all is not reused.

If any identity component differs, or a fingerprint is missing, nothing is
reused and one warning names the components that differ (never their values).

Trials that were running, pending or in an unknown state are not reused and are
not retried automatically; they are not counted. Failed and cancelled trials are
not reused either.

## Accounting

- Reused trials count toward `max_trials`, appear in `result.trials` with
  `trial.metadata["resumed_from_prior_run"] = True`, and the optimizer does not
  propose their configurations again.
- Their cost is part of the result totals, but is not charged again against this
  run's `cost_limit`: this run did not spend it. A trial whose cost was unknown
  stays unknown.
- `result.metadata["resume"]` summarises what was reused and what was not.

## Limits

Local runs only. A run that would use the managed cloud or a hybrid API
evaluator raises `ValueError("resume_from supports local runs only for now")`.
