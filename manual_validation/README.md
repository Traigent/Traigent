# Manual Validation

These checks are intentionally kept out of the default `tests/` suite. They are
local/manual harnesses for backend-driven validation and typically require a
localhost service or hands-on verification.

## Harnesses

They are **standalone scripts, not pytest modules** — their filenames
deliberately do not match `python_files = ["test_*.py"]`, so `pytest` collects
nothing from them even if pointed at this directory. Run them directly:

```bash
python manual_validation/custom_evaluator_metrics_check.py
python manual_validation/backend_measures_passthrough_check.py
```

Either command works from any working directory. Each script puts the repo root
on `sys.path` (so it runs against the checkout without a `pip install`) and
points `TRAIGENT_DATASET_ROOT` at this directory, writing its dataset to the
ignored `_run_artifacts/` beside it — the SDK rejects datasets outside that
root, which is what made an earlier `/tmp` dataset abort both scripts before
they ran.

Each script **exits non-zero when the property it checks is broken**, so it is
usable as a manual gate rather than a wall of output to squint at.

| Harness | What it verifies | What it does not |
|---|---|---|
| `custom_evaluator_metrics_check.py` | A custom evaluator's per-example `accuracy` reaches `TrialResult.metrics` for every trial, and reaches the per-example `measures` array of the backend payload. Prints every trial's config/metrics. | Nothing about the database. The run is `execution_mode="local"`: no backend session, no `configuration_runs` row. The `psql` recipe it prints is a follow-up for a *backend-tracked* run. |
| `backend_measures_passthrough_check.py` | Trial metrics are forwarded as `metadata.measures`, checked by calling `build_backend_metadata` — the same producer `BackendSessionManager` calls to build the payload it persists and submits. Flags missing basic fields and reports which LLM token/cost fields are present. | Anything on the wire. It never contacts a backend. |

Both previously lived in `tests/integration/` under `test_*.py` names. They
contributed zero collected tests there — `@traigent.optimize` returns a
non-function, so pytest emitted `PytestCollectionWarning` and collected nothing
— while their stale `execution_mode="edge_analytics"` raised at import and broke
collection for the whole suite. They now use `execution_mode="local"` so they
actually run. Automated coverage for both gaps is tracked separately; these
harnesses are a debugging aid, not a substitute for it.

`backend_measures_passthrough_check.py` used to claim it patched
`BackendIntegratedClient` and printed captured submissions. Under
`execution_mode="local"` the orchestrator never constructs that client, so the
patch target was dead, nothing was ever captured, and the script exited 0 no
matter what. It now checks the producer, which is the part that *is* verifiable
without a live backend.

## Adding a harness

**Name it `<subject>_check.py`** and drop it directly in this directory —
`.gitignore` un-ignores exactly `manual_validation/README.md` and
`manual_validation/*_check.py`, so `git add` picks it up as usual. Any subject
works, including one that starts with `test`. Add a row to the table above while
you are at it, and make it exit non-zero when its subject is broken.

The one carve-out: the negation is top level only (`manual_validation/*_check.py`,
never `**`), so a harness in a *subdirectory* stays ignored. Keep them flat here.

The negation is deliberately that narrow. A broader one (`*.py`/`*.md` at any
depth) sits *after* the global rules in `.gitignore` and therefore overrides
them, which quietly un-ignored `local/`, `local_settings.py`, `local_results/`,
`test_phase*.py` and `manual_validation_*.py` inside this directory — the exact
internal scratch files those rules exist to keep out.

Narrowness cuts both ways, and order decides which way. Git applies last match
wins, so the belt-and-braces re-ignore lines in that block (`test_*.py`,
`local_settings.py`, …) are listed **before** the two negations. Listed after,
`manual_validation/**/test_*.py` would beat `!manual_validation/*_check.py` and
silently drop `test_runner_check.py`. If you add a re-ignore line there, put it
above the negations.

Everything else here is a run artifact and stays ignored on purpose:
`_run_artifacts/`, `__pycache__/`, `.pyc`, run logs, and result/dataset dumps.
To track a file of some other type, check what is happening first, then force
it:

```bash
git check-ignore -v --no-index manual_validation/<file>   # shows the rule that matched
git add -f manual_validation/<file>
```

(This directory was once ignored wholesale, which made `git add` drop new
harnesses silently. If a `*_check.py` you add here does not show up in
`git status`, that regressed — fix the rule in `.gitignore` rather than working
around it.)
