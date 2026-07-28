# Manual Validation

These checks are intentionally kept out of the default `tests/` suite.

They are local/manual harnesses for backend-driven validation and typically
require a localhost service or hands-on verification.

Run them explicitly with:

```bash
RUN_MANUAL_VALIDATION=1 pytest manual_validation -o addopts=''
```

This keeps the main pytest surface clean while preserving the harnesses for
targeted debugging and release validation.

## Harnesses

Both of the harnesses below are **print-only scripts, not pytest modules** —
they contain no assertions, and their filenames deliberately do not match
`python_files = ["test_*.py"]`, so `pytest` collects nothing from them even if
pointed at this directory. Run them directly and read the output:

| Harness | What it verifies |
|---|---|
| `custom_evaluator_metrics_check.py` | A custom evaluator's per-example metrics survive the optimization pipeline and land in the database. Prints every trial's config/metrics, then the `psql` query to confirm `configuration_runs.measures` holds non-zero accuracy values. |
| `backend_measures_passthrough_check.py` | Trial metrics are forwarded to the backend as `metadata.measures`. Patches `BackendIntegratedClient` to capture submissions and prints each one, flagging missing basic fields and reporting which LLM token/cost fields are present. |

```bash
python manual_validation/custom_evaluator_metrics_check.py
python manual_validation/backend_measures_passthrough_check.py
```

Both previously lived in `tests/integration/` under `test_*.py` names. They
contributed zero collected tests there — `@traigent.optimize` returns a
non-function, so pytest emitted `PytestCollectionWarning` and collected nothing
— while their stale `execution_mode="edge_analytics"` raised at import and broke
collection for the whole suite. They now use `execution_mode="local"` so they
actually run. Automated coverage for both gaps is tracked separately; these
harnesses are a debugging aid, not a substitute for it.

> **Note:** `RUN_MANUAL_VALIDATION` is a documented convention that nothing
> currently reads. It gates nothing today — the harnesses above are excluded
> from the default suite by living outside `testpaths` and by their filenames.

## Adding a harness

**Files in this directory are tracked.** Drop your `.py` (and any `.md` notes)
here and `git add` it as usual — `.gitignore` un-ignores `*.py` and `*.md` at
any depth under `manual_validation/`. Add a row to the table above while you
are at it.

Everything else here is treated as a run artifact and stays ignored on purpose:
`__pycache__/`, `.pyc`, run logs, and result/dataset dumps the harnesses write
next to themselves. If you need to track a file of some other type, check what
is happening first and then force it:

```bash
git check-ignore -v --no-index manual_validation/<file>   # shows the rule that matched
git add -f manual_validation/<file>
```

(This directory was once ignored wholesale, which made `git add` refuse new
harnesses silently. If a file you add here does not show up in `git status`,
that regressed — fix the rule in `.gitignore` rather than working around it.)
