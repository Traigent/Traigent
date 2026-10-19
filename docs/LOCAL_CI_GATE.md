# Running the local checks

Most checks that run on a pull request can be run on your machine in seconds.
Running them before you push saves a CI round-trip.

## Quick start

```bash
make install-hooks   # once per clone: installs the pre-commit and pre-push hooks
make local-gate      # before every push: runs the checks below
```

`make local-gate` (which runs `scripts/local_gate.sh`) exits non-zero if
anything fails. Fix what it reports, then push.

## What it checks

| Check | Notes |
|-------|-------|
| Freshness of your branch against the base branch | Warns when your branch has fallen behind. |
| `ruff check` and `ruff format --check` | Run on the Python files your branch changes, like the pull request check. Fix formatting with `ruff format <files>`. |
| Test collection and a short unit-test smoke run | Runs when SDK code or tests changed. |
| Pull request tracking-line reminder | Reminds you to keep the tracking line in the pull request description (see the template). |
| Optional code-quality scan | Only for `release/*` and `hotfix/*` branches, or when `LOCAL_GATE_SONAR=1` is set. Requires your own scanner server and token; pushes to `release/*` and `hotfix/*` branches are blocked without scanner credentials. |

Useful switches:

```bash
LOCAL_GATE_SKIP=pytest,sonar scripts/local_gate.sh   # skip named steps
LOCAL_GATE_FULL_UNIT=1 scripts/local_gate.sh          # run the full unit tier
```

## Why ruff?

The pre-commit hooks format with `black` and `isort`, but the required pull
request check runs `ruff`, so the local gate mirrors `ruff`. It only looks at
the files your branch changes, so unrelated formatting debt elsewhere in the
tree does not block you.

## Bypassing

`git push --no-verify` skips the pre-push hook. Use it only when you have
verified the change another way.

## Which interpreter the gate uses

`scripts/local_gate.sh` prefers the repo's `.venv` (`.venv/bin/pytest`, `.venv/bin/ruff`) over any
`pytest`/`ruff` on `PATH`, because the pre-push hook does not activate the venv and `pyproject.toml`
`addopts` needs `pytest-xdist` from the `dev` extra. The gate prints the resolved tools in its header
and fails with an install hint if that interpreter cannot import `xdist`.
