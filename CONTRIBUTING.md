# Contributing to Traigent

Thank you for your interest in contributing. This guide covers how to report
problems, set up a development environment, and submit changes.

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
Additional contributor reference material lives in
[`docs/contributing/`](docs/contributing/CONTRIBUTING.md).

## Reporting bugs and proposing features

- **Bugs:** open an issue with the
  [bug report form](https://github.com/Traigent/Traigent/issues/new/choose).
  Include the Traigent version, Python version, and a minimal reproduction.
- **Features:** open an issue with the feature request form describing the
  problem you want to solve and your proposed approach. For larger changes,
  please discuss the idea in an issue before writing code.
- **Questions:** use [GitHub Discussions](https://github.com/Traigent/Traigent/discussions).
- **Security vulnerabilities:** do not open a public issue. See
  [SECURITY.md](SECURITY.md).

Please search existing issues first to avoid duplicates.

## Development setup

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
make install-hooks      # optional: install the git pre-commit and pre-push hooks
```

If you use `uv`, run `uv sync --all-extras --no-extra deepeval --dev`. The
`deepeval` extra is opt-in only; installing it alongside `--all-extras` breaks
test collection (see the Development section of the [README](README.md)).

## Running tests and checks

```bash
make test-unit          # unit tests (parallel)
make test               # full test suite (parallel)
make lint               # ruff, mypy, bandit
make format             # black and isort
make format-ruff        # ruff format and autofix (matches the CI preflight check)
make local-gate         # run the local pre-push checks before pushing
```

Run the tests related to your change as you work. Continuous integration runs
the broader suite on every pull request.

## Branch model

- Open pull requests against the `develop` branch.
- Maintainers promote `develop` to `main` for releases.
- Changes to protected branches are merged by maintainers.
- Create your branch from the latest `develop`, for example
  `feature/<short-description>` or `fix/<short-description>`.

## Commits and pull requests

- Keep pull requests focused: one logical change per PR.
- Write clear commit messages that explain what changed and why.
- Add or update tests for behavior changes, and update documentation and
  [CHANGELOG.md](CHANGELOG.md) when user-visible behavior changes.
- Follow the pull request template: summarize the change, link related issues,
  and note how you tested it.
- Make sure lint, format, and tests pass before requesting review.

## Contributor License Agreement

Traigent requires a signed Contributor License Agreement (CLA) before accepting
non-trivial external contributions. See
[CONTRIBUTOR-LICENSING.md](CONTRIBUTOR-LICENSING.md) for the policy and how to
request the CLA paperwork (opensource@traigent.ai).

## License

This project is dual-licensed under `AGPL-3.0-only` or a Traigent commercial
license (`AGPL-3.0-only OR LicenseRef-Traigent-Commercial`). See
[LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Accepted
contributions are distributed under these terms as described in
[CONTRIBUTOR-LICENSING.md](CONTRIBUTOR-LICENSING.md).
