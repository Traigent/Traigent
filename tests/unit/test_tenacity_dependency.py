"""Regression test for GitHub issue #1824: undeclared `tenacity` dependency.

litellm (a CORE dependency, see pyproject.toml [project.dependencies]) imports
`tenacity` *lazily* inside its public retry helpers —
`litellm.completion_with_retries()` / `acompletion_with_retries()` — but does
not hard-depend on it itself. Before this fix, `traigent` didn't declare
`tenacity` either, so a clean ``pip install traigent`` left that retry path
dead on arrival: calling `completion_with_retries()` raised
``ModuleNotFoundError: No module named 'tenacity'`` (surfaced by litellm as
``Exception: tenacity import failed ...``), causing un-retried transient
errors to silently score 0. This is the failure mode observed on litellm
1.91.0's retry path in #1824.

This test FAILS on the pre-fix state (no `tenacity` in pyproject/requirements,
and — in an environment where tenacity truly isn't installed — the retry
wrapper raising) and PASSES once `tenacity` is declared as a core dependency.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# File is at tests/unit/test_tenacity_dependency.py
# parents[0] = tests/unit/, parents[1] = tests/, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
REQUIREMENTS_CORE_PATH = _REPO_ROOT / "requirements" / "requirements.txt"


def _core_dependency_names_from_pyproject() -> list[str]:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps: list[str] = data["project"]["dependencies"]
    names = []
    for dep in deps:
        normalized = (
            dep.strip()
            .lower()
            .split("[")[0]
            .split(">=")[0]
            .split("==")[0]
            .split("<")[0]
            .split(",")[0]
            .strip()
        )
        names.append(normalized)
    return names


def test_tenacity_is_importable():
    """`tenacity` must be importable in this environment (i.e. actually
    installed), not merely referenced somewhere in litellm's optional extras."""
    import tenacity  # noqa: F401


def test_tenacity_declared_as_core_dependency_in_pyproject():
    """`tenacity` must be a CORE dependency in pyproject.toml, not an extra.

    litellm (which lazily imports tenacity inside its public
    `completion_with_retries`/`acompletion_with_retries` helpers) is itself a
    core dependency, so tenacity must be core too — gating it behind an
    optional extra would leave the same gap for anyone doing a plain
    `pip install traigent`.
    """
    core_deps = _core_dependency_names_from_pyproject()
    assert "tenacity" in core_deps, (
        "'tenacity' is missing from pyproject.toml [project].dependencies. "
        "litellm imports tenacity lazily inside its public "
        "completion_with_retries/acompletion_with_retries helpers but does "
        "not hard-depend on it, so a clean `pip install traigent` leaves "
        "that retry path dead on arrival (ModuleNotFoundError: No module "
        "named 'tenacity'). See GitHub issue #1824."
    )


def test_tenacity_declared_in_requirements_txt():
    """requirements/requirements.txt (the hand-maintained core mirror) must
    also declare tenacity, per requirements/README.md's synchronization
    contract with pyproject.toml."""
    requirements_text = REQUIREMENTS_CORE_PATH.read_text()
    assert "tenacity" in requirements_text, (
        "'tenacity' is missing from requirements/requirements.txt. "
        "See GitHub issue #1824."
    )


def test_litellm_completion_with_retries_does_not_raise_tenacity_import_error():
    """The actual failure mode from #1824: litellm's public
    `completion_with_retries()` helper raises when tenacity isn't importable
    (observed on litellm 1.91.0's retry path). Stub out the wrapped function
    so no real network/API call happens, and drive the real retry wrapper."""
    import litellm

    calls = {"count": 0}

    def _stub_completion(*args, **kwargs):
        calls["count"] += 1
        return "dummy-response"

    try:
        result = litellm.completion_with_retries(
            original_function=_stub_completion,
            num_retries=2,
        )
    except Exception as exc:  # pragma: no cover - only hit on regression
        assert "tenacity import failed" not in str(exc), (
            f"litellm.completion_with_retries() raised the tenacity-missing "
            f"error: {exc}. This is the exact #1824 failure mode: litellm "
            f"imports tenacity lazily inside this public retry helper but "
            f"traigent doesn't declare it, so this retry path is dead on "
            f"arrival."
        )
        raise

    assert result == "dummy-response"
    assert calls["count"] == 1
