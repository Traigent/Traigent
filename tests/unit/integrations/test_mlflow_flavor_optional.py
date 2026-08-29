"""A missing MLflow model flavor must not disable the whole integration.

`traigent/integrations/observability/mlflow.py` guards its imports with a
single `except ImportError` that sets ``MLFLOW_AVAILABLE = False``. Before the
fix for #2183, ``import mlflow.sklearn`` sat unguarded inside that block --
imported and never used anywhere in the file.

That is not hypothetical. The flavor modules (`mlflow.sklearn`,
`mlflow.pytorch`) import ``numpy`` and then ``pandas`` at module level, and
neither is a core dependency of this SDK: both are declared only in the
``analytics`` and ``ml`` extras, while ``mlflow`` lives in ``integrations``. So
``pip install traigent[integrations]`` -- without ``analytics`` -- raises
``ModuleNotFoundError: No module named 'pandas'`` on that line, which is an
``ImportError`` subclass, which the outer handler swallows. The user gets a
silently disabled MLflow integration and no diagnostic.

The integration calls only tracking-surface APIs (``start_run``,
``log_param``/``log_metric``/``log_dict``/``log_artifact``, ``set_tag``,
``search_runs``, ``MlflowClient``). None of them lives in a flavor module, so a
missing flavor should cost nothing.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys

import pytest

MODULE = "traigent.integrations.observability.mlflow"


@pytest.mark.parametrize("blocked", ["mlflow.sklearn", "mlflow.pytorch"])
def test_missing_flavor_module_does_not_disable_the_integration(
    blocked: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reimport the integration with one flavor module unimportable."""
    pytest.importorskip("mlflow")

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(blocked + "."):
            raise ModuleNotFoundError(f"No module named {blocked!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delitem(sys.modules, blocked, raising=False)

    module = importlib.import_module(MODULE)

    assert module.MLFLOW_AVAILABLE is True, (
        f"{blocked} being unimportable disabled the entire MLflow integration. "
        "Model flavors are optional; the integration uses only tracking APIs. "
        "Each flavor import needs its own nested try (see #2183)."
    )


def test_integration_reports_available_in_this_environment() -> None:
    """Baseline, so the parametrized tests above cannot pass vacuously.

    If MLflow were absent here, the tests above would assert True against a
    module that never took the guarded path at all.
    """
    pytest.importorskip("mlflow")
    module = importlib.import_module(MODULE)
    assert module.MLFLOW_AVAILABLE is True


def test_tracker_ends_runs_through_a_real_mlflow_api() -> None:
    """``end_optimization_run`` must call an API MLflow actually exposes.

    Until #2183 this called ``mlflow.finish()``. MLflow has never had a
    ``finish`` function -- the real name is ``end_run`` -- so every run against
    a genuine MLflow install raised ``AttributeError`` on completion. It went
    unnoticed because the mock class in this module's ``except ImportError``
    branch defined a matching ``finish()`` shim, so the only environment that
    exercised the line was the one where MLflow was absent.

    That shim is gone, and this test asserts against the installed package
    rather than the mock, so the two cannot drift apart again.
    """
    mlflow = pytest.importorskip("mlflow")

    assert hasattr(mlflow, "end_run"), "mlflow.end_run is the run-termination API"
    assert not hasattr(mlflow, "finish"), (
        "mlflow now exposes finish(); if upstream added it, revisit the comment "
        "at traigent/integrations/observability/mlflow.py:351"
    )

    module = importlib.import_module(MODULE)
    source = inspect.getsource(module.TraigentMLflowTracker.end_optimization_run)
    # Strip comments: the fix carries an explanatory comment naming the old call,
    # and scanning raw source would match that rather than executable code.
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "mlflow.end_run()" in code
    assert "mlflow.finish()" not in code


def test_the_offline_mock_matches_the_real_run_termination_api() -> None:
    """The mock must not define methods the real package lacks.

    A mock with a richer surface than the thing it stands in for turns a broken
    call into a passing test, which is exactly how the ``finish()`` defect
    survived. Assert the mock exposes ``end_run`` and not ``finish``.
    """
    module = importlib.import_module(MODULE)
    source = inspect.getsource(module)
    mock_start = source.index("except ImportError:")
    mock_block = source[mock_start:]

    mock_code = "\n".join(line.split("#", 1)[0] for line in mock_block.splitlines())
    assert "def end_run(" in mock_code
    assert "def finish(" not in mock_code, (
        "the offline mock defines finish(), which real MLflow does not expose; "
        "that mismatch is what hid the #2183 defect"
    )
