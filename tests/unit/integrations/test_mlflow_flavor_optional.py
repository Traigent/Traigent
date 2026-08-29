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
