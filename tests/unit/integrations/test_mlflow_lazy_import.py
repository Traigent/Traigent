"""Importing Traigent must never eagerly import the real ``mlflow`` package.

Owner-flagged defect (2026-09-25): the egress recorder
(``scripts/security/behaviour_audit.py``, now on develop) showed that with
MLflow installed, every ``@traigent.optimize`` run spawned a ``uname``
process. ``traigent/integrations/__init__.py`` and
``traigent/integrations/observability/__init__.py`` eagerly import
``traigent/integrations/observability/mlflow.py``, whose module top used to
do ``import mlflow`` (plus the unused ``mlflow.sklearn`` /
``mlflow.pytorch`` flavor modules) unconditionally -- paid on every
``import traigent`` whether or not MLflow tracking was ever configured.

These tests run in a fresh subprocess interpreter so a real ``import
mlflow`` performed elsewhere in the current test process (an already-loaded
test module, a plugin, pytest's own collection, …) cannot mask a
regression.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.stdout.strip().splitlines()


def test_import_traigent_does_not_import_mlflow() -> None:
    lines = _run("import sys\nimport traigent\nprint('mlflow' in sys.modules)\n")
    assert lines[-1] == "False", (
        "`import traigent` pulled the real mlflow package into sys.modules"
    )


def test_import_traigent_integrations_does_not_import_mlflow() -> None:
    lines = _run(
        "import sys\nimport traigent.integrations\nprint('mlflow' in sys.modules)\n"
    )
    assert lines[-1] == "False", (
        "`import traigent.integrations` pulled the real mlflow package into sys.modules"
    )


def test_import_observability_module_does_not_import_mlflow() -> None:
    lines = _run(
        "import sys\n"
        "import traigent.integrations.observability\n"
        "print('mlflow' in sys.modules)\n"
    )
    assert lines[-1] == "False", (
        "`import traigent.integrations.observability` pulled the real mlflow "
        "package into sys.modules"
    )


def test_mlflow_available_flag_is_computed_without_importing_mlflow() -> None:
    """MLFLOW_AVAILABLE must reflect a real spec lookup (importlib.util.find_spec),
    not the side effect of having actually imported the package."""
    lines = _run(
        "import sys\n"
        "from traigent.integrations.observability.mlflow import MLFLOW_AVAILABLE\n"
        "print(MLFLOW_AVAILABLE)\n"
        "print('mlflow' in sys.modules)\n"
    )
    assert lines[-2] == "True", (
        "MLFLOW_AVAILABLE should be True in this environment (mlflow-skinny "
        "is installed by the `integrations`/`all` extra) -- if this is False, "
        "either the environment lacks mlflow or find_spec() regressed"
    )
    assert lines[-1] == "False", (
        "reading MLFLOW_AVAILABLE must not itself import mlflow"
    )


def test_sensitivity_control_using_the_tracker_does_import_mlflow() -> None:
    """A regression test that can never fail is not evidence.

    Using the integration for real must still work end to end against a
    genuine MLflow run, and doing so is the only thing allowed to put
    ``mlflow`` in ``sys.modules``.
    """
    lines = _run(
        "import os\n"
        "import sys\n"
        "import tempfile\n"
        "from traigent.integrations.observability.mlflow import (\n"
        "    TraigentMLflowTracker,\n"
        ")\n"
        "print('before:', 'mlflow' in sys.modules)\n"
        "os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'\n"
        "tmp = tempfile.mkdtemp()\n"
        "tracker = TraigentMLflowTracker(\n"
        "    tracking_uri='file://' + tmp,\n"
        "    experiment_name='lazy-import-sensitivity',\n"
        ")\n"
        "print('after:', 'mlflow' in sys.modules)\n"
        "run_id = tracker.start_optimization_run(\n"
        "    function_name='f', objectives=['acc'], configuration_space={'x': [1, 2]}\n"
        ")\n"
        "tracker.end_optimization_run()\n"
        "print('run_id:', bool(run_id))\n"
    )
    payload = dict(line.split(": ", 1) for line in lines)
    assert payload["before"] == "False", (
        "mlflow was already imported before TraigentMLflowTracker was even "
        "constructed -- something upstream is still eager"
    )
    assert payload["after"] == "True", (
        "constructing TraigentMLflowTracker never actually reached the real "
        "mlflow package -- the lazy import path is broken, not just deferred"
    )
    assert payload["run_id"] == "True"
