"""Real-runtime integration coverage for the supported MLflow skinny package."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path

import pytest


@pytest.mark.integration
def test_documented_skinny_file_store_path_works_in_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal documented setup creates and starts its named experiment."""
    try:
        metadata.version("mlflow-skinny")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "mlflow-skinny is absent; install traigent[integrations] to run this smoke"
        )

    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("mlflow")
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("alembic")

    mlflow = importlib.import_module("mlflow")

    tracking_dir = (tmp_path / ".mlruns").resolve()
    tracking_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    experiment = mlflow.set_experiment("traigent_optimization")

    with mlflow.start_run(run_name="documented-skinny-path") as active_run:
        mlflow.log_metric("accuracy", 0.9)
        run_id = active_run.info.run_id

    recorded_run = mlflow.get_run(run_id)
    assert experiment.name == "traigent_optimization"
    assert recorded_run.data.metrics["accuracy"] == pytest.approx(0.9)


@pytest.mark.integration
def test_mlflow_skinny_file_tracking_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The integrations extra tracks and queries a run without full MLflow."""
    try:
        metadata.version("mlflow-skinny")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "mlflow-skinny is absent; install traigent[integrations] to run this smoke"
        )
    try:
        metadata.version("pandas")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "pandas is absent; install traigent[integrations] to run this smoke"
        )

    mlflow = importlib.import_module("mlflow")
    pandas = importlib.import_module("pandas")

    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("mlflow")

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_dir = (tmp_path / ".mlruns").resolve()
    tracking_dir.mkdir()

    from traigent.integrations.observability.mlflow import TraigentMLflowTracker

    tracker = TraigentMLflowTracker(
        tracking_uri=tracking_dir.as_uri(),
        experiment_name="traigent-skinny-runtime-smoke",
    )
    assert mlflow.get_tracking_uri() == tracking_dir.as_uri()
    run_id = tracker.start_optimization_run(
        function_name="skinny_runtime_smoke",
        objectives=["accuracy"],
        configuration_space={"temperature": [0.0, 1.0]},
        run_name="skinny-runtime",
    )
    try:
        mlflow.log_metric("accuracy", 0.9)
    finally:
        mlflow.end_run()

    runs = mlflow.search_runs(
        experiment_ids=[tracker.experiment_id],
        output_format="pandas",
    )

    assert isinstance(runs, pandas.DataFrame)
    matching_runs = runs.loc[runs["run_id"] == run_id]
    assert len(matching_runs) == 1
    assert matching_runs.iloc[0]["metrics.accuracy"] == pytest.approx(0.9)
