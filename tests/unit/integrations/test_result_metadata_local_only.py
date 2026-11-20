"""Per-example consistency labels never reach W&B or MLflow uploads."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import traigent.integrations.observability.mlflow as ml_mod
import traigent.integrations.observability.wandb as wb_mod
from traigent.integrations.observability.mlflow import TraigentMLflowTracker
from traigent.integrations.observability.wandb import TraigentWandBTracker
from traigent.utils.example_consistency import (
    EXAMPLE_CONSISTENCY_METADATA_KEY,
    without_local_only_metadata,
)

# Mocked tracker transports, gated while offline (#3186): connected lane.
pytestmark = pytest.mark.backend_online

EXAMPLE_ID = "ex-distinct-7f3a91"
LABEL = "flaky"
SENTINELS = (EXAMPLE_CONSISTENCY_METADATA_KEY, EXAMPLE_ID, LABEL)


def _metadata() -> dict:
    return {
        "ordinary_key": "ordinary_value",
        EXAMPLE_CONSISTENCY_METADATA_KEY: {
            "reps": 3,
            "examples": {EXAMPLE_ID: {"pass": 1, "fail": 2, "label": LABEL}},
        },
    }


def _result(metadata: dict) -> NS:
    return NS(
        best_config={},
        best_metrics={},
        trials=[],
        duration=1.0,
        status=NS(value="completed"),
        metadata=metadata,
    )


def _assert_clean(blob: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in blob


def test_helper_returns_detached_copy() -> None:
    meta = _metadata()
    cleaned = without_local_only_metadata(meta)
    assert cleaned == {"ordinary_key": "ordinary_value"}
    assert EXAMPLE_CONSISTENCY_METADATA_KEY in meta
    assert without_local_only_metadata(None) == {}


def test_wandb_completion_excludes_local_only_metadata(monkeypatch, tmp_path) -> None:
    m = MagicMock(name="wandb")
    saved: dict = {}

    def _save(path):
        with open(path) as f:
            saved[path] = f.read()

    m.save.side_effect = _save
    monkeypatch.setattr(wb_mod, "wandb", m)
    monkeypatch.chdir(tmp_path)
    tracker = object.__new__(TraigentWandBTracker)
    tracker.current_run = NS(id="rid")
    result = _result(_metadata())

    tracker.log_optimization_result(result)

    logged = json.dumps(m.log.call_args_list, default=str)
    _assert_clean(logged)
    assert m.log.call_args.args[0]["metadata/ordinary_key"] == "ordinary_value"
    artifact = saved["optimization_result.json"]
    _assert_clean(artifact)
    assert json.loads(artifact)["metadata"] == {"ordinary_key": "ordinary_value"}
    labels = result.metadata[EXAMPLE_CONSISTENCY_METADATA_KEY]
    assert labels["examples"][EXAMPLE_ID]["label"] == LABEL


def test_mlflow_completion_excludes_local_only_metadata(monkeypatch) -> None:
    m = MagicMock(name="mlflow")
    monkeypatch.setattr(ml_mod, "mlflow", m)
    tracker = object.__new__(TraigentMLflowTracker)
    tracker.current_run = NS(info=NS(run_id="rid"))
    result = _result(_metadata())

    tracker.log_optimization_result(result)

    calls = m.log_param.call_args_list + m.log_metric.call_args_list
    _assert_clean(json.dumps(calls, default=str))
    m.log_param.assert_any_call("metadata_ordinary_key", "ordinary_value")
    payload, name = m.log_dict.call_args.args
    assert name == "optimization_result.json"
    _assert_clean(json.dumps(payload, default=str))
    assert payload["metadata"] == {"ordinary_key": "ordinary_value"}
    labels = result.metadata[EXAMPLE_CONSISTENCY_METADATA_KEY]
    assert labels["examples"][EXAMPLE_ID]["label"] == LABEL
