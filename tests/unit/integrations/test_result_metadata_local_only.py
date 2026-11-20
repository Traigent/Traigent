"""The per-example consistency report is invisible to metadata serializers.

The report is stored in ``result.example_consistency_report``, not in
``result.metadata``. Code that serializes a result's metadata (the W&B and
MLflow trackers, the MCP result tool) is run here against a real result that
carries a report, with no per-sink filtering involved.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import traigent.integrations.observability.mlflow as ml_mod
import traigent.integrations.observability.wandb as wb_mod
from tests.unit.core.test_winner_stability_example_consistency import _run
from traigent.integrations.observability.mlflow import TraigentMLflowTracker
from traigent.integrations.observability.wandb import TraigentWandBTracker
from traigent.mcp.tools import _serialize_loaded_result

# Mocked tracker transports, gated while offline (#3186): connected lane.
pytestmark = pytest.mark.backend_online


async def _result_with_report():
    result = await _run(5)
    report = result.example_consistency_report
    assert report and report["examples"]
    result.metadata["ordinary_key"] = "ordinary_value"
    sentinels = [
        '"example_consistency"',
        '"example_consistency_report"',
        *report["examples"],
        *{row["label"] for row in report["examples"].values()},
    ]
    return result, sentinels


def _assert_clean(blob: str, sentinels: list[str]) -> None:
    for sentinel in sentinels:
        assert sentinel not in blob, sentinel


@pytest.mark.asyncio
async def test_metadata_does_not_carry_the_report() -> None:
    result, sentinels = await _result_with_report()

    _assert_clean(json.dumps(result.metadata, default=str), sentinels)


@pytest.mark.asyncio
async def test_wandb_completion_carries_no_report(monkeypatch, tmp_path) -> None:
    result, sentinels = await _result_with_report()
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

    tracker.log_optimization_result(result)

    _assert_clean(json.dumps(m.log.call_args_list, default=str), sentinels)
    assert m.log.call_args.args[0]["metadata/ordinary_key"] == "ordinary_value"
    _assert_clean(saved["optimization_result.json"], sentinels)


@pytest.mark.asyncio
async def test_mlflow_completion_carries_no_report(monkeypatch) -> None:
    result, sentinels = await _result_with_report()
    m = MagicMock(name="mlflow")
    monkeypatch.setattr(ml_mod, "mlflow", m)
    tracker = object.__new__(TraigentMLflowTracker)
    tracker.current_run = NS(info=NS(run_id="rid"))

    tracker.log_optimization_result(result)

    calls = m.log_param.call_args_list + m.log_metric.call_args_list
    _assert_clean(json.dumps(calls, default=str), sentinels)
    m.log_param.assert_any_call("metadata_ordinary_key", "ordinary_value")
    payload, name = m.log_dict.call_args.args
    assert name == "optimization_result.json"
    _assert_clean(json.dumps(payload, default=str), sentinels)


@pytest.mark.asyncio
async def test_mcp_result_payload_carries_no_report() -> None:
    result, sentinels = await _result_with_report()

    payload = _serialize_loaded_result(result)

    _assert_clean(json.dumps(payload, default=str), sentinels)
    assert payload["metadata"]["ordinary_key"] == "ordinary_value"
