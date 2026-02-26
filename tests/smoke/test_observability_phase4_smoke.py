"""Phase 4 observability smoke tests for SDK payload invariants."""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.types import TrialResult, TrialStatus

_ALLOWED_FAILURE_CLASSIFICATIONS = {
    "unknown",
    "below_trial_mean",
    "stable",
}

_ALLOWED_FAILURE_CLASSIFICATION_DETAILS = {
    "severe_regression",
    "regression",
    "below_trial_mean",
    "improved",
    "stable",
    "unknown",
}

_METRIC_KEY_PATTERN = re.compile(r"^[A-Za-z_]\w*$")


class _SmokeOrchestrator:
    def __init__(self) -> None:
        self._optimization_id = "obs-smoke-opt"
        self._workflow_traces_tracker = object()
        self._dataset_name = "obs_smoke_dataset"
        self.collect_workflow_span = MagicMock()


@pytest.mark.smoke
def test_observability_payload_invariants_from_trial_lifecycle() -> None:
    orchestrator = _SmokeOrchestrator()
    lifecycle = TrialLifecycle(orchestrator)

    trial_result = TrialResult(
        trial_id="trial_smoke_001",
        config={"temperature": 0.2},
        metrics={"score": 0.82, "total_cost": 0.05},
        status=TrialStatus.COMPLETED,
        duration=1.4,
        timestamp=datetime.now(UTC),
        metadata={
            "trial_number": 1,
            "examples_attempted": 3,
            "measures": [
                {"example_id": "ex_1", "metrics": {"score": 0.90}},
                {"example_id": "ex_2", "metrics": {"score": 0.70}},
                {"example_id": "ex_1", "metrics": {"score": 0.65}},
            ],
        },
    )

    now_ts = time.time()
    lifecycle._collect_workflow_span(
        trial_id="cfg_smoke_001",
        trial_result=trial_result,
        start_time=now_ts - 1.2,
        end_time=now_ts,
    )

    orchestrator.collect_workflow_span.assert_called_once()
    span = orchestrator.collect_workflow_span.call_args.args[0]

    lineage = span.metadata["lineage"]
    training_outcome = span.metadata["training_outcome"]

    # Cardinality and dedupe invariants.
    assert len(lineage["example_ids"]) == len(lineage["example_outcomes"])
    assert lineage["example_ids"] == ["ex_1", "ex_2"]

    # Trace linkage and classification invariants.
    for outcome in lineage["example_outcomes"]:
        assert outcome["failure_classification"] in _ALLOWED_FAILURE_CLASSIFICATIONS
        assert (
            outcome["failure_classification_detail"]
            in _ALLOWED_FAILURE_CLASSIFICATION_DETAILS
        )
        assert outcome["trace_linkage"]["trace_id"] == span.trace_id
        assert (
            outcome["trace_linkage"]["configuration_run_id"]
            == span.configuration_run_id
        )

    # Timestamp and metric key formatting invariants.
    assert not span.start_time.endswith("Z")
    assert not span.end_time.endswith("Z")
    for metric_key in training_outcome:
        assert _METRIC_KEY_PATTERN.match(metric_key) is not None

    assert span.status == "COMPLETED"
    assert isinstance(span.cost_usd, float)
    assert span.cost_usd >= 0.0
