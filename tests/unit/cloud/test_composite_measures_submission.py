"""Composite telemetry rides the measures wire channel end-to-end.

This is the INTEGRATION test for the composite telemetry feature: it proves the
full path from a composite run's §3.10 measures to the trial-submission HTTP
body, exercising the SAME mechanism every user follows.

The user-facing mechanism (traced in
``traigent/cloud/trial_operations.py``):

1. A trial's numeric metrics flow into
   ``TrialOperations.submit_trial_result_via_session(metrics=...)``.
2. There, ``_extract_measures_from_metrics`` splits off the reserved
   ``measures``/``summary_stats`` keys; the remaining numeric metrics
   (``clean_metrics``) are validated through ``MeasuresDict`` (≤ 50
   Python-identifier keys, numeric values only) and become the trial's
   per-trial measures in the posted body's ``metrics``.

So a user merges ``composite_measures(run)`` into the metrics their evaluated
function returns; those ``composite_*`` keys then ride the existing channel.
This test mocks the HTTP layer (mirroring
``tests/unit/cloud/test_trial_operations.py``) and asserts the posted body's
``metrics`` carries the flattened ``composite_*`` keys.

No new wire surface is introduced — the composite keys are ordinary numeric
metrics on a path that already exists.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.trial_operations import TrialOperations
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.knobs.patterns import binary_cascade
from traigent.knobs.runtime import StageRunner, execute_composite
from traigent.knobs.telemetry import merge_composite_measures

GATE = "router_margin_threshold"


def _stage(outputs: list[str]) -> StageRunner:
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


def _evaluate_one_item_metrics() -> dict[str, float]:
    """Mirror what a decorated function's evaluation produces for one trial.

    A composite is executed in-trial (deterministic stub stages, no LLM calls),
    the user's own score is recorded, and the composite telemetry is merged into
    the SAME metrics dict — exactly the documented integration recipe.
    """
    composite = binary_cascade(
        "answerer", base_stage="cheap", expert_stage="strong", threshold=GATE
    )
    # cheap split 2/3 -> margin 0.666 < theta 0.9 -> escalate to the expert arm.
    run = execute_composite(
        composite.structure,
        {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
        config={"variant": "strong"},
        calibrated_values={GATE: 0.9},
    )
    metrics: dict[str, float] = {"accuracy": 1.0}
    merge_composite_measures(metrics, run)
    return metrics


def _build_post_mock() -> tuple[Mock, Mock]:
    """Return (mock_session, mock_aiohttp_ClientSession) capturing post()."""
    mock_response = Mock()
    mock_response.status = 201

    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = Mock(return_value=mock_post_ctx)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_session, mock_session_ctx


@pytest.mark.asyncio
async def test_composite_measures_appear_in_posted_trial_metrics() -> None:
    """The flattened composite_* keys ride the posted trial-result body."""
    mock_client = Mock()
    mock_client.backend_config = Mock()
    mock_client.backend_config.backend_base_url = "https://api.example.com"
    mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
    mock_client.auth_manager = AsyncMock()
    mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
    mock_client._map_to_backend_status = Mock(return_value="COMPLETED")
    mock_client._normalize_execution_mode = Mock(return_value="hybrid")
    mock_client._sanitize_error_message = Mock(return_value="")

    ops = TrialOperations(mock_client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)

    # The user's evaluation produces metrics WITH the composite telemetry merged.
    metrics = _evaluate_one_item_metrics()
    # Pre-condition: the recipe actually produced composite keys to transport.
    assert metrics["composite_escalation_rate"] == 1.0
    assert metrics["composite_stage_selected"] == 1
    assert metrics["composite_gate_0_margin_pass_rate"] == 0.0

    mock_session, mock_session_ctx = _build_post_mock()

    with (
        patch(
            "traigent.cloud.trial_operations.is_backend_offline",
            return_value=False,
        ),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        # Real schema validation would require a full result envelope; the
        # measures-channel transport is the unit under test, so neutralize it
        # exactly as the sibling test in test_trial_operations.py does.
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
    ):
        mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
        mock_aiohttp.ClientTimeout = Mock()

        result = await ops.submit_trial_result_via_session(
            session_id="session_123",
            trial_id="trial_001",
            config={"variant": "strong"},
            metrics=metrics,
            status="completed",
            execution_mode="hybrid",
        )

    assert result is True

    # The posted body is the json= kwarg of session.post(...).
    posted_body = mock_session.post.call_args.kwargs["json"]
    posted_metrics = posted_body["metrics"]

    # The composite_* keys rode the channel as ordinary numeric metrics,
    # alongside the user's own metric.
    assert posted_metrics["accuracy"] == 1.0
    assert posted_metrics["composite_escalation_rate"] == 1.0
    assert posted_metrics["composite_stage_selected"] == 1
    assert posted_metrics["composite_gate_0_margin_pass_rate"] == 0.0

    # Content-free: no produced output value leaked onto the wire; every
    # transported metric value is a plain number.
    assert all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in posted_metrics.values()
    )


@pytest.mark.asyncio
async def test_composite_metrics_pass_measuresdict_on_the_submission_path() -> None:
    """The clean_metrics MeasuresDict validation accepts composite_* keys.

    ``submit_trial_result_via_session`` constructs ``MeasuresDict(clean_metrics)``
    before posting; a composite key that violated the identifier/numeric
    contract would trip a validation warning and submit unvalidated. This test
    asserts the composite keys validate cleanly (no warning path), proving the
    flattener's MeasuresDict-compliance holds on the REAL submission code.
    """
    mock_client = Mock()
    mock_client.backend_config = Mock()
    mock_client.backend_config.backend_base_url = "https://api.example.com"
    mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
    mock_client.auth_manager = AsyncMock()
    mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
    mock_client._map_to_backend_status = Mock(return_value="COMPLETED")
    mock_client._normalize_execution_mode = Mock(return_value="hybrid")
    mock_client._sanitize_error_message = Mock(return_value="")

    ops = TrialOperations(mock_client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)

    metrics = _evaluate_one_item_metrics()
    mock_session, mock_session_ctx = _build_post_mock()

    with (
        patch(
            "traigent.cloud.trial_operations.is_backend_offline",
            return_value=False,
        ),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        patch("traigent.cloud.trial_operations.logger") as mock_logger,
    ):
        mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
        mock_aiohttp.ClientTimeout = Mock()

        await ops.submit_trial_result_via_session(
            session_id="session_123",
            trial_id="trial_001",
            config={"variant": "strong"},
            metrics=metrics,
            status="completed",
            execution_mode="hybrid",
        )

    # No "Metrics validation warning" was emitted — composite keys validated.
    warning_msgs = [
        call.args[0]
        for call in mock_logger.warning.call_args_list
        if call.args and isinstance(call.args[0], str)
    ]
    assert not any("Metrics validation warning" in msg for msg in warning_msgs), (
        f"composite metrics tripped MeasuresDict validation: {warning_msgs}"
    )


# ---------------------------------------------------------------------------
# Upstream hop: a TUPLE-RETURNING decorated function drives the REAL evaluator,
# and the composite_* keys reach the posted trial-result body.
#
# The existing tests above start from ``submit_trial_result_via_session`` with a
# pre-merged metrics dict — they proved only the DOWNSTREAM half. The smoke that
# motivated this packet showed the UPSTREAM half was broken: the decorated
# function returned ``(output, metrics_dict)``, but tuple[1] was never extracted
# into the evaluator's trial metrics, so the composite_* keys never reached the
# wire (and the raw tuple poisoned the accuracy comparison). This test exercises
# the real LocalEvaluator lane end-to-end.
# ---------------------------------------------------------------------------


async def _trial_metrics_from_tuple_returning_function() -> dict[str, float]:
    """Run the REAL evaluator over a tuple-returning function (offline)."""
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="STRONG"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="STRONG"),
        ],
        name="composite_upstream_hop",
    )

    async def answer(text: str) -> tuple[str, dict[str, float]]:
        composite = binary_cascade(
            "answerer", base_stage="cheap", expert_stage="strong", threshold=GATE
        )
        run = execute_composite(
            composite.structure,
            {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
            config={"variant": "strong"},
            calibrated_values={GATE: 0.9},
        )
        metrics: dict[str, float] = {}
        merge_composite_measures(metrics, run)
        return str(run.output), metrics

    evaluator = LocalEvaluator(
        metrics=["accuracy"], detailed=True, execution_mode="edge_analytics"
    )
    result = await evaluator.evaluate(answer, {}, dataset)
    return result.metrics


@pytest.mark.asyncio
async def test_tuple_returning_function_metrics_reach_posted_body() -> None:
    """The composite_* keys produced upstream reach the posted trial body."""
    trial_metrics = await _trial_metrics_from_tuple_returning_function()

    # Pre-condition: the REAL evaluator surfaced the composite keys AND derived
    # accuracy from tuple[0] (== "STRONG" == expected) rather than the raw tuple.
    assert trial_metrics["accuracy"] == 1.0
    assert trial_metrics["composite_escalation_rate"] == 1.0
    assert trial_metrics["composite_stage_selected"] == 1
    assert trial_metrics["composite_gate_0_margin_pass_rate"] == 0.0

    mock_client = Mock()
    mock_client.backend_config = Mock()
    mock_client.backend_config.backend_base_url = "https://api.example.com"
    mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
    mock_client.auth_manager = AsyncMock()
    mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
    mock_client._map_to_backend_status = Mock(return_value="COMPLETED")
    mock_client._normalize_execution_mode = Mock(return_value="hybrid")
    mock_client._sanitize_error_message = Mock(return_value="")

    ops = TrialOperations(mock_client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)

    mock_session, mock_session_ctx = _build_post_mock()

    with (
        patch(
            "traigent.cloud.trial_operations.is_backend_offline",
            return_value=False,
        ),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
    ):
        mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
        mock_aiohttp.ClientTimeout = Mock()

        result = await ops.submit_trial_result_via_session(
            session_id="session_123",
            trial_id="trial_001",
            config={"variant": "strong"},
            metrics=trial_metrics,
            status="completed",
            execution_mode="hybrid",
        )

    assert result is True

    posted_metrics = mock_session.post.call_args.kwargs["json"]["metrics"]
    assert posted_metrics["accuracy"] == 1.0
    assert posted_metrics["composite_escalation_rate"] == 1.0
    assert posted_metrics["composite_stage_selected"] == 1
    assert posted_metrics["composite_gate_0_margin_pass_rate"] == 0.0
