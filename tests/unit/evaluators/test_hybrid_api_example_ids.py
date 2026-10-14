"""Hybrid example ids come from the dataset position, never the batch position.

Regression tests for the hybrid ``ex_{i}`` fallback, where ``i`` was the index
inside the batch: at the default ``batch_size=1`` every example was ``ex_0``,
and the outcome matrix silently merged them into one row.

No network and no LLM calls: the external service is a fake transport that
echoes the example ids it receives.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.metadata_helpers import _build_measures_full
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult
from traigent.hybrid.protocol import HybridExecuteResponse, ServiceCapabilities
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.exceptions import ValidationError
from traigent.utils.outcome_matrix import build_outcome_matrix


def _echo_transport(wrong_question: str = "q1") -> MagicMock:
    """A fake external service: echoes ids, fails one question, records requests."""
    transport = MagicMock()
    transport.sent_batches: list[list[str]] = []

    async def execute(request: Any) -> HybridExecuteResponse:
        transport.sent_batches.append([ex["example_id"] for ex in request.examples])
        return HybridExecuteResponse(
            request_id=request.request_id,
            execution_id="exec",
            status="completed",
            outputs=[
                {
                    "example_id": ex["example_id"],
                    "output": ex["data"].get("q"),
                    "metrics": {
                        "accuracy": (
                            0.0 if ex["data"].get("q") == wrong_question else 1.0
                        )
                    },
                }
                for ex in request.examples
            ],
            operational_metrics={"cost_usd": 0.0, "latency_ms": 1.0},
            # Non-empty quality metrics select the combined (one-call) mode.
            quality_metrics={"accuracy": 1.0},
        )

    transport.execute = AsyncMock(side_effect=execute)
    transport.capabilities = AsyncMock(
        return_value=ServiceCapabilities(
            version="1.0", supports_evaluate=False, supports_keep_alive=False
        )
    )
    return transport


def _evaluator(transport: MagicMock, batch_size: int = 1) -> HybridAPIEvaluator:
    ev = HybridAPIEvaluator(
        transport=transport, tunable_id="cap", keep_alive=False, batch_size=batch_size
    )
    ev._benchmark_id = "bench"
    return ev


def _dataset(*questions: str) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": q}, expected_output=q) for q in questions
        ],
        name="hybrid-ids",
    )


def _sent_ids(transport: MagicMock) -> list[str]:
    return [eid for batch in transport.sent_batches for eid in batch]


@pytest.mark.asyncio
async def test_default_batch_size_gives_each_example_its_own_id() -> None:
    transport = _echo_transport()
    result = await _evaluator(transport).evaluate(
        lambda: None, {}, _dataset("q0", "q1", "q2")
    )

    assert _sent_ids(transport) == ["ex_0", "ex_1", "ex_2"]
    assert [r.example_id for r in result.example_results] == ["ex_0", "ex_1", "ex_2"]


@pytest.mark.asyncio
async def test_ids_carry_the_batch_offset() -> None:
    transport = _echo_transport()
    await _evaluator(transport, batch_size=2).evaluate(
        lambda: None, {}, _dataset("q0", "q1", "q2", "q3", "q4")
    )

    assert transport.sent_batches == [["ex_0", "ex_1"], ["ex_2", "ex_3"], ["ex_4"]]


@pytest.mark.asyncio
async def test_single_batch_wire_ids_are_unchanged() -> None:
    """A dataset that fits one batch sends exactly the ids it always did."""
    transport = _echo_transport()
    await _evaluator(transport, batch_size=10).evaluate(
        lambda: None, {}, _dataset("q0", "q1", "q2")
    )

    assert transport.sent_batches == [["ex_0", "ex_1", "ex_2"]]


@pytest.mark.asyncio
async def test_input_ids_are_kept_and_metadata_ids_do_not_reach_the_wire() -> None:
    """Only ids in the inputs replace the positional fallback.

    ``metadata["example_id"]`` (where a JSONL row's top-level ``example_id``
    lands) is not sent: a single-batch run keeps the exact wire ids it had.
    """
    transport = _echo_transport()
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"q": "q0"}, metadata={"example_id": "c-A"}),
            EvaluationExample(input_data={"q": "q1", "example_id": "svc-1"}),
            EvaluationExample(input_data={"q": "q2", "input_id": "svc-2"}),
        ],
        name="hybrid-ids",
    )
    result = await _evaluator(transport, batch_size=10).evaluate(
        lambda: None, {}, dataset
    )

    assert transport.sent_batches == [["ex_0", "svc-1", "svc-2"]]
    assert [r.example_id for r in result.example_results] == [
        "ex_0",
        "svc-1",
        "svc-2",
    ]


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        ({"example_id": "a", "input_id": "b"}, "a"),
        ({"example_id": "", "input_id": "svc"}, "svc"),
        ({"example_id": None, "input_id": "svc"}, "svc"),
        ({"example_id": ""}, "ex_0"),
        ({"example_id": "", "input_id": ""}, "ex_0"),
    ],
    ids=[
        "example_id-wins",
        "empty-falls-to-input_id",
        "none-falls-to-input_id",
        "empty-is-absent",
        "both-empty",
    ],
)
@pytest.mark.asyncio
async def test_example_id_precedence_and_absent_values(
    input_data: dict[str, Any], expected: str
) -> None:
    """``example_id`` beats ``input_id``; only ``None`` / ``""`` count as absent."""
    transport = _echo_transport()
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"q": "q0", **input_data})],
        name="hybrid-ids",
    )
    await _evaluator(transport).evaluate(lambda: None, {}, dataset)

    assert _sent_ids(transport) == [expected]


@pytest.mark.asyncio
async def test_duplicate_ids_raise_before_any_request() -> None:
    transport = _echo_transport()
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"q": "q0", "example_id": "secret-id"}),
            EvaluationExample(input_data={"q": "q1"}),
            EvaluationExample(input_data={"q": "q2", "example_id": "secret-id"}),
        ],
        name="hybrid-ids",
    )

    with pytest.raises(ValidationError, match="positions 0 and 2") as excinfo:
        await _evaluator(transport).evaluate(lambda: None, {}, dataset)

    assert "secret-id" not in str(excinfo.value)
    transport.execute.assert_not_awaited()


@pytest.mark.parametrize(
    "bad_id",
    [1, 0, 1.0, True, False, [1, 2], {"a": 1}],
    ids=["int", "zero", "float", "bool", "false", "list", "dict"],
)
@pytest.mark.asyncio
async def test_non_string_ids_are_rejected_before_any_request(bad_id: Any) -> None:
    """The protocol's ids are strings echoed exactly; anything else fails loudly.

    A non-string id cannot be matched or de-duplicated reliably (``1`` vs
    ``"1"`` vs ``1.0`` vs ``[1.0]`` ...), so it is refused instead of guessed.
    """
    transport = _echo_transport()
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"q": "q0", "example_id": "a"}),
            EvaluationExample(input_data={"q": "q1", "example_id": bad_id}),
        ],
        name="hybrid-ids",
    )

    with pytest.raises(ValidationError) as excinfo:
        await _evaluator(transport).evaluate(lambda: None, {}, dataset)
    # Exact text: the position and the type, never the id's value.
    assert str(excinfo.value) == (
        "Hybrid example at position 1 has an example_id / input_id of type "
        f"{type(bad_id).__name__}; the hybrid protocol requires a string id"
    )
    transport.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_ids_that_break_truth_tests_get_the_validation_error() -> None:
    """``pandas.NA`` / NumPy arrays raise on ``==``; they still get the clear error."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    for bad_id, type_name in ((pd.NA, "NAType"), (np.array(["a", "b"]), "ndarray")):
        transport = _echo_transport()
        dataset = Dataset(
            examples=[EvaluationExample(input_data={"q": "q0", "example_id": bad_id})],
            name="hybrid-ids",
        )
        with pytest.raises(
            ValidationError, match=f"position 0 .* of type {type_name};"
        ):
            await _evaluator(transport).evaluate(lambda: None, {}, dataset)
        transport.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_outcome_matrix_keeps_every_hybrid_example() -> None:
    """End to end: the failing example survives into the outcome matrix."""
    transport = _echo_transport(wrong_question="q1")
    evaluation = await _evaluator(transport).evaluate(
        lambda: None, {}, _dataset("q0", "q1", "q2")
    )
    trial = TrialResult(
        trial_id="trial_0",
        config={},
        metrics={},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"example_results": list(evaluation.example_results)},
    )
    matrix = build_outcome_matrix(
        OptimizationResult(
            trials=[trial],
            best_config={},
            best_score=0.0,
            optimization_id="opt",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )

    assert matrix["example_count"] == 3
    accuracy = {
        row["example_id"]: row["cells"]["trial_0"]["accuracy"]
        for row in matrix["examples"]
    }
    assert accuracy == {"ex_0": 1.0, "ex_1": 0.0, "ex_2": 1.0}


def test_portal_measure_ids_for_hybrid_results_are_unchanged() -> None:
    """Backend measure ids for hybrid rows stay ``ex_<hash>_<idx>``.

    The fallback keeps its ``ex_`` prefix so the measure resolver never mistakes
    it for the SDK's ``example_<idx>`` mint; the portal's ids do not move.
    """
    results = [
        HybridExampleResult(example_id=f"ex_{i}", metrics={"accuracy": 1.0})
        for i in range(3)
    ]
    measures = _build_measures_full(results, "accuracy", dataset_name="hybrid-ids")

    dataset_hash = compute_dataset_hash("hybrid-ids")
    assert [m["example_id"] for m in measures] == [
        generate_stable_example_id(dataset_hash, i) for i in range(3)
    ]


def _trial_with(trial_id: str, ids: list[str]) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        metrics={},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={
            "example_results": [
                HybridExampleResult(example_id=eid, metrics={"accuracy": 1.0})
                for eid in ids
            ]
        },
    )


def test_outcome_matrix_warns_when_one_trial_repeats_an_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A collision is reported (content-free) instead of being merged silently.

    It must not raise: with content identity, identical rows legitimately share
    one id inside a trial.
    """
    result = OptimizationResult(
        trials=[
            _trial_with("trial_0", ["customer-id-a", "customer-id-a", "b"]),
            _trial_with("trial_1", ["customer-id-a", "b"]),
        ],
        best_config={},
        best_score=0.0,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )

    with caplog.at_level(logging.WARNING):
        matrix = build_outcome_matrix(result)

    assert matrix["example_count"] == 2
    warnings = [r for r in caplog.records if "outcome matrix" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "trial_0" in message and "1 " in message
    assert "trial_1" not in message
    assert "customer-id-a" not in message


def test_outcome_matrix_without_collisions_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = OptimizationResult(
        trials=[_trial_with("trial_0", ["a", "b"])],
        best_config={},
        best_score=0.0,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )

    with caplog.at_level(logging.WARNING):
        build_outcome_matrix(result)

    assert not [r for r in caplog.records if "outcome matrix" in r.getMessage()]
