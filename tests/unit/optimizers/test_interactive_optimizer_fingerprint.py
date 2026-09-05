"""InteractiveOptimizer direct-construction path (docs/user-guide/
interactive_optimization.md:100-116) carries the dataset content
fingerprint through to the session-create payload, the same way the
@optimize decorator path already does
(traigent/core/optimized_function.py:2007-2033).

Before this change, `InteractiveOptimizer.__init__` accepted
`artifact_fingerprints=` explicitly but had no way to derive it from a
dataset the caller passed in -- constructing it the documented way (only
`dataset_metadata=`) never populated the fingerprint at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.models import (
    OptimizationSessionStatus,
    SessionCreationResponse,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import (
    InteractiveOptimizer,
    RemoteGuidanceService,
)


def _example(input_data, expected_output):
    return EvaluationExample(input_data=input_data, expected_output=expected_output)


@pytest.fixture
def mock_remote_service():
    service = Mock(spec=RemoteGuidanceService)
    service.create_session = AsyncMock(
        return_value=SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
    )
    return service


@pytest.mark.asyncio
async def test_dataset_kwarg_carries_fingerprint_into_session_payload(
    mock_remote_service,
):
    dataset = Dataset(
        examples=[
            _example({"question": "a"}, "answer-a"),
            _example({"question": "b"}, "answer-b"),
        ]
    )

    optimizer = InteractiveOptimizer(
        config_space={"temperature": (0.0, 1.0)},
        objectives=["accuracy"],
        remote_service=mock_remote_service,
        dataset_metadata={"size": 2, "type": "qa"},
        dataset=dataset,
    )

    await optimizer.initialize_session(function_name="qa_agent", max_trials=10)

    mock_remote_service.create_session.assert_called_once()
    request = mock_remote_service.create_session.call_args[0][0]

    assert request.artifact_fingerprints is not None
    fingerprint = request.artifact_fingerprints["dataset"]
    assert fingerprint is not None
    assert fingerprint.startswith("fp1:")


@pytest.mark.asyncio
async def test_no_dataset_kwarg_leaves_fingerprints_none(mock_remote_service):
    """Documented usage (docs/user-guide/interactive_optimization.md) only
    passes dataset_metadata -- no materialized content is available, so no
    fingerprint should be invented."""
    optimizer = InteractiveOptimizer(
        config_space={"temperature": (0.0, 1.0)},
        objectives=["accuracy"],
        remote_service=mock_remote_service,
        dataset_metadata={"size": 10000, "type": "qa"},
    )

    await optimizer.initialize_session(function_name="qa_agent", max_trials=10)

    request = mock_remote_service.create_session.call_args[0][0]
    assert request.artifact_fingerprints is None


@pytest.mark.asyncio
async def test_explicit_artifact_fingerprints_takes_precedence_over_dataset(
    mock_remote_service,
):
    """Backward compatibility: an explicit artifact_fingerprints= (the
    pre-existing supported usage) must not be overridden by a dataset=
    kwarg."""
    explicit = {
        "dataset": "fp1:" + ("a" * 64),
        "agent": None,
        "evaluator": None,
        "config_space": None,
    }
    dataset = Dataset(examples=[_example({"q": 1}, "a")])

    optimizer = InteractiveOptimizer(
        config_space={"temperature": (0.0, 1.0)},
        objectives=["accuracy"],
        remote_service=mock_remote_service,
        artifact_fingerprints=explicit,
        dataset=dataset,
    )

    await optimizer.initialize_session(function_name="qa_agent", max_trials=10)

    request = mock_remote_service.create_session.call_args[0][0]
    assert request.artifact_fingerprints == explicit


@pytest.mark.asyncio
async def test_generator_dataset_is_not_consumed_and_yields_no_fingerprint(
    mock_remote_service,
):
    def gen():
        yield _example({"q": 1}, "a")
        yield _example({"q": 2}, "b")

    generator = gen()

    optimizer = InteractiveOptimizer(
        config_space={"temperature": (0.0, 1.0)},
        objectives=["accuracy"],
        remote_service=mock_remote_service,
        dataset_metadata={"size": 2},
        dataset=generator,
    )

    assert optimizer.artifact_fingerprints is None

    # Caller can still read every item -- construction never drained it.
    remaining = list(generator)
    assert len(remaining) == 2
