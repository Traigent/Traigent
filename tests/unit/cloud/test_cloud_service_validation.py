"""Validation tests for TraigentCloudService."""

import pytest

from traigent.cloud.service import OptimizationRequest, TraigentCloudService
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError as ValidationException


def _sample_dataset() -> Dataset:
    example = EvaluationExample(
        input_data={"prompt": "Hello"},
        expected_output="Hi",
        metadata={},
    )
    return Dataset(examples=[example])


@pytest.mark.asyncio
async def test_process_request_rejects_empty_function_name():
    service = TraigentCloudService()
    request = OptimizationRequest(
        function_name="",
        dataset=_sample_dataset(),
        configuration_space={"temperature": [0.1, 0.5]},
        objectives=["accuracy"],
    )

    with pytest.raises(ValidationException):
        await service.process_optimization_request(request)


@pytest.mark.asyncio
async def test_process_request_rejects_unknown_billing_tier():
    service = TraigentCloudService()
    request = OptimizationRequest(
        function_name="greet",
        dataset=_sample_dataset(),
        configuration_space={"temperature": [0.1, 0.5]},
        objectives=["accuracy"],
        billing_tier="unknown",
    )

    with pytest.raises(ValidationException):
        await service.process_optimization_request(request)
