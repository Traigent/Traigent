"""Tests for invoker parameter validation safeguards."""

from __future__ import annotations

import pytest

from traigent.invokers.base import BaseInvoker, InvocationResult
from traigent.invokers.batch import BatchInvoker
from traigent.invokers.local import LocalInvoker
from traigent.utils.exceptions import InvocationError


class DummyInvoker(BaseInvoker):
    """Concrete invoker used exclusively for validation tests."""

    async def invoke(self, func, config, input_data):
        return InvocationResult(result=None, is_successful=True)

    async def invoke_batch(self, func, config, input_batch):
        return []

    def supports_streaming(self) -> bool:
        return False

    def supports_batch(self) -> bool:
        return False


def test_base_invoker_accepts_valid_parameters() -> None:
    invoker = DummyInvoker(timeout=5.0, max_retries=2)
    assert invoker.timeout == 5.0
    assert invoker.max_retries == 2


@pytest.mark.parametrize("timeout", [0, -1, "bad", BaseInvoker.MAX_TIMEOUT_SECONDS + 1])
def test_base_invoker_rejects_invalid_timeout(timeout) -> None:
    with pytest.raises(InvocationError):
        DummyInvoker(timeout=timeout)


@pytest.mark.parametrize("retries", [-1, 1.5, BaseInvoker.MAX_RETRIES_LIMIT + 1])
def test_base_invoker_rejects_invalid_retry_count(retries) -> None:
    with pytest.raises(InvocationError):
        DummyInvoker(max_retries=retries)


def test_batch_invoker_rejects_invalid_concurrency_settings() -> None:
    with pytest.raises(InvocationError):
        BatchInvoker(max_workers=0)

    with pytest.raises(InvocationError):
        BatchInvoker(batch_size=0)

    with pytest.raises(InvocationError):
        BatchInvoker(batch_timeout=-5)

    with pytest.raises(InvocationError):
        BatchInvoker(batch_timeout=BaseInvoker.MAX_TIMEOUT_SECONDS + 10)


def test_local_invoker_validates_injection_mode_and_param() -> None:
    with pytest.raises(InvocationError):
        LocalInvoker(injection_mode="")

    with pytest.raises(InvocationError):
        LocalInvoker(injection_mode="parameter", config_param=" ")

    # Valid configuration should not raise
    invoker = LocalInvoker(injection_mode="parameter", config_param="cfg")
    assert invoker.injection_mode == "parameter"
    assert invoker.config_param == "cfg"


@pytest.mark.asyncio
async def test_dummy_invoker_invoke_returns_success() -> None:
    invoker = DummyInvoker(timeout=1.0)

    async def stub_func():
        return "ok"

    result = await invoker.invoke(stub_func, {}, {})
    assert result.is_successful
