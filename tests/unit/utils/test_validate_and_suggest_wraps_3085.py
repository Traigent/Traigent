"""validate_and_suggest preserves the wrapped function's identity (#3085)."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from traigent.utils.exceptions import ValidationError
from traigent.utils.validation import validate_and_suggest


def test_sync_wrapper_preserves_metadata() -> None:
    @validate_and_suggest
    def f(*, configuration_space=None, objectives=None, eval_dataset=None):
        """doc"""
        return "ok"

    assert f.__name__ == "f"
    assert f.__doc__ == "doc"
    assert hasattr(f, "__wrapped__")
    assert "configuration_space" in inspect.signature(f).parameters
    assert not inspect.iscoroutinefunction(f)


def test_async_wrapper_is_coroutine_function_and_validates() -> None:
    @validate_and_suggest
    async def f(*, configuration_space=None, objectives=None, eval_dataset=None):
        """doc"""
        return "ok"

    assert inspect.iscoroutinefunction(f)
    assert f.__name__ == "f"
    assert asyncio.run(f(configuration_space={"a": [1, 2]})) == "ok"
    with pytest.raises(ValidationError):
        asyncio.run(f(configuration_space="not-a-dict"))
