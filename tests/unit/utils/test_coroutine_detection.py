"""Tests for shared coroutine callable detection."""

from __future__ import annotations

import functools

from traigent.utils.function_identity import is_coroutine_callable


async def async_fn() -> str:
    return "async"


def sync_fn() -> str:
    return "sync"


def test_plain_async_def_is_coroutine_callable() -> None:
    assert is_coroutine_callable(async_fn) is True


def test_plain_sync_def_is_not_coroutine_callable() -> None:
    assert is_coroutine_callable(sync_fn) is False


def test_partial_async_def_is_coroutine_callable() -> None:
    assert is_coroutine_callable(functools.partial(async_fn)) is True


def test_nested_partial_async_def_is_coroutine_callable() -> None:
    partial_async = functools.partial(functools.partial(async_fn))

    assert is_coroutine_callable(partial_async) is True


def test_wraps_decorated_sync_wrapper_follows_wrapped_chain() -> None:
    @functools.wraps(async_fn)
    def wrapper() -> str:
        return "sync wrapper"

    assert is_coroutine_callable(wrapper) is True


def test_partial_sync_def_is_not_coroutine_callable() -> None:
    assert is_coroutine_callable(functools.partial(sync_fn)) is False
