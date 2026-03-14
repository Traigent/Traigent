"""Lightweight batch wrapping utilities for integration overrides.

These helpers provide simple batch and async batch execution around a single-call
function. They are intended to be attached to framework clients when native batch
APIs are missing so we can still offer a consistent `batch`/`abatch` surface.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

Call = Callable[..., Any]


class BatchWrapper:
    """Wrap a single-call function with batch helpers."""

    def __init__(
        self,
        single_call: Call,
        *,
        item_param: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        """
        Args:
            single_call: Callable that handles one item (sync or async).
            item_param: Optional name of the parameter that should receive each item
                when the item is not a mapping. If omitted, each item must be a
                mapping of call kwargs.
            concurrency: Optional max concurrent async calls for `abatch`.
                When None, calls are awaited sequentially.
        """
        self._single_call = single_call
        self._item_param = item_param
        self._concurrency = concurrency if concurrency and concurrency > 0 else None
        self._is_async_callable = inspect.iscoroutinefunction(single_call)

    def _build_kwargs(
        self, item: Any, shared_kwargs: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        base_kwargs = dict(shared_kwargs or {})
        if isinstance(item, Mapping):
            base_kwargs.update(item)
            return base_kwargs
        if self._item_param:
            base_kwargs[self._item_param] = item
            return base_kwargs
        raise TypeError(
            "BatchWrapper requires mapping items when item_param is not provided"
        )

    def batch(self, items: Iterable[Any], **shared_kwargs: Any) -> list[Any]:
        """Execute the wrapped callable over a collection synchronously."""
        if self._is_async_callable:
            raise TypeError("Wrapped callable is async; use abatch instead")

        results: list[Any] = []
        for item in items:
            call_kwargs = self._build_kwargs(item, shared_kwargs)
            results.append(self._single_call(**call_kwargs))
        return results

    async def _invoke_async(self, item: Any, shared_kwargs: Mapping[str, Any]) -> Any:
        call_kwargs = self._build_kwargs(item, shared_kwargs)
        result = self._single_call(**call_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def abatch(self, items: Iterable[Any], **shared_kwargs: Any) -> list[Any]:
        """Execute the wrapped callable over a collection asynchronously."""
        shared_map: Mapping[str, Any] = dict(shared_kwargs)

        # Sequential path for clarity and deterministic ordering
        if not self._concurrency:
            results = []
            for item in items:
                results.append(await self._invoke_async(item, shared_map))
            return results

        # Concurrency-limited path
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(item: Any) -> Any:
            async with semaphore:
                return await self._invoke_async(item, shared_map)

        tasks: list[asyncio.Task[Any]] = []
        for item in items:
            task = asyncio.create_task(run(item))
            tasks.append(task)
        return list(await asyncio.gather(*tasks))


def attach_batch_methods(
    obj: Any,
    *,
    method_name: str,
    item_param: str | None = None,
    concurrency: int | None = None,
    batch_attr: str = "batch",
    abatch_attr: str = "abatch",
) -> Any:
    """Attach batch/abatch helpers to an object if they are missing.

    Args:
        obj: Target instance to attach methods to.
        method_name: Name of the existing single-item method on the object.
        item_param: Optional parameter name to inject the item into when it is not a
            mapping (passed to BatchWrapper).
        concurrency: Optional max concurrent async calls for abatch.
        batch_attr: Attribute name to use for the sync batch method.
        abatch_attr: Attribute name to use for the async batch method.

    Returns:
        The same object for chaining.
    """
    single_call = getattr(obj, method_name)
    wrapper = BatchWrapper(single_call, item_param=item_param, concurrency=concurrency)

    if not hasattr(obj, batch_attr):
        setattr(obj, batch_attr, wrapper.batch)
    if not hasattr(obj, abatch_attr):
        setattr(obj, abatch_attr, wrapper.abatch)

    return obj
