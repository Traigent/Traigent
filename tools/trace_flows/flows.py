"""Collection of small instrumentation flows to exercise trace hooks.

Each flow is a callable (sync or async) that triggers key entrypoints so
TraceSync runtime logging can capture concept/sync/event actions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, Iterable

from traigent.api import functions as api_functions
from traigent.invokers.local import LocalInvoker

Flow = Callable[[], Any | Coroutine[Any, Any, Any]]


def flow_api_config_init() -> None:
    """Hit API configure + initialize entrypoints."""
    api_functions.configure(logging_level="WARNING", parallel_workers=1)
    api_functions.initialize(api_key=None, api_url=None)


async def flow_invoker_basic() -> None:
    """Invoke a simple function through LocalInvoker to emit invoker traces."""

    def demo_function(config: dict, text: str) -> str:
        return f"{text} | model={config.get('model', 'default')}"

    invoker = LocalInvoker(injection_mode="parameter", config_param="config")
    config = {"model": "demo-model", "temperature": 0.5}
    input_data = {"text": "hello flow"}

    await invoker.invoke(demo_function, config, input_data)


async def flow_invoker_batch() -> None:
    """Batch invocation path."""

    def demo_function(config: dict, text: str) -> str:
        return f"{text} | model={config.get('model', 'default')}"

    invoker = LocalInvoker(injection_mode="parameter", config_param="config")
    config = {"model": "demo-model", "temperature": 0.1}
    batch = [
        {"text": "batch-1"},
        {"text": "batch-2"},
    ]
    await invoker.invoke_batch(demo_function, config, batch)


FLOWS: Dict[str, Flow] = {
    "api_config_init": flow_api_config_init,
    "invoker_basic": flow_invoker_basic,
    "invoker_batch": flow_invoker_batch,
}


def list_flows(selected: Iterable[str] | None = None) -> dict[str, Flow]:
    if not selected:
        return FLOWS
    names = set(selected)
    return {name: fn for name, fn in FLOWS.items() if name in names}
