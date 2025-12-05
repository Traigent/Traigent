"""Minimal runtime trace demo using the bootstraped hooks.

This script:
- Forces trace logging into this repo under runtime/traces/runtime.log
- Enables TraceSync runtime and bootstrap
- Invokes a simple function through LocalInvoker to emit concept_action/event records
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "runtime" / "traces" / "runtime.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Force runtime config before import
os.environ.setdefault("TRACE_SYNC_ENABLED", "true")
os.environ.setdefault("TRACE_SYNC_SAMPLE_RATE", "1")
os.environ.setdefault("TRACE_SYNC_LOG", str(LOG_PATH))
os.environ.setdefault("TRAIGENT_TRACE_SYNC", "1")

from tools.trace_sync_bootstrap import TRACE_RUNTIME_AVAILABLE, enable_trace_sync
from traigent.invokers.local import LocalInvoker


def demo_function(config: dict, text: str) -> str:
    """Simple callable used for trace emission."""
    return f"{text} | model={config.get('model', 'default')}"


def main() -> None:
    enabled = enable_trace_sync()
    print(f"Trace runtime available: {TRACE_RUNTIME_AVAILABLE}")
    print(f"Bootstrap enabled: {enabled}")
    print(f"Log path: {LOG_PATH}")

    invoker = LocalInvoker(injection_mode="parameter", config_param="config")
    config = {"model": "demo-model", "temperature": 0.5}
    input_data = {"text": "hello trace"}

    result = invoker.validate_function(demo_function)
    invoker.validate_config(config)
    invoker.validate_input(input_data)

    invocation = invoker._provider.inject_config(demo_function, config, "config")
    output = invocation(**input_data)
    print(f"Direct invocation output: {output}")

    # Through invoke to trigger patched hook
    import asyncio

    async def run_invoke():
        return await invoker.invoke(demo_function, config, input_data)

    invoked = asyncio.run(run_invoke())
    print(f"Invoker status: {invoked.is_successful}, output={invoked.result}")


if __name__ == "__main__":
    main()
