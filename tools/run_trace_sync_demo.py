"""Demonstrate optional TraceSync instrumentation bootstrap without touching SDK code.

This script:
- Enables the TraceSync bootstrap (no-op if trace runtime is missing).
- Shows which entrypoints were patched.
- Executes lightweight API calls to prove patched functions still work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patched(obj: Any) -> bool:
    return bool(getattr(obj, "_trace_sync_patched", False))


def main() -> None:
    # Set runtime env before importing trace_sync (config is read at import time)
    os.environ.setdefault("TRAIGENT_TRACE_SYNC", "1")
    os.environ.setdefault("TRACE_SYNC_ENABLED", "true")  # enable runtime logger
    os.environ.setdefault("TRACE_SYNC_SAMPLE_RATE", "1")  # deterministic for demo
    os.environ.setdefault(
        "TRACE_SYNC_LOG", str(ROOT / "runtime" / "traces" / "runtime.log")
    )  # place logs in Traigent runtime/traces

    from tools.trace_sync_bootstrap import TRACE_RUNTIME_AVAILABLE, enable_trace_sync

    enabled = enable_trace_sync()
    print(f"Trace runtime available: {TRACE_RUNTIME_AVAILABLE}")
    print(f"Bootstrap enabled: {enabled}")

    from traigent.api import functions as api_functions
    from traigent.api import decorators as api_decorators
    from traigent.core import optimized_function as opt_func
    from traigent.core import orchestrator
    from traigent.invokers import local as local_invokers
    from traigent.invokers import batch as batch_invokers
    from traigent.evaluators import local as local_evaluators

    patched_status = {
        "api.configure": _patched(api_functions.configure),
        "api.initialize": _patched(api_functions.initialize),
        "api.optimize_decorator": _patched(api_decorators.optimize),
        "optimized_function.optimize": _patched(opt_func.OptimizedFunction.optimize),
        "orchestrator.optimize": _patched(
            orchestrator.OptimizationOrchestrator.optimize
        ),
        "local_invoker.invoke": _patched(local_invokers.LocalInvoker.invoke),
        "local_invoker.invoke_batch": _patched(
            local_invokers.LocalInvoker.invoke_batch
        ),
        "batch_invoker.invoke_batch": _patched(
            batch_invokers.BatchInvoker.invoke_batch
        ),
        "local_evaluator.evaluate": _patched(local_evaluators.LocalEvaluator.evaluate),
    }

    print("Patched entrypoints:")
    for name, status in patched_status.items():
        print(f"  - {name}: {status}")

    # Exercise a couple of patched functions to ensure normal behavior
    api_functions.configure(logging_level="WARNING", parallel_workers=1)
    api_functions.initialize(api_key=None, api_url=None)
    print("configure() and initialize() executed successfully under bootstrap.")


if __name__ == "__main__":
    main()
