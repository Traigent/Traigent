"""Bounded wait on blocking work run in a one-shot worker thread."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable


def run_in_thread_with_timeout[T](func: Callable[[], T], timeout: float) -> T:
    """Run ``func`` in a worker thread and wait at most ``timeout`` seconds.

    The executor is deliberately not used as a context manager: its exit runs
    ``shutdown(wait=True)``, which would hold a timeout until the slow call
    finished (#2844, #3142). An abandoned worker is left to finish on its own
    and queued work is cancelled.

    Raises:
        TimeoutError: If ``func`` does not finish within ``timeout`` seconds.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func)
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
