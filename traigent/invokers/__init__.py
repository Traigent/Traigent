"""Function invocation strategies for Traigent SDK.

This module provides different strategies for invoking functions with configurations,
separated from the evaluation logic to enable hybrid local/cloud architectures.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-INVOKERS REQ-INV-006 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.invokers.base import (
    BaseInvoker,
    InvocationResult,
    StreamingChunk,
    StreamingInvocationResult,
)
from traigent.invokers.batch import BatchInvoker
from traigent.invokers.local import LocalInvoker
from traigent.invokers.streaming import StreamingInvoker

__all__ = [
    "BaseInvoker",
    "InvocationResult",
    "StreamingChunk",
    "StreamingInvocationResult",
    "LocalInvoker",
    "BatchInvoker",
    "StreamingInvoker",
]
