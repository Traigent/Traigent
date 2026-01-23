"""Bridge implementations for external runtime execution.

This module provides bridges that allow Traigent to orchestrate optimization
trials in external runtimes (e.g., Node.js, Deno) via subprocess communication.
"""

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeConfig,
    JSBridgeError,
    JSProcessError,
    JSProtocolError,
    JSTrialResult,
    JSTrialTimeoutError,
)

__all__ = [
    "JSBridge",
    "JSBridgeConfig",
    "JSBridgeError",
    "JSProcessError",
    "JSProtocolError",
    "JSTrialResult",
    "JSTrialTimeoutError",
]
