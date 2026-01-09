"""TunedVariable abstractions for Traigent optimization.

This module provides utilities for working with tunable variables including:
- Callable auto-discovery from modules (P-5)
- Domain-specific variable presets
- Variable analysis utilities
"""

from __future__ import annotations

from .discovery import (
    CallableInfo,
    discover_callables,
    discover_callables_by_decorator,
    filter_by_signature,
)

__all__ = [
    "CallableInfo",
    "discover_callables",
    "discover_callables_by_decorator",
    "filter_by_signature",
]
