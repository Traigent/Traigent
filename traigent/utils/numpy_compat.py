"""Numpy compatibility utilities for JSON serialization.

This module provides utilities for handling numpy types in JSON serialization
contexts, with graceful degradation when numpy is not installed.
"""

# Traceability: CONC-Layer-Util CONC-Quality-Compatibility FUNC-STORAGE REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
    _SCALAR_TYPES: tuple[type[Any], ...] = (np.generic,)
    _ARRAY_TYPE: type[Any] = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    _SCALAR_TYPES = ()

    class _DummyArray:
        """Placeholder class when numpy is not available."""

        pass

    _ARRAY_TYPE = _DummyArray


def convert_numpy_value(value: Any) -> Any:
    """Convert numpy value to Python native type.

    Args:
        value: Possibly numpy scalar or array

    Returns:
        Python native equivalent (via .item() or .tolist())
        or the original value if not a numpy type
    """
    if _SCALAR_TYPES and isinstance(value, _SCALAR_TYPES):
        return value.item()
    if isinstance(value, _ARRAY_TYPE):
        return value.tolist()
    return value


def is_numpy_scalar(value: Any) -> bool:
    """Check if value is a numpy scalar type.

    Args:
        value: Value to check

    Returns:
        True if value is a numpy scalar, False otherwise
    """
    if not NUMPY_AVAILABLE:
        return False
    return isinstance(value, _SCALAR_TYPES)


def is_numpy_array(value: Any) -> bool:
    """Check if value is a numpy array.

    Args:
        value: Value to check

    Returns:
        True if value is a numpy array, False otherwise
    """
    if not NUMPY_AVAILABLE:
        return False
    return isinstance(value, _ARRAY_TYPE)


def is_numpy_type(value: Any) -> bool:
    """Check if value is any numpy type (scalar or array).

    Args:
        value: Value to check

    Returns:
        True if value is a numpy type, False otherwise
    """
    return is_numpy_scalar(value) or is_numpy_array(value)


__all__ = [
    "NUMPY_AVAILABLE",
    "convert_numpy_value",
    "is_numpy_scalar",
    "is_numpy_array",
    "is_numpy_type",
]
