"""Backward compatibility shim for playground module.

.. deprecated:: 0.9.0
    The playground has been moved to the traigent-ui plugin.
    Install with: pip install traigent-ui

This shim provides backward compatibility by:
1. Trying to import from traigent-ui plugin (preferred)
2. Falling back to embedded implementation (deprecated)
"""

from __future__ import annotations

import warnings as _warnings

_PLUGIN_AVAILABLE = False

# Try to import from the plugin first
try:
    from traigent_ui import (
        UIPlugin,
        launch_playground,
        launch_playground_subprocess,
    )

    _PLUGIN_AVAILABLE = True

except ImportError:
    # Plugin not installed - provide basic stubs
    def launch_playground(port: int = 8501, **kwargs) -> None:
        """Launch the playground (requires traigent-ui plugin)."""
        raise ImportError(
            "The playground requires the traigent-ui plugin. "
            "Install with: pip install traigent-ui"
        )

    def launch_playground_subprocess(port: int = 8501):
        """Launch the playground in subprocess (requires traigent-ui plugin)."""
        raise ImportError(
            "The playground requires the traigent-ui plugin. "
            "Install with: pip install traigent-ui"
        )

    UIPlugin = None

    _warnings.warn(
        "playground module is deprecated. "
        "Install traigent-ui plugin: pip install traigent-ui",
        DeprecationWarning,
        stacklevel=2,
    )


def is_ui_available() -> bool:
    """Check if the UI plugin is available."""
    return _PLUGIN_AVAILABLE


__all__ = [
    "launch_playground",
    "launch_playground_subprocess",
    "UIPlugin",
    "is_ui_available",
]
