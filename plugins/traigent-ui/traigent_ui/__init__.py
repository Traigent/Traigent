"""Traigent UI Plugin - Streamlit-based playground for optimization.

This plugin provides a Streamlit-based UI for:
- Interactive optimization configuration
- Result visualization and analysis
- Problem generation and management
- Quick start wizards
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def launch_playground(port: int = 8501, **kwargs) -> None:
    """Launch the Traigent playground Streamlit app.

    Args:
        port: Port to run the Streamlit app on (default: 8501)
        **kwargs: Additional arguments passed to streamlit
    """
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        raise ImportError(
            "Streamlit is required to launch the playground. "
            "Install with: pip install streamlit"
        ) from None

    # Get the path to the main app
    import traigent_ui.traigent_control_center as app_module

    app_path = app_module.__file__

    sys.argv = ["streamlit", "run", app_path, "--server.port", str(port)]
    sys.exit(stcli.main())


def launch_playground_subprocess(port: int = 8501) -> subprocess.Popen:
    """Launch the playground in a subprocess.

    Args:
        port: Port to run the Streamlit app on

    Returns:
        The subprocess handle
    """
    import traigent_ui.traigent_control_center as app_module

    app_path = app_module.__file__

    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    )


class UIPlugin:
    """Traigent UI Plugin.

    Provides Streamlit-based UI for optimization workflows.
    """

    name = "ui"
    version = "0.1.0"
    min_traigent_version = "0.9.0"
    features = [
        "playground",
        "visualization",
        "problem_generation",
        "result_browser",
    ]

    @classmethod
    def initialize(cls) -> None:
        """Initialize the UI plugin."""
        # UI plugin doesn't need initialization
        pass

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup UI plugin resources."""
        pass

    @classmethod
    def get_capabilities(cls) -> dict:
        """Return plugin capabilities."""
        return {
            "playground": True,
            "visualization": True,
            "problem_generation": True,
            "result_browser": True,
        }


__all__ = [
    "launch_playground",
    "launch_playground_subprocess",
    "UIPlugin",
]
