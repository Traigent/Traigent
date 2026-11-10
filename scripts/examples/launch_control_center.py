#!/usr/bin/env python3
"""
Launch script for Traigent Playground
Run from anywhere: python launch_control_center.py
"""

import os
import subprocess
import sys
from pathlib import Path

#: The Playground app ships in the traigent-ui plugin.
APP_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "traigent-ui"
    / "traigent_ui"
    / "traigent_control_center.py"
)


def main():
    """Launch the Traigent Playground Streamlit app."""

    print("🎯 Launching Traigent Playground...")
    print("==================================")

    # Check for streamlit
    try:
        import streamlit  # noqa: F401 - Import check only

        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"]
        )
        print("✅ Streamlit installed successfully")

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   You can set it in the app's Settings tab")
        print()

    # Find the streamlit app
    app_path = APP_PATH

    if not app_path.exists():
        print(f"❌ Error: Could not find {app_path}")
        sys.exit(1)

    # Launch streamlit
    print("🚀 Starting Streamlit server...")
    print("   Opening in browser: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Run from the app's directory so its relative imports resolve
    os.chdir(app_path.parent)

    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Traigent Playground...")


if __name__ == "__main__":
    main()
