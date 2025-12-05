#!/usr/bin/env python3
"""
Launch script for TraiGent Playground
Run from anywhere: python launch_control_center.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the TraiGent Playground Streamlit app."""

    print("🎯 Launching TraiGent Playground...")
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
    script_dir = Path(__file__).parent
    app_path = script_dir / "examples" / "traigent_control_center.py"

    if not app_path.exists():
        print(f"❌ Error: Could not find {app_path}")
        sys.exit(1)

    # Launch streamlit
    print("🚀 Starting Streamlit server...")
    print("   Opening in browser: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Change to project directory to ensure imports work
    os.chdir(script_dir)

    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n👋 Shutting down TraiGent Playground...")


if __name__ == "__main__":
    main()
