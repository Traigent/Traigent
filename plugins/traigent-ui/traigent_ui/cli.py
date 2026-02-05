"""CLI entry point for Traigent UI plugin."""

from __future__ import annotations

import argparse


def main() -> None:
    """Main CLI entry point for traigent-ui."""
    parser = argparse.ArgumentParser(
        description="Launch Traigent Playground UI",
        prog="traigent-ui",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit app on (default: 8501)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    from traigent_ui import launch_playground

    launch_playground(port=args.port)


if __name__ == "__main__":
    main()
