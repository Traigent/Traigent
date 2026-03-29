#!/usr/bin/env python3
"""Shared setup utilities for Traigent examples.

This module provides common setup code to reduce duplication across examples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


def is_mock_mode() -> bool:
    """Check if running in mock mode."""
    return os.getenv("TRAIGENT_MOCK_LLM", "").lower() in {"1", "true", "yes", "y"}


def setup_mock_environment(base_path: Path) -> None:
    """Set up the environment for mock mode execution.

    Args:
        base_path: The base path of the example (usually Path(__file__).parent)
    """
    if not is_mock_mode():
        return

    # Set HOME to avoid permission issues in sandboxed environments
    os.environ["HOME"] = str(base_path)

    # Create a local results directory
    results_dir = base_path / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)


def resolve_sdk_path(caller_file: str | Path | None = None) -> str | None:
    """Resolve the Traigent SDK root directory.

    Resolution order:
      1. ``TRAIGENT_SDK_PATH`` environment variable (explicit override)
      2. Walk parent directories of *caller_file* looking for the package
      3. Walk parent directories of *this* file as a last resort

    Returns:
        Absolute path to the SDK root, or ``None`` when the package is
        probably pip-installed.
    """
    env_path = os.environ.get("TRAIGENT_SDK_PATH")
    if env_path:
        return str(Path(env_path).resolve())

    anchors: list[Path] = []
    if caller_file is not None:
        anchors.append(Path(caller_file).resolve())
    anchors.append(Path(__file__).resolve())

    for anchor in anchors:
        for depth in range(1, 7):
            try:
                parent = anchor.parents[depth]
                if (parent / "traigent" / "__init__.py").exists():
                    return str(parent)
            except IndexError:
                break
    return None


def setup_traigent_import(caller_file: str | Path | None = None) -> None:
    """Set up sys.path to allow importing traigent from the repo.

    Checks ``TRAIGENT_SDK_PATH`` first, then falls back to parent-directory
    detection.  Pass *caller_file* (``__file__``) from example scripts so the
    walk starts from the correct location.
    """
    try:
        import traigent  # noqa: F401

        return  # Already importable
    except ImportError:
        pass

    sdk_root = resolve_sdk_path(caller_file)
    if sdk_root and sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)


def initialize_traigent(execution_mode: str = "edge_analytics") -> None:
    """Initialize traigent with appropriate settings.

    Args:
        execution_mode: The execution mode to use (default: edge_analytics)
    """
    import logging

    import traigent

    logger = logging.getLogger(__name__)

    if is_mock_mode():
        try:
            traigent.initialize(execution_mode=execution_mode)
        except Exception as e:
            # Log the error but continue - mock mode may work without full initialization
            logger.warning(
                f"Traigent initialization failed in mock mode: {e}. "
                "Continuing anyway - some features may be unavailable."
            )


def get_datasets_path() -> Path:
    """Get the path to the shared datasets directory."""
    return Path(__file__).resolve().parents[1] / "datasets"


def get_example_dataset(
    example_name: str, filename: str = "evaluation_set.jsonl"
) -> str:
    """Get the path to a dataset for a specific example.

    Args:
        example_name: Name of the example (e.g., "rag-optimization", "simple-prompt")
        filename: Name of the dataset file (default: "evaluation_set.jsonl")

    Returns:
        String path to the dataset file
    """
    return str(get_datasets_path() / example_name / filename)


# Quick setup function for examples
def quick_setup(base_path: Path, execution_mode: str = "edge_analytics") -> bool:
    """Perform all common setup steps for an example.

    Args:
        base_path: The base path of the example (usually Path(__file__).parent)
        execution_mode: The execution mode to use

    Returns:
        True if running in mock mode, False otherwise
    """
    mock = is_mock_mode()
    setup_mock_environment(base_path)
    setup_traigent_import()
    initialize_traigent(execution_mode)
    return mock


if __name__ == "__main__":
    try:
        print("Traigent Examples Setup Utilities")
        print(f"Mock mode: {is_mock_mode()}")
        print(f"Datasets path: {get_datasets_path()}")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
