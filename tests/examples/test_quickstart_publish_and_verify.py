"""Smoke tests for the publish-and-verify quickstart example."""

from __future__ import annotations

import importlib


def test_publish_and_verify_example_imports_without_running() -> None:
    module = importlib.import_module("traigent.examples.quickstart.publish_and_verify")

    assert module.EXPERIMENT_NAME == "Quickstart Publish and Verify"
    assert callable(module.build_answer)
