"""Helper utilities for test configuration."""

import json
from pathlib import Path
from typing import Any


def create_test_dataset(path: str, examples: list) -> None:
    """Create a test dataset file."""
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def create_test_config() -> dict[str, Any]:
    """Create a test configuration."""
    return {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 100}


def cleanup_test_files(*paths: str) -> None:
    """Clean up test files."""
    for path in paths:
        p = Path(path)
        if p.exists():
            p.unlink()


# Mock classes for testing when dependencies are missing
class MockOptimizationResult:
    """Mock optimization result for testing."""

    def __init__(self, best_score=0.9, best_config=None):
        self.best_score = best_score
        self.best_config = best_config or {"model": "gpt-4o-mini"}
        self.trials = []
        self.successful_trials = []
        self.status = "completed"
