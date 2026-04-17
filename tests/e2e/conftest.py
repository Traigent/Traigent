"""Configuration for end-to-end tests."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure async test execution
@pytest.fixture(scope="session")
def event_loop_policy():
    """Choose the loop policy once and let pytest-asyncio own loop lifecycles."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# Mark all async tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run performance benchmarks",
    )
