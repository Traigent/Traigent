"""Configuration for end-to-end tests."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Configure async test execution
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
