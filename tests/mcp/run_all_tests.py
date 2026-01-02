"""Comprehensive test runner for MCP integration tests.

This script runs all MCP tests with proper coverage reporting and cleanup.
"""

import asyncio
import sys
from pathlib import Path

import coverage
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_mcp_tests():
    """Run all MCP tests with coverage reporting."""
    print("Starting MCP Integration Test Suite")
    print("=" * 60)

    # Initialize coverage
    cov = coverage.Coverage(source=["traigent"])
    cov.start()

    # Test categories
    test_suites = [
        ("Unit Tests", "test_mcp_unit.py"),
        ("Integration Tests", "test_mcp_integration.py"),
        ("LLM Task Interpretation Tests", "test_llm_task_interpretation.py"),
    ]

    all_passed = True
    results = []

    for suite_name, test_file in test_suites:
        print(f"\n{suite_name}")
        print("-" * 40)

        # Run pytest for each test file
        test_path = Path(__file__).parent / test_file
        result = pytest.main(
            [str(test_path), "-v", "--tb=short", "--no-header", "--quiet"]
        )

        passed = result == 0
        all_passed = all_passed and passed
        results.append((suite_name, passed))

        print(
            f"{'✓' if passed else '✗'} {suite_name}: {'PASSED' if passed else 'FAILED'}"
        )

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for suite_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{suite_name:<40} {status}")

    # Coverage report
    print("\nCoverage Report:")
    print("-" * 40)
    cov.report(show_missing=True, skip_covered=False)

    # Generate HTML coverage report
    html_dir = Path(__file__).parent / "htmlcov"
    cov.html_report(directory=str(html_dir))
    print(f"\nDetailed HTML coverage report generated at: {html_dir}/index.html")

    return all_passed


async def run_cleanup_verification():
    """Verify that all resources are properly cleaned up after tests."""
    print("\nRunning Cleanup Verification")
    print("-" * 40)

    # Import test context
    from conftest import MCPTestContext, cleanup_mcp_resources

    # Create test context
    context = MCPTestContext([], [], [], {})

    # Simulate resource creation
    context.created_agents.extend(["agent-1", "agent-2", "agent-3"])
    context.created_sessions.extend(["session-1", "session-2"])
    context.temp_files.append(Path("/tmp/test_file.json"))

    print(f"Created {len(context.created_agents)} agents")
    print(f"Created {len(context.created_sessions)} sessions")
    print(f"Created {len(context.temp_files)} temp files")

    # Run cleanup
    async with cleanup_mcp_resources(context):
        pass

    # Verify cleanup
    assert len(context.active_resources) == 0
    print("✓ All resources cleaned up successfully")


def main():
    """Main entry point for test runner."""
    print("Traigent MCP Integration Test Suite")
    print("Version: 1.0.0")
    print("=" * 60)

    # Check for required dependencies
    import importlib.util

    missing_deps = []
    if importlib.util.find_spec("coverage") is None:
        missing_deps.append("coverage")
    if importlib.util.find_spec("pytest") is None:
        missing_deps.append("pytest")

    if missing_deps:
        print(f"Error: Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install pytest coverage")
        sys.exit(1)

    # Run tests
    all_passed = run_mcp_tests()

    # Run cleanup verification
    asyncio.run(run_cleanup_verification())

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All MCP tests passed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
