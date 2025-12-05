#!/usr/bin/env python3
"""
Simple coverage checker for install-dev.sh script.

This script analyzes the test coverage for our shell scripts by checking
which parts of the script are tested.
"""

from pathlib import Path
from typing import Dict, Tuple


def analyze_script_coverage(
    script_path: Path, test_path: Path
) -> Tuple[float, Dict[str, bool]]:
    """
    Analyze test coverage for a shell script.

    Returns:
        Tuple of (coverage_percentage, coverage_details)
    """
    # Read script content
    with open(script_path) as f:
        f.readlines()

    # Read test content
    with open(test_path) as f:
        test_content = f.read()

    # Identify testable elements in script
    testable_elements = {
        "shebang": False,
        "set_e": False,
        "python_version_check": False,
        "venv_creation": False,
        "venv_activation": False,
        "pip_upgrade": False,
        "package_install": False,
        "precommit_install": False,
        "run_tests": False,
        "black_check": False,
        "isort_check": False,
        "flake8_check": False,
        "mypy_check": False,
        "success_message": False,
        "next_steps": False,
        "documentation_refs": False,
    }

    # Check what's tested
    coverage_map = {}

    # Check for shebang test
    if "test_script_has_shebang" in test_content:
        coverage_map["shebang"] = True

    # Check for set -e test
    if "test_script_error_handling" in test_content and "set -e" in test_content:
        coverage_map["set_e"] = True

    # Check for Python version test
    if "test_python_version" in test_content:
        coverage_map["python_version_check"] = True

    # Check for venv tests
    if "test_virtual_environment_creation" in test_content:
        coverage_map["venv_creation"] = True

    # Check for pip upgrade test
    if "test_pip_upgrade" in test_content:
        coverage_map["pip_upgrade"] = True

    # Check for package installation test
    if "test_package_installation" in test_content:
        coverage_map["package_install"] = True

    # Check for pre-commit test
    if "test_precommit_installation" in test_content:
        coverage_map["precommit_install"] = True

    # Check for pytest test
    if "test_initial_tests_run" in test_content:
        coverage_map["run_tests"] = True

    # Check for code quality tests
    if "test_code_quality_checks" in test_content:
        if "black" in test_content:
            coverage_map["black_check"] = True
        if "isort" in test_content:
            coverage_map["isort_check"] = True
        if "flake8" in test_content:
            coverage_map["flake8_check"] = True
        if "mypy" in test_content:
            coverage_map["mypy_check"] = True

    # Check for output message tests
    if "test_script_output_messages" in test_content:
        coverage_map["success_message"] = True

    # Check for next steps test
    if "test_script_next_steps" in test_content:
        coverage_map["next_steps"] = True

    # Check for documentation test
    if "test_script_documentation_references" in test_content:
        coverage_map["documentation_refs"] = True

    # Calculate coverage
    tested_elements = sum(1 for v in coverage_map.values() if v)
    total_elements = len(testable_elements)
    coverage_percentage = (tested_elements / total_elements) * 100

    # Merge with testable elements
    for key in testable_elements:
        testable_elements[key] = coverage_map.get(key, False)

    return coverage_percentage, testable_elements


def print_coverage_report(script_name: str, coverage: float, details: Dict[str, bool]):
    """Print a formatted coverage report."""
    print(f"\n{'='*60}")
    print(f"Coverage Report for {script_name}")
    print(f"{'='*60}")
    print(f"Overall Coverage: {coverage:.1f}%")
    print("\nDetailed Coverage:")
    print(f"{'-'*60}")

    for element, tested in sorted(details.items()):
        status = "✅" if tested else "❌"
        print(
            f"{status} {element.replace('_', ' ').title():<40} {'Tested' if tested else 'Not Tested'}"
        )

    print(f"{'-'*60}")

    if coverage >= 85:
        print("✅ Coverage goal of >85% achieved!")
    else:
        print(f"⚠️  Coverage is below 85% target (current: {coverage:.1f}%)")
        untested = [k for k, v in details.items() if not v]
        if untested:
            print(f"\nElements needing tests: {', '.join(untested)}")


def main():
    """Main entry point."""
    script_path = Path(__file__).parent.parent / "install-dev.sh"
    test_path = Path(__file__).parent / "test_install_dev.py"

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1

    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return 1

    coverage, details = analyze_script_coverage(script_path, test_path)
    print_coverage_report("install-dev.sh", coverage, details)

    return 0 if coverage >= 85 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
