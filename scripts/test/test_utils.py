"""
Utility functions for testing scripts.

This module provides helper functions and mocks for testing shell scripts
and development tools.
"""

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock


class MockSubprocess:
    """Mock subprocess for testing shell command execution."""

    def __init__(self):
        """Initialize mock subprocess."""
        self.commands_run = []
        self.return_codes = {}
        self.outputs = {}

    def set_return_code(self, command: str, code: int):
        """Set return code for a specific command."""
        self.return_codes[command] = code

    def set_output(self, command: str, stdout: str, stderr: str = ""):
        """Set output for a specific command."""
        self.outputs[command] = (stdout, stderr)

    def run(self, args: List[str], **kwargs) -> MagicMock:
        """Mock subprocess.run method."""
        command = " ".join(args) if isinstance(args, list) else args
        self.commands_run.append(command)

        # Create mock result
        result = MagicMock()
        result.returncode = self.return_codes.get(command, 0)

        if command in self.outputs:
            stdout, stderr = self.outputs[command]
            result.stdout = stdout
            result.stderr = stderr
        else:
            result.stdout = ""
            result.stderr = ""

        return result

    def check_call(self, args: List[str], **kwargs):
        """Mock subprocess.check_call method."""
        result = self.run(args, **kwargs)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, args)

    def check_output(self, args: List[str], **kwargs) -> str:
        """Mock subprocess.check_output method."""
        result = self.run(args, **kwargs)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, args)
        return result.stdout


@contextmanager
def temp_directory():
    """Context manager for temporary directory creation and cleanup."""
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()

    try:
        os.chdir(temp_dir)
        yield Path(temp_dir)
    finally:
        os.chdir(original_cwd)
        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def mock_environment(**env_vars):
    """Context manager for temporarily setting environment variables."""
    original_env = os.environ.copy()

    try:
        os.environ.update(env_vars)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def create_mock_script(content: str, name: str = "test_script.sh") -> Path:
    """Create a mock shell script for testing."""
    with temp_directory() as temp_dir:
        script_path = temp_dir / name
        script_path.write_text(content)
        script_path.chmod(0o755)  # Make executable
        return script_path


def validate_shell_script(script_path: Path) -> Tuple[bool, str]:
    """Validate shell script syntax using bash -n."""
    try:
        result = subprocess.run(
            ["bash", "-n", str(script_path)], capture_output=True, text=True
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        return False, "Bash not found"


def run_shellcheck(script_path: Path) -> Tuple[bool, str]:
    """Run shellcheck on a script if available."""
    try:
        result = subprocess.run(
            ["shellcheck", str(script_path)], capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout + result.stderr
    except FileNotFoundError:
        return True, "shellcheck not available"


class ScriptTestCase:
    """Base class for script testing with common assertions."""

    @staticmethod
    def assert_executable(script_path: Path) -> bool:
        """Assert that a script is executable."""
        return os.access(script_path, os.X_OK)

    @staticmethod
    def assert_has_shebang(script_path: Path, expected: str = "#!/bin/bash") -> bool:
        """Assert that a script has the expected shebang."""
        with open(script_path) as f:
            first_line = f.readline().strip()
        return first_line == expected

    @staticmethod
    def assert_uses_set_e(script_path: Path) -> bool:
        """Assert that a script uses 'set -e' for error handling."""
        with open(script_path) as f:
            content = f.read()
        return "set -e" in content

    @staticmethod
    def assert_contains_text(script_path: Path, text: str) -> bool:
        """Assert that a script contains specific text."""
        with open(script_path) as f:
            content = f.read()
        return text in content

    @staticmethod
    def get_script_functions(script_path: Path) -> List[str]:
        """Extract function names from a bash script."""
        functions = []
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                # Match function definitions
                if line.startswith("function ") and "(" in line:
                    func_name = line.split()[1].split("(")[0]
                    functions.append(func_name)
                elif "() {" in line:
                    func_name = line.split("()")[0].strip()
                    functions.append(func_name)
        return functions

    @staticmethod
    def get_script_variables(script_path: Path) -> Dict[str, str]:
        """Extract variable assignments from a bash script."""
        variables = {}
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                # Match simple variable assignments
                if "=" in line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2 and parts[0].isidentifier():
                        var_name = parts[0]
                        var_value = parts[1].strip("\"'")
                        variables[var_name] = var_value
        return variables


def mock_python_version(version: str) -> MockSubprocess:
    """Create a mock subprocess that returns a specific Python version."""
    mock = MockSubprocess()
    mock.set_output("python3 --version", f"Python {version}")
    mock.set_output("python --version", f"Python {version}")
    return mock


def mock_successful_install() -> MockSubprocess:
    """Create a mock subprocess for successful installation flow."""
    mock = MockSubprocess()

    # Python version check
    mock.set_output("python3 --version", "Python 3.9.0")

    # Pip commands
    mock.set_output("pip install --upgrade pip", "Successfully upgraded pip")
    mock.set_output(
        "pip install -e .[dev,integrations,bayesian,docs]",
        "Successfully installed traigent",
    )

    # Pre-commit
    mock.set_output("pre-commit install", "pre-commit installed")

    # Tests
    mock.set_output("pytest tests/ -v", "All tests passed")

    # Code quality
    mock.set_output("black --check traigent tests examples", "All files formatted")
    mock.set_output("isort --check-only traigent tests examples", "All imports sorted")
    mock.set_output("flake8 traigent tests examples", "No issues found")
    mock.set_output("mypy traigent", "Success: no issues found")

    return mock
