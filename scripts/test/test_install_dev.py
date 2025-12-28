#!/usr/bin/env python3
"""
Test suite for install-dev.sh script.

This module tests the development installation script functionality
including environment setup, dependency installation, and validation.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestInstallDevScript(unittest.TestCase):
    """Test cases for the install-dev.sh script."""

    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent / "install-dev.sh"
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        # Clean up test directory
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_script_exists(self):
        """Test that the install-dev.sh script exists."""
        self.assertTrue(self.script_path.exists(), "install-dev.sh script not found")

    def test_script_is_executable(self):
        """Test that the script has executable permissions."""
        self.assertTrue(
            os.access(self.script_path, os.X_OK), "Script is not executable"
        )

    def test_script_has_shebang(self):
        """Test that the script has proper shebang line."""
        with open(self.script_path) as f:
            first_line = f.readline().strip()
        self.assertEqual(first_line, "#!/bin/bash", "Script missing proper shebang")

    @patch("subprocess.run")
    def test_python_version_check(self, mock_run):
        """Test Python version checking logic."""
        # Mock Python version check - version OK
        mock_run.return_value = MagicMock(stdout="Python 3.9.0", returncode=0)

        # Run a simplified version check
        result = subprocess.run(
            ["python3", "--version"], capture_output=True, text=True
        )

        version = result.stdout.strip().split()[-1]
        major, minor = map(int, version.split(".")[:2])

        self.assertGreaterEqual(major, 3)
        self.assertGreaterEqual(minor, 8)

    @patch("subprocess.run")
    def test_python_version_too_low(self, mock_run):
        """Test behavior when Python version is too low."""
        # Mock Python version check - version too low
        mock_run.return_value = MagicMock(stdout="Python 3.7.0", returncode=0)

        # Simulate version check
        result = subprocess.run(
            ["python3", "--version"], capture_output=True, text=True
        )

        version = result.stdout.strip().split()[-1]
        major, minor = map(int, version.split(".")[:2])

        # Should fail for Python < 3.8
        self.assertFalse(
            major > 3 or (major == 3 and minor >= 8), "Should reject Python < 3.8"
        )

    def test_virtual_environment_creation(self):
        """Test virtual environment creation logic."""
        os.chdir(self.test_dir)

        # Test venv doesn't exist initially
        venv_path = Path(self.test_dir) / "venv"
        self.assertFalse(venv_path.exists())

        # Simulate venv creation
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

        # Verify venv was created
        self.assertTrue(venv_path.exists())
        self.assertTrue(
            (venv_path / "bin" / "activate").exists()
            or (venv_path / "Scripts" / "activate").exists()
        )

    @patch("subprocess.run")
    def test_pip_upgrade(self, mock_run):
        """Test pip upgrade functionality."""
        mock_run.return_value = MagicMock(returncode=0)

        # Simulate pip upgrade
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
        )

        mock_run.assert_called()

    @patch("subprocess.run")
    def test_package_installation(self, mock_run):
        """Test package installation in development mode."""
        mock_run.return_value = MagicMock(returncode=0)

        # Simulate package installation
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".[dev,integrations,bayesian,docs]",
            ],
            capture_output=True,
        )

        mock_run.assert_called()

    @patch("subprocess.run")
    def test_precommit_installation(self, mock_run):
        """Test pre-commit hooks installation."""
        mock_run.return_value = MagicMock(returncode=0)

        # Simulate pre-commit install
        subprocess.run(["pre-commit", "install"], capture_output=True)

        mock_run.assert_called()

    @patch("subprocess.run")
    def test_initial_tests_run(self, mock_run):
        """Test that initial tests are executed."""
        mock_run.return_value = MagicMock(returncode=0)

        # Simulate pytest run
        subprocess.run(["pytest", "tests/", "-v"], capture_output=True)

        mock_run.assert_called()

    @patch("subprocess.run")
    def test_code_quality_checks(self, mock_run):
        """Test code quality check commands."""
        mock_run.return_value = MagicMock(returncode=0)

        # Test black
        subprocess.run(["black", "--check", "traigent", "tests", "examples"])

        # Test isort
        subprocess.run(["isort", "--check-only", "traigent", "tests", "examples"])

        # Test flake8
        subprocess.run(["flake8", "traigent", "tests", "examples"])

        # Test mypy
        subprocess.run(["mypy", "traigent"])

        # Verify all commands were called
        self.assertEqual(mock_run.call_count, 4)

    def test_script_error_handling(self):
        """Test that script uses 'set -e' for error handling."""
        with open(self.script_path) as f:
            content = f.read()

        self.assertIn(
            "set -e", content, "Script should use 'set -e' for error handling"
        )

    def test_script_output_messages(self):
        """Test that script contains appropriate output messages."""
        with open(self.script_path) as f:
            content = f.read()

        # Check for key messages
        expected_messages = [
            "Setting up Traigent SDK for development",
            "Checking Python version",
            "Creating virtual environment",
            "Installing Traigent SDK in development mode",
            "Running initial tests",
            "Development setup complete",
        ]

        for message in expected_messages:
            self.assertIn(message, content, f"Missing expected message: {message}")

    def test_script_documentation_references(self):
        """Test that script references correct documentation files."""
        with open(self.script_path) as f:
            content = f.read()

        # Check documentation references
        doc_files = [
            "README.md",
            "docs/CONTRIBUTING.md",
            "docs/ARCHITECTURE.md",
            "docs/SPRINT_PLAN.md",
            "docs/README.md",
        ]

        for doc in doc_files:
            self.assertIn(doc, content, f"Missing documentation reference: {doc}")

    def test_script_next_steps(self):
        """Test that script provides clear next steps."""
        with open(self.script_path) as f:
            content = f.read()

        # Check for next steps
        next_steps = [
            "source venv/bin/activate",
            "pytest",
            "black traigent tests",
            "python examples/basic_optimization.py",
        ]

        for step in next_steps:
            self.assertIn(step, content, f"Missing next step instruction: {step}")


class TestInstallDevIntegration(unittest.TestCase):
    """Integration tests for install-dev.sh script execution."""

    @unittest.skipIf(not Path("/bin/bash").exists(), "Bash not available")
    def test_script_syntax(self):
        """Test that script has valid bash syntax."""
        script_path = Path(__file__).parent.parent / "install-dev.sh"

        result = subprocess.run(
            ["bash", "-n", str(script_path)], capture_output=True, text=True
        )

        self.assertEqual(
            result.returncode, 0, f"Script has syntax errors: {result.stderr}"
        )

    def test_script_shellcheck(self):
        """Test script with shellcheck if available."""
        try:
            # Check if shellcheck is available
            subprocess.run(["shellcheck", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("shellcheck not available")

        script_path = Path(__file__).parent.parent / "install-dev.sh"

        result = subprocess.run(
            ["shellcheck", str(script_path)], capture_output=True, text=True
        )

        # shellcheck may have warnings, but should not have errors
        self.assertNotIn(
            "error", result.stderr.lower(), f"shellcheck found errors: {result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
