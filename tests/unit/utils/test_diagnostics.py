"""Unit tests for traigent.utils.diagnostics.

Tests for Traigent diagnostic utilities including system checks,
package validation, and environment configuration detection.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability
# CONC-Quality-Maintainability FUNC-ANALYTICS REQ-ANLY-011
# SYNC-Observability

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from traigent.utils.diagnostics import (
    DiagnosticReport,
    TraigentDiagnostics,
    diagnose,
    main,
)


class TestDiagnosticReport:
    """Tests for DiagnosticReport class."""

    @pytest.fixture
    def report(self) -> DiagnosticReport:
        """Create test report instance."""
        return DiagnosticReport()

    def test_initialization(self, report: DiagnosticReport) -> None:
        """Test report initializes with system information."""
        assert report.python_version == sys.version
        assert report.platform == sys.platform
        assert report.issues == []
        assert report.warnings == []
        assert report.successes == []
        assert report.recommendations == []

    def test_add_issue_without_fix(self, report: DiagnosticReport) -> None:
        """Test adding issue without fix suggestion."""
        report.add_issue("TestCategory", "Test issue message")

        assert len(report.issues) == 1
        assert report.issues[0]["category"] == "TestCategory"
        assert report.issues[0]["message"] == "Test issue message"
        assert report.issues[0]["fix"] is None

    def test_add_issue_with_fix(self, report: DiagnosticReport) -> None:
        """Test adding issue with fix suggestion."""
        report.add_issue("TestCategory", "Test issue", "Run fix command")

        assert len(report.issues) == 1
        assert report.issues[0]["category"] == "TestCategory"
        assert report.issues[0]["message"] == "Test issue"
        assert report.issues[0]["fix"] == "Run fix command"

    def test_add_warning(self, report: DiagnosticReport) -> None:
        """Test adding warning."""
        report.add_warning("TestCategory", "Test warning")

        assert len(report.warnings) == 1
        assert report.warnings[0]["category"] == "TestCategory"
        assert report.warnings[0]["message"] == "Test warning"

    def test_add_success(self, report: DiagnosticReport) -> None:
        """Test adding success."""
        report.add_success("TestCategory", "Test success")

        assert len(report.successes) == 1
        assert report.successes[0]["category"] == "TestCategory"
        assert report.successes[0]["message"] == "Test success"

    def test_add_recommendation(self, report: DiagnosticReport) -> None:
        """Test adding recommendation."""
        report.add_recommendation("Do this")

        assert len(report.recommendations) == 1
        assert report.recommendations[0] == "Do this"

    def test_to_dict(self, report: DiagnosticReport) -> None:
        """Test converting report to dictionary."""
        report.add_issue("Category1", "Issue1", "Fix1")
        report.add_warning("Category2", "Warning1")
        report.add_success("Category3", "Success1")
        report.add_recommendation("Rec1")

        result = report.to_dict()

        assert result["python_version"] == sys.version
        assert result["platform"] == sys.platform
        assert len(result["issues"]) == 1
        assert len(result["warnings"]) == 1
        assert len(result["successes"]) == 1
        assert len(result["recommendations"]) == 1
        assert result["summary"]["total_issues"] == 1
        assert result["summary"]["total_warnings"] == 1
        assert result["summary"]["total_successes"] == 1

    def test_to_dict_empty_report(self, report: DiagnosticReport) -> None:
        """Test converting empty report to dictionary."""
        result = report.to_dict()

        assert result["summary"]["total_issues"] == 0
        assert result["summary"]["total_warnings"] == 0
        assert result["summary"]["total_successes"] == 0

    def test_print_report_with_all_sections(
        self, report: DiagnosticReport, capsys
    ) -> None:
        """Test printing report with all sections populated."""
        report.add_success("Test", "All good")
        report.add_warning("Test", "Be careful")
        report.add_issue("Test", "Problem found", "Fix it")
        report.add_recommendation("Consider this")

        report.print_report()
        captured = capsys.readouterr()

        assert "Traigent Diagnostic Report" in captured.out
        assert "System Info:" in captured.out
        assert "Successes (1):" in captured.out
        assert "All good" in captured.out
        assert "Warnings (1):" in captured.out
        assert "Be careful" in captured.out
        assert "Issues (1):" in captured.out
        assert "Problem found" in captured.out
        assert "Fix: Fix it" in captured.out
        assert "Recommendations:" in captured.out
        assert "Consider this" in captured.out
        assert "issue(s) that need attention" in captured.out

    def test_print_report_with_no_issues(
        self, report: DiagnosticReport, capsys
    ) -> None:
        """Test printing report with no issues."""
        report.add_success("Test", "All good")

        report.print_report()
        captured = capsys.readouterr()

        assert "No critical issues found!" in captured.out

    def test_print_report_without_fix_suggestion(
        self, report: DiagnosticReport, capsys
    ) -> None:
        """Test printing issue without fix suggestion."""
        report.add_issue("Test", "Problem", None)

        report.print_report()
        captured = capsys.readouterr()

        assert "Problem" in captured.out
        # Fix line should not appear when fix is None
        assert "💡 Fix:" not in captured.out


class TestTraigentDiagnostics:
    """Tests for TraigentDiagnostics class."""

    def test_required_packages_list(self) -> None:
        """Test required packages list is defined."""
        assert len(TraigentDiagnostics.REQUIRED_PACKAGES) > 0
        assert (
            "traigent",
            "Traigent SDK",
            None,
        ) in TraigentDiagnostics.REQUIRED_PACKAGES

    def test_optional_packages_list(self) -> None:
        """Test optional packages list is defined."""
        assert isinstance(TraigentDiagnostics.OPTIONAL_PACKAGES, list)
        # May be empty, so just check type

    def test_environment_variables_list(self) -> None:
        """Test environment variables list is defined."""
        assert len(TraigentDiagnostics.ENVIRONMENT_VARIABLES) > 0
        # Check structure
        for (
            var_name,
            description,
            required,
        ) in TraigentDiagnostics.ENVIRONMENT_VARIABLES:
            assert isinstance(var_name, str)
            assert isinstance(description, str)
            assert isinstance(required, bool)

    @patch("sys.version_info", new=SimpleNamespace(major=3, minor=8, micro=10))
    def test_check_python_version_supported(self) -> None:
        """Test Python version check for supported version."""
        report = DiagnosticReport()
        TraigentDiagnostics._check_python_version(report)

        assert len(report.successes) == 1
        assert report.successes[0]["category"] == "Python"
        assert "3.8.10" in report.successes[0]["message"]
        assert len(report.issues) == 0

    @patch("sys.version_info", new=SimpleNamespace(major=3, minor=11, micro=5))
    def test_check_python_version_newer_supported(self) -> None:
        """Test Python version check for newer supported version."""
        report = DiagnosticReport()
        TraigentDiagnostics._check_python_version(report)

        assert len(report.successes) == 1
        assert len(report.issues) == 0

    @patch("sys.version_info", new=SimpleNamespace(major=3, minor=7, micro=9))
    def test_check_python_version_unsupported(self) -> None:
        """Test Python version check for unsupported version."""
        report = DiagnosticReport()
        TraigentDiagnostics._check_python_version(report)

        assert len(report.issues) == 1
        assert report.issues[0]["category"] == "Python"
        assert "3.7" in report.issues[0]["message"]
        assert "Install Python 3.8" in report.issues[0]["fix"]

    @patch("sys.version_info", new=SimpleNamespace(major=2, minor=7, micro=18))
    def test_check_python_version_python2(self) -> None:
        """Test Python version check for Python 2."""
        report = DiagnosticReport()
        TraigentDiagnostics._check_python_version(report)

        assert len(report.issues) == 1
        assert "2.7" in report.issues[0]["message"]

    def test_check_virtual_env_in_venv(self) -> None:
        """Test virtual environment check when in venv."""
        report = DiagnosticReport()

        with (
            patch.object(sys, "base_prefix", "/usr"),
            patch.object(sys, "prefix", "/usr/local/venv"),
        ):
            TraigentDiagnostics._check_virtual_env(report)

        assert len(report.successes) == 1
        assert "virtual environment" in report.successes[0]["message"]
        assert len(report.warnings) == 0

    def test_check_virtual_env_not_in_venv(self) -> None:
        """Test virtual environment check when not in venv."""
        report = DiagnosticReport()

        with (
            patch.object(sys, "base_prefix", "/usr"),
            patch.object(sys, "prefix", "/usr"),
        ):
            # Remove real_prefix if it exists
            if hasattr(sys, "real_prefix"):
                delattr(sys, "real_prefix")

            TraigentDiagnostics._check_virtual_env(report)

        assert len(report.warnings) == 1
        assert "Not running in virtual environment" in report.warnings[0]["message"]
        assert len(report.successes) == 0

    def test_check_virtual_env_with_real_prefix(self) -> None:
        """Test virtual environment check with real_prefix (virtualenv)."""
        report = DiagnosticReport()

        # Simulate virtualenv environment by setting real_prefix
        with patch.object(sys, "prefix", "/usr/local/venv"):
            # Add real_prefix attribute temporarily
            sys.real_prefix = "/usr"  # type: ignore[attr-defined]
            try:
                TraigentDiagnostics._check_virtual_env(report)
            finally:
                # Clean up
                if hasattr(sys, "real_prefix"):
                    delattr(sys, "real_prefix")

        assert len(report.successes) == 1
        assert len(report.warnings) == 0

    @patch("importlib.import_module")
    def test_check_packages_required_installed(self, mock_import: MagicMock) -> None:
        """Test checking required packages when installed."""
        report = DiagnosticReport()
        packages = [("test_package", "Test Package", "test-pkg")]

        TraigentDiagnostics._check_packages(report, packages, required=True)

        assert len(report.successes) == 1
        assert "Test Package is installed" in report.successes[0]["message"]
        assert len(report.issues) == 0
        mock_import.assert_called_once_with("test_package")

    @patch("importlib.import_module")
    def test_check_packages_required_missing(self, mock_import: MagicMock) -> None:
        """Test checking required packages when missing."""
        report = DiagnosticReport()
        packages = [("missing_package", "Missing Package", "missing-pkg")]
        mock_import.side_effect = ImportError("Module not found")

        TraigentDiagnostics._check_packages(report, packages, required=True)

        assert len(report.issues) == 1
        assert "Missing Package is not installed" in report.issues[0]["message"]
        assert "pip install missing-pkg" in report.issues[0]["fix"]
        assert len(report.successes) == 0

    @patch("importlib.import_module")
    def test_check_packages_optional_missing(self, mock_import: MagicMock) -> None:
        """Test checking optional packages when missing."""
        report = DiagnosticReport()
        packages = [("optional_package", "Optional Package", "optional-pkg")]
        mock_import.side_effect = ImportError("Module not found")

        TraigentDiagnostics._check_packages(report, packages, required=False)

        assert len(report.warnings) == 1
        msg = report.warnings[0]["message"]
        assert "Optional Package is not installed (optional)" in msg
        assert len(report.issues) == 0

    @patch("importlib.import_module")
    def test_check_packages_with_none_install_name(
        self, mock_import: MagicMock
    ) -> None:
        """Test checking packages with None install name."""
        report = DiagnosticReport()
        packages = [("package", "Package", None)]
        mock_import.side_effect = ImportError("Module not found")

        TraigentDiagnostics._check_packages(report, packages, required=True)

        assert "pip install package" in report.issues[0]["fix"]

    @patch.dict(
        os.environ,
        {"TEST_KEY": "test_value", "TEST_URL": "http://example.com"},
    )
    def test_check_environment_with_values(self) -> None:
        """Test environment variable check with values set."""
        report = DiagnosticReport()
        env_vars = [
            ("TEST_KEY", "Test key", True),
            ("TEST_URL", "Test URL", False),
        ]

        with patch.object(TraigentDiagnostics, "ENVIRONMENT_VARIABLES", env_vars):
            TraigentDiagnostics._check_environment(report)

        assert len(report.successes) == 2
        # KEY should be masked
        assert any("TEST_KEY is set" in s["message"] for s in report.successes)
        # Non-KEY should show value
        assert any(
            "TEST_URL = http://example.com" in s["message"] for s in report.successes
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_missing_required(self) -> None:
        """Test env variable check with required variable missing."""
        report = DiagnosticReport()
        env_vars = [("REQUIRED_VAR", "Required variable", True)]

        with patch.object(TraigentDiagnostics, "ENVIRONMENT_VARIABLES", env_vars):
            TraigentDiagnostics._check_environment(report)

        assert len(report.issues) == 1
        assert "REQUIRED_VAR not set" in report.issues[0]["message"]
        fix = report.issues[0]["fix"]
        assert "Add REQUIRED_VAR to .env file" in fix

    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_missing_optional(self) -> None:
        """Test env variable check with optional variable missing."""
        report = DiagnosticReport()
        env_vars = [("OPTIONAL_VAR", "Optional variable", False)]

        with patch.object(TraigentDiagnostics, "ENVIRONMENT_VARIABLES", env_vars):
            TraigentDiagnostics._check_environment(report)

        assert len(report.warnings) == 1
        msg = report.warnings[0]["message"]
        assert "OPTIONAL_VAR not set (optional" in msg

    @patch("traigent.initialize")
    @patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"}, clear=True)
    def test_check_traigent_config_success(self, mock_initialize: MagicMock) -> None:
        """Test Traigent configuration check when successful."""
        report = DiagnosticReport()

        TraigentDiagnostics._check_traigent_config(report)

        assert len(report.successes) >= 1
        assert any(
            "SDK initialized successfully" in s["message"] for s in report.successes
        )
        mock_initialize.assert_called_once_with(execution_mode="edge_analytics")

    @patch("traigent.initialize")
    @patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"})
    def test_check_traigent_config_with_mock_mode(
        self, mock_initialize: MagicMock
    ) -> None:
        """Test Traigent config check with mock LLM mode enabled."""
        report = DiagnosticReport()

        TraigentDiagnostics._check_traigent_config(report)

        assert len(report.successes) >= 2
        assert any("Mock LLM mode is enabled" in s["message"] for s in report.successes)

    @patch("traigent.initialize")
    @patch.dict(os.environ, {}, clear=True)
    def test_check_traigent_config_failure(self, mock_initialize: MagicMock) -> None:
        """Test Traigent config check when initialization fails."""
        report = DiagnosticReport()
        mock_initialize.side_effect = Exception("Init failed")

        TraigentDiagnostics._check_traigent_config(report)

        assert len(report.issues) == 1
        assert "Failed to initialize" in report.issues[0]["message"]
        assert "pip install -e ." in report.issues[0]["fix"]

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.unlink")
    def test_check_permissions_success(
        self,
        mock_unlink: MagicMock,
        mock_write: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        """Test file permissions check when successful."""
        report = DiagnosticReport()

        TraigentDiagnostics._check_permissions(report)

        # Should succeed for all test paths and exercise write/unlink mocks
        assert mock_write.called, "write_text should have been called to test permissions"
        assert len(report.successes) >= 1
        assert any("Can write to" in s["message"] for s in report.successes)

    @patch("pathlib.Path.mkdir")
    def test_check_permissions_failure(self, mock_mkdir: MagicMock) -> None:
        """Test file permissions check when permissions denied."""
        report = DiagnosticReport()
        mock_mkdir.side_effect = OSError("Permission denied")

        TraigentDiagnostics._check_permissions(report)

        assert len(report.warnings) >= 1
        assert any("Cannot write to" in w["message"] for w in report.warnings)

    @patch("socket.create_connection")
    def test_check_network_success(self, mock_socket: MagicMock) -> None:
        """Test network connectivity check when successful."""
        report = DiagnosticReport()

        TraigentDiagnostics._check_network(report)

        assert len(report.successes) >= 1
        assert any("Can connect to" in s["message"] for s in report.successes)

    @patch("socket.create_connection")
    def test_check_network_failure(self, mock_socket: MagicMock) -> None:
        """Test network connectivity check when connection fails."""
        report = DiagnosticReport()
        mock_socket.side_effect = TimeoutError("Connection timeout")

        TraigentDiagnostics._check_network(report)

        assert len(report.warnings) >= 1
        assert any("Cannot connect to" in w["message"] for w in report.warnings)

    @patch.dict(os.environ, {}, clear=True)
    def test_add_recommendations_with_issues(self) -> None:
        """Test recommendations when issues are present."""
        report = DiagnosticReport()
        report.add_issue("Test", "Test issue", "Fix it")

        TraigentDiagnostics._add_recommendations(report)

        assert any("Fix critical issues" in r for r in report.recommendations)

    @patch.dict(os.environ, {}, clear=True)
    def test_add_recommendations_without_mock_mode(self) -> None:
        """Test recommendations when mock LLM mode is not enabled."""
        report = DiagnosticReport()

        TraigentDiagnostics._add_recommendations(report)

        assert any("Enable mock LLM mode" in r for r in report.recommendations)

    @patch.dict(os.environ, {"TRAIGENT_MOCK_LLM": "true"})
    def test_add_recommendations_with_mock_mode(self) -> None:
        """Test recommendations when mock mode is enabled."""
        report = DiagnosticReport()

        TraigentDiagnostics._add_recommendations(report)

        # Should not recommend mock mode if already enabled
        assert not any("Enable mock mode" in r for r in report.recommendations)

    @patch.dict(os.environ, {}, clear=True)
    def test_add_recommendations_without_api_keys(self) -> None:
        """Test recommendations when no API keys are set."""
        report = DiagnosticReport()

        with patch.object(
            TraigentDiagnostics,
            "ENVIRONMENT_VARIABLES",
            [("TEST_API_KEY", "Test API", False)],
        ):
            TraigentDiagnostics._add_recommendations(report)

        assert any("Add API keys to .env file" in r for r in report.recommendations)

    def test_add_recommendations_includes_quickstart(self) -> None:
        """Test recommendations always include quickstart script."""
        report = DiagnosticReport()

        TraigentDiagnostics._add_recommendations(report)

        assert any("quickstart" in r for r in report.recommendations)

    @patch.object(TraigentDiagnostics, "_check_python_version")
    @patch.object(TraigentDiagnostics, "_check_virtual_env")
    @patch.object(TraigentDiagnostics, "_check_packages")
    @patch.object(TraigentDiagnostics, "_check_environment")
    @patch.object(TraigentDiagnostics, "_check_traigent_config")
    @patch.object(TraigentDiagnostics, "_check_permissions")
    @patch.object(TraigentDiagnostics, "_check_network")
    @patch.object(TraigentDiagnostics, "_add_recommendations")
    def test_run_diagnostics(
        self,
        mock_rec: MagicMock,
        mock_net: MagicMock,
        mock_perm: MagicMock,
        mock_config: MagicMock,
        mock_env: MagicMock,
        mock_pkg: MagicMock,
        mock_venv: MagicMock,
        mock_py: MagicMock,
    ) -> None:
        """Test run_diagnostics calls all check methods."""
        report = TraigentDiagnostics.run_diagnostics()

        assert isinstance(report, DiagnosticReport)
        mock_py.assert_called_once()
        mock_venv.assert_called_once()
        # _check_packages called twice (required and optional)
        assert mock_pkg.call_count == 2
        mock_env.assert_called_once()
        mock_config.assert_called_once()
        mock_perm.assert_called_once()
        mock_net.assert_called_once()
        mock_rec.assert_called_once()


class TestDiagnoseFunctions:
    """Tests for module-level functions."""

    @patch.object(TraigentDiagnostics, "run_diagnostics")
    def test_diagnose(self, mock_run: MagicMock) -> None:
        """Test diagnose function calls run_diagnostics."""
        mock_report = MagicMock(spec=DiagnosticReport)
        mock_run.return_value = mock_report

        result = diagnose()

        assert result is mock_report
        mock_run.assert_called_once()

    @patch.object(TraigentDiagnostics, "run_diagnostics")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.write_text")
    def test_main_success(
        self,
        mock_write: MagicMock,
        mock_file: MagicMock,
        mock_run: MagicMock,
        capsys,
    ) -> None:
        """Test main function with no issues."""
        mock_report = DiagnosticReport()
        mock_report.issues = []
        mock_run.return_value = mock_report

        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Running Traigent diagnostics" in captured.out
        assert "Full report saved to" in captured.out

    @patch.object(TraigentDiagnostics, "run_diagnostics")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_with_issues(
        self, mock_file: MagicMock, mock_run: MagicMock, capsys
    ) -> None:
        """Test main function with issues found."""
        mock_report = DiagnosticReport()
        mock_report.add_issue("Test", "Test issue", "Fix it")
        mock_run.return_value = mock_report

        exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Running Traigent diagnostics" in captured.out

    @patch.object(TraigentDiagnostics, "run_diagnostics")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_saves_report_to_file(
        self,
        mock_file: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test main function saves report to JSON file."""
        mock_report = DiagnosticReport()
        mock_run.return_value = mock_report

        main()

        # Verify file was opened for writing
        mock_file.assert_called_once()
        call_args = mock_file.call_args
        assert "traigent_diagnostic_report.json" in str(call_args)
        assert "w" in call_args[0] or "w" in call_args[1].get("mode", "")
