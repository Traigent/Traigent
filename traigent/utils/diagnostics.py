"""
Diagnostic utilities for Traigent SDK.

Provides tools to diagnose and troubleshoot Traigent installation and configuration issues.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


class DiagnosticReport:
    """Container for diagnostic results."""

    def __init__(self) -> None:
        self.python_version = sys.version
        self.platform = sys.platform
        self.issues: list[Any] = []
        self.warnings: list[Any] = []
        self.successes: list[Any] = []
        self.recommendations: list[Any] = []

    def add_issue(self, category: str, message: str, fix: str | None = None) -> None:
        """Add an issue to the report."""
        self.issues.append({"category": category, "message": message, "fix": fix})

    def add_warning(self, category: str, message: str) -> None:
        """Add a warning to the report."""
        self.warnings.append({"category": category, "message": message})

    def add_success(self, category: str, message: str) -> None:
        """Add a success to the report."""
        self.successes.append({"category": category, "message": message})

    def add_recommendation(self, message: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "issues": self.issues,
            "warnings": self.warnings,
            "successes": self.successes,
            "recommendations": self.recommendations,
            "summary": {
                "total_issues": len(self.issues),
                "total_warnings": len(self.warnings),
                "total_successes": len(self.successes),
            },
        }

    def print_report(self) -> None:
        """Print formatted report to console."""
        print("\n" + "=" * 60)
        print("🔍 Traigent Diagnostic Report")
        print("=" * 60)

        print("\n📊 System Info:")
        print(
            f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        print(f"  Platform: {self.platform}")

        if self.successes:
            print(f"\n✅ Successes ({len(self.successes)}):")
            for success in self.successes:
                print(f"  [{success['category']}] {success['message']}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  [{warning['category']}] {warning['message']}")

        if self.issues:
            print(f"\n❌ Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  [{issue['category']}] {issue['message']}")
                if issue["fix"]:
                    print(f"     💡 Fix: {issue['fix']}")

        if self.recommendations:
            print("\n💡 Recommendations:")
            for rec in self.recommendations:
                print(f"  • {rec}")

        print("\n" + "=" * 60)

        if not self.issues:
            print("✅ No critical issues found!")
        else:
            print(f"❌ Found {len(self.issues)} issue(s) that need attention.")

        print("=" * 60 + "\n")


class TraigentDiagnostics:
    """Diagnostic tool for Traigent SDK."""

    REQUIRED_PACKAGES = [
        ("traigent", "Traigent SDK", None),
        ("langchain", "LangChain", "langchain"),
        ("langchain_openai", "LangChain OpenAI", "langchain-openai"),
        ("langchain_chroma", "LangChain Chroma", "langchain-chroma"),
        ("openai", "OpenAI", "openai"),
        ("dotenv", "Python-dotenv", "python-dotenv"),
        ("numpy", "NumPy", "numpy"),
        ("pandas", "Pandas", "pandas"),
    ]

    OPTIONAL_PACKAGES: list[tuple[str, str, str | None]] = [
        ("mlflow", "MLflow", "mlflow"),
        ("wandb", "Weights & Biases", "wandb"),
        ("streamlit", "Streamlit", "streamlit"),
    ]

    ENVIRONMENT_VARIABLES = [
        ("OPENAI_API_KEY", "OpenAI API access", True),
        ("TRAIGENT_API_KEY", "Traigent cloud features", False),
        ("ANTHROPIC_API_KEY", "Anthropic Claude access", False),
        ("TRAIGENT_BACKEND_URL", "Traigent backend", False),
        ("TRAIGENT_MOCK_LLM", "Mock LLM mode testing", False),
        ("TRAIGENT_OFFLINE_MODE", "Offline backend mode", False),
    ]

    @classmethod
    def run_diagnostics(cls) -> DiagnosticReport:
        """Run complete diagnostics and return report."""
        report = DiagnosticReport()

        # Check Python version
        cls._check_python_version(report)

        # Check virtual environment
        cls._check_virtual_env(report)

        # Check required packages
        cls._check_packages(report, cls.REQUIRED_PACKAGES, required=True)

        # Check optional packages
        cls._check_packages(report, cls.OPTIONAL_PACKAGES, required=False)

        # Check environment variables
        cls._check_environment(report)

        # Check Traigent configuration
        cls._check_traigent_config(report)

        # Check file permissions
        cls._check_permissions(report)

        # Check network connectivity
        cls._check_network(report)

        # Add recommendations
        cls._add_recommendations(report)

        return report

    @classmethod
    def _check_python_version(cls, report: DiagnosticReport) -> None:
        """Check Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            report.add_success(
                "Python",
                f"Version {version.major}.{version.minor}.{version.micro} is supported",
            )
        else:
            report.add_issue(
                "Python",
                f"Version {version.major}.{version.minor} is not supported",
                "Install Python 3.8 or higher",
            )

    @classmethod
    def _check_virtual_env(cls, report: DiagnosticReport) -> None:
        """Check if running in virtual environment."""
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_venv:
            report.add_success("Environment", "Running in virtual environment")
        else:
            report.add_warning(
                "Environment",
                "Not running in virtual environment (recommended for isolation)",
            )

    @classmethod
    def _check_packages(
        cls,
        report: DiagnosticReport,
        packages: list[tuple[str, str, str | None]],
        required: bool,
    ) -> None:
        """Check if packages are installed."""
        for import_name, display_name, install_name in packages:
            try:
                importlib.import_module(import_name)
                report.add_success("Dependencies", f"{display_name} is installed")
            except ImportError:
                if required:
                    fix = f"pip install {install_name or import_name}"
                    report.add_issue(
                        "Dependencies", f"{display_name} is not installed", fix
                    )
                else:
                    report.add_warning(
                        "Dependencies", f"{display_name} is not installed (optional)"
                    )

    @classmethod
    def _check_environment(cls, report: DiagnosticReport) -> None:
        """Check environment variables."""
        for var_name, description, required in cls.ENVIRONMENT_VARIABLES:
            value = os.environ.get(var_name)

            if value:
                # Don't show actual key values for security
                if "KEY" in var_name:
                    report.add_success("Environment", f"{var_name} is set")
                else:
                    report.add_success("Environment", f"{var_name} = {value}")
            elif required:
                report.add_issue(
                    "Environment",
                    f"{var_name} not set ({description})",
                    f"Add {var_name} to .env file or export as environment variable",
                )
            else:
                report.add_warning(
                    "Environment", f"{var_name} not set (optional for {description})"
                )

    @classmethod
    def _check_traigent_config(cls, report: DiagnosticReport) -> None:
        """Check Traigent specific configuration."""
        try:
            import traigent

            # Check if Traigent can be initialized
            traigent.initialize(execution_mode="edge_analytics")
            report.add_success("Traigent", "SDK initialized successfully")

            # Check for mock LLM mode
            if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() == "true":
                report.add_success(
                    "Traigent", "Mock LLM mode is enabled (good for testing)"
                )

        except Exception as e:
            report.add_issue(
                "Traigent",
                f"Failed to initialize: {str(e)}",
                "Check installation with: pip install -e .",
            )

    @classmethod
    def _check_permissions(cls, report: DiagnosticReport) -> None:
        """Check file system permissions."""
        test_paths = [
            Path.home() / ".traigent",
            Path.cwd() / "logs",
            Path.cwd() / "data",
        ]

        for path in test_paths:
            try:
                # Try to create directory
                path.mkdir(parents=True, exist_ok=True)
                # Try to write a test file
                test_file = path / ".test_permission"
                test_file.write_text("test")
                test_file.unlink()
                report.add_success("Permissions", f"Can write to {path}")
            except Exception:
                report.add_warning(
                    "Permissions", f"Cannot write to {path} (may affect some features)"
                )

    @classmethod
    def _check_network(cls, report: DiagnosticReport) -> None:
        """Check network connectivity."""
        import socket

        test_hosts = [
            ("api.openai.com", 443, "OpenAI API"),
            ("github.com", 443, "GitHub"),
            ("pypi.org", 443, "PyPI"),
        ]

        for host, port, service in test_hosts:
            try:
                socket.create_connection((host, port), timeout=5)
                report.add_success("Network", f"Can connect to {service}")
            except Exception:
                report.add_warning(
                    "Network", f"Cannot connect to {service} (may affect some features)"
                )

    @classmethod
    def _add_recommendations(cls, report: DiagnosticReport) -> None:
        """Add recommendations based on diagnostics."""
        if report.issues:
            report.add_recommendation("Fix critical issues before proceeding")

        if not os.environ.get("TRAIGENT_MOCK_LLM"):
            report.add_recommendation(
                "Enable mock LLM mode for testing: export TRAIGENT_MOCK_LLM=true"
            )

        if not any(
            os.environ.get(var[0])
            for var in cls.ENVIRONMENT_VARIABLES
            if "KEY" in var[0]
        ):
            report.add_recommendation(
                "Add API keys to .env file for full functionality"
            )

        report.add_recommendation(
            "Run quickstart script for guided setup: python scripts/quickstart.py"
        )


def diagnose() -> DiagnosticReport:
    """Run diagnostics and return report."""
    return TraigentDiagnostics.run_diagnostics()


def main():
    """CLI entry point for diagnostics."""
    print("Running Traigent diagnostics...")
    report = diagnose()
    report.print_report()

    # Save report to file
    report_file = Path("traigent_diagnostic_report.json")
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"📄 Full report saved to: {report_file}")

    # Return exit code based on issues
    return 1 if report.issues else 0


if __name__ == "__main__":
    sys.exit(main())
