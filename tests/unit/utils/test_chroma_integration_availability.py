"""Regression tests for the temporary Chroma security hold."""

from __future__ import annotations

from unittest.mock import patch

from traigent.utils.diagnostics import TraigentDiagnostics, diagnose


def test_diagnose_reports_the_chroma_packaging_security_hold() -> None:
    with (
        patch.object(TraigentDiagnostics, "_check_python_version"),
        patch.object(TraigentDiagnostics, "_check_virtual_env"),
        patch.object(TraigentDiagnostics, "_check_packages"),
        patch.object(TraigentDiagnostics, "_check_environment"),
        patch.object(TraigentDiagnostics, "_check_traigent_config"),
        patch.object(TraigentDiagnostics, "_check_permissions"),
        patch.object(TraigentDiagnostics, "_check_network"),
        patch.object(TraigentDiagnostics, "_add_recommendations"),
    ):
        report = diagnose()

    chroma_warnings = [
        warning["message"]
        for warning in report.warnings
        if "Chroma packaging extra" in warning["message"]
    ]

    assert len(chroma_warnings) == 1
    warning = chroma_warnings[0]
    assert "GHSA-f4j7-r4q5-qw2c" in warning
    assert "temporarily withdrawn" in warning
    assert "manually managed compatible environments are not changed" in warning
    assert "pip install" not in warning
