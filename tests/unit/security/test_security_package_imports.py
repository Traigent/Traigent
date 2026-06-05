from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_audit_event_type_import_keeps_enterprise_modules_lazy() -> None:
    result = run_fresh_python(
        """
import sys

from traigent.security.audit import AuditEventType

assert AuditEventType.USER_LOGIN.value == "user_login"
for module_name in (
    "traigent.security.auth",
    "traigent.security.deployment",
    "traigent.security.encryption",
    "traigent.security.tenant",
    "jwt",
    "pyotp",
    "twilio",
):
    assert module_name not in sys.modules, module_name
"""
    )

    assert result.stderr == ""


def test_security_package_lazy_export_preserves_legacy_import_surface() -> None:
    result = run_fresh_python(
        """
from traigent.security import MultiFactorAuth

assert MultiFactorAuth.__name__ == "MultiFactorAuth"
"""
    )

    assert result.stderr == ""
