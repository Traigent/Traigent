"""#2835: info, doctor and algorithms must agree with the live state."""

from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner

from traigent.api.functions import get_version_info
from traigent.cli.doctor_command import DoctorReport, _run_key_checks
from traigent.cli.main import cli
from traigent.utils.error_handler import ErrorHandler, TraigentError

_VENDORS = ("wandb", "mlflow", "langchain", "langchain_core", "openai")
_KEY = "tg_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"  # pragma: allowlist secret


def test_integrations_need_the_vendor_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VENDORS:
        monkeypatch.setitem(sys.modules, name, None)  # import raises, find_spec None
    assert get_version_info()["integrations"] == {
        "langchain": False,
        "openai": False,
        "mlflow": False,
        "wandb": False,
    }


def test_info_names_missing_integrations_and_cloud_only_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _VENDORS:
        monkeypatch.setitem(sys.modules, name, None)
    result = CliRunner().invoke(cli, ["info"])
    assert result.exit_code == 0, result.output
    assert "Planned" not in result.output
    assert "traigent[integrations]" in result.output
    tpe_line = next(
        line for line in result.output.splitlines() if "Tpe Optimization" in line
    )
    assert "Cloud only" in tpe_line
    assert "Disabled" not in tpe_line


def test_doctor_auth_row_sees_stored_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key_with_source",
        classmethod(lambda cls: (_KEY, "stored CLI credentials")),
    )
    report = DoctorReport()
    _run_key_checks(report)
    auth = [c for c in report.checks if c.category == "Auth"]
    assert auth[0].status == "PASS"
    assert "stored CLI credentials" in auth[0].message
    assert not any("not set" in c.message for c in auth[:1])


def test_doctor_warns_only_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key_with_source",
        classmethod(lambda cls: (None, None)),
    )
    report = DoctorReport()
    _run_key_checks(report)
    first = next(c for c in report.checks if c.category == "Auth")
    assert first.status == "WARN"


def test_algorithms_marks_smart_rows_requires_login_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key_with_source",
        classmethod(lambda cls: (None, None)),
    )
    out = CliRunner().invoke(cli, ["algorithms"]).output
    assert "requires login" in out


def test_algorithms_keeps_connected_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_api_key_with_source",
        classmethod(lambda cls: (_KEY, "TRAIGENT_API_KEY")),
    )
    out = CliRunner().invoke(cli, ["algorithms"]).output
    assert "requires login" not in out


def test_error_handler_zero_accuracy_does_not_match_100_percent() -> None:
    with pytest.raises(TraigentError) as info:
        ErrorHandler.handle_error(RuntimeError("Pass rate 100.0% and 50.0%"))
    assert "0.0% accuracy" not in str(info.value)


def test_error_handler_zero_accuracy_still_matches_real_zero() -> None:
    with pytest.raises(TraigentError) as info:
        ErrorHandler.handle_error(RuntimeError("accuracy fell to 0.0%"))
    assert "0.0% accuracy" in str(info.value)


def test_error_handler_remedies_are_real() -> None:
    with pytest.raises(TraigentError) as info:
        ErrorHandler.handle_error(RuntimeError("something odd"))
    text = str(info.value)
    assert "traigent doctor" in text
    assert "verify_installation" not in text
    assert ErrorHandler.IMPORT_FIXES["traigent"] == "pip install --upgrade traigent"
