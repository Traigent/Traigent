"""Blank or padded TRAIGENT_API_KEY must not shadow stored credentials (#2987)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.admin.config import EnterpriseAdminConfig
from traigent.cloud.credential_manager import CredentialManager
from traigent.config.backend_config import BackendConfig
from traigent.core_metrics.config import CoreMetricsConfig
from traigent.evaluation.config import EvaluationConfig
from traigent.projects.config import ProjectManagementConfig
from traigent.prompts.config import PromptManagementConfig

KEY = "uk_abc"  # pragma: allowlist secret
STORED = "tg_stored_test_key"  # pragma: allowlist secret
_STORED_PATCH = (
    "traigent.cloud.credential_manager.CredentialManager.get_stored_api_key_only"
)


@pytest.mark.parametrize("blank", ["", " ", "   ", "\n", " \t\n"])
def test_blank_env_key_falls_through_to_stored(blank: str) -> None:
    with (
        patch.dict("os.environ", {"TRAIGENT_API_KEY": blank}, clear=True),
        patch(_STORED_PATCH, return_value=STORED),
    ):
        assert BackendConfig.get_api_key() == STORED


@pytest.mark.parametrize("raw", ["uk_abc\n", "  uk_abc  ", "\tuk_abc\r\n"])
def test_padded_env_key_is_stripped(raw: str) -> None:
    with (
        patch.dict("os.environ", {"TRAIGENT_API_KEY": raw}, clear=True),
        patch(_STORED_PATCH, return_value=STORED),
    ):
        assert BackendConfig.get_api_key() == "uk_abc"


def test_stored_key_is_stripped_and_blank_stored_is_unset() -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(_STORED_PATCH, return_value=" tg_k\n"),
    ):
        assert BackendConfig.get_api_key() == "tg_k"
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(_STORED_PATCH, return_value="  "),
    ):
        assert BackendConfig.get_api_key() is None


@pytest.mark.parametrize(
    "raw,expected", [(" ", None), ("", None), ("uk_abc\n", "uk_abc"), (None, None)]
)
def test_credential_manager_env_key(raw: str | None, expected: str | None) -> None:
    env = {} if raw is None else {"TRAIGENT_API_KEY": raw}
    with patch.dict("os.environ", env, clear=True):
        assert CredentialManager._get_env_api_key() == expected


@pytest.mark.parametrize(
    "cls",
    [
        EnterpriseAdminConfig,
        EvaluationConfig,
        ProjectManagementConfig,
        PromptManagementConfig,
        CoreMetricsConfig,
    ],
)
def test_service_configs_drop_blank_and_strip_padded_key(cls: type) -> None:
    kwargs = {"backend_origin": "https://api.example.com"}
    blank = cls(api_key="   ", **kwargs)
    assert "X-API-Key" not in blank.build_headers()
    padded = cls(api_key="uk_abc\n", **kwargs)
    assert padded.build_headers()["X-API-Key"] == "uk_abc"


def test_auth_manager_ignores_blank_env_key_and_strips_padding() -> None:
    from traigent.cloud.auth import AuthManager

    with patch.dict(
        "os.environ", {"TRAIGENT_API_KEY": "  \n"}, clear=True
    ):  # pragma: allowlist secret
        assert AuthManager()._provided_credentials is None
    with patch.dict(
        "os.environ", {"TRAIGENT_API_KEY": KEY + "\n"}, clear=True
    ):  # pragma: allowlist secret
        creds = AuthManager()._provided_credentials
        assert creds is not None and creds.api_key == KEY


def test_stored_credentials_strip_api_key() -> None:
    stored = {"api_key": f" {KEY}\n", "backend_url": "https://api.example.com"}
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(CredentialManager, "_load_cli_credentials", return_value=stored),
    ):
        assert CredentialManager.get_credentials()["api_key"] == KEY
    blank = {"api_key": "  "}
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(CredentialManager, "_load_cli_credentials", return_value=blank),
    ):
        assert "api_key" not in CredentialManager.get_credentials()


def test_analytics_jwt_selection_ignores_blank_env_key() -> None:
    from traigent.cloud.analytics_auth import _configured_jwt_without_api_key

    env = {"TRAIGENT_API_KEY": "   ", "TRAIGENT_JWT_TOKEN": "a.b.c"}
    with patch.dict("os.environ", env, clear=True):
        assert _configured_jwt_without_api_key() is True
