#!/usr/bin/env python3
"""Test CLI authentication wiring without live backend calls."""

from __future__ import annotations

from traigent.cli.auth_commands import TraigentAuthCLI
from traigent.cloud.auth import AuthManager
from traigent.cloud.credential_manager import CredentialManager
from traigent.cloud.resilient_client import ResilientClient
from traigent.config.backend_config import BackendConfig
from traigent.utils.logging import setup_logging


def test_integration(monkeypatch):
    """Verify CLI auth components are wired together deterministically."""
    setup_logging("INFO")

    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setattr(
        BackendConfig,
        "get_cloud_backend_url",
        staticmethod(lambda: "https://cloud.example.test"),
    )
    monkeypatch.setattr(
        BackendConfig,
        "get_cloud_api_url",
        staticmethod(lambda: "https://api.example.test"),
    )
    monkeypatch.setattr(
        BackendConfig,
        "get_backend_url",
        staticmethod(lambda: "https://backend.example.test"),
    )
    monkeypatch.setattr(
        CredentialManager,
        "get_api_key",
        staticmethod(lambda: "tr_test_key_1234567890"),
    )
    monkeypatch.setattr(
        CredentialManager,
        "get_auth_headers",
        staticmethod(lambda: {"Authorization": "Bearer test-token"}),
    )

    cli = TraigentAuthCLI()

    assert hasattr(cli, "auth_manager")
    assert isinstance(cli.auth_manager, AuthManager)
    assert cli.backend_url == "https://cloud.example.test"
    assert cli.backend_api_url == "https://api.example.test"

    assert CredentialManager.get_api_key() == "tr_test_key_1234567890"
    assert CredentialManager.get_auth_headers() == {
        "Authorization": "Bearer test-token"
    }

    client = ResilientClient()
    assert client.max_retries == 3

    assert BackendConfig.get_backend_url() == "https://backend.example.test"

    manager = AuthManager()
    for method in (
        "authenticate",
        "refresh_authentication",
        "get_auth_headers",
        "is_authenticated",
        "clear",
        "authenticate_with_result",
    ):
        assert hasattr(manager, method)
