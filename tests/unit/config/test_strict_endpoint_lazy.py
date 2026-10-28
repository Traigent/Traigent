"""Strict-endpoint mode: local and offline flows must not need an endpoint.

Follow-up to Traigent/Traigent#2770 (strict mode, commit 933cd938f). The
endpoint is resolved only when a backend call is actually needed, so offline
clients, ``initialize`` with a local config, observability in offline mode, the
sync manager's local previews, and the auth CLI's local commands all work with
nothing configured -- and no path makes a request to a vendor host.
"""

# Traceability: CONC-Security FUNC-SECURITY REQ-CLOUD-009

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.config.backend_config import (
    EndpointNotConfiguredError,
    endpoint_missing_message,
)

STORED_URL = (
    "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url"
)


@pytest.fixture
def strict_no_endpoint(tmp_path: Path) -> Iterator[None]:
    """Strict mode, nothing configured, and any socket connect fails the test."""
    env = {
        "TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": "1",
        "HOME": str(tmp_path),
        "TRAIGENT_RESULTS_FOLDER": str(tmp_path / "results"),
    }

    def _no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network connect attempted in strict mode")

    with (
        patch.dict("os.environ", env, clear=True),
        patch(STORED_URL, return_value=None),
        patch.object(socket.socket, "connect", _no_network),
        patch.object(socket, "create_connection", _no_network),
    ):
        yield


@pytest.fixture
def strict_offline(strict_no_endpoint: None) -> Iterator[None]:
    with patch.dict("os.environ", {"TRAIGENT_OFFLINE_MODE": "true"}):
        yield


class TestTraigentClientOffline:
    def test_offline_client_constructs_without_request(
        self, strict_no_endpoint: None
    ) -> None:
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(offline=True)
        assert client.backend_url is None
        assert client.no_egress is True
        assert client.backend_client is None

    def test_no_egress_client_constructs(self, strict_no_endpoint: None) -> None:
        from traigent.traigent_client import TraigentClient

        assert TraigentClient(no_egress=True).backend_url is None

    def test_offline_env_client_constructs(self, strict_offline: None) -> None:
        from traigent.traigent_client import TraigentClient

        assert TraigentClient().backend_url is None

    def test_online_client_still_requires_endpoint(
        self, strict_no_endpoint: None
    ) -> None:
        from traigent.traigent_client import TraigentClient

        with pytest.raises(EndpointNotConfiguredError):
            TraigentClient()

    def test_explicit_url_is_kept_when_offline(self, strict_no_endpoint: None) -> None:
        from traigent.traigent_client import TraigentClient

        client = TraigentClient(offline=True, backend_url="http://backend.internal")
        assert client.backend_url == "http://backend.internal"


class TestInitialize:
    def test_local_config_initializes_without_endpoint(
        self, strict_no_endpoint: None
    ) -> None:
        import traigent
        from traigent.api import functions

        functions._GLOBAL_CONFIG.pop("traigent_api_url", None)
        assert traigent.initialize(config=traigent.TraigentConfig.local_mode())
        assert "traigent_api_url" not in functions._GLOBAL_CONFIG

    def test_offline_env_initializes_without_endpoint(
        self, strict_offline: None
    ) -> None:
        import traigent

        assert traigent.initialize()

    def test_plain_initialize_still_requires_endpoint(
        self, strict_no_endpoint: None
    ) -> None:
        import traigent

        with pytest.raises(EndpointNotConfiguredError):
            traigent.initialize()

    def test_explicit_api_url_wins(self, strict_no_endpoint: None) -> None:
        import traigent
        from traigent.api import functions

        traigent.initialize(api_url="http://backend.internal")
        assert functions._GLOBAL_CONFIG["traigent_api_url"].startswith(
            "http://backend.internal"
        )


class TestObservabilityConfig:
    def test_offline_config_constructs(self, strict_offline: None) -> None:
        from traigent.observability.config import ObservabilityConfig

        config = ObservabilityConfig()
        assert config.offline_mode is True

    def test_explicit_offline_mode_constructs(self, strict_no_endpoint: None) -> None:
        from traigent.observability.config import ObservabilityConfig

        assert ObservabilityConfig(offline_mode=True).offline_mode is True

    def test_online_config_requires_endpoint(self, strict_no_endpoint: None) -> None:
        from traigent.observability.config import ObservabilityConfig

        with pytest.raises(EndpointNotConfiguredError):
            ObservabilityConfig(offline_mode=False)

    def test_explicit_empty_origin_still_invalid(
        self, strict_no_endpoint: None
    ) -> None:
        from traigent.observability.config import ObservabilityConfig

        with pytest.raises(ValueError):
            ObservabilityConfig(backend_origin="", offline_mode=False)


class TestSyncManagerLocalFlows:
    def _manager(self, tmp_path: Path):
        from traigent.cloud.sync_manager import SyncManager
        from traigent.config.types import TraigentConfig

        config = TraigentConfig.local_mode()
        config.local_storage_path = str(tmp_path / "results")
        return SyncManager(config)

    def test_constructs_without_endpoint(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        manager = self._manager(tmp_path)
        assert manager.get_sync_status()["completed_sessions"] == 0
        assert manager.get_cloud_analytics_preview()

    def test_transport_use_raises_endpoint_error(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        manager = self._manager(tmp_path)
        with pytest.raises(EndpointNotConfiguredError):
            _ = manager.base_url
        with pytest.raises(EndpointNotConfiguredError):
            _ = manager.session

    def test_configured_endpoint_resolves(self, tmp_path: Path) -> None:
        env = {
            "TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": "1",
            "TRAIGENT_BACKEND_URL": "https://backend.example.com",
            "HOME": str(tmp_path),
            "TRAIGENT_RESULTS_FOLDER": str(tmp_path / "results"),
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch("traigent.cloud.url_security._reject_unsafe_hostname"),
        ):
            manager = self._manager(tmp_path)
            assert manager.base_url.startswith("https://backend.example.com")
            assert manager.session is not None

    def test_preview_cloud_cli_works(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        from traigent.cli.local_commands import local_commands

        result = CliRunner().invoke(local_commands, ["preview-cloud"])
        assert result.exit_code == 0, result.output
        assert "traigent.ai" not in result.output

    def test_sync_dry_run_cli_works(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        from traigent.cli.local_commands import local_commands

        result = CliRunner().invoke(local_commands, ["sync", "--dry-run"])
        assert "EndpointNotConfiguredError" not in result.output
        assert "No Traigent backend endpoint" not in result.output
        assert result.exit_code == 0, result.output


class TestAuthCli:
    def _cli(self, tmp_path: Path):
        from traigent.cli import auth_commands

        with (
            patch.object(auth_commands, "TRAIGENT_CONFIG_DIR", tmp_path / ".t"),
            patch.object(auth_commands, "CREDENTIALS_FILE", tmp_path / ".t" / "c.json"),
        ):
            return auth_commands.TraigentAuthCLI()

    def test_constructs_without_endpoint(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        with pytest.raises(EndpointNotConfiguredError):
            _ = cli.backend_url
        with pytest.raises(EndpointNotConfiguredError):
            _ = cli.backend_api_url

    def test_status_works_locally(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        with patch.object(cli, "_load_stored_credentials", return_value=None):
            assert cli.status() is False

    def test_status_with_env_key_shows_not_configured(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        fake_key = "tg_x"  # pragma: allowlist secret
        with patch.dict("os.environ", {"TRAIGENT_API_KEY": fake_key}):
            assert cli.status() is True

    def test_logout_works_locally(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        with patch.object(cli, "_load_stored_credentials", return_value=None):
            assert asyncio.run(cli.logout()) is True

    def test_configure_accepts_an_endpoint(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        answers = iter(["https://backend.example.com", "3"])
        with (
            patch("traigent.cloud.url_security._reject_unsafe_hostname"),
            patch(
                "traigent.cli.auth_commands.Prompt.ask",
                side_effect=lambda *a, **k: next(answers),
            ),
        ):
            assert cli.configure() is True

    def test_configure_rejects_invalid_endpoint(
        self, strict_no_endpoint: None, tmp_path: Path
    ) -> None:
        cli = self._cli(tmp_path)
        with patch("traigent.cli.auth_commands.Prompt.ask", return_value="not a url"):
            assert cli.configure() is False


class TestPlanResolver:
    def test_strict_mode_does_not_fall_back_to_localhost(
        self, strict_no_endpoint: None
    ) -> None:
        from traigent.cli.plan_command import _resolve_backend_url

        with pytest.raises(EndpointNotConfiguredError):
            _resolve_backend_url(None)

    def test_strict_mode_explicit_flag_is_used(self, strict_no_endpoint: None) -> None:
        from traigent.cli.plan_command import _resolve_backend_url

        assert _resolve_backend_url("http://backend.internal") == (
            "http://backend.internal",
            True,
        )

    def test_default_mode_keeps_localhost_fallback(self) -> None:
        from traigent.cli.plan_command import _resolve_backend_url
        from traigent.config.backend_config import DEFAULT_LOCAL_URL

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(STORED_URL, return_value=None),
        ):
            assert _resolve_backend_url(None) == (DEFAULT_LOCAL_URL, False)

    def test_command_reports_missing_endpoint(self, strict_no_endpoint: None) -> None:
        from traigent.cli.plan_command import plan as plan_command

        result = CliRunner().invoke(
            plan_command,
            [
                "--task-description",
                "t",
                "--dataset-size",
                "10",
                "--objective",
                "accuracy",
                "--max-trials",
                "5",
                "--cost-limit",
                "1",
            ],
        )
        assert result.exit_code != 0
        assert "TRAIGENT_BACKEND_URL" in result.output


class TestMissingEndpointWarnings:
    def test_message_names_the_setting(self) -> None:
        assert "TRAIGENT_BACKEND_URL" in endpoint_missing_message("x")

    def test_analytics_mcp_backend_url_lookup_warns(
        self, strict_no_endpoint: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        from traigent.analytics_mcp.tools import _resolved_backend_url

        with caplog.at_level(logging.WARNING):
            assert _resolved_backend_url() is None
        assert any("TRAIGENT_BACKEND_URL" in r.getMessage() for r in caplog.records)
