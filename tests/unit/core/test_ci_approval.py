"""Tests for CI/CD approval checks (traigent.core.ci_approval)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from traigent.core.ci_approval import (
    _check_env_var_approval,
    _check_token_file_approval,
    _is_ci_environment,
    _sanitize_for_log,
    _validate_hmac_token,
    _validate_legacy_token,
    check_ci_approval,
)
from traigent.utils.exceptions import ConfigurationError, OptimizationError

# ---------------------------------------------------------------------------
# _is_ci_environment
# ---------------------------------------------------------------------------


class TestIsCiEnvironment:
    """Test CI environment detection for 10 providers."""

    @pytest.mark.parametrize(
        "env_var,env_value",
        [
            ("CI", "true"),
            ("CI", "1"),
            ("GITHUB_ACTIONS", "true"),
            ("JENKINS_URL", "https://jenkins.example.com"),
            ("GITLAB_CI", "true"),
            ("CIRCLECI", "true"),
            ("TRAVIS", "true"),
            ("BUILDKITE", "true"),
            ("TEAMCITY_VERSION", "2023.1"),
            ("AZURE_HTTP_USER_AGENT", "vsts-agent"),
            ("BITBUCKET_BUILD_NUMBER", "42"),
        ],
    )
    def test_detects_ci_providers(self, env_var: str, env_value: str) -> None:
        with patch.dict(os.environ, {env_var: env_value}, clear=True):
            assert _is_ci_environment() is True

    def test_not_ci_when_no_env_vars(self) -> None:
        ci_vars = [
            "CI",
            "GITHUB_ACTIONS",
            "JENKINS_URL",
            "GITLAB_CI",
            "CIRCLECI",
            "TRAVIS",
            "BUILDKITE",
            "TEAMCITY_VERSION",
            "AZURE_HTTP_USER_AGENT",
            "BITBUCKET_BUILD_NUMBER",
        ]
        env = {k: v for k, v in os.environ.items() if k not in ci_vars}
        with patch.dict(os.environ, env, clear=True):
            assert _is_ci_environment() is False


# ---------------------------------------------------------------------------
# _check_env_var_approval
# ---------------------------------------------------------------------------


class TestCheckEnvVarApproval:
    def test_approved_when_env_set(self) -> None:
        with patch.dict(
            os.environ, {"TRAIGENT_RUN_APPROVED": "1", "TRAIGENT_APPROVED_BY": "tester"}
        ):
            assert _check_env_var_approval() is True

    def test_approved_with_default_approver(self) -> None:
        env = {"TRAIGENT_RUN_APPROVED": "1"}
        # Remove TRAIGENT_APPROVED_BY if present
        cleaned = {k: v for k, v in os.environ.items() if k != "TRAIGENT_APPROVED_BY"}
        cleaned.update(env)
        with patch.dict(os.environ, cleaned, clear=True):
            assert _check_env_var_approval() is True

    def test_not_approved_when_env_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_RUN_APPROVED"}
        with patch.dict(os.environ, env, clear=True):
            assert _check_env_var_approval() is False


# ---------------------------------------------------------------------------
# _sanitize_for_log
# ---------------------------------------------------------------------------


class TestSanitizeForLog:
    def test_strips_unsafe_chars(self) -> None:
        assert _sanitize_for_log("user<script>alert(1)") == "userscriptalert1"

    def test_preserves_safe_chars(self) -> None:
        assert _sanitize_for_log("user-name_123@test.com") == "user-name_123@test.com"

    def test_truncates_long_values(self) -> None:
        result = _sanitize_for_log("a" * 100, max_len=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _validate_legacy_token
# ---------------------------------------------------------------------------


class TestValidateLegacyToken:
    def test_valid_token_with_future_expiry(self) -> None:
        # Use naive datetime — code uses datetime.now() (naive) for comparison
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        token = {"approved_by": "tester", "expires_at": future}
        assert _validate_legacy_token(token) is True

    def test_expired_token(self) -> None:
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        token = {"approved_by": "tester", "expires_at": past}
        assert _validate_legacy_token(token) is False

    def test_missing_approved_by(self) -> None:
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        assert _validate_legacy_token({"expires_at": future}) is False

    def test_missing_expires_at(self) -> None:
        assert _validate_legacy_token({"approved_by": "tester"}) is False

    def test_invalid_date_format(self) -> None:
        token = {"approved_by": "tester", "expires_at": "not-a-date"}
        assert _validate_legacy_token(token) is False

    def test_z_suffix_handled(self) -> None:
        # Z suffix gets replaced with +00:00, producing a tz-aware datetime
        # Code then uses datetime.now().replace(tzinfo=UTC) — use large delta to be safe
        future = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        token = {"approved_by": "tester", "expires_at": future}
        assert _validate_legacy_token(token) is True


# ---------------------------------------------------------------------------
# _validate_hmac_token
# ---------------------------------------------------------------------------


class TestValidateHmacToken:
    @staticmethod
    def _make_signed_token(
        approver: str = "ci-bot",
        expires_delta: timedelta = timedelta(hours=1),
        secret: str = "test-secret",  # pragma: allowlist secret
    ) -> dict:
        expires_iso = (datetime.now(UTC) + expires_delta).isoformat()
        nonce = "test-nonce-123"
        payload = f"{approver}|{expires_iso}|{nonce}".encode()
        sig = base64.b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        ).decode()
        return {
            "approver": approver,
            "expires_iso": expires_iso,
            "nonce": nonce,
            "signature": sig,
        }

    def test_valid_signed_token(self) -> None:
        token = self._make_signed_token()
        with patch.dict(
            os.environ, {"TRAIGENT_APPROVAL_SECRET": "test-secret"}  # pragma: allowlist secret
        ):
            assert _validate_hmac_token(token) is True

    def test_invalid_signature(self) -> None:
        token = self._make_signed_token()
        token["signature"] = "invalid-sig"
        with patch.dict(
            os.environ, {"TRAIGENT_APPROVAL_SECRET": "test-secret"}  # pragma: allowlist secret
        ):
            assert _validate_hmac_token(token) is False

    def test_expired_token(self) -> None:
        token = self._make_signed_token(expires_delta=timedelta(hours=-1))
        with patch.dict(
            os.environ, {"TRAIGENT_APPROVAL_SECRET": "test-secret"}  # pragma: allowlist secret
        ):
            assert _validate_hmac_token(token) is False

    def test_ttl_exceeds_24h(self) -> None:
        token = self._make_signed_token(expires_delta=timedelta(hours=25))
        with patch.dict(
            os.environ, {"TRAIGENT_APPROVAL_SECRET": "test-secret"}  # pragma: allowlist secret
        ):
            assert _validate_hmac_token(token) is False

    def test_missing_required_field(self) -> None:
        token = {"approver": "ci-bot", "expires_iso": "2030-01-01T00:00:00+00:00"}
        assert _validate_hmac_token(token) is False

    def test_no_secret_with_valid_expiry(self) -> None:
        """When secret is not set, token is accepted if not expired (with warning)."""
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        token = {
            "approver": "ci-bot",
            "expires_iso": future,
            "nonce": "n",
            "signature": "ignored",
        }
        env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_APPROVAL_SECRET"}
        with patch.dict(os.environ, env, clear=True):
            assert _validate_hmac_token(token) is True

    def test_no_secret_with_expired_token(self) -> None:
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        token = {
            "approver": "ci-bot",
            "expires_iso": past,
            "nonce": "n",
            "signature": "ignored",
        }
        env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_APPROVAL_SECRET"}
        with patch.dict(os.environ, env, clear=True):
            assert _validate_hmac_token(token) is False


# ---------------------------------------------------------------------------
# _check_token_file_approval
# ---------------------------------------------------------------------------


class TestCheckTokenFileApproval:
    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-file.token"
        assert _check_token_file_approval(missing, tmp_path) is False

    def test_valid_legacy_token_file(self, tmp_path: Path) -> None:
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        token = {"approved_by": "tester", "expires_at": future}
        token_file = tmp_path / "approval.token"
        token_file.write_text(json.dumps(token))
        assert _check_token_file_approval(token_file, tmp_path) is True

    def test_invalid_json_token_file(self, tmp_path: Path) -> None:
        token_file = tmp_path / "approval.token"
        token_file.write_text("not-json")
        assert _check_token_file_approval(token_file, tmp_path) is False


# ---------------------------------------------------------------------------
# check_ci_approval (integration-level)
# ---------------------------------------------------------------------------


class TestCheckCiApproval:
    def test_skips_when_not_edge_analytics(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = False
        # Should not raise
        check_ci_approval(config)

    def test_skips_in_mock_llm_mode(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=True),
            patch("traigent.core.ci_approval.is_production", return_value=False),
        ):
            check_ci_approval(config)

    def test_skips_in_mock_llm_production(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=True),
            patch("traigent.core.ci_approval.is_production", return_value=True),
        ):
            check_ci_approval(config)

    def test_skips_when_not_ci(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=False),
            patch("traigent.core.ci_approval._is_ci_environment", return_value=False),
        ):
            check_ci_approval(config)

    def test_raises_when_ci_no_approval(self, tmp_path: Path) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        config.get_local_storage_path.return_value = str(tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_RUN_APPROVED"}
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=False),
            patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
            patch.dict(os.environ, env, clear=True),
        ):
            with pytest.raises(OptimizationError, match="CI/CD Approval Required"):
                check_ci_approval(config)

    def test_raises_when_storage_path_none(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        config.get_local_storage_path.return_value = None
        env = {k: v for k, v in os.environ.items() if k != "TRAIGENT_RUN_APPROVED"}
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=False),
            patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
            patch.dict(os.environ, env, clear=True),
        ):
            with pytest.raises(ConfigurationError, match="Storage path not configured"):
                check_ci_approval(config)

    def test_passes_with_env_var_approval(self) -> None:
        config = MagicMock()
        config.is_edge_analytics_mode.return_value = True
        with (
            patch("traigent.core.ci_approval.is_mock_llm", return_value=False),
            patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
            patch.dict(os.environ, {"TRAIGENT_RUN_APPROVED": "1"}),
        ):
            check_ci_approval(config)
