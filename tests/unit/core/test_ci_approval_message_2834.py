"""#2834: the CI approval error teaches an example that actually validates."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.ci_approval import check_ci_approval
from traigent.utils.exceptions import OptimizationError


def _config(storage: Path) -> MagicMock:
    config = MagicMock()
    config.is_local_mode.return_value = True
    config.is_edge_analytics_mode.return_value = True
    config.get_local_storage_path.return_value = str(storage)
    return config


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "TRAIGENT_RUN_APPROVED"}


def _message(storage: Path) -> str:
    with (
        patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
        patch.dict(os.environ, _clean_env(), clear=True),
        pytest.raises(OptimizationError) as info,
    ):
        check_ci_approval(_config(storage))
    return str(info.value)


def test_example_token_in_message_validates_when_used(tmp_path: Path) -> None:
    message = _message(tmp_path)
    match = re.search(r"echo '(\{.*\})' > (\S+)", message)  # path has no spaces here
    assert match, message
    token = json.loads(match.group(1))
    expires = datetime.strptime(token["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    assert expires > datetime.now(UTC)
    Path(match.group(2)).write_text(match.group(1), encoding="utf-8")

    with (
        patch("traigent.core.ci_approval._is_ci_environment", return_value=True),
        patch.dict(os.environ, _clean_env(), clear=True),
    ):
        check_ci_approval(_config(tmp_path))  # must not raise


def test_message_names_the_path_actually_read(tmp_path: Path) -> None:
    custom = tmp_path / "results"
    custom.mkdir()
    message = _message(custom)
    assert str(custom / "approval.token") in message
    assert "~/.traigent/approval.token" not in message


def test_message_states_cap_and_real_environment_protection_remedy(
    tmp_path: Path,
) -> None:
    message = _message(tmp_path)
    assert "24 hours" in message
    assert "must still export TRAIGENT_RUN_APPROVED=1" in message


def test_expired_legacy_token_logs_the_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    past = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "approval.token").write_text(
        json.dumps({"approved_by": "me", "expires_at": past}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING):
        message = _message(tmp_path)
    assert "CI/CD Approval Required" in message
    assert any("expired" in r.getMessage() for r in caplog.records)


def test_over_ttl_legacy_token_logs_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    far = (datetime.now(UTC) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "approval.token").write_text(
        json.dumps({"approved_by": "me", "expires_at": far}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING):
        _message(tmp_path)
    assert any("24 hours" in r.getMessage() for r in caplog.records)


def test_message_quotes_a_path_with_spaces(tmp_path: Path) -> None:
    spaced = tmp_path / "my results"
    spaced.mkdir()
    message = _message(spaced)
    assert f"> '{spaced / 'approval.token'}'" in message


def test_legacy_token_without_expiry_is_logged_as_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "approval.token").write_text(
        json.dumps({"approved_by": "me", "expires_at": "not-a-date"}), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING):
        _message(tmp_path)
    assert any("missing or not a valid" in r.getMessage() for r in caplog.records)
