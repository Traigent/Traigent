"""The results request body is fitted under the backend size cap (#2964)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.trial_operations import _MAX_RESULTS_BODY_BYTES, TrialOperations

pytestmark = pytest.mark.backend_online


async def _submit(
    *, config: dict[str, Any], metrics: dict[str, Any], metadata: dict[str, Any]
) -> tuple[Any, Mock]:
    client = Mock()
    client.backend_config = Mock()
    client.backend_config.api_base_url = "https://api.example.com/api/v1"
    client.auth_manager = AsyncMock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    client._sanitize_error_message = Mock(return_value="")

    ops = TrialOperations(client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)

    response = Mock(status=201)
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(return_value=response)
    post_ctx.__aexit__ = AsyncMock(return_value=False)
    session = AsyncMock()
    session.post = Mock(return_value=post_ctx)
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("traigent.cloud.trial_operations.is_backend_offline", return_value=False),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as aio,
    ):
        client._ensure_session = AsyncMock(return_value=session)
        aio.ClientTimeout = Mock()
        result = await ops.submit_trial_result_via_session(
            session_id="s1",
            trial_id="t1",
            config=config,
            metrics=metrics,
            status="completed",
            metadata=metadata,
        )
    return result, session.post


def _measures(count: int) -> list[dict[str, Any]]:
    return [
        {"example_id": f"ex_{i}", "metrics": {"accuracy": 0.9}, "pad": "x" * 600}
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_measures_trimmed_so_whole_body_fits_with_large_config_and_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Measures alone (~700 KB) are under any measures-only budget, but with a
    # 250 KB config and 200 KB metadata the whole body exceeds the cap.
    config = {"prompt": "c" * 250_000}
    metadata = {"note": "m" * 200_000}
    with caplog.at_level("WARNING"):
        result, post = await _submit(
            config=config,
            metrics={"accuracy": 0.9, "measures": _measures(1000)},
            metadata=metadata,
        )

    assert result is True
    body = post.call_args.kwargs["json"]
    assert len(json.dumps(body).encode("utf-8")) <= _MAX_RESULTS_BODY_BYTES
    assert 0 < len(body["measures"]) < 1000
    assert "truncating per-example measures" in caplog.text


@pytest.mark.asyncio
async def test_body_too_large_without_measures_is_not_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR"):
        result, post = await _submit(
            config={"prompt": "c" * (_MAX_RESULTS_BODY_BYTES + 1000)},
            metrics={"accuracy": 0.9},
            metadata={},
        )

    assert result is False
    post.assert_not_called()
    assert "not sending it" in caplog.text
