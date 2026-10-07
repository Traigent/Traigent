"""Privacy boundary checks for trial spans at the HTTP send boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.api.types import TrialError, TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.integrations.observability import workflow_traces as wt


@pytest.mark.parametrize("send_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failed", [False, True], ids=["success", "failed"])
@pytest.mark.parametrize(
    "privacy_enabled", [False, True], ids=["privacy-off", "privacy-on"]
)
async def test_privacy_trial_span_sending_boundary_excludes_config_and_error_canaries(
    monkeypatch: pytest.MonkeyPatch,
    send_async: bool,
    failed: bool,
    privacy_enabled: bool,
) -> None:
    config = TraigentConfig()
    object.__setattr__(config, "privacy_enabled", privacy_enabled)
    spans = []
    orchestrator = SimpleNamespace(
        _workflow_traces_tracker=object(),
        _optimization_id="opt-canary-test",
        traigent_config=config,
        collect_workflow_span=spans.append,
    )
    lifecycle = TrialLifecycle.__new__(TrialLifecycle)
    lifecycle._orchestrator = orchestrator
    trial = TrialResult(
        trial_id="trial-1",
        config={
            "system_prompt": "CANARY-PROMPT-7731",
            "temperature": 0.7731,
            "max_retries": 37,
            "use_cache": True,
            "limits": {"max": 42},
        },
        metrics={"accuracy": 0.75, "total_cost": 0.125},
        status=TrialStatus.FAILED if failed else TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        error_message="CANARY-ERR-4419" if failed else None,
        error=(
            TrialError(
                message="CANARY-ERR-4419",
                error_type="SyntheticTrialError",
                traceback="CANARY-ERR-4419",
                timestamp=datetime.now(UTC),
                config={"system_prompt": "CANARY-PROMPT-7731"},
            )
            if failed
            else None
        ),
        metadata={"input_tokens": 4, "output_tokens": 2},
    )
    lifecycle._collect_workflow_span("run-1", trial, 1.0, 2.0)

    client = wt.WorkflowTracesClient("https://api.traigent.com")
    captured: list[dict] = []
    if send_async:
        response = MagicMock(status=200)
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"success": True})

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def post(self, _url, *, json, **_kwargs):
                captured.append(json)
                request = MagicMock()
                request.__aenter__ = AsyncMock(return_value=response)
                request.__aexit__ = AsyncMock(return_value=None)
                return request

        monkeypatch.setattr(wt.aiohttp, "ClientSession", lambda **_kwargs: Session())
        await client.ingest_traces_async(
            spans=spans, trace_id="opt-canary-test", configuration_run_id="run-1"
        )
    else:
        response = MagicMock(status_code=200)
        response.json.return_value = {"success": True}
        post = MagicMock(
            side_effect=lambda *_args, json, **_kwargs: (
                captured.append(json),
                response,
            )[1]
        )
        monkeypatch.setattr(wt.requests, "post", post)
        client.ingest_traces(
            spans=spans, trace_id="opt-canary-test", configuration_run_id="run-1"
        )

    outgoing_payload = captured[0]
    outgoing = str(outgoing_payload)
    outgoing_config = outgoing_payload["spans"]["spans"][0]["input_data"]["config"]
    if privacy_enabled:
        assert "CANARY-PROMPT-7731" not in outgoing
        assert "CANARY-ERR-4419" not in outgoing
        assert outgoing_config == dict.fromkeys(trial.config, "[REDACTED]")
        if failed:
            assert "SyntheticTrialError" in outgoing
    else:
        assert "CANARY-PROMPT-7731" in outgoing
        assert ("CANARY-ERR-4419" in outgoing) is failed
        assert outgoing_config == trial.config
