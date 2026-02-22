"""Live backend variant of the CTD execution mode behavioral test."""

from __future__ import annotations

import itertools
import os

import pytest

from tests.integration.test_ctd_execution_modes import (
    CTD_CASES,
    _assert_best_config,
    _grid_configurations,
    _run_behavior_validation,
)
from traigent.traigent_client import TraigentClient

# Only run cases that require an API key to exercise backend communication.
LIVE_CASES = [case for case in CTD_CASES if case["combo"].get("has_api_key")]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TRAIGENT_CTD_LIVE"),
    reason="Set TRAIGENT_CTD_LIVE=1 and provide TRAIGENT_API_KEY to run live backend tests",
)
@pytest.mark.parametrize("case", LIVE_CASES, ids=lambda payload: payload["id"])
async def test_ctd_execution_behavior_live(case: dict, monkeypatch) -> None:
    """Execute CTD scenarios against the real backend client."""
    combo = dict(case["combo"])  # shallow copy for safety

    # Ensure we are not in mock mode
    for key in ("TRAIGENT_MOCK_LLM", "TRAIGENT_USE_MOCK", "TRAIGENT_GENERATE_MOCKS"):
        monkeypatch.delenv(key, raising=False)

    # Apply force flags from combo
    for env_key in [
        "TRAIGENT_FORCE_LOCAL",
        "TRAIGENT_FORCE_HYBRID",
        "TRAIGENT_FORCE_CLOUD",
        "TRAIGENT_PRIVATE_DATA",
    ]:
        monkeypatch.delenv(env_key, raising=False)

    if combo.get("force_local"):
        monkeypatch.setenv("TRAIGENT_FORCE_LOCAL", "1")
    if combo.get("force_hybrid"):
        monkeypatch.setenv("TRAIGENT_FORCE_HYBRID", "1")
    if combo.get("force_cloud"):
        monkeypatch.setenv("TRAIGENT_FORCE_CLOUD", "1")
    if combo.get("privacy_flag"):
        monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "1")

    # Backend URL handling
    backend_url = combo.get("backend_url")
    if backend_url == "local":
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    elif backend_url == "prod":
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://api.traigent.ai")

    # We expect an API key to be available in the environment
    api_key = os.getenv("TRAIGENT_API_KEY")
    if not api_key:
        pytest.skip("TRAIGENT_API_KEY must be set for live backend test")

    # Instantiate real client
    client = TraigentClient(
        execution_mode=combo.get("explicit_mode", "auto"), api_key=api_key
    )

    # Run the same behavioral validation as the mocked test
    result = await _run_behavior_validation(case, client.execution_mode)

    scenario = case["scenario"]

    assert len(result.trials) == scenario.expected_trials
    assert result.best_score == pytest.approx(scenario.expected_best_score)
    assert result.success_rate == pytest.approx(1.0)

    _assert_best_config(result.best_config, scenario.expected_best_config)

    actual_configs = {tuple(sorted(trial.config.items())) for trial in result.trials}
    expected_configs = {
        tuple(sorted(cfg.items()))
        for cfg in itertools.islice(
            _grid_configurations(scenario.search_space), scenario.expected_trials
        )
    }
    assert actual_configs == expected_configs
