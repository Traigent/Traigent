"""Live backend smoke test for the interactive optimization session API."""

from __future__ import annotations

import os

import pytest

from traigent.cloud.client import TraigentCloudClient


def _resolve_backend_url() -> str | None:
    return os.getenv("TRAIGENT_BACKEND_URL") or os.getenv("TRAIGENT_API_URL")


def _score_config(config: dict[str, object]) -> dict[str, float]:
    model = str(config["model"])
    temperature = float(config["temperature"])

    accuracy = (
        0.9 - abs(temperature - 0.4) * 0.05
        if model == "gpt-4o"
        else 0.82 - abs(temperature - 0.2) * 0.04
    )
    cost = 0.35 if model == "gpt-4o" else 0.08

    return {"accuracy": accuracy, "cost": cost}


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TRAIGENT_HYBRID_LIVE"),
    reason=(
        "Set TRAIGENT_HYBRID_LIVE=1 plus TRAIGENT_API_KEY and "
        "TRAIGENT_BACKEND_URL or TRAIGENT_API_URL to run live hybrid session tests"
    ),
)
async def test_live_hybrid_session_round_trip() -> None:
    """Exercise create/next/result/finalize against the real backend."""
    api_key = os.getenv("TRAIGENT_API_KEY")
    backend_url = _resolve_backend_url()

    if not api_key:
        pytest.skip("TRAIGENT_API_KEY must be set for the live hybrid session test")

    if not backend_url:
        pytest.skip(
            "Set TRAIGENT_BACKEND_URL or TRAIGENT_API_URL for the live hybrid session test"
        )

    session_id: str | None = None

    async with TraigentCloudClient(
        api_key=api_key,
        base_url=backend_url,
        enable_fallback=False,
    ) as client:
        try:
            created = await client.create_optimization_session(
                "python_hybrid_live_smoke",
                configuration_space={
                    "model": {
                        "type": "categorical",
                        "choices": ["gpt-4o-mini", "gpt-4o"],
                    },
                    "temperature": {
                        "type": "float",
                        "low": 0.0,
                        "high": 1.0,
                        "step": 0.2,
                    },
                },
                objectives=["accuracy", "cost"],
                dataset_metadata={"size": 4, "suite": "python-hybrid-live-smoke"},
                max_trials=4,
                optimization_strategy={"algorithm": "optuna"},
            )

            session_id = created.session_id
            assert session_id
            assert created.status.value in {"created", "active"}

            next_trial = await client.get_next_trial(session_id)
            assert next_trial.should_continue is True
            assert next_trial.suggestion is not None

            suggestion = next_trial.suggestion
            assert suggestion.session_id == session_id
            assert suggestion.trial_number == 1
            assert suggestion.dataset_subset.indices
            assert all(0 <= index < 4 for index in suggestion.dataset_subset.indices)

            metrics = _score_config(suggestion.config)

            await client.submit_trial_result(
                session_id=session_id,
                trial_id=suggestion.trial_id,
                metrics=metrics,
                duration=0.01,
                status="completed",
                metadata={"suite": "python-hybrid-live-smoke"},
            )

            finalized = await client.finalize_optimization(session_id)
            assert finalized.session_id == session_id
            assert finalized.total_trials >= 1
            assert finalized.successful_trials >= 1
            assert isinstance(finalized.best_metrics, dict)
            assert "accuracy" in finalized.best_metrics
            assert finalized.stop_reason in {
                "max_trials_reached",
                "search_complete",
                "finalized",
            }
        finally:
            if session_id:
                try:
                    await client.delete_session(session_id, cascade=True)
                except Exception:
                    pass
