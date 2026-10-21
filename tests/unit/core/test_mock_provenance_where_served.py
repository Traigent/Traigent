"""#2516: mock provenance is recorded where LLM responses are actually served.

Dispatch-time or build-time samples of mock mode can be flipped by an agent or a
callback mid-run. These tests drive the public API with real LiteLLM
interception (``mock_response`` keeps the non-mock path offline).
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")

import traigent  # noqa: E402
from traigent.testing import disable_mock_mode  # noqa: E402
from traigent.utils.callbacks import OptimizationCallback  # noqa: E402
from traigent.utils.litellm_interceptor import (  # noqa: E402
    patch_litellm_for_metadata_capture,
)
from traigent.utils.results_table import _mock_run_notice  # noqa: E402


class _AfterTrial(OptimizationCallback):
    def __init__(self, action) -> None:
        self.action = action

    def on_optimization_start(self, config_space, objectives, algorithm) -> None:
        pass

    def on_trial_start(self, trial_number, config) -> None:
        pass

    def on_trial_complete(self, trial, progress) -> None:
        self.action()

    def on_optimization_complete(self, result) -> None:
        pass


@pytest.fixture(autouse=True)
def _interceptor(monkeypatch):
    patch_litellm_for_metadata_capture()
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    disable_mock_mode()
    yield
    disable_mock_mode()


async def _run(agent, callback, **options):
    @traigent.optimize(
        configuration_space={"x": [1]},
        objectives=["accuracy"],
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        offline=True,
        max_trials=1,
        **options,
    )
    def fn(q: str) -> str:
        return agent()

    return await fn.optimize(max_trials=1, algorithm="grid", callbacks=[callback])


def _call_llm() -> str:
    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="a",
    )
    return "a"


@pytest.mark.asyncio
async def test_agent_enables_mock_inside_trial_and_callback_disables_it(
    monkeypatch,
) -> None:
    def agent() -> str:
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
        return _call_llm()

    result = await _run(
        agent, _AfterTrial(lambda: monkeypatch.delenv("TRAIGENT_MOCK_LLM"))
    )

    assert result.metadata["mock_llm_run"] is True
    assert "MOCK RUN" in (_mock_run_notice(result) or "")


@pytest.mark.asyncio
async def test_winner_rerun_served_by_mock_is_reflected_in_final_provenance(
    monkeypatch,
) -> None:
    result = await _run(
        _call_llm,
        _AfterTrial(lambda: monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")),
        winner_stability_reps=1,
    )

    assert result.metadata["mock_llm_run"] is True
    assert result.metadata["mock_llm_mixed"] is True
    notice = _mock_run_notice(result) or ""
    assert "mix simulated and real results" in notice


@pytest.mark.asyncio
async def test_real_calls_only_stay_unlabelled_even_if_mock_is_on_at_the_end(
    monkeypatch,
) -> None:
    result = await _run(
        _call_llm,
        _AfterTrial(lambda: monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")),
    )

    assert result.metadata["mock_llm_run"] is False
    assert "mock_llm_mixed" not in result.metadata
    assert _mock_run_notice(result) is None


def test_mixed_wording_never_claims_all_cost_is_synthetic() -> None:
    from types import SimpleNamespace

    result = SimpleNamespace(metadata={"mock_llm_run": True, "mock_llm_mixed": True})
    notice = _mock_run_notice(result) or ""
    assert notice
    for banned in ("synthetic estimate", "not real spend", "not meaningful"):
        assert banned not in notice
