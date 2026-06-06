from __future__ import annotations

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import traigent.utils.cost_calculator as cost_calculator
from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.cost_calculator import UnknownModelError

FAKE_MODEL = "us.fake.unpriced-model-v9:0"


class CountedFunction:
    def __init__(self) -> None:
        self.call_count = 0
        self.__name__ = "counted_function"

    def __call__(self, text: str = "", **_kwargs: object) -> str:
        self.call_count += 1
        return text


@pytest.fixture(autouse=True)
def reset_cost_preflight_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_JSON", raising=False)
    monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", raising=False)
    monkeypatch.delenv("TRAIGENT_COST_APPROVED", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("traigent.core.optimized_function._COST_WARNING_EMITTED", True)
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None
    yield
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None


@pytest.fixture
def one_example_dataset() -> Dataset:
    return Dataset(
        [EvaluationExample(input_data={"text": "hello"}, expected_output="hello")]
    )


def _completed_result() -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"model": "gpt-4o"},
        best_score=1.0,
        optimization_id="cost-preflight-test",
        duration=0.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=datetime.now(UTC),
        metadata={},
    )


async def _run_with_mocked_orchestrator(
    opt_func: OptimizedFunction,
    *,
    cost_approved: Any | None = None,
) -> OptimizationResult:
    with patch(
        "traigent.core.optimized_function.OptimizationOrchestrator"
    ) as mock_orchestrator_cls:
        mock_orchestrator = mock_orchestrator_cls.return_value
        mock_orchestrator.optimize = AsyncMock(return_value=_completed_result())
        optimize_kwargs: dict[str, Any] = {}
        if cost_approved is not None:
            optimize_kwargs["cost_approved"] = cost_approved
        result = await opt_func.optimize(progress_bar=False, **optimize_kwargs)
        mock_orchestrator.optimize.assert_awaited_once()
        return result


@pytest.mark.asyncio
async def test_cost_preflight_blocks_unpriced_models_non_interactive_by_default(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    counted = CountedFunction()
    opt_func = OptimizedFunction(
        func=counted,
        configuration_space={"model": ["gpt-4o", FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with patch("traigent.core.optimized_function.is_mock_llm", return_value=False):
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_cls:
            with pytest.raises(UnknownModelError, match="non-interactive shell"):
                await opt_func.optimize(progress_bar=False)

    assert counted.call_count == 0
    mock_orchestrator_cls.assert_not_called()


@pytest.mark.asyncio
async def test_cost_preflight_env_approved_warns_with_exact_unpriced_models(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": ["gpt-4o", FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("approval prompt should be skipped"),
    )
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    preflight_warnings = [
        str(warning.message)
        for warning in captured
        if "results will report $0" in str(warning.message)
    ]
    assert preflight_warnings == [
        "Cost for `us.fake.unpriced-model-v9:0` is unavailable — results will "
        "report $0 for it. Set TRAIGENT_CUSTOM_MODEL_PRICING_JSON, or contact "
        "Traigent to add coverage."
    ]
    assert "`gpt-4o`" not in preflight_warnings[0]


@pytest.mark.asyncio
async def test_cost_preflight_cost_approved_param_skips_prompt_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("approval prompt should be skipped"),
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func, cost_approved=True)

    assert [
        warning
        for warning in captured
        if FAKE_MODEL in str(warning.message)
        and "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cost_approved", ["false", "true"])
async def test_cost_preflight_string_cost_approved_param_blocks_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    one_example_dataset: Dataset,
    cost_approved: str,
) -> None:
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("non-interactive preflight should not prompt"),
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_cls,
        caplog.at_level("WARNING", logger="traigent.core.cost_enforcement"),
    ):
        with pytest.raises(UnknownModelError, match="non-interactive shell"):
            await opt_func.optimize(progress_bar=False, cost_approved=cost_approved)

    assert (
        f"Ignoring non-boolean cost_approved='{cost_approved}' (type: str)"
        in caplog.text
    )
    mock_orchestrator_cls.assert_not_called()


@pytest.mark.asyncio
async def test_cost_preflight_token_approval_skips_prompt_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    one_example_dataset: Dataset,
) -> None:
    token_path = tmp_path / "traigent" / "cost_approval.token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("approved")
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("approval prompt should be skipped"),
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert [
        warning
        for warning in captured
        if FAKE_MODEL in str(warning.message)
        and "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["n", ""])
async def test_cost_preflight_interactive_decline_blocks_before_any_trial(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
    choice: str,
) -> None:
    counted = CountedFunction()
    opt_func = OptimizedFunction(
        func=counted,
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": choice)

    with patch("traigent.core.optimized_function.is_mock_llm", return_value=False):
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_cls:
            with pytest.raises(UnknownModelError, match="declined"):
                await opt_func.optimize(progress_bar=False)

    assert counted.call_count == 0
    mock_orchestrator_cls.assert_not_called()


@pytest.mark.asyncio
async def test_cost_preflight_interactive_eof_blocks_before_any_trial(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    counted = CountedFunction()
    opt_func = OptimizedFunction(
        func=counted,
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with patch("traigent.core.optimized_function.is_mock_llm", return_value=False):
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_cls:
            with pytest.raises(UnknownModelError, match="interrupted"):
                await opt_func.optimize(progress_bar=False)

    assert counted.call_count == 0
    mock_orchestrator_cls.assert_not_called()


@pytest.mark.asyncio
async def test_cost_preflight_interactive_keyboard_interrupt_blocks_before_any_trial(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    counted = CountedFunction()
    opt_func = OptimizedFunction(
        func=counted,
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def raise_keyboard_interrupt() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_keyboard_interrupt)

    with patch("traigent.core.optimized_function.is_mock_llm", return_value=False):
        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as mock_orchestrator_cls:
            with pytest.raises(UnknownModelError, match="interrupted"):
                await opt_func.optimize(progress_bar=False)

    assert counted.call_count == 0
    mock_orchestrator_cls.assert_not_called()


@pytest.mark.asyncio
async def test_cost_preflight_interactive_yes_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert [
        warning
        for warning in captured
        if FAKE_MODEL in str(warning.message)
        and "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_cost_preflight_strict_blocks_before_any_trial(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    counted = CountedFunction()
    opt_func = OptimizedFunction(
        func=counted,
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("strict mode should not prompt"),
    )

    with patch("traigent.core.optimized_function.is_mock_llm", return_value=False):
        with pytest.raises(UnknownModelError, match=FAKE_MODEL):
            await opt_func.optimize(progress_bar=False)

    assert counted.call_count == 0


@pytest.mark.asyncio
async def test_cost_preflight_fully_priced_space_has_no_warning(
    one_example_dataset: Dataset,
) -> None:
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": ["gpt-4o", "gpt-4o-mini"]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert not [
        warning
        for warning in captured
        if "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_cost_preflight_custom_pricing_json_counts_as_covered(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    monkeypatch.setenv(
        "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
        json.dumps(
            {
                FAKE_MODEL: {
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 2e-6,
                }
            }
        ),
    )
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": pytest.fail("covered model should not prompt"),
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert not [
        warning
        for warning in captured
        if "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_cost_preflight_mock_mode_skips_warning_and_strict_block(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"model": [FAKE_MODEL]},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=True),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert not [
        warning
        for warning in captured
        if "results will report $0" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_cost_preflight_checks_default_model_when_model_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
    one_example_dataset: Dataset,
) -> None:
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    opt_func = OptimizedFunction(
        func=CountedFunction(),
        configuration_space={"temperature": [0.0, 0.5]},
        default_config={"model": FAKE_MODEL},
        eval_dataset=one_example_dataset,
        max_trials=1,
    )

    with (
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        await _run_with_mocked_orchestrator(opt_func)

    assert [
        warning
        for warning in captured
        if FAKE_MODEL in str(warning.message)
        and "results will report $0" in str(warning.message)
    ]
