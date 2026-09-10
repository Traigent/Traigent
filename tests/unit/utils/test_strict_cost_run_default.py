"""Strict cost accounting defaults on for runs whose objectives include cost.

Without it, a call whose model has no price is recorded as $0.00 and the
optimizer, which minimizes cost, ranks that configuration cheapest: measured on
develop 42db2603, an unpriced model won ``best_config`` over a priced one with
equal accuracy, whichever order the configuration space listed them in.

Contract under test:
- An explicit ``TRAIGENT_STRICT_COST_ACCOUNTING`` always wins (``false`` opts out).
- Otherwise a run with a cost objective is strict at runtime: an unpriced call
  with no provider-reported cost fails the run with an actionable message.
- A provider-reported cost (OpenRouter's ``response_cost``) still prices the call,
  and the pre-run coverage preflight ignores the run default, so such models are
  not blocked before any response exists.
- Runs without a cost objective keep the non-strict behavior.
- Every run records where its price table came from in ``result.metadata``.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

import traigent.utils.cost_calculator as cost_calculator
from traigent.api.decorators import optimize
from traigent.core.optimized_function import _objectives_include_cost
from traigent.utils.env_config import (
    is_strict_cost_accounting,
    strict_cost_accounting_origin,
    strict_cost_accounting_run_default,
)

UNPRICED_MODEL = "acme/private-llm-v9"


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class _LLMResponse:
    """litellm-style response; ``provider_cost`` mimics OpenRouter's reported cost."""

    def __init__(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider_cost: float | None = None,
    ) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(input_tokens, output_tokens)
        self.model = model
        if provider_cost is not None:
            self._hidden_params = {"response_cost": provider_cost}


@pytest.fixture(autouse=True)
def _isolated_cost_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("TRAIGENT_ENV", "test")
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_COST_TRACKING", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_JSON", raising=False)
    monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", raising=False)
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")  # no interactive prompt
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    cost_calculator.reset_unpriced_runtime_models()
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None
    yield
    cost_calculator.reset_unpriced_runtime_models()
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None


def _error_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    err: BaseException | None = exc
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        parts.append(f"{type(err).__name__}: {err}")
        err = err.__cause__ or err.__context__
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Precedence of the explicit setting over the run-scoped default              #
# --------------------------------------------------------------------------- #
def test_unset_env_outside_a_run_is_not_strict() -> None:
    assert is_strict_cost_accounting() is False
    assert strict_cost_accounting_origin() == "default_off"


def test_run_default_turns_strict_on_when_env_unset() -> None:
    with strict_cost_accounting_run_default(True):
        assert is_strict_cost_accounting() is True
        assert strict_cost_accounting_origin() == "cost_objective"


def test_run_default_off_keeps_non_strict() -> None:
    with strict_cost_accounting_run_default(False):
        assert is_strict_cost_accounting() is False
        assert strict_cost_accounting_origin() == "default_off"


def test_explicit_false_opts_out_of_the_run_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    with strict_cost_accounting_run_default(True):
        assert is_strict_cost_accounting() is False
        assert strict_cost_accounting_origin() == "env"


def test_explicit_true_is_strict_without_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    assert is_strict_cost_accounting() is True
    assert strict_cost_accounting_origin() == "env"


def test_blank_env_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "  ")
    with strict_cost_accounting_run_default(True):
        assert is_strict_cost_accounting() is True
        assert strict_cost_accounting_origin() == "cost_objective"


def test_preflight_view_ignores_the_run_default() -> None:
    with strict_cost_accounting_run_default(True):
        assert is_strict_cost_accounting(include_run_default=False) is False


def test_preflight_view_still_honours_explicit_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    assert is_strict_cost_accounting(include_run_default=False) is True


def test_run_default_resets_on_exit_and_on_error() -> None:
    with strict_cost_accounting_run_default(True):
        pass
    assert is_strict_cost_accounting() is False

    with pytest.raises(RuntimeError):
        with strict_cost_accounting_run_default(True):
            raise RuntimeError("boom")
    assert is_strict_cost_accounting() is False


def test_run_default_reaches_worker_threads_that_reenter_the_callers_context() -> None:
    """Mirrors the evaluator, which runs sync examples via ``copy_context().run``."""
    seen: list[bool] = []
    with strict_cost_accounting_run_default(True):
        ctx = contextvars.copy_context()
    worker = threading.Thread(
        target=lambda: seen.append(ctx.run(is_strict_cost_accounting))
    )
    worker.start()
    worker.join()
    assert seen == [True]


# --------------------------------------------------------------------------- #
# End to end: a real offline optimization run                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_cost_objective_fails_closed_on_an_unpriced_call_by_default() -> None:
    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", UNPRICED_MODEL, 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc_info:  # noqa: B017 - must fail closed
            await fn.optimize(progress_bar=False)

    text = _error_chain_text(exc_info.value)
    assert "'cost' is an objective" in text
    assert "TRAIGENT_STRICT_COST_ACCOUNTING=false" in text
    assert "TRAIGENT_CUSTOM_MODEL_PRICING_JSON" in text
    # The run-scoped default must not outlive the run.
    assert is_strict_cost_accounting() is False


@pytest.mark.asyncio
async def test_provider_reported_cost_prices_an_unpriced_model_and_passes_preflight() -> (
    None
):
    """OpenRouter case: the model is in the configuration space and has no table
    price, but each response reports its cost. The preflight must not block it,
    and the reported cost is what the run records."""

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"model": [UNPRICED_MODEL]},
        injection_mode="parameter",
        config_param="config",
        offline=True,
    )
    def fn(question: str = "", config: dict | None = None):
        model = (config or {}).get("model", UNPRICED_MODEL)
        return _LLMResponse("A", model, 700, 300, provider_cost=0.0123)

    # The suite runs with TRAIGENT_MOCK_LLM=true, which skips the preflight;
    # switch it off so this test really exercises the preflight carve-out.
    with (
        warnings.catch_warnings(),
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
    ):
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials, "expected the trial to run"
    costs = [(t.metrics or {}).get("cost") for t in result.trials]
    assert all(c is not None and c > 0 for c in costs), costs
    pricing = result.metadata["pricing"]
    assert pricing["strict_cost_accounting"] is True
    assert pricing["strict_cost_accounting_origin"] == "cost_objective"
    assert pricing["usage_captured"] is True


@pytest.mark.asyncio
async def test_explicit_false_restores_the_non_strict_warning_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", UNPRICED_MODEL, 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert "UNPRICED_MODEL_RUNTIME" in result.warning_codes
    assert result.metadata["pricing"]["strict_cost_accounting"] is False
    assert result.metadata["pricing"]["strict_cost_accounting_origin"] == "env"


@pytest.mark.asyncio
async def test_run_without_a_cost_objective_stays_non_strict() -> None:
    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", UNPRICED_MODEL, 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials
    pricing = result.metadata["pricing"]
    assert pricing["strict_cost_accounting"] is False
    assert pricing["strict_cost_accounting_origin"] == "default_off"


@pytest.mark.asyncio
async def test_every_run_records_its_price_table_source() -> None:
    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return "A"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    pricing = result.metadata["pricing"]
    assert pricing["price_table_source"] in {"local", "remote", "unknown"}
    assert {
        "price_table_env_forced",
        "price_table_fallback_reason",
        "litellm_local_model_cost_map",
    } <= set(pricing)


def test_pricing_provenance_never_raises_without_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cost_calculator, "LITELLM_AVAILABLE", False)
    assert cost_calculator.get_pricing_provenance()["price_table_source"] == "unknown"


# --------------------------------------------------------------------------- #
# Review follow-up: the two remaining ways a strict run still recorded $0     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_captured_tokens_with_no_model_name_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """700/300 tokens and an empty model name: the SDK knows there was spend.

    Before this guard the empty-model branch in metrics_tracker logged a
    warning, recorded $0, and the run completed — the same "unpriced call
    scores free" defect as an unknown model, reached by a different door.

    The specific message is raised per trial, which the trial lifecycle turns
    into a failed trial (logged verbatim); the run then fails closed through
    the CostEnforcer, whose message names the same opt-out. Both surfaces are
    asserted, because the run-level error chain does not carry the per-trial
    exception object.
    """

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", "", 700, 300)

    with warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc_info:  # noqa: B017 - must fail closed
            await fn.optimize(progress_bar=False)

    run_error = _error_chain_text(exc_info.value)
    assert "Cost extraction failed" in run_error
    assert "TRAIGENT_STRICT_COST_ACCOUNTING=false" in run_error
    assert "no model name" in caplog.text
    assert "tokens were captured (in=700, out=300)" in caplog.text
    assert "TRAIGENT_STRICT_COST_ACCOUNTING=false" in caplog.text
    assert is_strict_cost_accounting() is False


@pytest.mark.asyncio
async def test_captured_tokens_with_no_model_name_are_zero_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", "", 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials
    costs = [float((t.metrics or {}).get("cost") or 0.0) for t in result.trials]
    assert costs == [0.0] * len(costs)
    assert result.metadata["pricing"]["strict_cost_accounting"] is False


@pytest.mark.asyncio
async def test_cost_objective_with_no_usage_captured_fails_closed() -> None:
    """No tokens on any trial: the $0 cost column is unmeasured, not cheap."""

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return "A"

    # The suite runs with TRAIGENT_MOCK_LLM=true, and a mock run has no spend
    # to measure, so the guard only warns there; switch it off to exercise the
    # real-run path.
    with (
        warnings.catch_warnings(),
        patch("traigent.core.optimized_function.is_mock_llm", return_value=False),
    ):
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc_info:  # noqa: B017 - must fail closed
            await fn.optimize(progress_bar=False)

    text = _error_chain_text(exc_info.value)
    assert "no LLM usage was captured" in text
    assert "TRAIGENT_STRICT_COST_ACCOUNTING=false" in text
    assert is_strict_cost_accounting() is False


@pytest.mark.asyncio
async def test_cost_objective_with_no_usage_warns_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return "A"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" in result.warning_codes
    assert result.metadata["pricing"]["usage_captured"] is False


@pytest.mark.asyncio
async def test_no_cost_objective_and_no_usage_is_not_a_warning() -> None:
    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return "A"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" not in result.warning_codes
    assert result.metadata["pricing"]["usage_captured"] is False


def test_cost_per_1k_counts_as_a_cost_objective() -> None:
    """``insights.py`` already treats ``cost_per_1k`` as a cost metric."""
    assert _objectives_include_cost(["accuracy", "cost_per_1k"]) is True
    assert _objectives_include_cost(["accuracy", "latency"]) is False


@pytest.mark.asyncio
async def test_mock_llm_run_warns_instead_of_failing_without_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mock run has no spend to measure, so the guard warns rather than fails.

    Strict is on (cost objective, env unset) and nothing is captured — the
    real-run path would raise. The pre-run cost preflight already skips mock
    mode for the same reason.
    """
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return "A"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    assert result.trials
    pricing = result.metadata["pricing"]
    assert pricing["strict_cost_accounting"] is True
    assert pricing["usage_captured"] is False
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" in result.warning_codes
