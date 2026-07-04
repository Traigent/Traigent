"""Regression tests for cost-metric wiring (#1423), runtime cost-coverage (#1407),
and unpriced-response-id honesty (#1597).

#1423: the per-config ``cost`` metric / objective and the input_cost/output_cost
breakdown must be wired to the SDK's already-correct ``total_cost`` (non-zero,
model-differentiated), the OpenRouter-prefix pricing helper must price via the
normalized id, and ``cost_efficiency`` must not divide by zero.

#1407: a model HARD-CODED in the optimized function body that no pricing table
covers must surface to the USER and the result object at runtime (non-strict) and
must fail closed under strict accounting.

#1597: OpenRouter (or any provider) can return a response-model id litellm
genuinely cannot price (e.g. a model released after litellm's bundled pricing
snapshot, like a future-dated ``anthropic/claude-*`` id). The candidate-ladder
normalization from #1423 already prices every *known* id via the underlying
litellm entry; for the residual genuinely-unknown case, cost must never be
silently indistinguishable from verified-free: ``ExampleMetrics.cost.unpriced``
marks the example, the runtime registry counts occurrences per model (not just
membership), and the result-level warning is quantified and states the $0 is
UNKNOWN spend rather than free usage.

These tests assert externally-observable behavior on a real (mocked-LLM) pipeline;
they never assert ``== 0`` or trivially-true conditions.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import pytest

import traigent.utils.cost_calculator as cost_calculator
from traigent.api.decorators import optimize
from traigent.api.types import TrialResult, TrialStatus
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils.cost_calculator import (
    UnknownModelError,
    completion_cost,
    prompt_cost,
)
from traigent.utils.insights import _calculate_cost_efficiency


# --------------------------------------------------------------------------- #
# Test doubles: minimal litellm-style response objects                        #
# --------------------------------------------------------------------------- #
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
    """OpenAI/litellm-style response with usage and optional provider cost."""

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
            # OpenRouter / litellm report per-call cost here when the model is
            # missing from the pricing table.
            self._hidden_params = {"response_cost": provider_cost}


@pytest.fixture(autouse=True)
def _reset_unpriced_registry():
    cost_calculator.reset_unpriced_runtime_models()
    yield
    cost_calculator.reset_unpriced_runtime_models()


@pytest.fixture(autouse=True)
def _non_prod_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAIGENT_ENV", "test")
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    yield


# --------------------------------------------------------------------------- #
# #1423 (a): OpenRouter-prefix pricing helper                                 #
# --------------------------------------------------------------------------- #
def test_completion_cost_prices_openrouter_prefix_nonzero():
    """completion_cost('openrouter/openai/gpt-4o-mini') returns non-zero.

    litellm.completion_cost raises 'This model isn't mapped yet' for this id even
    though litellm.model_cost HAS the underlying gpt-4o-mini entry. The SDK helper
    must price via the normalized id.
    """
    mini = completion_cost(model="openrouter/openai/gpt-4o-mini", output_tokens=1000)
    assert mini > 0.0

    # And it is model-differentiated: gpt-4o output costs strictly more than mini.
    big = completion_cost(model="openrouter/openai/gpt-4o", output_tokens=1000)
    assert big > mini

    # Prompt side too.
    assert prompt_cost(model="openrouter/openai/gpt-4o-mini", input_tokens=1000) > 0.0


def test_completion_cost_still_zero_for_genuinely_unpriced():
    """An id with no pricing anywhere returns 0.0 (query-mode, not a raise)."""
    assert completion_cost(model="acme/private-unmapped-v9", output_tokens=1000) == 0.0


# --------------------------------------------------------------------------- #
# #1423 (b): input_cost / output_cost breakdown back-filled                   #
# --------------------------------------------------------------------------- #
def test_breakdown_backfilled_when_only_total_cost_reported():
    """OpenRouter path: total_cost authoritative, breakdown reconstructed.

    The provider reports total_cost via _hidden_params['response_cost'] but no
    input/output split. The breakdown must be re-derived from the calculator
    while total_cost stays exactly as reported.
    """
    reported_total = 0.0144275
    metrics = extract_llm_metrics(
        _LLMResponse("AI", "openrouter/openai/gpt-4o", 700, 300, reported_total),
        model_name="openrouter/openai/gpt-4o",
    )
    assert metrics.cost.total_cost == pytest.approx(reported_total)
    assert metrics.cost.input_cost > 0.0
    assert metrics.cost.output_cost > 0.0
    # Output tokens are cheaper-priced per-token but there are fewer of them;
    # the key invariant is the split sums to the authoritative total.
    assert metrics.cost.input_cost + metrics.cost.output_cost == pytest.approx(
        reported_total
    )


# --------------------------------------------------------------------------- #
# #1423 (c): cost_efficiency divide-by-zero guard                             #
# --------------------------------------------------------------------------- #
def _trial(metrics: dict[str, float]) -> TrialResult:
    return TrialResult(
        trial_id="t",
        config={"model": "gpt-4o"},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def test_cost_efficiency_not_inf_when_cost_zero():
    """cost_efficiency returns a finite 0.0 (not inf) when cost is missing/zero."""
    eff = _calculate_cost_efficiency(_trial({"accuracy": 0.6, "cost": 0.0}), "accuracy")
    assert eff == 0.0


def test_cost_efficiency_falls_back_to_total_cost_and_ranks():
    """With total_cost present, efficiency is finite and ranks cheaper config higher."""
    cheap = _calculate_cost_efficiency(
        _trial({"accuracy": 0.5, "total_cost": 0.0007}), "accuracy"
    )
    pricey = _calculate_cost_efficiency(
        _trial({"accuracy": 0.6, "total_cost": 0.0125}), "accuracy"
    )
    assert cheap != float("inf") and pricey != float("inf")
    # 0.5/0.0007 ~= 714 vs 0.6/0.0125 = 48 -> cheaper config is more efficient.
    assert cheap > pricey


# --------------------------------------------------------------------------- #
# #1423 (1): per-config cost metric / objective wired to total_cost           #
# real, mocked-LLM optimize run, model-differentiated, ordered                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_per_config_cost_nonzero_and_objective_prefers_cheaper():
    """A real local optimize run reports non-zero per-config cost ordered by model.

    Both models score identically (accuracy tie) so the minimize-cost objective is
    the only differentiator. The optimizer must (a) report a non-zero, model-
    differentiated per-config cost and (b) prefer the cheaper model on the tie.
    """
    dataset = [{"input": {"question": "q"}, "expected_output": "A"} for _ in range(2)]

    @optimize(
        eval_dataset=dataset,
        objectives=["accuracy", "cost"],
        configuration_space={"model": ["gpt-4o", "gpt-4o-mini"]},
        offline=True,
    )
    def fn(question: str = "", model: str = "gpt-4o", **_cfg):
        return {"text": "A", "raw_response": _LLMResponse("A", model, 700, 300)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    costs = {t.config["model"]: t.metrics["cost"] for t in result.trials}
    assert costs["gpt-4o"] > 0.0
    assert costs["gpt-4o-mini"] > 0.0
    # gpt-4o is materially (>5x) more expensive than gpt-4o-mini for equal tokens.
    assert costs["gpt-4o"] > costs["gpt-4o-mini"] * 5
    # Aggregate total_cost stays correct (sum of per-trial totals), unchanged.
    assert result.total_cost == pytest.approx(
        sum(t.metrics["total_cost"] for t in result.trials)
    )
    # The cost objective is live: on the accuracy tie the cheaper model wins.
    assert result.best_config["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_per_config_cost_wired_from_provider_reported_total():
    """OpenRouter-style provider-reported total still yields a non-zero per-config cost."""
    dataset = [{"input": {"question": "q"}, "expected_output": "A"}]

    @optimize(
        eval_dataset=dataset,
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1, 0.5]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return {
            "text": "A",
            "raw_response": _LLMResponse(
                "A", "openrouter/openai/gpt-4o", 700, 300, provider_cost=0.0144275
            ),
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    for trial in result.trials:
        # Per-config cost is wired to the provider-reported total (a small
        # pre-existing 6-decimal rounding applies to stored trial metrics).
        assert trial.metrics["cost"] == pytest.approx(0.0144275, abs=1e-5)
        assert trial.metrics["cost"] > 0.0
        assert trial.metrics["total_cost"] == pytest.approx(0.0144275, abs=1e-5)


# --------------------------------------------------------------------------- #
# #1407: unpriced fixed-in-code model — runtime warning + strict fail-closed  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_unpriced_fixed_in_code_model_surfaces_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-strict: an unpriced model fixed in the function body warns the user.

    The model is NOT in the configuration space (preflight cannot see it), so the
    only signal is the runtime collection lifted onto the OptimizationResult.
    """
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")  # no interactive prompt
    dataset = [{"input": {"question": "q"}, "expected_output": "A"}]

    @optimize(
        eval_dataset=dataset,
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        # Model hard-coded in body, unpriced, discoverable on the response object.
        return _LLMResponse("A", "acme/private-llm-v9", 700, 300)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await fn.optimize(progress_bar=False)

    assert result.warnings, "expected a user-visible unpriced-model warning"
    assert any("acme/private-llm-v9" in w for w in result.warnings)
    assert "UNPRICED_MODEL_RUNTIME" in result.warning_codes
    assert "acme/private-llm-v9" in result.metadata.get("unpriced_models_runtime", [])
    assert any(
        "acme/private-llm-v9" in str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
    )


@pytest.mark.asyncio
async def test_unpriced_fixed_in_code_model_fails_closed_under_strict(
    monkeypatch: pytest.MonkeyPatch,
):
    """Strict accounting: an unpriced fixed-in-code model fails the run, never $0."""
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    dataset = [{"input": {"question": "q"}, "expected_output": "A"}]

    @optimize(
        eval_dataset=dataset,
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        return _LLMResponse("A", "acme/private-llm-v9", 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc_info:  # noqa: B017 - run must fail closed
            await fn.optimize(progress_bar=False)

    # The run failed closed instead of silently recording $0. The failure must be
    # cost-related: either the UnknownModelError itself (in the __cause__ chain) or
    # an OptimizationError whose message reports the cost-extraction failure.
    chain: list[BaseException] = []
    err: BaseException | None = exc_info.value
    while err is not None:
        chain.append(err)
        err = err.__cause__
    messages = " | ".join(str(e).lower() for e in chain)
    assert any(isinstance(e, UnknownModelError) for e in chain) or (
        "private-llm-v9" in messages
        or "no known pricing" in messages
        or "cost extraction failed" in messages
    )


def test_unpriced_runtime_model_recorded_at_evaluator_level():
    """The LocalEvaluator records an unpriced runtime model id (non-strict)."""
    import asyncio

    evaluator = LocalEvaluator(metrics=["accuracy", "cost"])

    async def fn(question: str = "", **_cfg):
        return _LLMResponse("A", "acme/private-llm-v9", 700, 300)

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"question": "q"}, expected_output="A")]
    )
    asyncio.run(evaluator.evaluate(fn, {"temperature": 0.1}, dataset))
    assert cost_calculator.get_unpriced_runtime_models() == ["acme/private-llm-v9"]


# --------------------------------------------------------------------------- #
# #1597: OpenRouter/future-dated unpriced response-model ids — never silently  #
# indistinguishable from verified-free; per-example marker + quantified       #
# runtime warning that aggregation cannot mistake for $0-because-free.        #
# --------------------------------------------------------------------------- #
def test_example_metrics_cost_marked_unpriced_for_unmapped_model():
    """A genuinely-unmapped response-model id (e.g. a future OpenRouter release
    not yet in litellm's bundled pricing map) marks ``cost.unpriced`` True
    instead of looking identical to a verified-free ($0) call.
    """
    metrics = extract_llm_metrics(
        _LLMResponse("A", "anthropic/claude-4.8-opus-20260528", 700, 300),
        model_name="anthropic/claude-4.8-opus-20260528",
    )
    assert metrics.cost.total_cost == 0.0
    assert metrics.cost.unpriced is True


def test_example_metrics_cost_not_marked_unpriced_for_priced_model():
    """A model litellm can price (via the OpenRouter-prefix candidate ladder,
    #1423) is NOT marked unpriced — the priced case is unchanged by #1597.
    """
    metrics = extract_llm_metrics(
        _LLMResponse("A", "openrouter/openai/gpt-4o-mini", 700, 300),
        model_name="openrouter/openai/gpt-4o-mini",
    )
    assert metrics.cost.total_cost > 0.0
    assert metrics.cost.unpriced is False


def test_unpriced_runtime_occurrences_counts_calls_per_model():
    """The runtime registry counts occurrences per model (#1597), not just
    set membership, so a systematic per-trial pattern is distinguishable from
    a single unlucky call.
    """
    cost_calculator.record_unpriced_runtime_model("acme/private-llm-v9")
    cost_calculator.record_unpriced_runtime_model("acme/private-llm-v9")
    cost_calculator.record_unpriced_runtime_model("acme/other-v2")

    assert cost_calculator.get_unpriced_runtime_models() == [
        "acme/other-v2",
        "acme/private-llm-v9",
    ]
    assert cost_calculator.get_unpriced_runtime_occurrences() == {
        "acme/other-v2": 1,
        "acme/private-llm-v9": 2,
    }


@pytest.mark.asyncio
async def test_unpriced_runtime_warning_is_quantified_and_not_treated_as_free(
    monkeypatch: pytest.MonkeyPatch,
):
    """The result-level warning states call counts and that $0 means UNKNOWN
    spend — never silently equating it with verified-free usage (#1597).

    This is the "aggregation consuming per-trial cost" requirement made
    concrete: the OptimizationResult surface a user inspects after a run must
    be quantitative (how many calls were affected), not just a bare model-id
    mention, and must say the reported cost is a lower bound.
    """
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")  # no interactive prompt
    dataset = [
        {"input": {"question": "q1"}, "expected_output": "A"},
        {"input": {"question": "q2"}, "expected_output": "A"},
    ]

    @optimize(
        eval_dataset=dataset,
        objectives=["accuracy", "cost"],
        configuration_space={"temperature": [0.1]},
        offline=True,
    )
    def fn(question: str = "", temperature: float = 0.1, **_cfg):
        # Hard-coded, genuinely-unmapped response model id (future OpenRouter
        # release) — discoverable on the response object, invisible to the
        # config-space preflight.
        return _LLMResponse("A", "anthropic/claude-4.8-opus-20260528", 700, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await fn.optimize(progress_bar=False)

    counts = result.metadata.get("unpriced_models_runtime_call_counts", {})
    assert counts.get("anthropic/claude-4.8-opus-20260528") == 2

    warning_text = " ".join(result.warnings)
    assert "anthropic/claude-4.8-opus-20260528" in warning_text
    assert "2 call" in warning_text  # quantified, not just a bare model-id mention
    assert "UNKNOWN" in warning_text
    assert "lower bound" in warning_text
