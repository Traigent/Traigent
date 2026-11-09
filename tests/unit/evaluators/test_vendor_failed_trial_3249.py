"""Traigent#3249: a trial whose every example fails on a vendor error is a failed trial.

No call returned a response, so nothing was left unpriced: with a cost objective
the run must record the trial as failed and keep the earlier completed trials,
not abort with "Cost extraction failed ... An unpriced call is not recorded as
$0". A call that DID return a response but cannot be priced must still raise.
Keyless: LiteLLM ``mock_response``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

litellm = pytest.importorskip("litellm")

import traigent  # noqa: E402
from traigent import Choices  # noqa: E402
from traigent.api.types import TrialResult, TrialStatus  # noqa: E402
from traigent.core.cost_enforcement import (  # noqa: E402
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
    Permit,
)
from traigent.core.orchestrator import OptimizationOrchestrator  # noqa: E402

MODEL = "openrouter/openai/gpt-4o-mini"
UNPRICED_MODEL = "openrouter/acme/unpriced-model-3249"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.chdir(tmp_path)


def _write_dataset(rows: int) -> None:
    Path("ds.jsonl").write_text(
        "".join(
            json.dumps({"input": {"q": f"q{i}"}, "output": "x"}) + "\n"
            for i in range(rows)
        )
    )


def test_trial_whose_every_example_hits_a_vendor_error_is_a_failed_trial() -> None:
    _write_dataset(3)
    calls = {"n": 0}

    @traigent.optimize(
        configuration_space={"model": Choices([MODEL]), "temperature": [0.0, 0.5]},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> str:
        calls["n"] += 1
        # The provider call itself raises (no response) from the 4th call on.
        r = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response=(
                "paris"
                if calls["n"] <= 3
                else litellm.RateLimitError(
                    message="HTTP 429: rate limited",
                    llm_provider="openrouter",
                    model=MODEL,
                )
            ),
            num_retries=0,
        )
        return r.choices[0].message.content

    result = ask.optimize_sync(max_trials=2, algorithm="grid")

    statuses = [str(getattr(t.status, "value", t.status)) for t in result.trials]
    assert statuses == ["completed", "failed"]
    assert result.trials[0].metrics["cost"] > 0
    assert not result.trials[0].metrics.get("cost_unmeasured")
    assert ask.get_optimization_results() is not None


def test_returned_but_unpriced_call_still_aborts_under_a_cost_objective() -> None:
    # Guardrail: a response came back and cannot be priced -> fail closed.
    _write_dataset(1)

    @traigent.optimize(
        configuration_space={"model": Choices([UNPRICED_MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> str:
        r = litellm.completion(
            model=UNPRICED_MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response="paris",
        )
        return r.choices[0].message.content

    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        ask.optimize_sync(max_trials=1, algorithm="grid")


def _strict_enforcer() -> CostEnforcer:
    with patch.dict(os.environ, {"TRAIGENT_STRICT_COST_ACCOUNTING": "true"}):
        return CostEnforcer(config=CostEnforcerConfig(fallback_trial_limit=5))


def test_enforcer_strict_records_failed_vendor_error_trial_without_raising() -> None:
    enforcer = _strict_enforcer()
    enforcer.track_cost(
        None,
        permit=Permit(id=0, amount=0.0, active=True),
        trial_failed=True,
        vendor_error_only=True,
    )
    # The unknown spend is still bounded: the run falls back to trial-count limits.
    assert enforcer.get_status().unknown_cost_mode is True


@pytest.mark.parametrize(
    ("trial_failed", "vendor_error_only"),
    [(True, False), (False, True), (False, False)],
)
def test_enforcer_strict_still_raises_unless_failed_and_vendor_error_only(
    trial_failed: bool, vendor_error_only: bool
) -> None:
    enforcer = _strict_enforcer()
    with pytest.raises(CostTrackingRequiredError):
        enforcer.track_cost(
            None,
            permit=Permit(id=0, amount=0.0, active=True),
            trial_failed=trial_failed,
            vendor_error_only=vendor_error_only,
        )


def _trial(status: TrialStatus, **meta: int) -> TrialResult:
    from datetime import UTC, datetime

    return TrialResult(
        trial_id="t",
        config={},
        metrics={},
        status=status,
        duration=0.0,
        timestamp=datetime.now(UTC),
        metadata=dict(meta),
    )


@pytest.mark.parametrize(
    ("status", "attempts", "no_response", "expected"),
    [
        (TrialStatus.FAILED, 3, 3, True),
        # One call returned a response (it may carry usage that cannot be
        # priced) next to later provider errors: stays fail-closed.
        (TrialStatus.FAILED, 4, 3, False),
        # No provider error evidence at all.
        (TrialStatus.FAILED, 0, 0, False),
        # A completed trial with unknown cost is never exempt.
        (TrialStatus.COMPLETED, 3, 3, False),
    ],
)
def test_only_failed_trials_where_every_call_raised_are_exempt(
    status: TrialStatus, attempts: int, no_response: int, expected: bool
) -> None:
    trial = _trial(
        status,
        vendor_call_attempts=attempts,
        vendor_call_no_response_attempts=no_response,
        vendor_call_responses_captured=0,
    )
    assert OptimizationOrchestrator._failed_on_vendor_errors_only(trial) is expected


def test_failed_trial_with_a_returned_unpriced_response_and_later_errors_still_aborts() -> (
    None
):
    # Guardrail: a response came back (unpriced) and later calls raised.
    _write_dataset(1)
    calls = {"n": 0}

    @traigent.optimize(
        configuration_space={"model": Choices([UNPRICED_MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> str:
        calls["n"] += 1
        litellm.completion(
            model=UNPRICED_MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response="paris",
        )
        litellm.completion(
            model=UNPRICED_MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response=litellm.RateLimitError(
                message="HTTP 429: rate limited",
                llm_provider="openrouter",
                model=UNPRICED_MODEL,
            ),
            num_retries=0,
        )
        return "unreachable"

    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        ask.optimize_sync(max_trials=1, algorithm="grid")


def test_captured_response_outside_the_ledger_or_unknown_state_is_not_exempt() -> None:
    base = {"vendor_call_attempts": 1, "vendor_call_no_response_attempts": 1}
    captured = _trial(TrialStatus.FAILED, **base, vendor_call_responses_captured=1)
    unknown = _trial(TrialStatus.FAILED, **base)
    assert OptimizationOrchestrator._failed_on_vendor_errors_only(captured) is False
    assert OptimizationOrchestrator._failed_on_vendor_errors_only(unknown) is False


def test_failed_trial_with_unpriced_capture_outside_ledger_then_429_still_aborts() -> (
    None
):
    # The response is recorded by the LangChain-style capture path only (no
    # ledger attempt), then a LiteLLM call raises 429 through the ledger.
    from traigent.utils.langchain_interceptor import capture_langchain_response

    _write_dataset(1)

    @traigent.optimize(
        configuration_space={"model": Choices([UNPRICED_MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> str:
        capture_langchain_response(
            litellm.ModelResponse(
                model=UNPRICED_MODEL,
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            )
        )
        litellm.completion(
            model=UNPRICED_MODEL,
            messages=[{"role": "user", "content": q}],
            mock_response=litellm.RateLimitError(
                message="HTTP 429: rate limited",
                llm_provider="openrouter",
                model=UNPRICED_MODEL,
            ),
            num_retries=0,
        )
        return "unreachable"

    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        ask.optimize_sync(max_trials=1, algorithm="grid")


@pytest.mark.parametrize("bad_prompt_tokens", [-5, True, 10**400, "12"])
def test_invalid_usage_count_aborts_under_a_cost_objective_3252(
    bad_prompt_tokens: object,
) -> None:
    """Traigent#3252: a returned response with an unusable token count is not $0.

    A response DID come back (so #3249's vendor-error exemption must not apply),
    its usage is unknown, and the cost objective must fail closed exactly like
    any other returned-but-unpriced call.
    """
    from types import SimpleNamespace

    _write_dataset(1)

    @traigent.optimize(
        configuration_space={"model": Choices([MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> Any:
        return SimpleNamespace(
            model=MODEL,
            usage=SimpleNamespace(
                prompt_tokens=bad_prompt_tokens,
                completion_tokens=7,
                total_tokens=18,
            ),
        )

    with pytest.raises(Exception, match="Cost extraction failed|unpriced"):
        ask.optimize_sync(max_trials=1, algorithm="grid")


def test_valid_usage_counts_complete_under_a_cost_objective_3252() -> None:
    """Control for the invalid-count test: valid counts price and complete."""
    from types import SimpleNamespace

    _write_dataset(1)

    @traigent.optimize(
        configuration_space={"model": Choices([MODEL])},
        eval_dataset="ds.jsonl",
        objectives=["accuracy", "cost"],
        offline=True,
    )
    def ask(q: str) -> Any:
        return SimpleNamespace(
            model=MODEL,
            usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=7, total_tokens=18
            ),
        )

    result = ask.optimize_sync(max_trials=1, algorithm="grid")
    assert result.trials[0].metrics["cost"] > 0
    assert not result.trials[0].metrics.get("cost_unmeasured")
