"""An unmeasured cost is never stored, displayed or ranked as $0 (#2517, #2518).

The reported shape: the optimized function calls a provider SDK client the
interceptors do not wrap, so no usage reaches the evaluator. The local lane then
estimated tokens from the dataset input and stored ``cost: 0.0`` /
``total_cost: 0.0`` as if it were a measurement, and the results table printed
``$0.00000`` for every trial. A run whose model Traigent cannot price stopped at
10 trials with nothing saying why.

Contract under test:
- ``CostMetrics.unmeasured`` is the typed "no usage captured" state; trial-level
  cost keys are left OUT (never ``0.0``) and ``cost_unmeasured`` says why.
- A genuinely measured ``$0`` stays ``0.0``.
- The results table says ``unmeasured``.
- A run cut by a limit records ``metadata["stop_detail"]`` and prints it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

import traigent  # noqa: E402
from traigent.api.decorators import optimize  # noqa: E402
from traigent.evaluators.base import BaseEvaluator  # noqa: E402
from traigent.evaluators.metrics_tracker import (  # noqa: E402
    RESERVED_METRIC_KEYS,
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    TokenMetrics,
    extract_llm_metrics,
)
from traigent.utils import cost_calculator  # noqa: E402

UNPRICED_MODEL = "acme/private-llm-v9"


def _example(*, unmeasured: bool, cost: float = 0.0) -> ExampleMetrics:
    return ExampleMetrics(
        tokens=TokenMetrics(
            input_tokens=7, output_tokens=2, total_tokens=9, estimated=unmeasured
        ),
        cost=CostMetrics(
            input_cost=cost / 2, output_cost=cost / 2, unmeasured=unmeasured
        ),
    )


# --------------------------------------------------------------------------- #
# Tracker level: the typed state propagates into the trial metrics            #
# --------------------------------------------------------------------------- #
def test_unmeasured_examples_leave_the_cost_keys_out_of_the_trial() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for _ in range(3):
        tracker.add_example_metrics(_example(unmeasured=True))
    formatted = tracker.format_for_backend()

    assert formatted["cost"] is None
    assert formatted["cost_per_example_mean"] is None
    assert formatted["cost_unmeasured"] == 1.0
    # Tokens stay, labelled as estimates.
    assert formatted["tokens_estimated"] == 1.0


def test_a_measured_zero_cost_stays_zero() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(_example(unmeasured=False, cost=0.0))
    formatted = tracker.format_for_backend()

    assert formatted["cost"] == 0.0
    assert formatted["cost_unmeasured"] == 0.0


def test_mixed_trial_sums_only_the_measured_cost_and_is_flagged() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(_example(unmeasured=False, cost=0.002))
    tracker.add_example_metrics(_example(unmeasured=True))
    formatted = tracker.format_for_backend()

    assert formatted["cost"] == pytest.approx(0.002)
    assert formatted["cost_unmeasured"] == 1.0


def test_unmeasured_flag_is_a_reserved_metric_key() -> None:
    assert "cost_unmeasured" in RESERVED_METRIC_KEYS


def test_registry_cost_metric_is_none_when_nothing_was_measured() -> None:
    class _Evaluator(BaseEvaluator):  # minimal concrete evaluator
        async def evaluate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    evaluator = _Evaluator(metrics=["accuracy", "cost"])
    cost = evaluator._compute_cost(
        ["a"], ["a"], [None], example_metrics=[_example(unmeasured=True)]
    )
    assert cost is None


def test_string_output_without_usage_is_unmeasured_but_real_usage_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)

    no_usage = extract_llm_metrics(response="plain text", model_name="gpt-4o-mini")
    assert no_usage.cost.unmeasured is True

    class _Usage:
        prompt_tokens = 30
        completion_tokens = 2
        total_tokens = 32

    class _Response:
        usage = _Usage()
        model = "gpt-4o-mini"

    with_usage = extract_llm_metrics(response=_Response(), model_name="gpt-4o-mini")
    assert with_usage.cost.unmeasured is False
    assert with_usage.cost.total_cost > 0


def test_mock_llm_mode_makes_no_unmeasured_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    metrics = extract_llm_metrics(response="plain text", model_name="gpt-4o-mini")
    assert metrics.cost.unmeasured is False


# --------------------------------------------------------------------------- #
# End to end: a raw OpenAI client the interceptors do not wrap (#2517/#2518)  #
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_run_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("TRAIGENT_ENV", "test")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    # Accept the unmeasured run instead of failing it, to inspect what is stored.
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    cost_calculator.reset_unpriced_runtime_models()
    dataset = tmp_path / "qa.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"input": {"question": f"What is {i} + 3?"}, "output": "ok"})
            for i in range(3)
        )
    )
    return dataset


def _raw_client() -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 29,
                    "completion_tokens": 2,
                    "total_tokens": 31,
                },
            },
        )

    return openai.OpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        max_retries=0,
    )


def _optimized(dataset: Path, space: dict[str, list[Any]]) -> Any:
    client = _raw_client()

    @optimize(
        configuration_space=space,
        objectives=["accuracy", "cost"],
        eval_dataset=str(dataset),
    )
    def answer(question: str) -> str:
        cfg = traigent.get_config()
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=cfg["temperature"],
            messages=[
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    return answer


def test_uncaptured_raw_openai_calls_are_stored_as_unmeasured_not_zero(
    real_run_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answer = _optimized(
        real_run_env,
        {"model": ["gpt-4o-mini", "gpt-4.1-nano"], "temperature": [0.0, 0.7]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = answer.optimize_sync(algorithm="grid", max_trials=4)

    assert len(result.trials) == 4
    for trial in result.trials:
        assert "cost" not in trial.metrics, trial.metrics
        assert "total_cost" not in trial.metrics, trial.metrics
        assert trial.metrics["cost_unmeasured"] == 1.0
        assert trial.metrics["tokens_estimated"] == 1.0

    out = capsys.readouterr().out
    assert "unmeasured" in out
    assert "$0.00000" not in out


def test_unpriced_default_run_says_why_it_stopped_at_ten_trials(
    real_run_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answer = _optimized(
        real_run_env,
        {
            "model": [UNPRICED_MODEL],
            "temperature": [round(0.1 * i, 1) for i in range(12)],
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = answer.optimize_sync(algorithm="grid")  # no max_trials

    assert len(result.trials) == 10
    detail = result.metadata["stop_detail"]
    assert "Stopped at 10 trials" in detail
    assert "unmeasured" in detail
    assert "max_trials" in detail
    assert all("cost" not in t.metrics for t in result.trials)

    out = capsys.readouterr().out
    assert "Stopped at 10 trials" in out
    assert "$0.00000" not in out


def test_default_max_trials_stop_is_named_as_the_sdk_default(
    real_run_env: Path,
) -> None:
    class _Usage:
        prompt_tokens = 30
        completion_tokens = 2
        total_tokens = 32

    class _Message:
        content = "ok"

    class _Choice:
        message = _Message()

    class _Response:
        usage = _Usage()
        choices = [_Choice()]
        model = "gpt-4o-mini"

    @optimize(
        configuration_space={"temperature": [round(0.1 * i, 1) for i in range(12)]},
        objectives=["accuracy", "cost"],
        eval_dataset=str(real_run_env),
    )
    def answer(question: str) -> Any:
        traigent.get_config()
        return _Response()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_run = answer.optimize_sync(algorithm="grid")
        explicit_run = answer.optimize_sync(algorithm="grid", max_trials=3)

    assert default_run.stop_reason == "max_trials_reached"
    assert "SDK default max_trials=10" in default_run.metadata["stop_detail"]
    assert explicit_run.stop_reason == "max_trials_reached"
    assert "max_trials=3 reached" in explicit_run.metadata["stop_detail"]
    assert "default" not in explicit_run.metadata["stop_detail"]
