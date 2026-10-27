import logging

import pytest

import traigent
from traigent.core.objectives import create_default_objectives
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def _disable_backend_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setattr(
        "traigent.core.backend_session_manager.BackendSessionManager.create_backend_client",
        staticmethod(lambda _config: None),
    )


@pytest.mark.asyncio
async def test_local_evaluator_accuracy_exact_match() -> None:
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    examples = [
        EvaluationExample({"query": "1+1"}, "2"),
        EvaluationExample({"query": "2+2"}, "4"),
    ]
    dataset = Dataset(examples, name="math", description="basic math")

    def func(input_data):
        expression = input_data["query"]
        return str(eval(expression))

    result = await evaluator.evaluate(func, {}, dataset)

    assert result.metrics is not None
    assert result.metrics.get("accuracy") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_local_evaluator_accuracy_mismatch() -> None:
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    examples = [
        EvaluationExample({"query": "1+1"}, "2"),
        EvaluationExample({"query": "2+2"}, "4"),
    ]
    dataset = Dataset(examples, name="math", description="basic math")

    def func(input_data):
        if input_data["query"] == "1+1":
            return "2"
        return "5"  # wrong on purpose

    result = await evaluator.evaluate(func, {}, dataset)

    assert result.metrics is not None
    assert result.metrics.get("accuracy") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_local_evaluator_paraphrases_fail_under_default_exact_match() -> None:
    """Issue #891: the default ``LocalEvaluator`` accuracy is exact /
    case-insensitive match, not semantic similarity. Two paraphrased
    answers must therefore score 0.0 under the default contract."""
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    examples = [
        EvaluationExample(
            {"question": "What is the capital of France?"},
            "Paris",
        ),
        EvaluationExample(
            {"question": "Who wrote Hamlet?"},
            "Shakespeare",
        ),
    ]
    dataset = Dataset(examples, name="qa", description="paraphrased qa")

    def paraphrasing_agent(input_data):  # pragma: no branch - test fixture
        question = input_data["question"].lower()
        if "capital" in question:
            # Semantically correct paraphrase of "Paris".
            return "Paris is the capital of France"
        # Semantically correct paraphrase of "Shakespeare".
        return "William Shakespeare wrote it"

    result = await evaluator.evaluate(paraphrasing_agent, {}, dataset)

    assert result.metrics is not None
    # Both paraphrases differ character-for-character from the expected
    # output. Under the documented contract (exact / case-insensitive
    # match) accuracy must be 0.0.
    assert result.metrics.get("accuracy") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_local_evaluator_paraphrases_pass_with_user_supplied_semantic_scorer() -> (
    None
):
    """Issue #891: when users want semantic scoring they must supply
    their own scorer via ``metric_functions`` (the documented public
    surface). With a token-overlap scorer standing in for an embedding
    model, the same paraphrases that fail under exact-match pass."""

    def semantic_overlap_scorer(output: str, expected: str) -> float:
        """User-supplied semantic-style scorer.

        Returns 1.0 when every whitespace token of the expected output
        appears (case-insensitively) somewhere in the actual output.
        Stands in for an embedding-similarity score in a unit test that
        must not depend on external API keys.
        """
        if not output or not expected:
            return 0.0
        out_tokens = {tok.lower() for tok in str(output).split()}
        exp_tokens = {tok.lower() for tok in str(expected).split()}
        if not exp_tokens:
            return 0.0
        overlap = exp_tokens & out_tokens
        return 1.0 if overlap == exp_tokens else 0.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=False,
        metric_functions={"accuracy": semantic_overlap_scorer},
    )

    examples = [
        EvaluationExample(
            {"question": "What is the capital of France?"},
            "Paris",
        ),
        EvaluationExample(
            {"question": "Who wrote Hamlet?"},
            "Shakespeare",
        ),
    ]
    dataset = Dataset(examples, name="qa", description="paraphrased qa")

    def paraphrasing_agent(input_data):
        question = input_data["question"].lower()
        if "capital" in question:
            return "Paris is the capital of France"
        return "William Shakespeare wrote it"

    result = await evaluator.evaluate(paraphrasing_agent, {}, dataset)

    assert result.metrics is not None
    # With a user-supplied semantic-style scorer, paraphrases pass.
    assert result.metrics.get("accuracy") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_optimize_scoring_function_called_with_prediction_expected_plain_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)
    calls: list[tuple[str, str]] = []

    def scorer(prediction: str, expected: str) -> float:
        calls.append((prediction, expected))
        return 0.75 if prediction == expected else 0.0

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g4"),
        objectives=["accuracy"],
        configuration_space={"style": ["a", "b"]},
        scoring_function=scorer,
    )
    def agent(text: str) -> str:
        return "YES"

    result = await agent.optimize(algorithm="grid", max_trials=2)

    assert calls == [("YES", "YES"), ("YES", "YES")]
    assert result.best_score == pytest.approx(0.75)
    assert len(result.trials) == 2
    assert all(
        trial.metrics["accuracy"] == pytest.approx(0.75) for trial in result.trials
    )


@pytest.mark.asyncio
async def test_optimize_scoring_function_uses_unpacked_tuple_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)
    calls: list[tuple[str, str]] = []

    def scorer(prediction: str, expected: str) -> float:
        calls.append((prediction, expected))
        return 0.4 if prediction == expected else 0.0

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g4"),
        objectives=["accuracy"],
        configuration_space={"style": ["a"]},
        scoring_function=scorer,
    )
    def agent(text: str) -> tuple[str, dict[str, float]]:
        return "YES", {"user_metric": 1.0}

    result = await agent.optimize(algorithm="grid", max_trials=1)

    assert calls == [("YES", "YES")]
    assert result.best_score == pytest.approx(0.4)
    assert result.trials[0].metrics["accuracy"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_optimize_scoring_function_binds_to_custom_primary_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)
    calls: list[tuple[str, str]] = []

    def scorer(prediction: str, expected: str) -> float:
        calls.append((prediction, expected))
        return 0.9 if prediction == expected else 0.0

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g4"),
        objectives=create_default_objectives(
            ["quality"], orientations={"quality": "maximize"}
        ),
        configuration_space={"style": ["a", "b"]},
        scoring_function=scorer,
    )
    def agent(text: str) -> str:
        return "YES"

    result = await agent.optimize(algorithm="grid", max_trials=2)

    assert calls == [("YES", "YES"), ("YES", "YES")]
    assert result.best_score == pytest.approx(0.9)
    assert len(result.trials) == 2
    assert all(
        trial.metrics["quality"] == pytest.approx(0.9) for trial in result.trials
    )


@pytest.mark.asyncio
async def test_optimize_dict_returning_scoring_function_merges_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)

    def scorer(prediction: str, expected: str) -> dict[str, float]:
        return {
            "accuracy": 1.0 if prediction == expected else 0.0,
            "f1": 0.5,
        }

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g4"),
        objectives=["accuracy"],
        configuration_space={"style": ["a"]},
        scoring_function=scorer,
    )
    def agent(text: str) -> str:
        return "YES"

    result = await agent.optimize(algorithm="grid", max_trials=1)

    assert result.best_score == pytest.approx(1.0)
    assert result.trials[0].metrics["accuracy"] == pytest.approx(1.0)
    assert result.trials[0].metrics["f1"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_optimize_three_argument_metric_function_receives_metrics_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_backend_tracking(monkeypatch)
    calls: list[tuple[str, str, float]] = []

    def scorer(prediction: str, expected: str, metrics: dict[str, float]) -> float:
        calls.append((prediction, expected, metrics["total_tokens"]))
        return 0.6 if prediction == expected and "total_tokens" in metrics else 0.0

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "YES")], name="g4"),
        objectives=["accuracy"],
        configuration_space={"style": ["a"]},
        metric_functions={"accuracy": scorer},
    )
    def agent(text: str) -> str:
        return "YES"

    result = await agent.optimize(algorithm="grid", max_trials=1)

    assert len(calls) == 1
    prediction, expected, total_tokens = calls[0]
    assert (prediction, expected) == ("YES", "YES")
    assert total_tokens == pytest.approx(2.0)
    assert result.best_score == pytest.approx(0.6)
    assert result.trials[0].metrics["accuracy"] == pytest.approx(0.6)


def _label_dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"q": "a"}, "label-a"),
            EvaluationExample({"q": "b"}, "label-b"),
        ],
        name="labels",
        description="rows carry a gold label the function never echoes",
    )


@pytest.mark.asyncio
async def test_nonobjective_accuracy_not_computed_or_uploaded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exact-match accuracy is computed and uploaded only when it is a
    configured metric; otherwise it must not be manufactured as 0.0."""
    evaluator = LocalEvaluator(
        metrics=["pass_rate"],
        metric_functions={"pass_rate": lambda output, expected, **_: 1.0},
        detailed=True,
        execution_mode="local",
    )

    def func(input_data):
        return "encoded-metrics-string"

    caplog.set_level(logging.INFO, logger="traigent")
    first = await evaluator.evaluate(func, {}, _label_dataset())
    await evaluator.evaluate(func, {}, _label_dataset())

    assert first.metrics is not None
    assert first.metrics.get("accuracy") is None
    assert first.metrics.get("score") is None
    assert first.metrics["pass_rate"] == pytest.approx(1.0)
    for example in first.example_results:
        assert "accuracy" not in (example.metrics or {})
    stats = (first.summary_stats or {}).get("metrics", {})
    assert stats.get("accuracy", {}).get("count", 0) == 0
    notices = [
        r for r in caplog.records if "Built-in exact-match accuracy" in r.getMessage()
    ]
    assert len(notices) == 1


@pytest.mark.asyncio
async def test_declared_accuracy_still_computed_and_zero_is_kept() -> None:
    evaluator = LocalEvaluator(
        metrics=["accuracy"], detailed=True, execution_mode="local"
    )
    result = await evaluator.evaluate(
        lambda input_data: "never-matches", {}, _label_dataset()
    )
    assert result.metrics["accuracy"] == 0.0
    assert result.summary_stats["metrics"]["accuracy"]["count"] == 2


@pytest.mark.asyncio
async def test_supplied_accuracy_survives_when_not_a_configured_metric() -> None:
    evaluator = LocalEvaluator(
        metrics=["pass_rate"],
        metric_functions={
            "pass_rate": lambda output, expected, **_: 1.0,
            "accuracy": lambda output, expected, **_: 0.5,
        },
        detailed=True,
        execution_mode="local",
    )
    result = await evaluator.evaluate(lambda input_data: "x", {}, _label_dataset())
    assert result.metrics["accuracy"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_failed_attempt_without_output_is_a_miss_in_aggregate_and_tracker() -> (
    None
):
    """Declared built-in accuracy: an evaluator failure with no output counts
    as 0 in the aggregate and in both tracker formats."""

    def func(input_data):
        if input_data["q"] == "b":
            raise RuntimeError("boom")
        return "label-a"

    evaluator = LocalEvaluator(
        metrics=["accuracy"], detailed=True, execution_mode="local"
    )
    result = await evaluator.evaluate(func, {}, _label_dataset())

    assert result.metrics["accuracy"] == pytest.approx(0.5)
    stats = result.summary_stats["metrics"]["accuracy"]
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(0.5)


def _three_label_dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"q": "a"}, "label-a"),
            EvaluationExample({"q": "b"}, "label-b"),
            EvaluationExample({"q": "c"}, "label-c"),
        ],
        name="labels3",
        description="three gold labels",
    )


def _resume_prefill_for_first_row(stored_metrics: dict):
    from traigent.core.resume import ResumePrefill, config_key

    dataset = _three_label_dataset()
    row = {
        "input_data": dataset.examples[0].input_data,
        "expected_output": dataset.examples[0].expected_output,
        "actual_output": "label-a",
        "metrics": stored_metrics,
        "execution_time": 0.01,
    }
    return ResumePrefill({config_key({}): {"example_0": row}})


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_resumed_row_with_gold_but_no_stored_accuracy_is_unmeasured(
    configured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed successful row with gold and no stored accuracy is never 1.0:
    it stays out of the accuracy denominator in the backend payload and the
    summary stats, whether or not accuracy is a configured metric."""
    from traigent.evaluators.base import bind_resume_prefill, reset_resume_prefill

    def func(input_data):
        return {"b": "label-b", "c": "wrong"}[input_data["q"]]

    evaluator = LocalEvaluator(
        metrics=["accuracy"] if configured else ["pass_rate"],
        metric_functions=(
            None if configured else {"pass_rate": lambda output, expected, **_: 1.0}
        ),
        detailed=True,
        execution_mode="local",
    )
    from traigent.evaluators.metrics_tracker import MetricsTracker

    payloads: list[dict] = []
    original = MetricsTracker.format_for_backend

    def spy(self, *args, **kwargs):
        payload = original(self, *args, **kwargs)
        payloads.append(payload)
        return payload

    monkeypatch.setattr(MetricsTracker, "format_for_backend", spy)
    token = bind_resume_prefill(
        _resume_prefill_for_first_row({"total_cost": 0.0, "input_tokens": 1})
    )
    try:
        result = await evaluator.evaluate(func, {}, _three_label_dataset())
    finally:
        reset_resume_prefill(token)

    resumed = [
        r
        for r in result.example_results
        if (r.metadata or {}).get("resumed_example_from_prior_run")
    ]
    assert len(resumed) == 1  # the partially resumed evaluation really resumed
    stats = result.summary_stats["metrics"].get("accuracy", {})
    (payload,) = payloads  # the tracker's submission payload
    if configured:
        # Two executed rows (one hit, one miss); the resumed row is excluded
        # from the tracker's accuracy, never counted as a derived 1.0.
        assert payload["accuracy"] == pytest.approx(0.5)
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(0.5)
    else:
        assert payload["accuracy"] is None
        assert result.metrics.get("accuracy") is None
        assert stats.get("count", 0) == 0
