import pytest

import traigent
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
        objectives=["quality"],
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
