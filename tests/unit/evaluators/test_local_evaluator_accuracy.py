import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


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
