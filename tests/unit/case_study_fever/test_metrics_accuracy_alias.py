from paper_experiments.case_study_fever.dataset import load_case_study_dataset
from paper_experiments.case_study_fever.metrics import (
    _mock_example_metrics,
    _real_example_metrics,
)


def _sample_config() -> dict:
    return {
        "model": "gpt-4o-mini",
        "evidence_selector": "dense",
        "consistency_checker": "consistency",
        "retriever_k": 3,
        "temperature": 0.3,
        "verdict_threshold": 0.6,
    }


def test_mock_metrics_alias_accuracy():
    dataset = load_case_study_dataset()
    claim = str(dataset.examples[0].input_data.get("claim", ""))

    metrics = _mock_example_metrics(
        claim=claim,
        config=_sample_config(),
        llm_metrics=None,
    )

    assert "accuracy" in metrics
    assert metrics["accuracy"] == metrics["label_accuracy"]


def test_real_metrics_alias_accuracy():
    dataset = load_case_study_dataset()
    example = dataset.examples[0]
    expected = example.expected_output

    output = {
        "verdict": str(expected),
        "justification": "Because the evidence supports the claim.",
        "evidence": [{"page": example.metadata.get("page", "Example_Page"), "line": 1}],
    }

    metrics = _real_example_metrics(
        output=output,
        expected=expected,
        example=example,
        config=_sample_config(),
        llm_metrics=None,
    )

    assert "accuracy" in metrics
    assert metrics["accuracy"] == metrics["label_accuracy"]
