"""End-to-end SDK tests for feature shipping and backend metadata shape."""

from datetime import datetime

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.metrics.content_features import SimhashFeatureExtractor


class TestExampleFeatureWorkflow:
    """Verify the SDK ships features, not example scores."""

    def test_feature_extractor_produces_one_payload_row_per_example(self):
        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data={"text": "Python programming"},
                    expected_output="code",
                ),
                EvaluationExample(
                    input_data={"text": "Machine learning"},
                    expected_output="ml",
                ),
                EvaluationExample(
                    input_data={"text": "Customer support email"},
                    expected_output="support",
                ),
            ],
            name="e2e_feature_dataset",
        )

        rows = SimhashFeatureExtractor().extract_dataset_features(dataset)

        assert len(rows) == 3
        assert len({row["example_id"] for row in rows}) == 3
        assert all(len(row["feature"]) == 16 for row in rows)

    def test_backend_metadata_keeps_stable_ids_but_omits_content_scores(self):
        trial_result = TrialResult(
            trial_id="trial_e2e",
            config={"model": "gpt-4"},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=2.0,
            timestamp=datetime.now(),
        )

        example_results = []
        for _ in range(2):
            example = type("ExampleResult", (), {})()
            example.metrics = {"score": 0.9, "latency": 0.5}
            example.execution_time = 0.5
            example_results.append(example)

        trial_result.metadata = {"example_results": example_results}

        metadata = build_backend_metadata(
            trial_result,
            "accuracy",
            TraigentConfig(execution_mode="edge_analytics"),
            dataset_name="e2e_feature_dataset",
            content_scores={"uniqueness": {0: 0.1}, "novelty": {0: 0.2}},
        )

        assert "measures" in metadata
        assert len(metadata["measures"]) == 2
        for measure in metadata["measures"]:
            assert measure["example_id"].startswith("ex_")
            metrics = measure["metrics"]
            assert "score" in metrics
            assert "content_uniqueness" not in metrics
            assert "content_novelty" not in metrics
