"""Unit tests for deterministic content feature extraction."""

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.metrics.content_features import SimhashFeatureExtractor


class TestSimhashFeatureExtractor:
    """Tests for the SDK-side feature shipping contract."""

    def test_extract_dataset_features_uses_stable_example_ids(self):
        extractor = SimhashFeatureExtractor()
        dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"text": "Hello world"}),
                EvaluationExample(input_data={"text": "Goodbye world"}),
            ],
            name="feature_test_dataset",
        )

        rows = extractor.extract_dataset_features(dataset)

        assert len(rows) == 2
        assert rows[0]["example_id"].startswith("ex_")
        assert rows[1]["example_id"].startswith("ex_")
        assert rows[0]["example_id"] != rows[1]["example_id"]
        assert len(rows[0]["feature"]) == 16

    def test_compute_feature_is_deterministic(self):
        extractor = SimhashFeatureExtractor()

        left = extractor.compute_feature({"prompt": "Explain AI briefly"})
        right = extractor.compute_feature({"prompt": "Explain AI briefly"})

        assert left == right

    def test_similar_inputs_produce_closer_hashes_than_distinct_inputs(self):
        extractor = SimhashFeatureExtractor()

        base = int(extractor.compute_feature({"text": "Python programming basics"}), 16)
        similar = int(
            extractor.compute_feature({"text": "Python programming fundamentals"}),
            16,
        )
        distinct = int(
            extractor.compute_feature({"text": "Tropical fish aquarium care"}),
            16,
        )

        def hamming(left: int, right: int) -> int:
            return (left ^ right).bit_count()

        assert hamming(base, similar) < hamming(base, distinct)
