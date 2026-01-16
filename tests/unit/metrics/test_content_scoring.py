"""Unit tests for content-based example scoring."""

import pytest

try:
    from traigent.metrics.content_scoring import ContentScorer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ContentScorer = None  # type: ignore


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestContentScorer:
    """Test ContentScorer class."""

    def test_initialization(self):
        """ContentScorer should initialize successfully."""
        scorer = ContentScorer()
        assert scorer is not None
        assert scorer.tfidf_vectorizer is not None

    def test_compute_uniqueness_scores_identical_examples(self):
        """Identical examples should have low uniqueness scores."""
        scorer = ContentScorer()
        example_inputs = ["Hello world", "Hello world", "Hello world"]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        # All scores should be close to 0 (not unique)
        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 0.1, f"Expected low uniqueness, got {score}"

    def test_compute_uniqueness_scores_unique_examples(self):
        """Completely different examples should have high uniqueness scores."""
        scorer = ContentScorer()
        example_inputs = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "Machine learning requires data and algorithms",
        ]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        # All scores should be relatively high (unique)
        assert len(scores) == 3
        for idx, score in scores.items():
            assert score > 0.3, f"Expected high uniqueness, got {score}"

    def test_compute_uniqueness_scores_mixed_similarity(self):
        """Examples with varying similarity should have varied scores."""
        scorer = ContentScorer()
        example_inputs = [
            "I love cats and dogs",
            "I love cats and puppies",  # Similar to first
            "Python programming is fun",  # Very different
        ]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        assert len(scores) == 3
        # All scores should be in valid range
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0

        # Third example should be more unique than first two
        assert scores[2] > scores[0] or scores[2] > scores[1]

    def test_compute_novelty_scores_basic(self):
        """Novelty scores should be computed for all examples."""
        scorer = ContentScorer()
        example_inputs = [
            "Hello world",
            "Goodbye world",
            "Hello there",
        ]
        scores = scorer.compute_novelty_scores(example_inputs)

        # Should return scores for all examples
        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_compute_novelty_scores_typical_vs_novel(self):
        """Examples far from average should have higher novelty."""
        scorer = ContentScorer()
        example_inputs = [
            "Python is a programming language used for web development",
            "Java is a programming language used for enterprise applications",
            "JavaScript is a programming language used for web browsers",
            "The colorful butterfly landed gently on the blooming flower",  # Very different topic
        ]
        scores = scorer.compute_novelty_scores(example_inputs)

        assert len(scores) == 4
        # Last example should be more novel than at least one of the first three
        # (it's on a completely different topic)
        assert scores[3] >= min(scores[0], scores[1], scores[2])

    def test_compute_all_scores(self):
        """compute_all_scores should return both uniqueness and novelty."""
        scorer = ContentScorer()
        example_inputs = ["Hello world", "Goodbye world", "Hello there"]
        all_scores = scorer.compute_all_scores(example_inputs)

        # Should have both score types
        assert "uniqueness" in all_scores
        assert "novelty" in all_scores

        # Each should have scores for all examples
        assert len(all_scores["uniqueness"]) == 3
        assert len(all_scores["novelty"]) == 3

        # All scores should be in valid range
        for score_type in ["uniqueness", "novelty"]:
            for idx, score in all_scores[score_type].items():
                assert 0.0 <= score <= 1.0

    def test_single_example_fallback(self):
        """Single example should return neutral scores."""
        scorer = ContentScorer()
        example_inputs = ["Single example"]
        uniqueness = scorer.compute_uniqueness_scores(example_inputs)
        novelty = scorer.compute_novelty_scores(example_inputs)

        # Should return neutral score (0.5)
        assert uniqueness[0] == 0.5
        assert novelty[0] == 0.5

    def test_empty_strings_fallback(self):
        """Empty strings should return neutral scores without crashing."""
        scorer = ContentScorer()
        example_inputs = ["", "", ""]
        uniqueness = scorer.compute_uniqueness_scores(example_inputs)
        novelty = scorer.compute_novelty_scores(example_inputs)

        # Should return neutral scores (fallback)
        assert len(uniqueness) == 3
        assert len(novelty) == 3
        for idx in range(3):
            assert uniqueness[idx] == 0.5
            assert novelty[idx] == 0.5

    def test_score_consistency(self):
        """Same inputs should produce same scores (deterministic)."""
        scorer = ContentScorer()
        example_inputs = ["Hello", "World", "Test"]

        scores1 = scorer.compute_uniqueness_scores(example_inputs)
        scores2 = scorer.compute_uniqueness_scores(example_inputs)

        # Scores should be identical
        for idx in range(3):
            assert scores1[idx] == scores2[idx]

    def test_per_thread_instantiation_pattern(self):
        """Test recommended pattern: new instance per thread."""
        # This is the recommended pattern for thread safety
        def process_examples(inputs):
            scorer = ContentScorer()  # New instance per call
            return scorer.compute_uniqueness_scores(inputs)

        inputs = ["Example 1", "Example 2", "Example 3"]
        scores = process_examples(inputs)

        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestContentScorerEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_texts(self):
        """Should handle very long input texts."""
        scorer = ContentScorer()
        example_inputs = [
            "word " * 1000,  # 1000 words
            "different " * 1000,
            "text " * 1000,
        ]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_special_characters(self):
        """Should handle special characters gracefully."""
        scorer = ContentScorer()
        example_inputs = [
            "Hello @#$% world!",
            "Special chars: <>?{}[]|\\",
            "Unicode: 你好世界 🌍",
        ]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0

    def test_numeric_strings(self):
        """Should handle numeric strings."""
        scorer = ContentScorer()
        example_inputs = ["123 456", "789 012", "345 678"]
        scores = scorer.compute_uniqueness_scores(example_inputs)

        assert len(scores) == 3
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0


@pytest.mark.skipif(SKLEARN_AVAILABLE, reason="Test for sklearn not available")
class TestContentScorerWithoutSklearn:
    """Test ContentScorer behavior when scikit-learn is not installed."""

    def test_initialization_fails_without_sklearn(self):
        """Should raise ImportError if sklearn not available."""
        with pytest.raises(ImportError, match="scikit-learn is required"):
            ContentScorer()
