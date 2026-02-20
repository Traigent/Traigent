"""Tests for smart dataset subset selection algorithms."""

from unittest.mock import patch

import pytest

from traigent.cloud.subset_selection import (
    DiverseSampling,
    HighConfidenceSampling,
    RepresentativeSampling,
    SmartSubsetSelector,
    SubsetSelectionResult,
)
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"text": "Urgent: Server is down immediately!"},
            expected_output="urgent",
        ),
        EvaluationExample(
            input_data={"text": "Meeting scheduled for tomorrow"},
            expected_output="scheduling",
        ),
        EvaluationExample(
            input_data={"text": "Invoice #12345 is overdue"}, expected_output="finance"
        ),
        EvaluationExample(
            input_data={"text": "Please review the attached document"},
            expected_output="general",
        ),
        EvaluationExample(
            input_data={"text": "ASAP: Database connection failed"},
            expected_output="urgent",
        ),
        EvaluationExample(
            input_data={"text": "Calendar invite for next week"},
            expected_output="scheduling",
        ),
        EvaluationExample(
            input_data={"text": "Payment reminder for account"},
            expected_output="finance",
        ),
        EvaluationExample(
            input_data={"text": "General inquiry about services"},
            expected_output="general",
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


@pytest.fixture
def large_dataset():
    """Create larger dataset for testing."""
    examples = []
    categories = ["urgent", "scheduling", "finance", "general"]

    for i in range(100):
        category = categories[i % len(categories)]
        examples.append(
            EvaluationExample(
                input_data={"text": f"Test message {i} for {category}"},
                expected_output=category,
            )
        )

    return Dataset(examples=examples, name="large_test_dataset")


@pytest.fixture
def empty_dataset():
    """Create empty dataset for testing."""
    return Dataset(examples=[], name="empty_dataset")


class TestDiverseSampling:
    """Test cases for diverse sampling strategy."""

    def test_diverse_sampling_initialization(self):
        """Test DiverseSampling initialization."""
        sampler = DiverseSampling(random_seed=123)
        assert sampler.random_seed == 123

    @pytest.mark.asyncio
    async def test_select_subset_smaller_target(self, sample_dataset):
        """Test subset selection with target smaller than dataset."""
        sampler = DiverseSampling()
        result = await sampler.select_subset(sample_dataset, target_size=4)

        assert isinstance(result, SubsetSelectionResult)
        assert result.selected_size == 4
        assert result.original_size == 8
        assert result.reduction_ratio == 0.5
        assert result.selection_strategy == "diverse_sampling"
        assert 0.0 <= result.diversity_score <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
        # Verify selected examples are actual dataset members
        for ex in result.selected_examples:
            assert ex in sample_dataset.examples

    @pytest.mark.asyncio
    async def test_select_subset_larger_target(self, sample_dataset):
        """Test subset selection with target larger than dataset."""
        sampler = DiverseSampling()
        result = await sampler.select_subset(sample_dataset, target_size=10)

        assert result.selected_size == 8  # Original size
        assert result.reduction_ratio == 0.0
        assert len(result.selected_examples) == 8

    @pytest.mark.asyncio
    async def test_select_subset_with_clustering(self, large_dataset):
        """Test subset selection using clustering."""
        sampler = DiverseSampling()
        result = await sampler.select_subset(
            large_dataset, target_size=20, use_clustering=True
        )

        assert result.selected_size == 20
        assert result.original_size == 100
        assert result.reduction_ratio == 0.8
        assert 0.0 <= result.diversity_score <= 1.0

    @pytest.mark.asyncio
    async def test_select_subset_similarity_based(self, sample_dataset):
        """Test subset selection using similarity-based sampling."""
        sampler = DiverseSampling()
        result = await sampler.select_subset(
            sample_dataset, target_size=4, use_clustering=False
        )

        assert result.selected_size == 4
        assert result.diversity_score > 0

    @pytest.mark.asyncio
    async def test_diverse_sampling_empty_dataset(self, empty_dataset):
        """Ensure empty datasets are handled safely."""
        sampler = DiverseSampling()
        result = await sampler.select_subset(empty_dataset, target_size=5)

        assert result.selected_size == 0
        assert result.original_size == 0
        assert result.reduction_ratio == 0.0

    def test_extract_text_features(self, sample_dataset):
        """Test text feature extraction."""
        sampler = DiverseSampling()
        features = sampler._extract_text_features(sample_dataset.examples)

        assert len(features) == 8
        assert "urgent" in features[0].lower()
        assert "server" in features[0].lower()

    @pytest.mark.asyncio
    async def test_cluster_based_selection_error_handling(self, sample_dataset):
        """Test error handling in cluster-based selection."""
        sampler = DiverseSampling()

        # Mock TfidfVectorizer to raise an error
        with patch(
            "traigent.cloud.subset_selection.TfidfVectorizer"
        ) as mock_vectorizer:
            mock_vectorizer.return_value.fit_transform.side_effect = ValueError(
                "TF-IDF failed"
            )

            # Should fallback to random sampling
            with patch("random.sample") as mock_random:
                mock_random.return_value = [0, 1, 2]
                indices = await sampler._cluster_based_selection(
                    ["text1", "text2", "text3", "text4"], target_size=3
                )
                assert indices == [0, 1, 2]


class TestRepresentativeSampling:
    """Test cases for representative sampling strategy."""

    def test_representative_sampling_initialization(self):
        """Test RepresentativeSampling initialization."""
        sampler = RepresentativeSampling(random_seed=456)
        assert sampler.random_seed == 456

    @pytest.mark.asyncio
    async def test_select_subset_balanced(self, sample_dataset):
        """Test balanced subset selection."""
        sampler = RepresentativeSampling()
        result = await sampler.select_subset(
            sample_dataset, target_size=4, balance_outputs=True
        )

        assert result.selected_size == 4
        assert result.selection_strategy == "representative_sampling"
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.diversity_score <= 1.0
        # Verify selected examples are actual dataset members
        for ex in result.selected_examples:
            assert ex in sample_dataset.examples

    @pytest.mark.asyncio
    async def test_select_subset_unbalanced(self, sample_dataset):
        """Test unbalanced subset selection."""
        sampler = RepresentativeSampling()
        result = await sampler.select_subset(
            sample_dataset, target_size=4, balance_outputs=False
        )

        assert result.selected_size == 4

    @pytest.mark.asyncio
    async def test_representative_sampling_empty_dataset(self, empty_dataset):
        """Representative sampling should handle empty datasets."""
        sampler = RepresentativeSampling()
        result = await sampler.select_subset(empty_dataset, target_size=3)

        assert result.selected_size == 0
        assert result.reduction_ratio == 0.0

    def test_stratified_sampling(self, sample_dataset):
        """Test stratified sampling to maintain distribution."""
        sampler = RepresentativeSampling()
        selected = sampler._stratified_sampling(sample_dataset.examples, target_size=4)

        assert len(selected) == 4

        # Check that we have representation from different categories
        outputs = [ex.expected_output for ex in selected]
        unique_outputs = set(outputs)
        assert len(unique_outputs) >= 2  # Should have multiple categories

    def test_stratified_sampling_proportional(self, large_dataset):
        """Test stratified sampling maintains proportions."""
        sampler = RepresentativeSampling()
        selected = sampler._stratified_sampling(large_dataset.examples, target_size=20)

        # Should maintain roughly equal proportions since dataset is balanced
        outputs = [ex.expected_output for ex in selected]
        output_counts = {}
        for output in outputs:
            output_counts[output] = output_counts.get(output, 0) + 1

        # Each category should have roughly 5 examples (20/4 categories)
        for count in output_counts.values():
            assert 3 <= count <= 7  # Allow some variance


class TestHighConfidenceSampling:
    """Test cases for high confidence sampling strategy."""

    def test_high_confidence_sampling_initialization(self):
        """Test HighConfidenceSampling initialization."""
        sampler = HighConfidenceSampling(random_seed=789)
        assert sampler.random_seed == 789

    @pytest.mark.asyncio
    async def test_select_subset_prioritize_difficult(self, sample_dataset):
        """Test subset selection prioritizing difficult examples."""
        sampler = HighConfidenceSampling()
        result = await sampler.select_subset(
            sample_dataset, target_size=4, prioritize_difficult=True
        )

        assert result.selected_size == 4
        assert result.selection_strategy == "high_confidence_sampling"
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.diversity_score <= 1.0
        # Verify selected examples are actual dataset members
        for ex in result.selected_examples:
            assert ex in sample_dataset.examples

    @pytest.mark.asyncio
    async def test_select_subset_prioritize_easy(self, sample_dataset):
        """Test subset selection prioritizing easy examples."""
        sampler = HighConfidenceSampling()
        result = await sampler.select_subset(
            sample_dataset, target_size=4, prioritize_difficult=False
        )

        assert result.selected_size == 4

    @pytest.mark.asyncio
    async def test_high_confidence_sampling_empty_dataset(self, empty_dataset):
        """High confidence sampling should handle empty datasets."""
        sampler = HighConfidenceSampling()
        result = await sampler.select_subset(empty_dataset, target_size=2)

        assert result.selected_size == 0
        assert result.reduction_ratio == 0.0

    def test_score_examples(self, sample_dataset):
        """Test example scoring by difficulty."""
        sampler = HighConfidenceSampling()
        scores = sampler._score_examples(sample_dataset.examples)

        assert len(scores) == 8
        assert all(isinstance(score, float) for score in scores)
        assert all(score >= 0 for score in scores)

        # Examples with difficulty keywords should have higher scores
        urgent_indices = [
            i
            for i, ex in enumerate(sample_dataset.examples)
            if "urgent" in ex.expected_output.lower()
        ]
        if urgent_indices:
            urgent_scores = [scores[i] for i in urgent_indices]
            general_indices = [
                i
                for i, ex in enumerate(sample_dataset.examples)
                if "general" in ex.expected_output.lower()
            ]
            if general_indices:
                general_scores = [scores[i] for i in general_indices]
                # Urgent examples might have higher difficulty scores
                assert max(urgent_scores) >= min(general_scores)


class TestSmartSubsetSelector:
    """Test cases for smart subset selector."""

    def test_smart_subset_selector_initialization(self):
        """Test SmartSubsetSelector initialization."""
        selector = SmartSubsetSelector()
        assert selector.diverse_sampler is not None
        assert selector.representative_sampler is not None
        assert selector.confidence_sampler is not None

    @pytest.mark.asyncio
    async def test_select_optimal_subset_auto(self, sample_dataset):
        """Test optimal subset selection with auto strategy."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            sample_dataset, target_reduction=0.5, strategy="auto"
        )

        assert isinstance(result_dataset, Dataset)
        assert len(result_dataset.examples) == 4  # 50% reduction from 8
        assert result_dataset.name.startswith("test_dataset_subset_")

    @pytest.mark.asyncio
    async def test_select_optimal_subset_diverse(self, sample_dataset):
        """Test optimal subset selection with diverse strategy."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            sample_dataset, target_reduction=0.5, strategy="diverse"
        )

        assert len(result_dataset.examples) == 4
        assert "diverse" in result_dataset.name

    @pytest.mark.asyncio
    async def test_select_optimal_subset_representative(self, sample_dataset):
        """Test optimal subset selection with representative strategy."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            sample_dataset, target_reduction=0.5, strategy="representative"
        )

        assert len(result_dataset.examples) == 4
        assert "representative" in result_dataset.name

    @pytest.mark.asyncio
    async def test_select_optimal_subset_confident(self, sample_dataset):
        """Test optimal subset selection with confident strategy."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            sample_dataset, target_reduction=0.5, strategy="confident"
        )

        assert len(result_dataset.examples) == 4
        assert "confident" in result_dataset.name

    @pytest.mark.asyncio
    async def test_select_optimal_subset_unknown_strategy(self, sample_dataset):
        """Test optimal subset selection with unknown strategy."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            sample_dataset, target_reduction=0.5, strategy="unknown"
        )

        # Should default to diverse
        assert len(result_dataset.examples) == 4

    @pytest.mark.asyncio
    async def test_select_optimal_subset_empty_dataset(self, empty_dataset):
        """Smart selector should handle empty datasets without errors."""
        selector = SmartSubsetSelector()
        result_dataset = await selector.select_optimal_subset(
            empty_dataset, target_reduction=0.5, strategy="auto"
        )

        assert len(result_dataset.examples) == 0

    def test_choose_optimal_strategy_high_diversity(self):
        """Test strategy choice for high output diversity."""
        selector = SmartSubsetSelector()

        # Create dataset with high output diversity
        examples = [
            EvaluationExample(
                input_data={"text": f"text{i}"}, expected_output=f"output{i}"
            )
            for i in range(10)
        ]
        dataset = Dataset(examples=examples, name="high_diversity")

        strategy = selector._choose_optimal_strategy(dataset, target_size=5)
        assert strategy == "diverse"

    def test_choose_optimal_strategy_low_diversity(self):
        """Test strategy choice for low output diversity."""
        selector = SmartSubsetSelector()

        # Create dataset with low output diversity
        examples = [
            EvaluationExample(
                input_data={"text": f"text{i}"}, expected_output="same_output"
            )
            for i in range(10)
        ]
        dataset = Dataset(examples=examples, name="low_diversity")

        strategy = selector._choose_optimal_strategy(dataset, target_size=5)
        assert strategy == "representative"

    def test_choose_optimal_strategy_complex_inputs(self):
        """Test strategy choice for complex inputs."""
        selector = SmartSubsetSelector()

        # Create dataset with complex inputs
        examples = [
            EvaluationExample(
                input_data={"text": "x" * 600},  # Long complex input
                expected_output=f"output{i % 3}",
            )
            for i in range(10)
        ]
        dataset = Dataset(examples=examples, name="complex_inputs")

        strategy = selector._choose_optimal_strategy(dataset, target_size=5)
        assert strategy == "confident"


class TestSubsetSelectionResult:
    """Test cases for SubsetSelectionResult dataclass."""

    def test_subset_selection_result_creation(self):
        """Test creation of SubsetSelectionResult."""
        examples = [
            EvaluationExample(input_data={"text": "test"}, expected_output="result")
        ]

        result = SubsetSelectionResult(
            selected_examples=examples,
            selection_strategy="test_strategy",
            original_size=10,
            selected_size=5,
            reduction_ratio=0.5,
            diversity_score=0.8,
            confidence_score=0.9,
        )

        assert result.selected_examples == examples
        assert result.selection_strategy == "test_strategy"
        assert result.original_size == 10
        assert result.selected_size == 5
        assert result.reduction_ratio == 0.5
        assert result.diversity_score == 0.8
        assert result.confidence_score == 0.9

    def test_subset_selection_result_properties(self):
        """Test properties of SubsetSelectionResult."""
        result = SubsetSelectionResult(
            selected_examples=[],
            selection_strategy="test",
            original_size=100,
            selected_size=30,
            reduction_ratio=0.7,
            diversity_score=0.75,
            confidence_score=0.85,
        )

        assert result.reduction_ratio == 0.7
        assert result.diversity_score > 0.5
        assert result.confidence_score > 0.8
