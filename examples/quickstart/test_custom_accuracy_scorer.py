"""
Tests for custom_accuracy_scorer function.

Run with: pytest examples/quickstart/test_custom_accuracy_scorer.py -v
"""

import pytest
from scorers import custom_accuracy_scorer


@pytest.mark.unit
class TestCustomAccuracyScorerPositive:
    """Positive test cases - should return 1.0 (pass)"""

    def test_exact_match(self):
        assert custom_accuracy_scorer("Paris", "Paris", {}) == 1.0

    def test_contains_at_end(self):
        assert (
            custom_accuracy_scorer("The capital of France is Paris.", "Paris", {})
            == 1.0
        )

    def test_contains_at_start(self):
        assert (
            custom_accuracy_scorer("Jupiter is the largest planet", "Jupiter", {})
            == 1.0
        )

    def test_contains_in_middle(self):
        assert (
            custom_accuracy_scorer("The answer is 4, which is correct", "4", {}) == 1.0
        )

    def test_case_insensitive_upper_output(self):
        assert custom_accuracy_scorer("PARIS", "paris", {}) == 1.0

    def test_case_insensitive_upper_expected(self):
        assert custom_accuracy_scorer("paris", "PARIS", {}) == 1.0

    def test_case_insensitive_mixed(self):
        assert custom_accuracy_scorer("PaRiS", "pArIs", {}) == 1.0

    def test_multi_word_exact(self):
        assert (
            custom_accuracy_scorer("Leonardo da Vinci", "Leonardo da Vinci", {}) == 1.0
        )

    def test_multi_word_contained(self):
        assert (
            custom_accuracy_scorer(
                "The artist Leonardo da Vinci painted it", "Leonardo da Vinci", {}
            )
            == 1.0
        )

    def test_with_trailing_punctuation(self):
        assert (
            custom_accuracy_scorer("William Shakespeare.", "William Shakespeare", {})
            == 1.0
        )

    def test_with_surrounding_text_and_punctuation(self):
        assert (
            custom_accuracy_scorer("The chemical symbol for water is H2O.", "H2O", {})
            == 1.0
        )

    def test_number_as_string(self):
        assert custom_accuracy_scorer("4", "4", {}) == 1.0

    def test_number_in_sentence(self):
        assert custom_accuracy_scorer("World War II ended in 1945.", "1945", {}) == 1.0


@pytest.mark.unit
class TestCustomAccuracyScorerNegative:
    """Negative test cases - should return 0.0 (fail)"""

    def test_empty_output(self):
        assert custom_accuracy_scorer("", "Paris", {}) == 0.0

    def test_empty_expected(self):
        assert custom_accuracy_scorer("Paris", "", {}) == 0.0

    def test_both_empty(self):
        assert custom_accuracy_scorer("", "", {}) == 0.0

    def test_none_like_empty_output(self):
        assert custom_accuracy_scorer("", "test", {}) == 0.0

    def test_completely_different_strings(self):
        assert custom_accuracy_scorer("Hello world", "Goodbye universe", {}) == 0.0

    def test_partial_match_not_full_word(self):
        assert custom_accuracy_scorer("Par", "Paris", {}) == 0.0

    def test_reversed_containment(self):
        assert (
            custom_accuracy_scorer("Paris", "The capital of France is Paris", {}) == 0.0
        )

    def test_different_numbers(self):
        assert (
            custom_accuracy_scorer(
                "299,792 kilometers", "299,792,458 meters per second", {}
            )
            == 0.0
        )

    def test_similar_but_not_contained(self):
        assert (
            custom_accuracy_scorer("William Shakespear", "William Shakespeare", {})
            == 0.0
        )

    def test_substring_of_expected_not_full(self):
        assert (
            custom_accuracy_scorer("Leonardo painted this", "Leonardo da Vinci", {})
            == 0.0
        )

    def test_wrong_answer_entirely(self):
        assert custom_accuracy_scorer("London", "Paris", {}) == 0.0

    def test_numeric_mismatch(self):
        assert custom_accuracy_scorer("5", "4", {}) == 0.0

    def test_whitespace_only_output(self):
        assert custom_accuracy_scorer("   ", "Paris", {}) == 0.0


@pytest.mark.unit
class TestCustomAccuracyScorerEdgeCases:
    """Edge cases and boundary conditions"""

    def test_llm_metrics_ignored(self):
        assert (
            custom_accuracy_scorer("Paris", "Paris", {"tokens": 100, "latency": 0.5})
            == 1.0
        )

    def test_llm_metrics_empty_dict(self):
        assert custom_accuracy_scorer("Paris", "Paris", {}) == 1.0

    def test_single_character_match(self):
        assert custom_accuracy_scorer("4", "4", {}) == 1.0

    def test_single_character_in_string(self):
        assert custom_accuracy_scorer("The answer is 4", "4", {}) == 1.0

    def test_special_characters(self):
        assert custom_accuracy_scorer("H2O is water", "H2O", {}) == 1.0

    def test_newlines_in_output(self):
        assert custom_accuracy_scorer("The answer is:\nParis", "Paris", {}) == 1.0

    def test_tabs_in_output(self):
        assert custom_accuracy_scorer("Answer:\tParis", "Paris", {}) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
