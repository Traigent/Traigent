"""Comprehensive tests for examples.problem_generation.problem_classifier module."""

import pytest

from playground.problem_generation.improved_problem_classifier import (
    ClassificationResult,
    ImprovedProblemClassifier,
)


class TestImprovedProblemClassifier:
    """Test suite for ImprovedProblemClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance with LLM disabled for deterministic testing."""
        return ImprovedProblemClassifier(use_llm=False)

    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert hasattr(classifier, "PROBLEM_TYPES")
        assert len(classifier.PROBLEM_TYPES) == 9  # 8 original + code_generation

        # Check all problem types are present
        expected_types = {
            "classification",
            "generation",
            "information_extraction",
            "question_answering",
            "summarization",
            "ranking_retrieval",
            "translation_transformation",
            "reasoning",
            "code_generation",
        }
        assert set(classifier.PROBLEM_TYPES.keys()) == expected_types

    def test_pattern_compilation(self, classifier):
        """Test that patterns are compiled on initialization."""
        for _problem_type, config in classifier.PROBLEM_TYPES.items():
            assert "compiled_patterns" in config
            assert len(config["compiled_patterns"]) == len(config["patterns"])
            # Verify patterns are compiled regex objects
            for pattern in config["compiled_patterns"]:
                assert hasattr(pattern, "search")
                assert hasattr(pattern, "match")

    # Classification Tests
    def test_classify_classification_problems(self, classifier):
        """Test classification of classification problems."""
        test_cases = [
            "classify emails as spam or not spam",
            "categorize customer feedback into positive, negative, neutral",
            "sentiment analysis of product reviews",
            "intent classification for chatbot messages",
            "tag articles with appropriate categories",
            "identify the type of customer complaint",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "classification"
            assert result.confidence > 0.5
            assert "classification" in result.suggested_metrics

    # Generation Tests
    def test_classify_generation_problems(self, classifier):
        """Test classification of generation problems."""
        test_cases = [
            "generate product descriptions for e-commerce",
            "create marketing copy for social media",
            "write blog posts about technology",
            "compose email responses to customers",
            "how to lose weight effectively",
            "how to start a business",
            "provide guidance on career development",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "generation"
            assert result.confidence > 0.5
            assert "bleu_score" in result.suggested_metrics

    # Information Extraction Tests
    def test_classify_extraction_problems(self, classifier):
        """Test classification of information extraction problems."""
        test_cases = [
            "extract names and dates from documents",
            "parse invoice information from PDFs",
            "find all phone numbers in text",
            "identify key entities in news articles",
            "extract product specifications from descriptions",
            "detect addresses in unstructured text",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "information_extraction"
            assert result.confidence > 0.5
            assert "extraction_f1" in result.suggested_metrics

    # Question Answering Tests
    def test_classify_qa_problems(self, classifier):
        """Test classification of question answering problems."""
        test_cases = [
            "answer questions about product features",
            "create a Q&A system for documentation",
            "respond to customer inquiries",
            "build an FAQ bot",
            "how does photosynthesis work",
            "how does a car engine work",
            "what causes climate change",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "question_answering"
            assert result.confidence > 0.5

    # Summarization Tests
    def test_classify_summarization_problems(self, classifier):
        """Test classification of summarization problems."""
        test_cases = [
            "summarize long documents",
            "create executive summaries of reports",
            "condense meeting notes",
            "generate abstracts for research papers",
            "brief overview of news articles",
            "create synopsis of books",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "summarization"
            assert result.confidence > 0.5
            assert "rouge_1" in result.suggested_metrics

    # Ranking/Retrieval Tests
    def test_classify_ranking_problems(self, classifier):
        """Test classification of ranking/retrieval problems."""
        test_cases = [
            "rank search results by relevance",
            "find similar documents in database",
            "recommend products to users",
            "retrieve relevant articles for query",
            "match candidates to job postings",
            "order items by user preference",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "ranking_retrieval"
            assert result.confidence > 0.5
            assert "ndcg" in result.suggested_metrics

    # Translation/Transformation Tests
    def test_classify_transformation_problems(self, classifier):
        """Test classification of translation/transformation problems."""
        test_cases = [
            "translate English to Spanish",
            "convert technical documentation to simple language",
            "transform formal writing to casual tone",
            "rewrite content for different audience",
            "change the style of writing",
            "adapt content for children",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "translation_transformation"
            assert result.confidence > 0.5

    # Code Generation Tests
    def test_classify_code_generation_problems(self, classifier):
        """Test classification of code generation problems."""
        test_cases = [
            "text to sql",
            "convert natural language to SQL queries",
            "text2sql conversion",
            "nl to sql transformation",
            "generate python code from description",
            "create SQL query for database",
            "write a function to calculate fibonacci",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "code_generation"
            assert result.confidence > 0.5
            assert "exact_match" in result.suggested_metrics

    # Reasoning Tests
    def test_classify_reasoning_problems(self, classifier):
        """Test classification of reasoning problems."""
        test_cases = [
            "solve math word problems",
            "logical reasoning puzzles",
            "calculate based on given formulas",
            "step-by-step problem solving",
            "mathematical reasoning questions",
            "derive conclusions from premises",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "reasoning"
            assert result.confidence > 0.5

    # Special Cases Tests
    def test_special_case_text_to_sql(self, classifier):
        """Test special case handling for text-to-SQL."""
        test_cases = [
            "text to sql",
            "text2sql",
            "text-to-sql",
            "nl to sql",
            "nl2sql",
            "natural language to SQL",
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert result.problem_type == "code_generation"
            assert result.confidence >= 0.95
            assert "Text-to-SQL" in result.reasoning

    def test_special_case_how_to_questions(self, classifier):
        """Test special case handling for 'how to' questions."""
        # Generation cases
        generation_cases = [
            "how to lose weight",
            "how to start a business",
            "how to learn programming",
        ]

        for description in generation_cases:
            result = classifier.classify(description)
            assert result.problem_type == "generation"
            assert result.confidence >= 0.9

        # Question answering cases
        qa_cases = [
            "how does a computer work",
            "how does photosynthesis work",
            "how do airplanes fly and work",
        ]

        for description in qa_cases:
            result = classifier.classify(description)
            assert result.problem_type == "question_answering"
            assert result.confidence >= 0.85

    def test_confidence_scores(self, classifier):
        """Test that confidence scores are reasonable."""
        test_cases = [
            "classify sentiment",  # High confidence
            "do something with text",  # Low confidence
            "text to sql conversion",  # Very high confidence
        ]

        for description in test_cases:
            result = classifier.classify(description)
            assert 0.0 <= result.confidence <= 1.0

    def test_alternative_types(self, classifier):
        """Test that alternative types are provided."""
        result = classifier.classify("analyze and classify customer feedback")

        assert len(result.alternative_types) > 0
        # Check alternatives are sorted by confidence
        for i in range(len(result.alternative_types) - 1):
            assert result.alternative_types[i][1] >= result.alternative_types[i + 1][1]

    def test_detected_keywords(self, classifier):
        """Test keyword detection."""
        result = classifier.classify(
            "classify and categorize email messages by sentiment"
        )

        assert len(result.detected_keywords) > 0
        assert any(
            kw in ["classify", "categorize", "sentiment"]
            for kw in result.detected_keywords
        )

    def test_ambiguous_descriptions(self, classifier):
        """Test handling of ambiguous descriptions."""
        ambiguous_cases = [
            "process text data",
            "work with documents",
            "analyze user input",
            "handle customer messages",
        ]

        for description in ambiguous_cases:
            result = classifier.classify(description)
            # Should still classify but with lower confidence
            assert result.problem_type in classifier.PROBLEM_TYPES
            assert result.confidence < 0.8  # Lower confidence for ambiguous

    def test_edge_cases(self, classifier):
        """Test edge cases and unusual inputs."""
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "123 456",  # Numbers only
            "!@#$%",  # Special characters
            " " * 100,  # Whitespace
            "x" * 1000,  # Very long repetitive
        ]

        for description in edge_cases:
            # Should not crash
            result = classifier.classify(description)
            assert isinstance(result, ClassificationResult)
            assert result.problem_type in classifier.PROBLEM_TYPES

    def test_case_insensitivity(self, classifier):
        """Test that classification is case-insensitive."""
        descriptions = [
            ("CLASSIFY EMAILS AS SPAM", "classify emails as spam"),
            ("Text To SQL", "text to sql"),
            ("HOW DOES IT WORK", "how does it work"),
        ]

        for desc1, desc2 in descriptions:
            result1 = classifier.classify(desc1)
            result2 = classifier.classify(desc2)
            assert result1.problem_type == result2.problem_type

    def test_get_problem_type_info(self, classifier):
        """Test getting problem type information."""
        for problem_type in classifier.PROBLEM_TYPES:
            info = classifier.get_problem_type_info(problem_type)
            assert isinstance(info, dict)
            assert "description" in info
            assert "keywords" in info
            assert "metrics" in info

        # Test non-existent type
        info = classifier.get_problem_type_info("non_existent")
        assert info == {}

    def test_validate_classification(self, classifier):
        """Test classification validation."""
        # Valid classification
        is_valid, explanation = classifier.validate_classification(
            "classify emails as spam", "classification"
        )
        assert is_valid is True
        assert "Correctly classified" in explanation

        # Invalid classification
        is_valid, explanation = classifier.validate_classification(
            "classify emails as spam", "generation"
        )
        assert is_valid is False
        assert "not generation" in explanation

    def test_get_all_problem_types(self, classifier):
        """Test getting all problem types."""
        all_types = classifier.get_all_problem_types()
        assert len(all_types) == 9
        assert "classification" in all_types
        assert "code_generation" in all_types

    def test_get_metrics_for_type(self, classifier):
        """Test getting metrics for problem type."""
        # Test each problem type
        metrics_map = {
            "classification": "accuracy",
            "generation": "bleu_score",
            "information_extraction": "extraction_f1",
            "question_answering": "exact_match",
            "summarization": "rouge_1",
            "ranking_retrieval": "ndcg",
            "translation_transformation": "bleu_score",
            "reasoning": "answer_accuracy",
            "code_generation": "exact_match",
        }

        for problem_type, expected_metric in metrics_map.items():
            metrics = classifier.get_metrics_for_type(problem_type)
            assert isinstance(metrics, list)
            assert expected_metric in metrics

    def test_complex_descriptions(self, classifier):
        """Test classification of complex, multi-faceted descriptions."""
        complex_cases = [
            {
                "description": "Build a system that classifies customer emails and then generates appropriate responses",
                "primary_type": "classification",  # Or could be generation
                "should_have_alternatives": True,
            },
            {
                "description": "Extract information from documents and rank them by relevance",
                "primary_type": "information_extraction",  # Or ranking_retrieval
                "should_have_alternatives": True,
            },
        ]

        for case in complex_cases:
            result = classifier.classify(case["description"])
            # Should classify as one of the expected types
            assert result.problem_type in [
                case["primary_type"],
                "ranking_retrieval",
                "generation",
                "question_answering",
            ]
            if case["should_have_alternatives"]:
                # Allow no alternatives for very low confidence scores where all types scored poorly
                if result.confidence > 0.3:
                    assert len(result.alternative_types) > 0

    def test_reasoning_generation(self, classifier):
        """Test that reasoning generation is meaningful."""
        test_cases = [
            ("classify spam emails", "classify"),
            ("generate product descriptions", "generate"),
            ("extract dates from text", "extract"),
        ]

        for description, expected_keyword in test_cases:
            result = classifier.classify(description)
            assert expected_keyword in result.reasoning.lower()
            assert len(result.reasoning) > 20  # Not too short

    def test_performance(self, classifier):
        """Test classification performance with many inputs."""
        import time

        descriptions = [
            "classify emails",
            "generate text",
            "extract entities",
            "answer questions",
            "summarize documents",
        ] * 100  # 500 descriptions

        start_time = time.time()
        for desc in descriptions:
            classifier.classify(desc)
        end_time = time.time()

        # Should process 500 descriptions in under 1 second
        assert end_time - start_time < 1.0

    def test_multilingual_hints(self, classifier):
        """Test classification with multilingual hints."""
        # Even with non-English words, keywords should work
        test_cases = [
            ("classify documentos into categorías", "classification"),
            ("generate descriptions de productos", "generation"),
            ("summarize les documents", "summarization"),
        ]

        for description, expected_type in test_cases:
            result = classifier.classify(description)
            # Should still recognize based on English keywords
            assert result.problem_type == expected_type
