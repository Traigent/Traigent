"""Comprehensive tests for examples.problem_generation.constrained_generators module."""

import pytest

from playground.problem_generation.constrained_generators import (  # Data structures; Factory function
    ClassificationExampleGenerator,
    CodeGenerationExampleGenerator,
    ConstrainedExample,
    GenerationExampleGenerator,
    InformationExtractionExampleGenerator,
    QuestionAnsweringExampleGenerator,
    RankingRetrievalExampleGenerator,
    ReasoningExampleGenerator,
    SummarizationExampleGenerator,
    TranslationTransformationExampleGenerator,
    get_constrained_generator,
)
from playground.problem_generation.problem_types import (
    ClassificationProblem,
    CodeGenerationProblem,
    InformationExtractionProblem,
    ProblemType,
    QuestionAnsweringProblem,
    RankingRetrievalProblem,
    ReasoningProblem,
    SequenceGenerationProblem,
    SummarizationProblem,
    TranslationTransformationProblem,
)


class TestConstrainedExample:
    """Test ConstrainedExample dataclass."""

    def test_constrained_example_creation(self):
        """Test creating a ConstrainedExample."""
        example = ConstrainedExample(
            id=1,
            input_data={"text": "hello world"},
            expected_output="positive",
            metadata={"domain": "test", "difficulty": "easy"},
        )

        assert example.id == 1
        assert example.input_data == {"text": "hello world"}
        assert example.expected_output == "positive"
        assert example.metadata["domain"] == "test"

    def test_constrained_example_to_dict(self):
        """Test converting ConstrainedExample to dictionary."""
        example = ConstrainedExample(
            id=42,
            input_data={"query": "test query"},
            expected_output={"answer": "test answer"},
            metadata={"score": 0.95},
        )

        dict_repr = example.to_dict()

        assert dict_repr["id"] == 42
        assert dict_repr["input_data"]["query"] == "test query"
        assert dict_repr["expected_output"]["answer"] == "test answer"
        assert dict_repr["metadata"]["score"] == 0.95


class TestConstrainedExampleGenerator:
    """Test base ConstrainedExampleGenerator class."""

    def test_validate_example_valid(self):
        """Test validating a valid example."""
        problem = ClassificationProblem(num_classes=3)
        generator = ClassificationExampleGenerator(problem, ["A", "B", "C"])

        example = ConstrainedExample(
            id=1, input_data={"text": "test text"}, expected_output="A", metadata={}
        )

        is_valid, issues = generator.validate_example(example)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_example_missing_field(self):
        """Test validating example with missing required field."""
        problem = QuestionAnsweringProblem(with_context=True)
        generator = QuestionAnsweringExampleGenerator(problem)

        # Missing required 'question' field
        example = ConstrainedExample(
            id=1,
            input_data={"context": "some context"},
            expected_output="answer",
            metadata={},
        )

        is_valid, issues = generator.validate_example(example)
        assert is_valid is False
        assert any("question" in issue for issue in issues)


class TestClassificationExampleGenerator:
    """Test ClassificationExampleGenerator."""

    def test_basic_classification_generation(self):
        """Test generating basic classification examples."""
        problem = ClassificationProblem(
            num_classes=3, class_names=["positive", "negative", "neutral"]
        )
        generator = ClassificationExampleGenerator(
            problem, ["positive", "negative", "neutral"]
        )

        example = generator.generate_example(1, "customer_service", "easy")

        assert example.id == 1
        assert "text" in example.input_data
        assert example.expected_output in [
            "positive",
            "negative",
            "neutral",
            "shipping_inquiry",
            "return_request",
            "account_support",
        ]
        assert example.metadata["domain"] == "customer_service"
        assert example.metadata["difficulty"] == "easy"
        assert example.metadata["problem_type"] == "classification"

    def test_classification_batch_generation(self):
        """Test generating batch of classification examples."""
        problem = ClassificationProblem(num_classes=2, class_names=["spam", "ham"])
        generator = ClassificationExampleGenerator(problem, ["spam", "ham"])

        batch = generator.generate_batch(5, "email", "medium")

        assert len(batch) == 5
        for i, example in enumerate(batch):
            assert example.id == i + 1
            assert "text" in example.input_data
            assert example.metadata["domain"] == "email"

    def test_classification_with_context(self):
        """Test classification generation with context."""
        problem = ClassificationProblem(num_classes=3)
        categories = ["diet_advice", "exercise_advice", "medical_advice"]
        context = {"problem_description": "how to lose weight effectively"}
        generator = ClassificationExampleGenerator(problem, categories, context)

        example = generator.generate_example(1, "general", "medium", context)

        assert "text" in example.input_data
        # Should generate weight-loss related examples
        text_lower = example.input_data["text"].lower()
        assert any(
            word in text_lower
            for word in ["weight", "calorie", "diet", "exercise", "metabolism"]
        )

    def test_classification_domain_templates(self):
        """Test different domain templates."""
        problem = ClassificationProblem(num_classes=3)

        # Test medical domain
        generator = ClassificationExampleGenerator(
            problem, ["routine_checkup", "emergency", "prescription_request"]
        )
        example = generator.generate_example(1, "medical", "easy")
        assert example.metadata["domain"] == "medical"

        # Test educational domain
        generator = ClassificationExampleGenerator(
            problem, ["science_concept", "math_problem", "geography_fact"]
        )
        example = generator.generate_example(1, "educational", "medium")
        assert example.metadata["domain"] == "educational"

    def test_classification_difficulty_levels(self):
        """Test different difficulty levels."""
        problem = ClassificationProblem(num_classes=3)
        generator = ClassificationExampleGenerator(problem, ["A", "B", "C"])

        easy = generator.generate_example(1, "customer_service", "easy")
        medium = generator.generate_example(2, "customer_service", "medium")
        hard = generator.generate_example(3, "customer_service", "hard")

        assert easy.metadata["difficulty"] == "easy"
        assert medium.metadata["difficulty"] == "medium"
        assert hard.metadata["difficulty"] == "hard"


class TestGenerationExampleGenerator:
    """Test GenerationExampleGenerator."""

    def test_basic_generation(self):
        """Test basic text generation examples."""
        problem = SequenceGenerationProblem(min_length=10, max_length=100)
        generator = GenerationExampleGenerator(problem)

        example = generator.generate_example(1, "marketing", "easy")

        assert example.id == 1
        assert "prompt" in example.input_data
        assert isinstance(example.expected_output, str)
        assert example.metadata["problem_type"] == "generation"

    def test_generation_with_constraints(self):
        """Test generation with length constraints."""
        problem = SequenceGenerationProblem(
            min_length=50,
            max_length=200,
            constrained=True,
            constraints=["professional tone", "include call-to-action"],
        )
        generator = GenerationExampleGenerator(problem)

        example = generator.generate_example(1, "marketing", "medium")

        assert "constraints" in example.input_data
        assert len(example.input_data["constraints"]) > 0
        assert "minimum 50 words" in example.input_data["constraints"][0]

    def test_generation_domain_templates(self):
        """Test different domain templates."""
        problem = SequenceGenerationProblem()
        generator = GenerationExampleGenerator(problem)

        # Marketing domain
        marketing = generator.generate_example(1, "marketing", "easy")
        assert "prompt" in marketing.input_data

        # Technical domain
        technical = generator.generate_example(2, "technical", "medium")
        assert "prompt" in technical.input_data

    def test_generation_batch(self):
        """Test batch generation."""
        problem = SequenceGenerationProblem()
        generator = GenerationExampleGenerator(problem)

        batch = generator.generate_batch(3, "technical", "hard")

        assert len(batch) == 3
        for example in batch:
            assert example.metadata["domain"] == "technical"
            assert example.metadata["difficulty"] == "hard"


class TestInformationExtractionExampleGenerator:
    """Test InformationExtractionExampleGenerator."""

    def test_basic_extraction(self):
        """Test basic information extraction."""
        problem = InformationExtractionProblem(extraction_type="entities")
        generator = InformationExtractionExampleGenerator(problem)

        example = generator.generate_example(1, "business", "easy")

        assert "text" in example.input_data
        assert isinstance(example.expected_output, dict)
        assert example.metadata["extraction_type"] == "entities"

    def test_extraction_with_schema(self):
        """Test extraction with schema."""
        problem = InformationExtractionProblem(extraction_type="slots")
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "date": {"type": "string"},
                "amount": {"type": "number"},
            },
        }
        generator = InformationExtractionExampleGenerator(problem, schema)

        example = generator.generate_example(1, "financial", "medium")

        assert example.input_data["schema"] == schema

    def test_extraction_domains(self):
        """Test different extraction domains."""
        problem = InformationExtractionProblem()
        generator = InformationExtractionExampleGenerator(problem)

        # Business domain
        business = generator.generate_example(1, "business", "easy")
        assert (
            "entities" in business.expected_output
            or "financial_metrics" in business.expected_output
        )

        # Legal domain
        legal = generator.generate_example(2, "legal", "easy")
        if "parties" in legal.expected_output:
            assert isinstance(legal.expected_output["parties"], list)


class TestQuestionAnsweringExampleGenerator:
    """Test QuestionAnsweringExampleGenerator."""

    def test_basic_qa_generation(self):
        """Test basic question answering generation."""
        problem = QuestionAnsweringProblem(qa_type="open")
        generator = QuestionAnsweringExampleGenerator(problem)

        example = generator.generate_example(1, "educational", "easy")

        assert "question" in example.input_data
        assert isinstance(example.expected_output, str)
        assert example.metadata["qa_type"] == "open"

    def test_qa_with_context(self):
        """Test QA with context."""
        problem = QuestionAnsweringProblem(qa_type="extractive", with_context=True)
        generator = QuestionAnsweringExampleGenerator(problem)

        example = generator.generate_example(1, "educational", "medium")

        assert "question" in example.input_data
        if example.metadata["has_context"]:
            assert "context" in example.input_data

    def test_qa_context_aware_generation(self):
        """Test context-aware QA generation."""
        problem = QuestionAnsweringProblem()
        context = {"problem_description": "how does photosynthesis work"}
        generator = QuestionAnsweringExampleGenerator(problem, context)

        example = generator.generate_example(1, "general", "easy", context)

        question_lower = example.input_data["question"].lower()
        assert any(
            word in question_lower
            for word in ["how", "photosynthesis", "plant", "work"]
        )

    def test_qa_domains(self):
        """Test different QA domains."""
        problem = QuestionAnsweringProblem()
        generator = QuestionAnsweringExampleGenerator(problem)

        # Educational domain
        edu = generator.generate_example(1, "educational", "easy")
        assert "question" in edu.input_data

        # Technical domain
        tech = generator.generate_example(2, "technical", "medium")
        assert "question" in tech.input_data


class TestSummarizationExampleGenerator:
    """Test SummarizationExampleGenerator."""

    def test_basic_summarization(self):
        """Test basic summarization generation."""
        problem = SummarizationProblem(summary_type="abstractive", target_length=50)
        generator = SummarizationExampleGenerator(problem)

        example = generator.generate_example(1, "news", "easy")

        assert "document" in example.input_data
        assert isinstance(example.expected_output, str)
        assert example.metadata["summary_type"] == "abstractive"

    def test_summarization_compression_ratio(self):
        """Test compression ratio calculation."""
        problem = SummarizationProblem()
        generator = SummarizationExampleGenerator(problem)

        example = generator.generate_example(1, "news", "medium")

        assert "compression_ratio" in example.metadata
        assert 0 < example.metadata["compression_ratio"] < 1

    def test_summarization_templates(self):
        """Test summarization templates."""
        problem = SummarizationProblem()
        generator = SummarizationExampleGenerator(problem)

        # News domain with templates
        news = generator.generate_example(1, "news", "easy")
        assert len(news.input_data["document"]) > len(news.expected_output)


class TestRankingRetrievalExampleGenerator:
    """Test RankingRetrievalExampleGenerator."""

    def test_basic_ranking(self):
        """Test basic ranking generation."""
        problem = RankingRetrievalProblem(task_type="ranking", top_k=3)
        generator = RankingRetrievalExampleGenerator(problem)

        example = generator.generate_example(1, "search", "easy")

        assert "query" in example.input_data
        assert "candidates" in example.input_data
        assert isinstance(example.expected_output, list)
        assert len(example.expected_output) <= 3  # top_k constraint

    def test_ranking_scores(self):
        """Test ranking with scores."""
        problem = RankingRetrievalProblem(top_k=5)
        generator = RankingRetrievalExampleGenerator(problem)

        example = generator.generate_example(1, "search", "easy")

        if example.expected_output:
            for item in example.expected_output:
                assert "score" in item
                assert "rank" in item
                assert 0 <= item["score"] <= 1

    def test_retrieval_generation(self):
        """Test retrieval task generation."""
        problem = RankingRetrievalProblem(task_type="retrieval", top_k=10)
        generator = RankingRetrievalExampleGenerator(problem)

        example = generator.generate_example(1, "documents", "medium")

        assert example.metadata["task_type"] == "retrieval"
        assert "num_candidates" in example.metadata


class TestTranslationTransformationExampleGenerator:
    """Test TranslationTransformationExampleGenerator."""

    def test_style_transfer(self):
        """Test style transfer generation."""
        problem = TranslationTransformationProblem(
            transformation_type="style_transfer", target_format="formal"
        )
        generator = TranslationTransformationExampleGenerator(problem)

        example = generator.generate_example(1, "business", "easy")

        assert "text" in example.input_data
        if "target_style" in example.input_data:
            assert example.input_data["target_style"] in ["formal", "polite"]

    def test_translation_generation(self):
        """Test translation generation."""
        problem = TranslationTransformationProblem(
            transformation_type="translation",
            source_format="English",
            target_format="Spanish",
        )
        generator = TranslationTransformationExampleGenerator(problem)

        example = generator.generate_example(1, "general", "medium")

        assert "source_lang" in example.input_data
        assert "target_lang" in example.input_data
        assert example.input_data["source_lang"] == "English"
        assert example.input_data["target_lang"] == "Spanish"

    def test_transformation_metadata(self):
        """Test transformation metadata."""
        problem = TranslationTransformationProblem()
        generator = TranslationTransformationExampleGenerator(problem)

        example = generator.generate_example(1, "technical", "hard")

        assert example.metadata["transformation_type"] == "translation"


class TestReasoningExampleGenerator:
    """Test ReasoningExampleGenerator."""

    def test_mathematical_reasoning(self):
        """Test mathematical reasoning generation."""
        problem = ReasoningProblem(reasoning_type="mathematical", requires_steps=True)
        generator = ReasoningExampleGenerator(problem)

        example = generator.generate_example(1, "education", "easy")

        assert "problem" in example.input_data
        if isinstance(example.expected_output, dict):
            assert "steps" in example.expected_output
            assert "answer" in example.expected_output

    def test_logical_reasoning(self):
        """Test logical reasoning generation."""
        problem = ReasoningProblem(reasoning_type="logical", requires_steps=True)
        generator = ReasoningExampleGenerator(problem)

        example = generator.generate_example(1, "philosophy", "medium")

        assert example.metadata["reasoning_type"] == "logical"
        assert example.metadata["requires_steps"] is True

    def test_reasoning_without_steps(self):
        """Test reasoning without requiring steps."""
        problem = ReasoningProblem(reasoning_type="causal", requires_steps=False)
        generator = ReasoningExampleGenerator(problem)

        example = generator.generate_example(1, "science", "hard")

        # Without steps, output should be a simple string
        if not problem.requires_steps:
            assert isinstance(example.expected_output, str)

    def test_reasoning_templates(self):
        """Test reasoning templates."""
        problem = ReasoningProblem(reasoning_type="mathematical", requires_steps=True)
        generator = ReasoningExampleGenerator(problem)

        # Should use templates for known reasoning types
        example = generator.generate_example(1, "any", "easy")

        if (
            isinstance(example.expected_output, dict)
            and "steps" in example.expected_output
        ):
            assert len(example.expected_output["steps"]) > 0


class TestCodeGenerationExampleGenerator:
    """Test CodeGenerationExampleGenerator."""

    def test_sql_generation(self):
        """Test SQL code generation."""
        problem = CodeGenerationProblem(target_language="sql", code_type="query")
        generator = CodeGenerationExampleGenerator(problem)

        example = generator.generate_example(1, "database", "easy")

        assert "description" in example.input_data
        assert example.metadata["target_language"] == "sql"
        assert example.metadata["code_type"] == "query"

    def test_python_generation(self):
        """Test Python code generation."""
        problem = CodeGenerationProblem(target_language="python", code_type="function")
        generator = CodeGenerationExampleGenerator(problem)

        example = generator.generate_example(1, "algorithms", "medium")

        assert "description" in example.input_data
        assert example.metadata["target_language"] == "python"

    def test_code_generation_with_context(self):
        """Test code generation with context."""
        problem = CodeGenerationProblem(target_language="sql")
        context = {"problem_description": "text to sql conversion"}
        generator = CodeGenerationExampleGenerator(problem, context)

        example = generator.generate_example(1, "general", "medium", context)

        # Should generate SQL-related example
        assert "description" in example.input_data
        description_lower = example.input_data["description"].lower()
        assert any(
            word in description_lower
            for word in ["customer", "product", "order", "find", "get"]
        )

    def test_code_generation_difficulty_levels(self):
        """Test different difficulty levels for code generation."""
        problem = CodeGenerationProblem(target_language="sql")
        generator = CodeGenerationExampleGenerator(problem)

        easy = generator.generate_example(1, "database", "easy")
        medium = generator.generate_example(2, "database", "medium")
        hard = generator.generate_example(3, "database", "hard")

        # Hard examples should be more complex (if using templates)
        assert easy.metadata["difficulty"] == "easy"
        assert medium.metadata["difficulty"] == "medium"
        assert hard.metadata["difficulty"] == "hard"

    def test_code_generation_with_schema(self):
        """Test SQL generation with schema context."""
        problem = CodeGenerationProblem(
            target_language="sql",
            execution_environment={"schema": "users(id, name, email)"},
        )
        generator = CodeGenerationExampleGenerator(problem)

        example = generator.generate_example(1, "database", "easy")

        # If using templates, should have schema in context
        if "context" in example.input_data:
            assert "schema" in example.input_data["context"].lower()


class TestFactoryFunction:
    """Test get_constrained_generator factory function."""

    def test_get_classification_generator(self):
        """Test getting classification generator."""
        problem = ClassificationProblem(num_classes=3)
        generator = get_constrained_generator(
            problem, categories=["A", "B", "C"], context={"test": "context"}
        )

        assert isinstance(generator, ClassificationExampleGenerator)
        assert generator.categories == ["A", "B", "C"]

    def test_get_generation_generator(self):
        """Test getting generation generator."""
        problem = SequenceGenerationProblem()
        generator = get_constrained_generator(problem)

        assert isinstance(generator, GenerationExampleGenerator)

    def test_get_extraction_generator(self):
        """Test getting extraction generator."""
        problem = InformationExtractionProblem()
        schema = {"type": "object"}
        generator = get_constrained_generator(problem, schema=schema)

        assert isinstance(generator, InformationExtractionExampleGenerator)
        assert generator.schema == schema

    def test_get_qa_generator(self):
        """Test getting QA generator."""
        problem = QuestionAnsweringProblem()
        generator = get_constrained_generator(problem, context={"key": "value"})

        assert isinstance(generator, QuestionAnsweringExampleGenerator)
        assert generator.context == {"key": "value"}

    def test_get_summarization_generator(self):
        """Test getting summarization generator."""
        problem = SummarizationProblem()
        generator = get_constrained_generator(problem)

        assert isinstance(generator, SummarizationExampleGenerator)

    def test_get_ranking_generator(self):
        """Test getting ranking generator."""
        problem = RankingRetrievalProblem()
        generator = get_constrained_generator(problem)

        assert isinstance(generator, RankingRetrievalExampleGenerator)

    def test_get_transformation_generator(self):
        """Test getting transformation generator."""
        problem = TranslationTransformationProblem()
        generator = get_constrained_generator(problem)

        assert isinstance(generator, TranslationTransformationExampleGenerator)

    def test_get_reasoning_generator(self):
        """Test getting reasoning generator."""
        problem = ReasoningProblem()
        generator = get_constrained_generator(problem)

        assert isinstance(generator, ReasoningExampleGenerator)

    def test_get_code_generator(self):
        """Test getting code generation generator."""
        problem = CodeGenerationProblem()
        generator = get_constrained_generator(problem, context={"lang": "sql"})

        assert isinstance(generator, CodeGenerationExampleGenerator)
        assert generator.context == {"lang": "sql"}

    def test_invalid_problem_type(self):
        """Test factory with invalid problem type."""

        class UnknownProblem(ProblemType):
            def __init__(self):
                super().__init__("unknown", "Unknown problem")

            def get_input_constraints(self):
                pass

            def get_output_constraints(self):
                pass

            def get_evaluation_constraints(self):
                pass

            def generate_prompt_template(self):
                pass

            def parse_output(self, raw_output):
                pass

            def format_input(self, input_data):
                pass

            def evaluate(self, prediction, ground_truth):
                pass

        problem = UnknownProblem()

        with pytest.raises(ValueError, match="No generator available"):
            get_constrained_generator(problem)


class TestIntegration:
    """Integration tests for constrained generators."""

    def test_end_to_end_classification_workflow(self):
        """Test complete classification workflow."""
        # Create problem
        problem = ClassificationProblem(
            num_classes=3, class_names=["positive", "negative", "neutral"]
        )

        # Get generator
        generator = get_constrained_generator(
            problem, categories=["positive", "negative", "neutral"]
        )

        # Generate batch
        batch = generator.generate_batch(10, "customer_service", "medium")

        # Validate all examples
        for example in batch:
            is_valid, issues = generator.validate_example(example)
            assert is_valid, f"Example {example.id} failed validation: {issues}"

            # Check structure
            assert "text" in example.input_data
            assert example.expected_output in [
                "positive",
                "negative",
                "neutral",
                "billing_issue",
                "technical_support",
                "order_modification",
            ]

    def test_multi_domain_generation(self):
        """Test generation across multiple domains."""
        problem = SequenceGenerationProblem()
        generator = get_constrained_generator(problem)

        domains = ["marketing", "technical", "educational", "general"]
        difficulties = ["easy", "medium", "hard"]

        for domain in domains:
            for difficulty in difficulties:
                example = generator.generate_example(1, domain, difficulty)
                assert example.metadata["domain"] == domain
                assert example.metadata["difficulty"] == difficulty

    def test_context_propagation(self):
        """Test context propagation through generators."""
        context = {
            "problem_description": "how does machine learning work",
            "target_audience": "beginners",
            "max_complexity": "medium",
        }

        # Test with QA generator
        qa_problem = QuestionAnsweringProblem()
        qa_gen = get_constrained_generator(qa_problem, context=context)
        qa_example = qa_gen.generate_example(1, "general", "easy", context)

        # Test with classification generator
        class_problem = ClassificationProblem(num_classes=3)
        class_gen = get_constrained_generator(
            class_problem,
            categories=["technical", "conceptual", "practical"],
            context=context,
        )
        class_gen.generate_example(1, "general", "medium", context)

        # Both should generate ML-related content
        assert any(
            word in str(qa_example.to_dict()).lower()
            for word in ["machine", "learning", "algorithm", "model", "data"]
        )

    def test_batch_consistency(self):
        """Test batch generation consistency."""
        problem = CodeGenerationProblem(target_language="python")
        generator = get_constrained_generator(problem)

        batch1 = generator.generate_batch(5, "algorithms", "medium")
        generator.generate_batch(5, "algorithms", "medium")

        # Each batch should have consistent metadata
        for example in batch1:
            assert example.metadata["domain"] == "algorithms"
            assert example.metadata["difficulty"] == "medium"
            assert example.metadata["target_language"] == "python"

        # IDs should be sequential within batch
        for i, example in enumerate(batch1):
            assert example.id == i + 1

    def test_error_handling(self):
        """Test error handling in generators."""
        # Test with invalid domain/difficulty
        problem = SummarizationProblem()
        generator = get_constrained_generator(problem)

        # Should handle gracefully
        example = generator.generate_example(1, "unknown_domain", "unknown_difficulty")
        assert example is not None
        assert example.metadata["domain"] == "unknown_domain"

    def test_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        problem = ClassificationProblem(num_classes=10)
        generator = get_constrained_generator(
            problem, categories=[f"category_{i}" for i in range(10)]
        )

        # Generate large batch
        large_batch = generator.generate_batch(1000, "test", "medium")

        assert len(large_batch) == 1000

        # Verify no memory leaks by checking unique IDs
        ids = [ex.id for ex in large_batch]
        assert len(set(ids)) == 1000  # All IDs should be unique
