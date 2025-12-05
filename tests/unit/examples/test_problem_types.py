"""Comprehensive tests for examples.problem_generation.problem_types module."""

import pytest

from playground.problem_generation.problem_types import (  # Problem type implementations; Constraints; Enums; Base class; Registry functions
    Cardinality,
    ClassificationProblem,
    CodeGenerationProblem,
    EvaluationConstraints,
    InformationExtractionProblem,
    InputConstraints,
    OptimizationDirection,
    OutputConstraints,
    OutputFormat,
    ProblemType,
    QuestionAnsweringProblem,
    RankingRetrievalProblem,
    ReasoningProblem,
    RegressionProblem,
    SequenceGenerationProblem,
    SummarizationProblem,
    TranslationTransformationProblem,
    get_problem_type,
    list_problem_types,
    register_problem_type,
)


class TestEnums:
    """Test enum definitions."""

    def test_output_format_values(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.DISCRETE.value == "discrete"
        assert OutputFormat.CONTINUOUS.value == "continuous"
        assert OutputFormat.SEQUENCE.value == "sequence"
        assert OutputFormat.STRUCTURED.value == "structured"
        assert OutputFormat.MIXED.value == "mixed"

    def test_cardinality_values(self):
        """Test Cardinality enum values."""
        assert Cardinality.SINGLE.value == "single"
        assert Cardinality.MULTIPLE.value == "multiple"
        assert Cardinality.VARIABLE.value == "variable"

    def test_optimization_direction_values(self):
        """Test OptimizationDirection enum values."""
        assert OptimizationDirection.MAXIMIZE.value == "maximize"
        assert OptimizationDirection.MINIMIZE.value == "minimize"


class TestConstraints:
    """Test constraint classes."""

    def test_input_constraints_creation(self):
        """Test InputConstraints creation and validation."""
        constraints = InputConstraints(
            format="text",
            min_length=10,
            max_length=100,
            required_fields=["text", "context"],
            optional_fields=["metadata"],
            validation_rules=[lambda x: isinstance(x, dict), lambda x: "text" in x],
        )

        assert constraints.format == "text"
        assert constraints.min_length == 10
        assert constraints.max_length == 100
        assert len(constraints.required_fields) == 2
        assert len(constraints.validation_rules) == 2

    def test_input_constraints_validate(self):
        """Test InputConstraints validation method."""
        constraints = InputConstraints(
            validation_rules=[
                lambda x: isinstance(x, dict),
                lambda x: "text" in x,
                lambda x: len(x.get("text", "")) > 0,
            ]
        )

        # Valid input
        assert constraints.validate({"text": "hello"}) is True

        # Invalid inputs
        assert constraints.validate("not a dict") is False
        assert constraints.validate({"no_text": "field"}) is False
        assert constraints.validate({"text": ""}) is False

    def test_output_constraints_creation(self):
        """Test OutputConstraints creation."""
        constraints = OutputConstraints(
            format=OutputFormat.DISCRETE,
            cardinality=Cardinality.SINGLE,
            domain=["cat", "dog", "bird"],
            structure_schema={"type": "string", "enum": ["cat", "dog", "bird"]},
            validation_rules=[lambda x: x in ["cat", "dog", "bird"]],
        )

        assert constraints.format == OutputFormat.DISCRETE
        assert constraints.cardinality == Cardinality.SINGLE
        assert len(constraints.domain) == 3

    def test_evaluation_constraints(self):
        """Test EvaluationConstraints."""
        constraints = EvaluationConstraints(
            primary_metrics=["accuracy", "f1_score"],
            secondary_metrics=["precision", "recall"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

        assert len(constraints.primary_metrics) == 2
        assert len(constraints.secondary_metrics) == 2
        assert constraints.get_all_metrics() == [
            "accuracy",
            "f1_score",
            "precision",
            "recall",
        ]


class TestClassificationProblem:
    """Test ClassificationProblem implementation."""

    def test_basic_classification(self):
        """Test basic classification problem."""
        problem = ClassificationProblem(
            num_classes=3, class_names=["positive", "negative", "neutral"]
        )

        assert problem.num_classes == 3
        assert problem.class_names == ["positive", "negative", "neutral"]
        assert problem.multi_label is False
        assert problem.name == "classification"

    def test_multi_label_classification(self):
        """Test multi-label classification."""
        problem = ClassificationProblem(num_classes=5, multi_label=True)

        assert problem.name == "multi_label_classification"
        assert problem.multi_label is True

        # Check output constraints
        output_constraints = problem.get_output_constraints()
        assert output_constraints.cardinality == Cardinality.MULTIPLE

    def test_classification_prompt_template(self):
        """Test prompt template generation."""
        problem = ClassificationProblem(
            num_classes=3, class_names=["cat", "dog", "bird"]
        )

        template = problem.generate_prompt_template()
        assert "cat" in template
        assert "dog" in template
        assert "bird" in template
        assert "{text}" in template

    def test_classification_parse_output(self):
        """Test output parsing."""
        problem = ClassificationProblem(
            num_classes=3, class_names=["positive", "negative", "neutral"]
        )

        # Test various output formats
        assert problem.parse_output("positive") == 0
        assert problem.parse_output("Negative") == 1
        assert problem.parse_output("2") == 2
        assert problem.parse_output("The sentiment is neutral") == 2

    def test_classification_evaluation(self):
        """Test classification evaluation."""
        problem = ClassificationProblem(num_classes=3)

        # Perfect prediction
        metrics = problem.evaluate(prediction=1, ground_truth=1)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

        # Wrong prediction
        metrics = problem.evaluate(prediction=0, ground_truth=1)
        assert metrics["accuracy"] == 0.0

    def test_multi_label_evaluation(self):
        """Test multi-label classification evaluation."""
        problem = ClassificationProblem(num_classes=5, multi_label=True)

        # Test with overlapping predictions
        metrics = problem.evaluate(prediction=[0, 1, 2], ground_truth=[1, 2, 3])

        assert metrics["accuracy"] == 0.0  # Not exact match
        assert 0 < metrics["precision"] < 1.0
        assert 0 < metrics["recall"] < 1.0
        assert "hamming_loss" in metrics


class TestRegressionProblem:
    """Test RegressionProblem implementation."""

    def test_scalar_regression(self):
        """Test scalar regression problem."""
        problem = RegressionProblem(output_dim=1, output_range=(0.0, 1.0))

        assert problem.name == "regression"
        assert problem.output_dim == 1
        assert problem.output_range == (0.0, 1.0)

    def test_vector_regression(self):
        """Test vector regression problem."""
        problem = RegressionProblem(output_dim=3, output_range=(-1.0, 1.0))

        assert problem.name == "vector_regression"
        assert problem.output_dim == 3

    def test_regression_parse_output(self):
        """Test regression output parsing."""
        # Scalar regression
        problem = RegressionProblem(output_dim=1, output_range=(0.0, 10.0))

        assert problem.parse_output("5.5") == 5.5
        assert problem.parse_output("15.0") == 10.0  # Clipped to range
        assert problem.parse_output("-5") == 0.0  # Clipped to range

        # Vector regression
        problem = RegressionProblem(output_dim=3)
        output = problem.parse_output("1.0, 2.0, 3.0")
        assert output == [1.0, 2.0, 3.0]

    def test_regression_evaluation(self):
        """Test regression evaluation metrics."""
        problem = RegressionProblem(output_dim=1)

        metrics = problem.evaluate(prediction=5.0, ground_truth=4.0)
        assert metrics["mean_squared_error"] == 1.0
        assert metrics["mean_absolute_error"] == 1.0
        assert "r2_score" in metrics


class TestSequenceGenerationProblem:
    """Test SequenceGenerationProblem implementation."""

    def test_basic_generation(self):
        """Test basic sequence generation."""
        problem = SequenceGenerationProblem(min_length=10, max_length=100)

        assert problem.name == "sequence_generation"
        assert problem.min_length == 10
        assert problem.max_length == 100

    def test_constrained_generation(self):
        """Test constrained generation."""
        problem = SequenceGenerationProblem(
            constrained=True, constraints=["Must include keywords", "Professional tone"]
        )

        assert problem.name == "constrained_generation"
        assert len(problem.constraints) == 2

    def test_generation_evaluation(self):
        """Test generation evaluation."""
        problem = SequenceGenerationProblem(constraints=["keyword1", "keyword2"])

        prediction = "This text contains keyword1 and keyword2"
        ground_truth = "This text has keyword1 and also keyword2"

        metrics = problem.evaluate(prediction, ground_truth)
        assert "bleu_score" in metrics
        assert "rouge_score" in metrics
        assert "constraint_satisfaction" in metrics
        assert metrics["constraint_satisfaction"] == 1.0  # Both keywords present


class TestInformationExtractionProblem:
    """Test InformationExtractionProblem implementation."""

    def test_entity_extraction(self):
        """Test entity extraction setup."""
        problem = InformationExtractionProblem(extraction_type="entities")

        assert problem.name == "entities_extraction"
        assert problem.extraction_type == "entities"

    def test_extraction_with_schema(self):
        """Test extraction with schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "location": {"type": "string"},
            },
        }

        problem = InformationExtractionProblem(extraction_type="slots", schema=schema)

        assert problem.schema == schema

    def test_extraction_parse_output(self):
        """Test extraction output parsing."""
        problem = InformationExtractionProblem()

        # Test JSON parsing
        json_output = '{"entities": [{"type": "person", "value": "John"}]}'
        parsed = problem.parse_output(json_output)
        assert isinstance(parsed, dict)
        assert "entities" in parsed

    def test_extraction_evaluation(self):
        """Test extraction evaluation."""
        problem = InformationExtractionProblem()

        # Test slot-based evaluation
        prediction = {"name": "John", "age": 30}
        ground_truth = {"name": "John", "age": 30, "city": "NYC"}

        metrics = problem.evaluate(prediction, ground_truth)
        assert metrics["extraction_precision"] == 1.0  # All predicted are correct
        assert metrics["extraction_recall"] < 1.0  # Missing city


class TestQuestionAnsweringProblem:
    """Test QuestionAnsweringProblem implementation."""

    def test_open_qa(self):
        """Test open question answering."""
        problem = QuestionAnsweringProblem(qa_type="open", with_context=False)

        assert problem.name == "open_qa"
        assert problem.with_context is False

    def test_extractive_qa_with_context(self):
        """Test extractive QA with context."""
        problem = QuestionAnsweringProblem(qa_type="extractive", with_context=True)

        assert problem.name == "extractive_qa_with_context"
        constraints = problem.get_input_constraints()
        assert "context" in constraints.required_fields

    def test_qa_parse_output(self):
        """Test QA output parsing."""
        # Boolean answer type
        problem = QuestionAnsweringProblem(answer_type="boolean")
        assert problem.parse_output("Yes, that's correct") == "yes"
        assert problem.parse_output("No") == "no"

        # Number answer type
        problem = QuestionAnsweringProblem(answer_type="number")
        assert problem.parse_output("The answer is 42") == 42.0
        assert problem.parse_output("approximately 3.14") == 3.14

    def test_qa_evaluation(self):
        """Test QA evaluation metrics."""
        problem = QuestionAnsweringProblem()

        metrics = problem.evaluate(
            prediction="The capital of France is Paris",
            ground_truth="Paris is the capital of France",
        )

        assert metrics["exact_match"] == 0.0  # Different strings
        assert metrics["f1_score"] > 0.5  # High token overlap


class TestSummarizationProblem:
    """Test SummarizationProblem implementation."""

    def test_abstractive_summarization(self):
        """Test abstractive summarization."""
        problem = SummarizationProblem(summary_type="abstractive", target_length=50)

        assert problem.summary_type == "abstractive"
        assert problem.target_length == 50

    def test_extractive_summarization(self):
        """Test extractive summarization."""
        problem = SummarizationProblem(summary_type="extractive", compression_ratio=0.2)

        template = problem.generate_prompt_template()
        assert "extract" in template.lower()

    def test_summarization_evaluation(self):
        """Test summarization evaluation."""
        problem = SummarizationProblem()

        prediction = "This is a short summary"
        ground_truth = "This is a brief summary"

        metrics = problem.evaluate(prediction, ground_truth)
        assert "rouge_1" in metrics
        assert "rouge_2" in metrics
        assert "rouge_l" in metrics
        assert "compression_ratio" in metrics


class TestRankingRetrievalProblem:
    """Test RankingRetrievalProblem implementation."""

    def test_ranking_setup(self):
        """Test ranking problem setup."""
        problem = RankingRetrievalProblem(task_type="ranking", top_k=5)

        assert problem.task_type == "ranking"
        assert problem.top_k == 5

    def test_ranking_parse_output(self):
        """Test ranking output parsing."""
        problem = RankingRetrievalProblem(top_k=3)

        # Test JSON format
        json_output = '[{"item": "A", "score": 0.9}, {"item": "B", "score": 0.8}]'
        parsed = problem.parse_output(json_output)
        assert len(parsed) == 2
        assert parsed[0]["item"] == "A"

        # Test numbered list format
        list_output = "1. First item\n2. Second item\n3. Third item"
        parsed = problem.parse_output(list_output)
        assert len(parsed) == 3

    def test_ranking_evaluation(self):
        """Test ranking evaluation metrics."""
        problem = RankingRetrievalProblem(top_k=3)

        prediction = [
            {"item": "A", "rank": 1},
            {"item": "B", "rank": 2},
            {"item": "C", "rank": 3},
        ]
        ground_truth = [
            {"item": "B", "rank": 1},
            {"item": "A", "rank": 2},
            {"item": "D", "rank": 3},
        ]

        metrics = problem.evaluate(prediction, ground_truth)
        assert "precision_at_k" in metrics
        assert "mrr" in metrics
        assert "ndcg" in metrics


class TestTranslationTransformationProblem:
    """Test TranslationTransformationProblem implementation."""

    def test_translation_setup(self):
        """Test translation problem setup."""
        problem = TranslationTransformationProblem(
            transformation_type="translation",
            source_format="English",
            target_format="Spanish",
        )

        assert problem.transformation_type == "translation"
        assert problem.source_format == "English"
        assert problem.target_format == "Spanish"

    def test_style_transfer(self):
        """Test style transfer setup."""
        problem = TranslationTransformationProblem(
            transformation_type="style_transfer", target_format="formal"
        )

        template = problem.generate_prompt_template()
        assert "formal" in template

    def test_transformation_evaluation(self):
        """Test transformation evaluation."""
        problem = TranslationTransformationProblem()

        prediction = "Hello world"
        ground_truth = "Hello world!"

        metrics = problem.evaluate(prediction, ground_truth)
        assert "bleu_score" in metrics
        assert "semantic_similarity" in metrics
        assert metrics["semantic_similarity"] > 0.9  # Very similar


class TestReasoningProblem:
    """Test ReasoningProblem implementation."""

    def test_logical_reasoning(self):
        """Test logical reasoning setup."""
        problem = ReasoningProblem(reasoning_type="logical", requires_steps=True)

        assert problem.reasoning_type == "logical"
        assert problem.requires_steps is True

    def test_mathematical_reasoning(self):
        """Test mathematical reasoning."""
        problem = ReasoningProblem(reasoning_type="mathematical", domain="algebra")

        assert problem.domain == "algebra"
        template = problem.generate_prompt_template()
        assert "mathematical" in template

    def test_reasoning_parse_output(self):
        """Test reasoning output parsing."""
        problem = ReasoningProblem(requires_steps=True)

        # Test structured output
        json_output = """{
            "steps": ["Step 1", "Step 2"],
            "answer": "42",
            "explanation": "Because..."
        }"""

        parsed = problem.parse_output(json_output)
        assert isinstance(parsed, dict)
        assert "steps" in parsed
        assert len(parsed["steps"]) == 2

    def test_reasoning_evaluation(self):
        """Test reasoning evaluation."""
        problem = ReasoningProblem(requires_steps=True)

        prediction = {"steps": ["Add 2+2", "Get 4"], "answer": "4"}
        ground_truth = {"steps": ["2+2", "=4"], "answer": "4"}

        metrics = problem.evaluate(prediction, ground_truth)
        assert metrics["answer_accuracy"] == 1.0
        assert "reasoning_validity" in metrics


class TestCodeGenerationProblem:
    """Test CodeGenerationProblem implementation."""

    def test_python_code_generation(self):
        """Test Python code generation setup."""
        problem = CodeGenerationProblem(target_language="python", code_type="function")

        assert problem.target_language == "python"
        assert problem.code_type == "function"
        assert problem.name == "code_generation"

    def test_sql_generation(self):
        """Test SQL generation setup."""
        problem = CodeGenerationProblem(
            target_language="sql",
            code_type="query",
            execution_environment={"schema": "users(id, name, email)"},
        )

        assert problem.target_language == "sql"
        assert "schema" in problem.execution_environment

        # Check metrics include SQL-specific ones
        constraints = problem.get_evaluation_constraints()
        assert "execution_match" in constraints.primary_metrics

    def test_code_parse_output(self):
        """Test code output parsing."""
        problem = CodeGenerationProblem()

        # Test with markdown code blocks
        code_with_markdown = """```python
def hello():
    return "world"
```"""

        parsed = problem.parse_output(code_with_markdown)
        assert "```" not in parsed
        assert "def hello():" in parsed

    def test_code_generation_evaluation(self):
        """Test code generation evaluation."""
        problem = CodeGenerationProblem(target_language="python")

        prediction = "def add(a, b):\n    return a + b"
        ground_truth = "def add(x, y):\n    return x + y"

        metrics = problem.evaluate(prediction, ground_truth)
        assert metrics["exact_match"] == 0.0  # Different parameter names
        assert metrics["token_similarity"] > 0.5  # High similarity


class TestProblemTypeRegistry:
    """Test problem type registry functions."""

    def test_list_problem_types(self):
        """Test listing all problem types."""
        types = list_problem_types()

        assert isinstance(types, list)
        assert len(types) >= 9  # At least the built-in types
        assert "classification" in types
        assert "code_generation" in types

    def test_get_problem_type(self):
        """Test getting problem type instances."""
        # Test classification
        problem = get_problem_type("classification", num_classes=5)
        assert isinstance(problem, ClassificationProblem)
        assert problem.num_classes == 5

        # Test with invalid type
        with pytest.raises(ValueError, match="Unknown problem type"):
            get_problem_type("invalid_type")

    def test_register_custom_problem_type(self):
        """Test registering custom problem type."""

        class CustomProblem(ProblemType):
            def __init__(self):
                super().__init__("custom", "Custom problem type")

            def get_input_constraints(self):
                return InputConstraints()

            def get_output_constraints(self):
                return OutputConstraints(OutputFormat.MIXED)

            def get_evaluation_constraints(self):
                return EvaluationConstraints(["custom_metric"])

            def generate_prompt_template(self):
                return "Custom prompt: {input}"

            def parse_output(self, raw_output):
                return raw_output

            def format_input(self, input_data):
                return str(input_data)

            def evaluate(self, prediction, ground_truth):
                return {"custom_metric": 1.0}

        # Register custom type
        register_problem_type("custom", CustomProblem)

        # Should be in list now
        assert "custom" in list_problem_types()

        # Should be able to get instance
        problem = get_problem_type("custom")
        assert isinstance(problem, CustomProblem)
        assert problem.name == "custom"
