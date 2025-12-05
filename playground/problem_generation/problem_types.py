"""
Problem Type System for TraiGent SDK.

This module provides a structured, constraint-based approach for defining
and working with different types of LLM optimization problems. It focuses
on abstract mathematical formulations that are commonly solved by LLMs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class OutputFormat(Enum):
    """Supported output formats for problem types."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    STRUCTURED = "structured"
    SEQUENCE = "sequence"
    MIXED = "mixed"


class Cardinality(Enum):
    """Output cardinality types."""

    SINGLE = "single"
    MULTIPLE = "multiple"
    VARIABLE = "variable"


class OptimizationDirection(Enum):
    """Optimization direction for metrics."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class InputConstraints:
    """Constraints for problem inputs."""

    format: str = "text"  # "text", "structured", "mixed"
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: List[Callable[[Any], bool]] = field(default_factory=list)

    def validate(self, input_data: Any) -> bool:
        """Validate input against constraints."""
        for rule in self.validation_rules:
            if not rule(input_data):
                return False
        return True


@dataclass
class OutputConstraints:
    """Constraints for problem outputs."""

    format: OutputFormat
    cardinality: Cardinality = Cardinality.SINGLE
    domain: Union[List[Any], range, Dict[str, Any]] = field(default_factory=list)
    structure_schema: Optional[Dict[str, Any]] = None
    validation_rules: List[Callable[[Any], bool]] = field(default_factory=list)

    def validate(self, output: Any) -> bool:
        """Validate output against constraints."""
        for rule in self.validation_rules:
            if not rule(output):
                return False
        return True


@dataclass
class EvaluationConstraints:
    """Constraints for problem evaluation."""

    primary_metrics: List[str]
    secondary_metrics: List[str] = field(default_factory=list)
    constraints: List[Callable[[Any, Any], bool]] = field(default_factory=list)
    optimization_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def get_all_metrics(self) -> List[str]:
        """Get all metrics (primary + secondary)."""
        return self.primary_metrics + self.secondary_metrics


class ProblemType(ABC):
    """Abstract base class for all problem types."""

    def __init__(self, name: str, description: str):
        """Initialize problem type.

        Args:
            name: Name of the problem type
            description: Description of what this problem type does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints for this problem type."""
        pass

    @abstractmethod
    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints for this problem type."""
        pass

    @abstractmethod
    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints for this problem type."""
        pass

    @abstractmethod
    def generate_prompt_template(self) -> str:
        """Generate a prompt template for this problem type."""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> Any:
        """Parse raw LLM output into structured format.

        Args:
            raw_output: Raw string output from LLM

        Returns:
            Parsed output in appropriate format
        """
        pass

    @abstractmethod
    def format_input(self, input_data: Any) -> str:
        """Format input data for LLM consumption.

        Args:
            input_data: Input data in structured format

        Returns:
            Formatted string for LLM input
        """
        pass

    @abstractmethod
    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate prediction against ground truth.

        Args:
            prediction: Model prediction
            ground_truth: Expected output

        Returns:
            Dictionary of metric names to values
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data against constraints."""
        constraints = self.get_input_constraints()
        return constraints.validate(input_data)

    def validate_output(self, output: Any) -> bool:
        """Validate output against constraints."""
        constraints = self.get_output_constraints()
        return constraints.validate(output)

    def get_metrics(self) -> List[str]:
        """Get all metrics for this problem type."""
        constraints = self.get_evaluation_constraints()
        return constraints.get_all_metrics()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ClassificationProblem(ProblemType):
    """Classification problem type (discrete output space)."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        multi_label: bool = False,
        hierarchical: bool = False,
    ):
        """Initialize classification problem.

        Args:
            num_classes: Number of classes
            class_names: Optional names for classes
            multi_label: Whether multiple labels can be assigned
            hierarchical: Whether classes form a hierarchy
        """
        name = "classification"
        if multi_label:
            name = "multi_label_classification"
        elif hierarchical:
            name = "hierarchical_classification"

        super().__init__(
            name=name, description=f"Classify inputs into {num_classes} categories"
        )

        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.multi_label = multi_label
        self.hierarchical = hierarchical

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="text",
            min_length=1,
            validation_rules=[lambda x: isinstance(x, dict) and "text" in x],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        if self.multi_label:
            return OutputConstraints(
                format=OutputFormat.DISCRETE,
                cardinality=Cardinality.MULTIPLE,
                domain=list(range(self.num_classes)),
                structure_schema={"type": "array", "items": {"type": "integer"}},
                validation_rules=[
                    lambda x: isinstance(x, list),
                    lambda x: all(
                        isinstance(i, int) and 0 <= i < self.num_classes for i in x
                    ),
                ],
            )
        else:
            return OutputConstraints(
                format=OutputFormat.DISCRETE,
                cardinality=Cardinality.SINGLE,
                domain=list(range(self.num_classes)),
                structure_schema={"type": "integer"},
                validation_rules=[
                    lambda x: isinstance(x, (int, str)),
                    lambda x: (isinstance(x, int) and 0 <= x < self.num_classes)
                    or (isinstance(x, str) and x in self.class_names),
                ],
            )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        if self.multi_label:
            return EvaluationConstraints(
                primary_metrics=["f1_score", "precision", "recall"],
                secondary_metrics=["accuracy", "hamming_loss"],
                optimization_direction=OptimizationDirection.MAXIMIZE,
            )
        else:
            return EvaluationConstraints(
                primary_metrics=["accuracy"],
                secondary_metrics=["f1_score", "precision", "recall"],
                optimization_direction=OptimizationDirection.MAXIMIZE,
            )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.multi_label:
            return f"""Classify the following text into one or more of these categories: {', '.join(self.class_names)}

Text: {{text}}

Categories (list all that apply):"""
        else:
            return f"""Classify the following text into one of these categories: {', '.join(self.class_names)}

Text: {{text}}

Category:"""

    def parse_output(self, raw_output: str) -> Any:
        """Parse classification output."""
        output = raw_output.strip()

        if self.multi_label:
            # Parse multiple labels
            labels = []
            for class_name in self.class_names:
                if class_name.lower() in output.lower():
                    labels.append(self.class_names.index(class_name))
            return labels
        else:
            # Parse single label
            output_lower = output.lower()
            for i, class_name in enumerate(self.class_names):
                if class_name.lower() in output_lower:
                    return i
            # Try to parse as integer
            try:
                return int(output)
            except ValueError:
                return output  # Return as string if can't parse

    def format_input(self, input_data: Any) -> str:
        """Format input for classification."""
        if isinstance(input_data, dict) and "text" in input_data:
            return input_data["text"]
        return str(input_data)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate classification prediction."""
        metrics = {}

        if self.multi_label:
            # Multi-label metrics
            pred_set = set(prediction) if isinstance(prediction, list) else {prediction}
            true_set = (
                set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
            )

            intersection = pred_set & true_set
            pred_set | true_set

            metrics["accuracy"] = 1.0 if pred_set == true_set else 0.0
            metrics["precision"] = (
                len(intersection) / len(pred_set) if pred_set else 0.0
            )
            metrics["recall"] = len(intersection) / len(true_set) if true_set else 0.0
            metrics["f1_score"] = (
                2
                * metrics["precision"]
                * metrics["recall"]
                / (metrics["precision"] + metrics["recall"])
                if metrics["precision"] + metrics["recall"] > 0
                else 0.0
            )
            metrics["hamming_loss"] = len(pred_set ^ true_set) / self.num_classes
        else:
            # Single-label metrics
            metrics["accuracy"] = 1.0 if prediction == ground_truth else 0.0
            # For single-label, f1/precision/recall are same as accuracy
            metrics["f1_score"] = metrics["accuracy"]
            metrics["precision"] = metrics["accuracy"]
            metrics["recall"] = metrics["accuracy"]

        return metrics


class RegressionProblem(ProblemType):
    """Regression problem type (continuous output space)."""

    def __init__(
        self,
        output_dim: int = 1,
        output_range: Optional[tuple] = None,
        probabilistic: bool = False,
    ):
        """Initialize regression problem.

        Args:
            output_dim: Output dimensionality (1 for scalar)
            output_range: Optional (min, max) range for outputs
            probabilistic: Whether to output probability distributions
        """
        name = "regression"
        if output_dim > 1:
            name = "vector_regression"
        if probabilistic:
            name = "probabilistic_regression"

        super().__init__(
            name=name, description=f"Predict continuous values (dim={output_dim})"
        )

        self.output_dim = output_dim
        self.output_range = output_range
        self.probabilistic = probabilistic

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="mixed", validation_rules=[lambda x: isinstance(x, dict)]
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        validation_rules = []

        if self.output_dim == 1:
            validation_rules.append(lambda x: isinstance(x, (int, float)))
            if self.output_range:
                min_val, max_val = self.output_range
                validation_rules.append(lambda x: min_val <= float(x) <= max_val)
        else:
            validation_rules.append(
                lambda x: isinstance(x, list) and len(x) == self.output_dim
            )
            validation_rules.append(
                lambda x: all(isinstance(v, (int, float)) for v in x)
            )

        return OutputConstraints(
            format=OutputFormat.CONTINUOUS,
            cardinality=(
                Cardinality.SINGLE if self.output_dim == 1 else Cardinality.MULTIPLE
            ),
            domain=self.output_range or (-float("inf"), float("inf")),
            validation_rules=validation_rules,
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["mean_squared_error", "mean_absolute_error"],
            secondary_metrics=["r2_score", "explained_variance"],
            optimization_direction=OptimizationDirection.MINIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.output_dim == 1:
            range_str = (
                f" (between {self.output_range[0]} and {self.output_range[1]})"
                if self.output_range
                else ""
            )
            return f"""Predict the numerical value{range_str} for the following input:

Input: {{input}}

Predicted value:"""
        else:
            return f"""Predict {self.output_dim} numerical values for the following input:

Input: {{input}}

Predicted values (comma-separated):"""

    def parse_output(self, raw_output: str) -> Any:
        """Parse regression output."""
        output = raw_output.strip()

        if self.output_dim == 1:
            # Parse single value
            try:
                value = float(output.split()[0])
                if self.output_range:
                    min_val, max_val = self.output_range
                    value = max(min_val, min(value, max_val))  # Clip to range
                return value
            except (ValueError, IndexError):
                return 0.0
        else:
            # Parse multiple values
            try:
                values = [float(v.strip()) for v in output.split(",")][
                    : self.output_dim
                ]
                # Pad with zeros if not enough values
                while len(values) < self.output_dim:
                    values.append(0.0)
                return values
            except ValueError:
                return [0.0] * self.output_dim

    def format_input(self, input_data: Any) -> str:
        """Format input for regression."""
        if isinstance(input_data, dict):
            # Format as key-value pairs
            parts = []
            for key, value in input_data.items():
                parts.append(f"{key}: {value}")
            return ", ".join(parts)
        return str(input_data)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate regression prediction."""

        metrics = {}

        if self.output_dim == 1:
            pred = float(prediction) if isinstance(prediction, (int, float)) else 0.0
            true = (
                float(ground_truth) if isinstance(ground_truth, (int, float)) else 0.0
            )

            error = pred - true
            metrics["mean_squared_error"] = error**2
            metrics["mean_absolute_error"] = abs(error)
            metrics["r2_score"] = 1.0 - (error**2) / (true**2 + 1e-8)
            metrics["explained_variance"] = 1.0 - abs(error) / (abs(true) + 1e-8)
        else:
            # Vector regression metrics
            pred_vec = (
                prediction if isinstance(prediction, list) else [0.0] * self.output_dim
            )
            true_vec = (
                ground_truth
                if isinstance(ground_truth, list)
                else [0.0] * self.output_dim
            )

            mse = (
                sum((p - t) ** 2 for p, t in zip(pred_vec, true_vec)) / self.output_dim
            )
            mae = sum(abs(p - t) for p, t in zip(pred_vec, true_vec)) / self.output_dim

            metrics["mean_squared_error"] = mse
            metrics["mean_absolute_error"] = mae

            # Simple R2 for vectors
            ss_res = sum((p - t) ** 2 for p, t in zip(pred_vec, true_vec))
            ss_tot = sum(t**2 for t in true_vec) + 1e-8
            metrics["r2_score"] = 1.0 - ss_res / ss_tot
            metrics["explained_variance"] = 1.0 - mae / (
                sum(abs(t) for t in true_vec) / self.output_dim + 1e-8
            )

        return metrics


class SequenceGenerationProblem(ProblemType):
    """Sequence generation problem type."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        constrained: bool = False,
        constraints: Optional[List[str]] = None,
    ):
        """Initialize sequence generation problem.

        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            constrained: Whether generation is constrained
            constraints: List of constraints to apply
        """
        name = "constrained_generation" if constrained else "sequence_generation"

        super().__init__(
            name=name, description="Generate sequences with optional constraints"
        )

        self.min_length = min_length
        self.max_length = max_length
        self.constrained = constrained
        self.constraints = constraints or []

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="text", validation_rules=[lambda x: isinstance(x, (str, dict))]
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        validation_rules = [lambda x: isinstance(x, str)]

        if self.min_length:
            validation_rules.append(lambda x: len(x.split()) >= self.min_length)
        if self.max_length:
            validation_rules.append(lambda x: len(x.split()) <= self.max_length)

        return OutputConstraints(
            format=OutputFormat.SEQUENCE,
            cardinality=Cardinality.VARIABLE,
            validation_rules=validation_rules,
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["bleu_score", "rouge_score"],
            secondary_metrics=["length_ratio", "constraint_satisfaction"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        constraints_str = ""
        if self.constraints:
            constraints_str = "\n\nConstraints:\n" + "\n".join(
                f"- {c}" for c in self.constraints
            )

        length_str = ""
        if self.min_length and self.max_length:
            length_str = f" (between {self.min_length} and {self.max_length} words)"
        elif self.min_length:
            length_str = f" (at least {self.min_length} words)"
        elif self.max_length:
            length_str = f" (at most {self.max_length} words)"

        return f"""Generate text{length_str} based on the following input:

Input: {{input}}{constraints_str}

Generated text:"""

    def parse_output(self, raw_output: str) -> str:
        """Parse generation output."""
        return raw_output.strip()

    def format_input(self, input_data: Any) -> str:
        """Format input for generation."""
        if isinstance(input_data, dict):
            if "prompt" in input_data:
                return input_data["prompt"]
            elif "text" in input_data:
                return input_data["text"]
        return str(input_data)

    def evaluate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate generated sequence."""
        metrics = {}

        # Simple token overlap metrics (simplified BLEU/ROUGE)
        pred_tokens = set(prediction.lower().split())
        true_tokens = set(ground_truth.lower().split())

        if pred_tokens and true_tokens:
            overlap = len(pred_tokens & true_tokens)
            precision = overlap / len(pred_tokens)
            recall = overlap / len(true_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )

            metrics["bleu_score"] = precision  # Simplified BLEU
            metrics["rouge_score"] = f1  # Simplified ROUGE-L
        else:
            metrics["bleu_score"] = 0.0
            metrics["rouge_score"] = 0.0

        # Length ratio
        pred_len = len(prediction.split())
        true_len = len(ground_truth.split())
        metrics["length_ratio"] = pred_len / true_len if true_len > 0 else 0.0

        # Constraint satisfaction
        if self.constraints:
            satisfied = sum(
                1 for c in self.constraints if c.lower() in prediction.lower()
            )
            metrics["constraint_satisfaction"] = satisfied / len(self.constraints)
        else:
            metrics["constraint_satisfaction"] = 1.0

        return metrics


class InformationExtractionProblem(ProblemType):
    """Information extraction problem type."""

    def __init__(
        self, extraction_type: str = "entities", schema: Optional[Dict[str, Any]] = None
    ):
        """Initialize information extraction problem.

        Args:
            extraction_type: Type of extraction (entities, relations, slots)
            schema: Expected output schema
        """
        super().__init__(
            name=f"{extraction_type}_extraction",
            description=f"Extract {extraction_type} from text",
        )

        self.extraction_type = extraction_type
        self.schema = schema or {}

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="text",
            min_length=1,
            validation_rules=[lambda x: isinstance(x, (str, dict))],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        return OutputConstraints(
            format=OutputFormat.STRUCTURED,
            cardinality=Cardinality.VARIABLE,
            structure_schema=self.schema,
            validation_rules=[lambda x: isinstance(x, (dict, list))],
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=[
                "extraction_f1",
                "extraction_precision",
                "extraction_recall",
            ],
            secondary_metrics=["exact_match"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.extraction_type == "entities":
            return """Extract all named entities from the following text and categorize them:

Text: {text}

Entities (format as JSON):"""
        elif self.extraction_type == "relations":
            return """Extract relationships between entities in the following text:

Text: {text}

Relations (format as JSON with subject, relation, object):"""
        elif self.extraction_type == "slots":
            slots_str = (
                ", ".join(self.schema.keys()) if self.schema else "relevant information"
            )
            return f"""Extract the following information from the text: {slots_str}

Text: {{text}}

Extracted information (JSON):"""
        else:
            return """Extract structured information from the following text:

Text: {text}

Extracted information (JSON):"""

    def parse_output(self, raw_output: str) -> Any:
        """Parse extraction output."""
        import json

        output = raw_output.strip()

        # Try to parse as JSON
        try:
            # Find JSON block if wrapped in markdown
            if "```" in output:
                start = output.find("{")
                end = output.rfind("}") + 1
                if start >= 0 and end > start:
                    output = output[start:end]

            return json.loads(output)
        except json.JSONDecodeError:
            # Fallback: try to extract key-value pairs
            if self.extraction_type == "slots" and self.schema:
                extracted = {}
                for slot in self.schema:
                    # Simple pattern matching
                    import re

                    pattern = rf"{slot}[:\s]+([^,\n]+)"
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        extracted[slot] = match.group(1).strip()
                return extracted

            return {}

    def format_input(self, input_data: Any) -> str:
        """Format input for extraction."""
        if isinstance(input_data, dict) and "text" in input_data:
            return input_data["text"]
        return str(input_data)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate extraction results."""
        metrics = {}

        if isinstance(prediction, dict) and isinstance(ground_truth, dict):
            # Slot-based evaluation
            pred_keys = set(prediction.keys())
            true_keys = set(ground_truth.keys())

            correct = sum(
                1
                for k in pred_keys & true_keys
                if str(prediction[k]).lower() == str(ground_truth[k]).lower()
            )

            precision = correct / len(pred_keys) if pred_keys else 0.0
            recall = correct / len(true_keys) if true_keys else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            metrics["extraction_precision"] = precision
            metrics["extraction_recall"] = recall
            metrics["extraction_f1"] = f1
            metrics["exact_match"] = 1.0 if prediction == ground_truth else 0.0

        elif isinstance(prediction, list) and isinstance(ground_truth, list):
            # Entity/relation evaluation
            pred_set = {str(p) for p in prediction}
            true_set = {str(t) for t in ground_truth}

            correct = len(pred_set & true_set)
            precision = correct / len(pred_set) if pred_set else 0.0
            recall = correct / len(true_set) if true_set else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            metrics["extraction_precision"] = precision
            metrics["extraction_recall"] = recall
            metrics["extraction_f1"] = f1
            metrics["exact_match"] = 1.0 if pred_set == true_set else 0.0

        else:
            # Fallback
            metrics["extraction_precision"] = 0.0
            metrics["extraction_recall"] = 0.0
            metrics["extraction_f1"] = 0.0
            metrics["exact_match"] = 0.0

        return metrics


class QuestionAnsweringProblem(ProblemType):
    """Question answering problem type."""

    def __init__(
        self,
        qa_type: str = "open",
        with_context: bool = True,
        answer_type: str = "text",
    ):
        """Initialize QA problem.

        Args:
            qa_type: Type of QA (open, extractive, multiple_choice)
            with_context: Whether context is provided
            answer_type: Expected answer type (text, number, boolean, etc.)
        """
        name = f"{qa_type}_qa"
        if with_context:
            name += "_with_context"

        super().__init__(name=name, description=f"{qa_type} question answering")

        self.qa_type = qa_type
        self.with_context = with_context
        self.answer_type = answer_type

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        required_fields = ["question"]
        if self.with_context:
            required_fields.append("context")

        return InputConstraints(
            format="structured",
            required_fields=required_fields,
            validation_rules=[lambda x: isinstance(x, dict) and "question" in x],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        if self.answer_type == "boolean":
            return OutputConstraints(
                format=OutputFormat.DISCRETE,
                domain=["yes", "no", "true", "false"],
                validation_rules=[
                    lambda x: str(x).lower() in ["yes", "no", "true", "false"]
                ],
            )
        elif self.answer_type == "number":
            return OutputConstraints(
                format=OutputFormat.CONTINUOUS,
                validation_rules=[
                    lambda x: isinstance(x, (int, float))
                    or x.replace(".", "").replace("-", "").isdigit()
                ],
            )
        else:
            return OutputConstraints(
                format=OutputFormat.SEQUENCE, cardinality=Cardinality.SINGLE
            )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        if self.qa_type == "extractive":
            return EvaluationConstraints(
                primary_metrics=["exact_match", "f1_score"],
                secondary_metrics=["partial_match"],
                optimization_direction=OptimizationDirection.MAXIMIZE,
            )
        else:
            return EvaluationConstraints(
                primary_metrics=["answer_relevance", "answer_accuracy"],
                secondary_metrics=["answer_completeness"],
                optimization_direction=OptimizationDirection.MAXIMIZE,
            )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.with_context:
            return """Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:"""
        else:
            return """Answer the following question:

Question: {question}

Answer:"""

    def parse_output(self, raw_output: str) -> Any:
        """Parse QA output."""
        output = raw_output.strip()

        if self.answer_type == "boolean":
            output_lower = output.lower()
            if "yes" in output_lower or "true" in output_lower:
                return "yes"
            elif "no" in output_lower or "false" in output_lower:
                return "no"
            return output

        elif self.answer_type == "number":
            # Extract first number from output
            import re

            numbers = re.findall(r"-?\d+\.?\d*", output)
            if numbers:
                try:
                    return float(numbers[0])
                except ValueError:
                    pass
            return output

        else:
            # Clean up answer
            if output.startswith("Answer:"):
                output = output[7:].strip()
            return output

    def format_input(self, input_data: Any) -> str:
        """Format input for QA."""
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
            if self.with_context and "context" in input_data:
                return f"Context: {input_data['context']}\nQuestion: {question}"
            return question
        return str(input_data)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate QA prediction."""
        metrics = {}

        pred_str = str(prediction).lower().strip()
        true_str = str(ground_truth).lower().strip()

        # Exact match
        metrics["exact_match"] = 1.0 if pred_str == true_str else 0.0

        # Token-level F1
        pred_tokens = set(pred_str.split())
        true_tokens = set(true_str.split())

        if pred_tokens and true_tokens:
            overlap = len(pred_tokens & true_tokens)
            precision = overlap / len(pred_tokens)
            recall = overlap / len(true_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            metrics["f1_score"] = f1
        else:
            metrics["f1_score"] = metrics["exact_match"]

        # Answer relevance (simplified - based on keyword overlap)
        metrics["answer_relevance"] = metrics["f1_score"]

        # Answer accuracy (same as F1 for now)
        metrics["answer_accuracy"] = metrics["f1_score"]

        # Answer completeness (based on length ratio)
        if len(true_str) > 0:
            length_ratio = min(len(pred_str) / len(true_str), 1.0)
            metrics["answer_completeness"] = length_ratio
        else:
            metrics["answer_completeness"] = 1.0 if len(pred_str) > 0 else 0.0

        # Partial match
        metrics["partial_match"] = (
            1.0 if any(token in pred_tokens for token in true_tokens) else 0.0
        )

        return metrics


class SummarizationProblem(ProblemType):
    """Summarization problem type."""

    def __init__(
        self,
        summary_type: str = "abstractive",
        target_length: Optional[int] = None,
        compression_ratio: Optional[float] = None,
    ):
        """Initialize summarization problem.

        Args:
            summary_type: Type of summary (abstractive, extractive)
            target_length: Target length in words/tokens
            compression_ratio: Target compression ratio (0.1 = 10% of original)
        """
        super().__init__(
            name="summarization",
            description="Condense text while preserving key information",
        )

        self.summary_type = summary_type
        self.target_length = target_length
        self.compression_ratio = compression_ratio or 0.3

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="text",
            min_length=100,  # Need enough text to summarize
            validation_rules=[
                lambda x: isinstance(x, (str, dict)),
                lambda x: len(str(x)) > 50,  # Meaningful content
            ],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        validation_rules = [lambda x: isinstance(x, str), lambda x: len(x.strip()) > 0]

        if self.target_length:
            validation_rules.append(
                lambda x: len(x.split())
                <= self.target_length * 1.2  # Allow 20% variance
            )

        return OutputConstraints(
            format=OutputFormat.SEQUENCE,
            cardinality=Cardinality.SINGLE,
            validation_rules=validation_rules,
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["rouge_1", "rouge_2", "rouge_l"],
            secondary_metrics=[
                "bleu_score",
                "compression_ratio",
                "information_retention",
            ],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        length_str = ""
        if self.target_length:
            length_str = f" (approximately {self.target_length} words)"
        elif self.compression_ratio:
            length_str = f" (approximately {int(self.compression_ratio * 100)}% of original length)"

        if self.summary_type == "extractive":
            return f"""Extract the most important sentences{length_str} from the following text:

Text: {{document}}

Key sentences:"""
        else:
            return f"""Summarize the following text{length_str}:

Text: {{document}}

Summary:"""

    def parse_output(self, raw_output: str) -> str:
        """Parse summary output."""
        return raw_output.strip()

    def format_input(self, input_data: Any) -> str:
        """Format input for summarization."""
        if isinstance(input_data, dict):
            return input_data.get("document", input_data.get("text", str(input_data)))
        return str(input_data)

    def evaluate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate summary quality."""
        metrics = {}

        # Simplified ROUGE scores (based on word overlap)
        pred_words = set(prediction.lower().split())
        true_words = set(ground_truth.lower().split())

        if pred_words and true_words:
            overlap = len(pred_words & true_words)
            precision = overlap / len(pred_words)
            recall = overlap / len(true_words)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )

            metrics["rouge_1"] = f1  # Simplified ROUGE-1
            metrics["rouge_l"] = f1  # Simplified ROUGE-L

            # Bigram ROUGE-2 (simplified)
            pred_bigrams = set(
                zip(prediction.lower().split()[:-1], prediction.lower().split()[1:])
            )
            true_bigrams = set(
                zip(ground_truth.lower().split()[:-1], ground_truth.lower().split()[1:])
            )
            if pred_bigrams and true_bigrams:
                bigram_overlap = len(pred_bigrams & true_bigrams)
                bigram_precision = bigram_overlap / len(pred_bigrams)
                bigram_recall = bigram_overlap / len(true_bigrams)
                metrics["rouge_2"] = (
                    2
                    * bigram_precision
                    * bigram_recall
                    / (bigram_precision + bigram_recall)
                    if bigram_precision + bigram_recall > 0
                    else 0
                )
            else:
                metrics["rouge_2"] = 0.0
        else:
            metrics["rouge_1"] = 0.0
            metrics["rouge_2"] = 0.0
            metrics["rouge_l"] = 0.0

        # BLEU score (simplified)
        metrics["bleu_score"] = metrics["rouge_1"]  # Simplified

        # Compression ratio
        original_length = len(ground_truth.split())
        summary_length = len(prediction.split())
        metrics["compression_ratio"] = (
            summary_length / original_length if original_length > 0 else 0.0
        )

        # Information retention (simplified - based on key word coverage)
        metrics["information_retention"] = recall  # How much of original is retained

        return metrics


class RankingRetrievalProblem(ProblemType):
    """Ranking and retrieval problem type."""

    def __init__(
        self,
        task_type: str = "ranking",
        top_k: int = 10,
        similarity_metric: str = "cosine",
    ):
        """Initialize ranking/retrieval problem.

        Args:
            task_type: Type of task (ranking, retrieval, recommendation)
            top_k: Number of top results to return
            similarity_metric: Metric for similarity (cosine, euclidean, etc.)
        """
        super().__init__(
            name="ranking_retrieval",
            description="Order or retrieve items based on relevance",
        )

        self.task_type = task_type
        self.top_k = top_k
        self.similarity_metric = similarity_metric

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="structured",
            required_fields=["query", "candidates"],
            validation_rules=[
                lambda x: isinstance(x, dict),
                lambda x: "query" in x and "candidates" in x,
                lambda x: isinstance(x.get("candidates"), list),
            ],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        return OutputConstraints(
            format=OutputFormat.STRUCTURED,
            cardinality=Cardinality.MULTIPLE,
            structure_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": ["string", "object"]},
                        "score": {"type": "number"},
                        "rank": {"type": "integer"},
                    },
                },
            },
            validation_rules=[
                lambda x: isinstance(x, list),
                lambda x: all(isinstance(item, (dict, tuple)) for item in x),
                lambda x: len(x) <= self.top_k,
            ],
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["ndcg", "map", "mrr"],
            secondary_metrics=["precision_at_k", "recall_at_k", "hit_rate"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.task_type == "retrieval":
            return f"""Find the top {self.top_k} most relevant items for the query:

Query: {{query}}

Candidates:
{{candidates}}

Relevant items (ranked by relevance):"""
        elif self.task_type == "recommendation":
            return f"""Recommend the top {self.top_k} items based on:

User preferences: {{query}}

Available items:
{{candidates}}

Recommendations:"""
        else:  # ranking
            return f"""Rank the following items by relevance to the query (top {self.top_k}):

Query: {{query}}

Items to rank:
{{candidates}}

Ranked results:"""

    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse ranking output."""
        import json

        output = raw_output.strip()

        # Try to parse as JSON array
        try:
            if "```" in output:
                start = output.find("[")
                end = output.rfind("]") + 1
                if start >= 0 and end > start:
                    output = output[start:end]

            results = json.loads(output)
            if isinstance(results, list):
                return results[: self.top_k]
        except json.JSONDecodeError:
            pass

        # Fallback: parse as numbered list
        results = []
        lines = output.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and i < self.top_k:
                # Remove numbering if present
                import re

                line = re.sub(r"^\d+\.?\s*", "", line)
                results.append(
                    {
                        "item": line,
                        "score": 1.0 - (i * 0.1),  # Decreasing score
                        "rank": i + 1,
                    }
                )

        return results

    def format_input(self, input_data: Any) -> str:
        """Format input for ranking."""
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            candidates = input_data.get("candidates", [])

            # Format candidates list
            if isinstance(candidates, list):
                formatted_candidates = "\n".join(
                    f"{i+1}. {item}" for i, item in enumerate(candidates)
                )
                return f"Query: {query}\n\nCandidates:\n{formatted_candidates}"

        return str(input_data)

    def evaluate(
        self, prediction: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Evaluate ranking quality."""
        metrics = {}

        # Convert to ranked lists
        pred_items = []
        true_items = []

        for item in prediction:
            if isinstance(item, dict):
                pred_items.append(item.get("item", item))
            else:
                pred_items.append(item)

        for item in ground_truth:
            if isinstance(item, dict):
                true_items.append(item.get("item", item))
            else:
                true_items.append(item)

        # Calculate metrics
        k = min(self.top_k, len(pred_items), len(true_items))

        if k == 0:
            metrics["ndcg"] = 0.0
            metrics["map"] = 0.0
            metrics["mrr"] = 0.0
            metrics["precision_at_k"] = 0.0
            metrics["recall_at_k"] = 0.0
            metrics["hit_rate"] = 0.0
            return metrics

        # Precision@k
        correct_at_k = sum(1 for item in pred_items[:k] if item in true_items[:k])
        metrics["precision_at_k"] = correct_at_k / k

        # Recall@k
        metrics["recall_at_k"] = correct_at_k / len(true_items) if true_items else 0.0

        # Hit Rate
        metrics["hit_rate"] = (
            1.0 if any(item in true_items for item in pred_items[:k]) else 0.0
        )

        # MRR (Mean Reciprocal Rank)
        for i, item in enumerate(pred_items):
            if item in true_items:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0.0

        # Simplified NDCG and MAP
        metrics["ndcg"] = metrics["precision_at_k"]  # Simplified
        metrics["map"] = metrics["precision_at_k"]  # Simplified

        return metrics


class TranslationTransformationProblem(ProblemType):
    """Translation and transformation problem type."""

    def __init__(
        self,
        transformation_type: str = "translation",
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        preserve_meaning: bool = True,
    ):
        """Initialize translation/transformation problem.

        Args:
            transformation_type: Type (translation, style_transfer, format_conversion)
            source_format: Source language/style/format
            target_format: Target language/style/format
            preserve_meaning: Whether to preserve semantic meaning
        """
        super().__init__(
            name="translation_transformation",
            description="Convert text from one form to another",
        )

        self.transformation_type = transformation_type
        self.source_format = source_format
        self.target_format = target_format
        self.preserve_meaning = preserve_meaning

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        required_fields = ["text"]
        if self.transformation_type == "translation":
            required_fields.extend(["source_lang", "target_lang"])

        return InputConstraints(
            format="structured",
            required_fields=required_fields,
            validation_rules=[
                lambda x: isinstance(x, (str, dict)),
                lambda x: len(str(x)) > 0,
            ],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        return OutputConstraints(
            format=OutputFormat.SEQUENCE,
            cardinality=Cardinality.SINGLE,
            validation_rules=[
                lambda x: isinstance(x, str),
                lambda x: len(x.strip()) > 0,
            ],
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["bleu_score", "semantic_similarity"],
            secondary_metrics=["fluency", "adequacy", "style_preservation"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.transformation_type == "translation":
            return """Translate the following text from {source_lang} to {target_lang}:

Text: {text}

Translation:"""
        elif self.transformation_type == "style_transfer":
            return f"""Transform the following text to {self.target_format or 'target'} style:

Original text: {{text}}

Transformed text:"""
        elif self.transformation_type == "format_conversion":
            return f"""Convert the following from {self.source_format or 'source'} to {self.target_format or 'target'} format:

Input: {{text}}

Output:"""
        else:
            return """Transform the following text as requested:

Input: {text}
Requirement: {transformation_type}

Output:"""

    def parse_output(self, raw_output: str) -> str:
        """Parse transformation output."""
        return raw_output.strip()

    def format_input(self, input_data: Any) -> str:
        """Format input for transformation."""
        if isinstance(input_data, dict):
            text = input_data.get("text", "")
            if "source_lang" in input_data and "target_lang" in input_data:
                return f"Translate from {input_data['source_lang']} to {input_data['target_lang']}: {text}"
            return text
        return str(input_data)

    def evaluate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate transformation quality."""
        metrics = {}

        # BLEU score (simplified)
        pred_words = prediction.lower().split()
        true_words = ground_truth.lower().split()

        if pred_words and true_words:
            # Unigram precision
            matches = sum(1 for word in pred_words if word in true_words)
            precision = matches / len(pred_words)

            # Length penalty
            length_ratio = len(pred_words) / len(true_words)
            brevity_penalty = min(1.0, length_ratio)

            metrics["bleu_score"] = brevity_penalty * precision
        else:
            metrics["bleu_score"] = 0.0

        # Semantic similarity (simplified - word overlap with fuzzy matching)
        if pred_words and true_words:
            pred_set = set(pred_words)
            true_set = set(true_words)

            # For each predicted word, find if there's a similar word in true set
            matches = 0
            for pred_word in pred_set:
                for true_word in true_set:
                    # Exact match or fuzzy match (word contained in another)
                    if (
                        pred_word == true_word
                        or pred_word in true_word
                        or true_word in pred_word
                    ):
                        matches += 1
                        break

            # Use the higher of two ratios for more generous scoring
            ratio1 = matches / len(pred_set)
            ratio2 = matches / len(true_set)
            metrics["semantic_similarity"] = max(ratio1, ratio2)
        else:
            metrics["semantic_similarity"] = 0.0

        # Fluency (simplified - based on length consistency)
        metrics["fluency"] = (
            min(len(pred_words), len(true_words))
            / max(len(pred_words), len(true_words))
            if pred_words and true_words
            else 0.0
        )

        # Adequacy (how much information is preserved)
        metrics["adequacy"] = metrics["semantic_similarity"]

        # Style preservation (simplified)
        metrics["style_preservation"] = 1.0 if self.preserve_meaning else 0.5

        return metrics


class ReasoningProblem(ProblemType):
    """Reasoning problem type."""

    def __init__(
        self,
        reasoning_type: str = "logical",
        requires_steps: bool = True,
        domain: Optional[str] = None,
    ):
        """Initialize reasoning problem.

        Args:
            reasoning_type: Type of reasoning (logical, mathematical, causal, common_sense)
            requires_steps: Whether step-by-step reasoning is required
            domain: Specific domain (math, science, etc.)
        """
        super().__init__(
            name="reasoning",
            description="Solve problems requiring logical or mathematical reasoning",
        )

        self.reasoning_type = reasoning_type
        self.requires_steps = requires_steps
        self.domain = domain

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="structured",
            required_fields=["problem"],
            optional_fields=["context", "constraints"],
            validation_rules=[
                lambda x: isinstance(x, (str, dict)),
                lambda x: len(str(x)) > 10,  # Non-trivial problem
            ],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        if self.requires_steps:
            schema = {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": ["string", "number", "object"]},
                    "explanation": {"type": "string"},
                },
            }
            return OutputConstraints(
                format=OutputFormat.STRUCTURED,
                cardinality=Cardinality.SINGLE,
                structure_schema=schema,
            )
        else:
            return OutputConstraints(
                format=OutputFormat.MIXED, cardinality=Cardinality.SINGLE
            )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        return EvaluationConstraints(
            primary_metrics=["answer_accuracy", "reasoning_validity"],
            secondary_metrics=[
                "step_correctness",
                "logic_consistency",
                "solution_efficiency",
            ],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        step_instruction = (
            " Show your step-by-step reasoning." if self.requires_steps else ""
        )

        if self.reasoning_type == "mathematical":
            return f"""Solve the following mathematical problem.{step_instruction}

Problem: {{problem}}

Solution:"""
        elif self.reasoning_type == "logical":
            return f"""Solve the following logical reasoning problem.{step_instruction}

Problem: {{problem}}
{{context}}

Solution:"""
        elif self.reasoning_type == "causal":
            return f"""Analyze the cause-and-effect relationships.{step_instruction}

Scenario: {{problem}}

Analysis:"""
        else:  # common_sense
            return f"""Apply common sense reasoning to solve this problem.{step_instruction}

Situation: {{problem}}

Reasoning:"""

    def parse_output(self, raw_output: str) -> Any:
        """Parse reasoning output."""
        import json

        output = raw_output.strip()

        if self.requires_steps:
            # Try to parse as structured output
            try:
                if "```" in output:
                    start = output.find("{")
                    end = output.rfind("}") + 1
                    if start >= 0 and end > start:
                        output = output[start:end]

                return json.loads(output)
            except json.JSONDecodeError:
                # Fallback: extract steps and answer
                lines = output.split("\n")
                steps = []
                answer = ""

                in_steps = False
                for line in lines:
                    line = line.strip()
                    if "step" in line.lower() or line.startswith(("1.", "2.", "3.")):
                        in_steps = True

                    if in_steps and line:
                        steps.append(line)
                    elif "answer" in line.lower() or "therefore" in line.lower():
                        answer = line.split(":", 1)[-1].strip() if ":" in line else line

                return {
                    "steps": steps,
                    "answer": answer or output,
                    "explanation": output,
                }
        else:
            # Just return the answer
            return output

    def format_input(self, input_data: Any) -> str:
        """Format input for reasoning."""
        if isinstance(input_data, dict):
            problem = input_data.get("problem", "")
            context = input_data.get("context", "")
            constraints = input_data.get("constraints", [])

            formatted = f"Problem: {problem}"
            if context:
                formatted += f"\nContext: {context}"
            if constraints:
                formatted += f"\nConstraints: {', '.join(constraints)}"

            return formatted

        return str(input_data)

    def evaluate(self, prediction: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate reasoning quality."""
        metrics = {}

        # Extract answers for comparison
        pred_answer = prediction
        true_answer = ground_truth

        if isinstance(prediction, dict):
            pred_answer = prediction.get("answer", prediction)
            pred_steps = prediction.get("steps", [])
        else:
            pred_steps = []

        if isinstance(ground_truth, dict):
            true_answer = ground_truth.get("answer", ground_truth)
            true_steps = ground_truth.get("steps", [])
        else:
            true_steps = []

        # Answer accuracy
        pred_str = str(pred_answer).lower().strip()
        true_str = str(true_answer).lower().strip()

        # For numerical answers
        try:
            pred_num = float(pred_str)
            true_num = float(true_str)
            metrics["answer_accuracy"] = (
                1.0 if abs(pred_num - true_num) < 0.001 else 0.0
            )
        except ValueError:
            # For text answers
            metrics["answer_accuracy"] = 1.0 if pred_str == true_str else 0.0

        # Reasoning validity (simplified - based on having steps)
        if self.requires_steps:
            metrics["reasoning_validity"] = 1.0 if len(pred_steps) > 0 else 0.0
        else:
            metrics["reasoning_validity"] = metrics["answer_accuracy"]

        # Step correctness (simplified - based on step count similarity)
        if pred_steps and true_steps:
            step_ratio = min(len(pred_steps), len(true_steps)) / max(
                len(pred_steps), len(true_steps)
            )
            metrics["step_correctness"] = step_ratio
        else:
            metrics["step_correctness"] = 0.0 if self.requires_steps else 1.0

        # Logic consistency (simplified)
        metrics["logic_consistency"] = metrics["reasoning_validity"]

        # Solution efficiency (fewer steps is better)
        if pred_steps:
            efficiency = 1.0 / (1.0 + len(pred_steps) / 10.0)  # Decay with more steps
            metrics["solution_efficiency"] = efficiency
        else:
            metrics["solution_efficiency"] = 0.5

        return metrics


class CodeGenerationProblem(ProblemType):
    """Code generation problem type."""

    def __init__(
        self,
        target_language: str = "python",
        code_type: str = "function",  # function, class, script, query
        execution_environment: Optional[Dict[str, Any]] = None,
        syntax_constraints: Optional[List[str]] = None,
    ):
        """Initialize code generation problem.

        Args:
            target_language: Programming language to generate
            code_type: Type of code to generate
            execution_environment: Environment for code execution (e.g., database schema for SQL)
            syntax_constraints: Specific syntax requirements
        """
        super().__init__(
            name="code_generation",
            description=f"Generate {target_language} {code_type} from natural language",
        )

        self.target_language = target_language
        self.code_type = code_type
        self.execution_environment = execution_environment or {}
        self.syntax_constraints = syntax_constraints or []

    def get_input_constraints(self) -> InputConstraints:
        """Get input constraints."""
        return InputConstraints(
            format="structured",
            required_fields=["description"],
            optional_fields=["context", "requirements", "examples"],
            validation_rules=[
                lambda x: isinstance(x, (str, dict)),
                lambda x: bool(x.get("description", x) if isinstance(x, dict) else x),
            ],
        )

    def get_output_constraints(self) -> OutputConstraints:
        """Get output constraints."""
        return OutputConstraints(
            format=OutputFormat.SEQUENCE,
            cardinality=Cardinality.SINGLE,
            validation_rules=[
                lambda x: isinstance(x, str),
                lambda x: len(x.strip()) > 0,
            ],
        )

    def get_evaluation_constraints(self) -> EvaluationConstraints:
        """Get evaluation constraints."""
        primary_metrics = ["exact_match", "syntax_validity"]

        # Add language-specific metrics
        if self.target_language.lower() in ["sql", "mysql", "postgresql"]:
            primary_metrics.extend(["execution_match", "query_efficiency"])
        else:
            primary_metrics.extend(["execution_match", "semantic_similarity"])

        return EvaluationConstraints(
            primary_metrics=primary_metrics,
            secondary_metrics=["functional_correctness", "code_quality"],
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )

    def generate_prompt_template(self) -> str:
        """Generate prompt template."""
        if self.target_language.lower() in ["sql", "mysql", "postgresql"]:
            return """Convert the following natural language query to SQL:

Description: {description}
{f"Database Schema: {context}" if "context" in "{}" else ""}

SQL Query:"""
        elif self.code_type == "function":
            return f"""Generate a {self.target_language} function based on:

Description: {{description}}
{{f"Requirements: {{requirements}}" if "requirements" in "{{}}" else ""}}

{self.target_language.capitalize()} code:"""
        else:
            return f"""Generate {self.target_language} {self.code_type}:

Description: {{description}}
{{f"Context: {{context}}" if "context" in "{{}}" else ""}}

Code:"""

    def parse_output(self, output: str) -> str:
        """Parse and clean generated code."""
        # Remove markdown code blocks if present
        if "```" in output:
            lines = output.split("\n")
            code_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)

            return "\n".join(code_lines).strip()

        return output.strip()

    def format_input(self, input_data: Any) -> str:
        """Format input data for code generation."""
        if isinstance(input_data, dict):
            description = input_data.get("description", "")
            context = input_data.get("context", "")
            requirements = input_data.get("requirements", "")
            examples = input_data.get("examples", "")

            formatted = description
            if context:
                formatted += f"\n\nContext: {context}"
            if requirements:
                formatted += f"\n\nRequirements: {requirements}"
            if examples:
                formatted += f"\n\nExamples: {examples}"
            return formatted
        return str(input_data)

    def evaluate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate code generation metrics."""
        metrics = {}

        # Exact match
        metrics["exact_match"] = float(prediction.strip() == ground_truth.strip())

        # Normalized similarity (ignoring whitespace differences)
        pred_normalized = " ".join(prediction.split())
        truth_normalized = " ".join(ground_truth.split())
        metrics["normalized_match"] = float(pred_normalized == truth_normalized)

        # Token-level similarity (with enhanced fuzzy matching for code)
        pred_tokens = prediction.split()
        truth_tokens = ground_truth.split()

        if pred_tokens or truth_tokens:
            # For code, use more generous matching
            pred_set = set(pred_tokens)
            truth_set = set(truth_tokens)

            # Direct matches
            direct_matches = len(pred_set & truth_set)

            # Fuzzy matches for code structure
            fuzzy_matches = 0
            for pred_token in pred_set - truth_set:
                for truth_token in truth_set - pred_set:
                    # Structure similarity (like "add(a," vs "add(x,")
                    if (
                        len(pred_token) > 3
                        and len(truth_token) > 3
                        and (
                            pred_token[:3] == truth_token[:3]  # Same start
                            or pred_token[-3:] == truth_token[-3:]
                        )
                    ):  # Same end
                        fuzzy_matches += 1
                        break

            total_matches = direct_matches + fuzzy_matches * 0.8  # Weight fuzzy matches

            # Use F1-like scoring
            precision = total_matches / len(pred_set) if pred_set else 0
            recall = total_matches / len(truth_set) if truth_set else 0

            if precision + recall > 0:
                metrics["token_similarity"] = (
                    2 * precision * recall / (precision + recall)
                )
            else:
                metrics["token_similarity"] = 0.0
        else:
            metrics["token_similarity"] = 1.0

        # Syntax validity (basic check - would need language-specific parsers)
        metrics["syntax_validity"] = (
            1.0  # Placeholder - requires language-specific validation
        )

        # Execution match (placeholder - requires execution environment)
        metrics["execution_match"] = 0.0  # Requires custom implementation

        # Semantic similarity (placeholder - requires deeper analysis)
        metrics["semantic_similarity"] = metrics["token_similarity"]  # Simplified

        return metrics


# Problem Type Registry
_PROBLEM_TYPE_REGISTRY = {
    "classification": ClassificationProblem,
    "regression": RegressionProblem,
    "generation": SequenceGenerationProblem,
    "extraction": InformationExtractionProblem,
    "information_extraction": InformationExtractionProblem,  # Alias
    "question_answering": QuestionAnsweringProblem,
    "summarization": SummarizationProblem,
    "ranking_retrieval": RankingRetrievalProblem,
    "translation_transformation": TranslationTransformationProblem,
    "reasoning": ReasoningProblem,
    "code_generation": CodeGenerationProblem,
}


def register_problem_type(name: str, problem_class: type) -> None:
    """Register a custom problem type.

    Args:
        name: Name for the problem type
        problem_class: Class that extends ProblemType
    """
    if not issubclass(problem_class, ProblemType):
        raise ValueError(f"{problem_class} must extend ProblemType")
    _PROBLEM_TYPE_REGISTRY[name] = problem_class


def get_problem_type(name: str, **kwargs) -> ProblemType:
    """Get a problem type instance by name.

    Args:
        name: Name of the problem type
        **kwargs: Arguments to pass to problem type constructor

    Returns:
        Instance of the requested problem type
    """
    if name not in _PROBLEM_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown problem type: {name}. Available: {list(_PROBLEM_TYPE_REGISTRY.keys())}"
        )

    problem_class = _PROBLEM_TYPE_REGISTRY[name]
    return problem_class(**kwargs)


def list_problem_types() -> List[str]:
    """List all available problem types."""
    return list(_PROBLEM_TYPE_REGISTRY.keys())
