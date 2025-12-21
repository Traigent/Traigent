"""Evaluation dataset format for Haystack pipeline optimization.

This module defines the evaluation dataset format used by the optimizer
to run pipelines against test cases and compute quality metrics.

Example usage:
    from traigent.integrations.haystack import EvaluationDataset

    # Simple format: list of dicts
    data = [
        {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence..."},
        {"input": {"query": "Define ML"}, "expected": "Machine Learning..."},
    ]
    dataset = EvaluationDataset.from_dicts(data)

    # Use with optimizer (Story 3.2+)
    for example in dataset:
        output = pipeline.run(**example.input)
        score = metric(output, example.expected)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.utils.exceptions import DatasetValidationError

if TYPE_CHECKING:
    from traigent.evaluators.base import Dataset as CoreDataset


@dataclass
class EvaluationExample:
    """A single evaluation example.

    Attributes:
        input: Pipeline input as a dict (kwargs to pipeline.run())
        expected: Expected output for comparison (format depends on metric)
        metadata: Optional metadata for logging/debugging
        id: Optional identifier for this example
    """

    input: dict[str, Any]
    expected: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        id_str = f", id='{self.id}'" if self.id else ""
        return f"EvaluationExample(input={self.input!r}, expected={self.expected!r}{id_str})"


@dataclass
class EvaluationDataset:
    """Collection of evaluation examples for pipeline optimization.

    EvaluationDataset wraps a list of EvaluationExample objects and provides
    sequence-like access (iteration, indexing, length).

    Example usage:
        # From list of dicts (simple)
        data = [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence..."},
            {"input": {"query": "Define ML"}, "expected": "Machine Learning..."},
        ]
        dataset = EvaluationDataset.from_dicts(data)

        # From EvaluationExample objects (explicit)
        dataset = EvaluationDataset(examples=[
            EvaluationExample(input={"query": "What?"}, expected="Answer..."),
        ])

        # Iteration
        for example in dataset:
            print(example.input, example.expected)

        # Indexing
        first = dataset[0]
        print(len(dataset))
    """

    examples: list[EvaluationExample] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)

    def __getitem__(self, index: int) -> EvaluationExample:
        """Get example by index."""
        return self.examples[index]

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return f"EvaluationDataset(examples={len(self.examples)})"

    def to_core_dataset(self) -> CoreDataset:
        """Convert to core Dataset type for orchestrator compatibility.

        This method bridges the Haystack-specific EvaluationDataset with
        the core Traigent evaluator infrastructure, enabling HaystackEvaluator
        to work with the existing OptimizationOrchestrator and TrialLifecycle.

        Returns:
            A traigent.evaluators.base.Dataset instance with converted examples.

        Example:
            >>> haystack_dataset = EvaluationDataset.from_dicts([...])
            >>> core_dataset = haystack_dataset.to_core_dataset()
            >>> # Now usable with BaseEvaluator.evaluate()
        """
        from traigent.evaluators.base import Dataset as CoreDataset
        from traigent.evaluators.base import EvaluationExample as CoreExample

        core_examples = [
            CoreExample(
                input_data=ex.input,
                expected_output=ex.expected,
                metadata=ex.metadata,
            )
            for ex in self.examples
        ]
        return CoreDataset(examples=core_examples)

    @classmethod
    def from_core_dataset(cls, core_dataset: CoreDataset) -> EvaluationDataset:
        """Create EvaluationDataset from a core Dataset.

        This method bridges the core Traigent Dataset with Haystack-specific
        EvaluationDataset, enabling callers to pass core Datasets to
        HaystackEvaluator.evaluate().

        Args:
            core_dataset: A traigent.evaluators.base.Dataset instance.

        Returns:
            EvaluationDataset instance with converted examples.

        Example:
            >>> from traigent.evaluators.base import Dataset, EvaluationExample
            >>> core = Dataset(examples=[EvaluationExample(input_data={...}, ...)])
            >>> haystack_dataset = EvaluationDataset.from_core_dataset(core)
        """
        examples = [
            EvaluationExample(
                input=ex.input_data,
                expected=ex.expected_output,
                metadata=getattr(ex, "metadata", None) or {},
                id=getattr(ex, "id", None),
            )
            for ex in core_dataset.examples
        ]
        return cls(examples=examples)

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> EvaluationDataset:
        """Create EvaluationDataset from list of dicts.

        This is the primary factory method for creating datasets from
        user-provided data. It validates the format and converts to
        EvaluationExample objects.

        Args:
            data: List of dicts with 'input' and 'expected' keys.
                  Optional keys: 'metadata', 'id'.

        Returns:
            EvaluationDataset instance with validated examples.

        Raises:
            ValueError: If dataset is empty.
            DatasetValidationError: If entries are missing required keys
                or have invalid types.

        Example:
            data = [
                {"input": {"query": "Q1"}, "expected": "A1"},
                {"input": {"query": "Q2"}, "expected": "A2", "id": "test-002"},
            ]
            dataset = EvaluationDataset.from_dicts(data)
        """
        validate_dataset(data)

        examples = []
        for entry in data:
            examples.append(
                EvaluationExample(
                    input=entry["input"],
                    expected=entry["expected"],
                    metadata=entry.get("metadata", {}),
                    id=entry.get("id"),
                )
            )

        return cls(examples=examples)


def validate_dataset(data: list[dict[str, Any]]) -> None:
    """Validate evaluation dataset format.

    Checks that the dataset is non-empty and each entry has the required
    'input' and 'expected' keys with correct types.

    Args:
        data: Raw dataset as list of dicts.

    Raises:
        ValueError: If dataset is empty.
        DatasetValidationError: If entries are missing required keys
            or have invalid types. Error details include:
            - index: The problematic entry index
            - missing_key: The key that was missing (if applicable)
            - entry: The problematic entry data (if applicable)

    Example:
        >>> validate_dataset([{"input": {"q": "test"}, "expected": "answer"}])
        # No exception raised

        >>> validate_dataset([])
        ValueError: Evaluation dataset cannot be empty...

        >>> validate_dataset([{"expected": "answer"}])
        DatasetValidationError: Entry at index 0 is missing required key 'input'...
    """
    if not data:
        raise ValueError(
            "Evaluation dataset cannot be empty. "
            "Provide at least one example with 'input' and 'expected' keys."
        )

    for i, entry in enumerate(data):
        _validate_entry(i, entry)


def _validate_entry(index: int, entry: Any) -> None:
    """Validate a single dataset entry.

    Args:
        index: Entry index for error reporting.
        entry: The entry to validate.

    Raises:
        DatasetValidationError: If entry is invalid.
    """
    # Check entry is a dict
    if not isinstance(entry, dict):
        raise DatasetValidationError(
            f"Entry at index {index} must be a dict, got {type(entry).__name__}",
            details={"index": index, "type": type(entry).__name__},
        )

    # Check required 'input' key
    if "input" not in entry:
        raise DatasetValidationError(
            f"Entry at index {index} is missing required key 'input'. "
            f"Each entry must have 'input' and 'expected' keys.",
            details={"index": index, "entry": entry, "missing_key": "input"},
        )

    # Check required 'expected' key
    if "expected" not in entry:
        raise DatasetValidationError(
            f"Entry at index {index} is missing required key 'expected'. "
            f"Each entry must have 'input' and 'expected' keys.",
            details={"index": index, "entry": entry, "missing_key": "expected"},
        )

    # Validate 'input' is a dict (pipeline inputs are kwargs)
    if not isinstance(entry["input"], dict):
        raise DatasetValidationError(
            f"Entry at index {index}: 'input' must be a dict (pipeline kwargs), "
            f"got {type(entry['input']).__name__}",
            details={
                "index": index,
                "input_type": type(entry["input"]).__name__,
                "hint": "Use {'input': {'query': 'your query'}, 'expected': ...}",
            },
        )
