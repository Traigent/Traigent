"""Dataset Converter for Traigent SDK and OptiGen Backend Integration.

This module provides utilities to convert between SDK JSONL format and
backend example set format, enabling seamless data exchange while
supporting privacy-preserving operations.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

# Optional dependency for HTTP client
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

    class aiohttp:  # type: ignore[no-redef]
        class ClientSession:
            def __init__(self, *args, **kwargs) -> None:
                raise ImportError("aiohttp not available") from None


from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import safe_write_text, validate_path

logger = get_logger(__name__)

_EXAMPLE_SET_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


@dataclass
class ConversionResult:
    """Result of dataset conversion operation."""

    success: bool
    converted_count: int
    skipped_count: int
    error_count: int
    errors: list[str]
    metadata: dict[str, Any]


@dataclass
class ExampleSetMetadata:
    """Metadata for backend example set."""

    example_set_id: str
    name: str
    type: str  # "input-only" or "input-output"
    description: str
    total_examples: int
    created_from: str
    privacy_mode: bool


class DatasetConverter:
    """Converter between SDK datasets and backend example sets."""

    def __init__(self, backend_base_url: str = "http://localhost:5000") -> None:
        """Initialize dataset converter.

        Args:
            backend_base_url: Backend API base URL. Must use http(s) scheme without
                embedded credentials, query parameters, or fragments.
        """
        self.backend_base_url = self._validate_backend_base_url(backend_base_url)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        if AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    @staticmethod
    def _validate_backend_base_url(backend_base_url: str) -> str:
        """Validate and normalize backend base URL."""
        if not isinstance(backend_base_url, str):
            raise ValueError("backend_base_url must be a string")

        candidate = backend_base_url.strip()
        if any(ord(ch) < 32 for ch in candidate):
            raise ValueError("backend_base_url cannot contain control characters")

        if not candidate:
            raise ValueError("backend_base_url cannot be empty")

        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"backend_base_url must include scheme and host, got '{backend_base_url}'"
            )

        if parsed.username or parsed.password:
            raise ValueError("backend_base_url cannot include user credentials")

        if any(ch in parsed.netloc for ch in {" ", "\t", "\n", "\r"}):
            raise ValueError("backend_base_url cannot contain whitespace in host")

        if parsed.query or parsed.fragment:
            raise ValueError(
                "backend_base_url cannot include query parameters or fragments"
            )

        if parsed.params:
            raise ValueError("backend_base_url cannot include parameters")

        if parsed.path and ".." in Path(parsed.path).parts:
            raise ValueError("backend_base_url cannot contain path traversal segments")

        sanitized_path = parsed.path.rstrip("/")
        normalized = parsed._replace(
            path=sanitized_path, query="", fragment="", params=""
        )
        candidate = normalized.geturl()

        return candidate.rstrip("/")

    @staticmethod
    def _validate_example_set_id(example_set_id: str) -> str:
        """Validate example set identifier for safe backend usage."""
        if not isinstance(example_set_id, str):
            raise ValueError("example_set_id must be a string")

        candidate = example_set_id.strip()
        if not candidate:
            raise ValueError("example_set_id cannot be empty")

        if any(ord(ch) < 32 for ch in candidate):
            raise ValueError("example_set_id cannot contain control characters")

        if "/" in candidate or "\\" in candidate:
            raise ValueError("example_set_id cannot contain path separators")

        if not _EXAMPLE_SET_ID_PATTERN.match(candidate):
            try:
                uuid.UUID(candidate)
            except (ValueError, AttributeError) as error:
                raise ValueError(
                    "example_set_id must be a UUID or contain only alphanumeric, '.', '_', or '-' characters"
                ) from error

        return candidate

    # SDK Dataset to Backend Example Set Conversion

    def sdk_dataset_to_backend_examples(
        self, dataset: Dataset, privacy_mode: bool = False
    ) -> tuple[list[dict[str, Any]], ExampleSetMetadata]:
        """Convert SDK dataset to backend example format.

        Args:
            dataset: SDK dataset
            privacy_mode: If True, excludes sensitive data

        Returns:
            Tuple of (examples list, metadata)
        """
        logger.info(
            f"Converting SDK dataset '{dataset.name}' to backend format (privacy={privacy_mode})"
        )

        examples = []
        error_count = 0

        for i, example in enumerate(dataset.examples):
            try:
                backend_example = self._convert_evaluation_example_to_backend(
                    example, i, privacy_mode
                )
                examples.append(backend_example)
            except Exception as e:
                logger.warning(f"Failed to convert example {i}: {e}")
                error_count += 1

        # Determine example set type
        example_type = self._determine_example_set_type(dataset)

        # Create metadata
        metadata = ExampleSetMetadata(
            example_set_id=str(uuid.uuid4()),
            name=dataset.name or f"SDK_Dataset_{int(uuid.uuid4().hex[:8], 16)}",
            type=example_type,
            description=dataset.description
            or f"Converted from SDK dataset with {len(examples)} examples",
            total_examples=len(examples),
            created_from="sdk_dataset",
            privacy_mode=privacy_mode,
        )

        logger.info(f"Converted {len(examples)} examples ({error_count} errors)")
        return examples, metadata

    def sdk_dataset_to_csv(
        self,
        dataset: Dataset,
        include_metadata: bool = True,
        privacy_mode: bool = False,
    ) -> str:
        """Convert SDK dataset to CSV format for backend upload.

        Args:
            dataset: SDK dataset
            include_metadata: Include metadata columns
            privacy_mode: Exclude sensitive data

        Returns:
            CSV string
        """
        logger.debug(f"Converting SDK dataset to CSV (privacy={privacy_mode})")

        output = io.StringIO()

        # Determine columns
        columns = ["input", "output"]
        if include_metadata:
            columns.extend(["explanation", "tags"])

        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for example in dataset.examples:
            row = {
                "input": self._serialize_input_data(example.input_data, privacy_mode),
                "output": (
                    example.expected_output
                    if hasattr(example, "expected_output")
                    else ""
                ),
            }

            if include_metadata:
                row["explanation"] = ""
                row["tags"] = self._serialize_metadata_as_tags(
                    getattr(example, "metadata", {}), privacy_mode
                )

            writer.writerow(row)

        return output.getvalue()

    async def upload_sdk_dataset_to_backend(
        self,
        dataset: Dataset,
        agent_id: str | None = None,
        example_set_name: str | None = None,
        privacy_mode: bool = False,
    ) -> ConversionResult:
        """Upload SDK dataset to backend as example set.

        Args:
            dataset: SDK dataset
            agent_id: Optional agent ID to associate with
            example_set_name: Custom example set name
            privacy_mode: Enable privacy-preserving upload

        Returns:
            Conversion result
        """
        logger.info(f"Uploading SDK dataset to backend (privacy={privacy_mode})")

        try:
            # Convert to backend format
            examples, metadata = self.sdk_dataset_to_backend_examples(
                dataset, privacy_mode
            )

            if example_set_name:
                metadata.name = example_set_name

            # Create example set via API
            example_set_id = await self._create_backend_example_set(metadata, agent_id)

            # Upload examples
            result = await self._upload_examples_to_backend(example_set_id, examples)

            return ConversionResult(
                success=True,
                converted_count=result.get("added", 0),
                skipped_count=result.get("skipped", 0),
                error_count=result.get("invalid", 0),
                errors=[],
                metadata={"example_set_id": example_set_id, "backend_response": result},
            )

        except Exception as e:
            logger.error(f"Failed to upload dataset to backend: {e}")
            return ConversionResult(
                success=False,
                converted_count=0,
                skipped_count=0,
                error_count=len(dataset.examples),
                errors=[str(e)],
                metadata={},
            )

    # Backend Example Set to SDK Dataset Conversion

    async def backend_example_set_to_sdk_dataset(
        self, example_set_id: str, dataset_name: str | None = None
    ) -> Dataset:
        """Convert backend example set to SDK dataset.

        Args:
            example_set_id: Backend example set ID consisting of a UUID or
                alphanumeric string with '.', '_', or '-' characters.
            dataset_name: Optional dataset name

        Returns:
            SDK dataset
        """
        normalized_example_set_id = self._validate_example_set_id(example_set_id)

        logger.info(
            f"Converting backend example set {normalized_example_set_id} to SDK dataset"
        )

        try:
            # Fetch example set from backend
            example_set_data = await self._fetch_backend_example_set(
                normalized_example_set_id
            )
            examples_data = await self._fetch_backend_examples(
                normalized_example_set_id
            )

            # Convert examples
            sdk_examples = []
            for example_data in examples_data:
                sdk_example = self._convert_backend_example_to_evaluation_example(
                    example_data
                )
                sdk_examples.append(sdk_example)

            # Create SDK dataset
            dataset = Dataset(
                examples=sdk_examples,
                name=dataset_name
                or example_set_data.get(
                    "name", f"Backend_Dataset_{normalized_example_set_id}"
                ),
                description=example_set_data.get("description", ""),
            )

            logger.info(
                f"Converted backend example set to SDK dataset with {len(sdk_examples)} examples"
            )
            return dataset

        except Exception as e:
            logger.error(f"Failed to convert backend example set: {e}")
            raise

    def backend_examples_to_jsonl(
        self, examples: list[dict[str, Any]], output_path: Path | None = None
    ) -> str:
        """Convert backend examples to JSONL format.

        Args:
            examples: Backend examples
            output_path: Optional output file path

        Returns:
            JSONL string
        """
        logger.debug(f"Converting {len(examples)} backend examples to JSONL")

        jsonl_lines = []
        for example in examples:
            jsonl_line = {
                "input": self._deserialize_input_data(example.get("input", "")),
                "output": example.get("output", ""),
            }

            # Add metadata if available
            if example.get("tags"):
                jsonl_line["metadata"] = self._deserialize_tags_to_metadata(
                    example["tags"]
                )

            jsonl_lines.append(json.dumps(jsonl_line))

        jsonl_content = "\n".join(jsonl_lines)

        # Write to file if path provided
        if output_path:
            base_dir = (
                output_path.parent
                if output_path.is_absolute()
                else Path.cwd().resolve()
            )
            validated_path = validate_path(output_path, base_dir, must_exist=False)
            safe_write_text(validated_path, jsonl_content, base_dir)
            logger.info(f"Saved JSONL to {validated_path}")

        return jsonl_content

    # Privacy-Preserving Operations

    def create_privacy_metadata(
        self, dataset: Dataset, include_sample: bool = True, sample_size: int = 3
    ) -> dict[str, Any]:
        """Create privacy-preserving metadata for dataset.

        Args:
            dataset: SDK dataset
            include_sample: Include sample examples (anonymized)
            sample_size: Number of sample examples

        Returns:
            Privacy-safe metadata
        """
        logger.debug("Creating privacy metadata for dataset")

        metadata = {
            "name": dataset.name,
            "description": dataset.description,
            "total_examples": len(dataset.examples),
            "type": self._determine_example_set_type(dataset),
            "privacy_safe": True,
        }

        if dataset.examples:
            # Analyze structure without exposing content
            first_example = dataset.examples[0]
            metadata["input_structure"] = self._analyze_input_structure(
                first_example.input_data
            )
            metadata["output_type"] = (
                type(first_example.expected_output).__name__
                if hasattr(first_example, "expected_output")
                else "unknown"
            )

            # Length statistics
            input_lengths = []
            output_lengths = []
            for example in dataset.examples:
                input_lengths.append(len(str(example.input_data)))
                if hasattr(example, "expected_output") and example.expected_output:
                    output_lengths.append(len(str(example.expected_output)))

            metadata["statistics"] = {
                "avg_input_length": (
                    sum(input_lengths) / len(input_lengths) if input_lengths else 0
                ),
                "avg_output_length": (
                    sum(output_lengths) / len(output_lengths) if output_lengths else 0
                ),
                "input_length_range": (
                    [min(input_lengths), max(input_lengths)]
                    if input_lengths
                    else [0, 0]
                ),
                "output_length_range": (
                    [min(output_lengths), max(output_lengths)]
                    if output_lengths
                    else [0, 0]
                ),
            }

            # Include anonymized samples
            if include_sample:
                samples = []
                for _i, example in enumerate(dataset.examples[:sample_size]):
                    samples.append(
                        {
                            "input_type": type(example.input_data).__name__,
                            "input_sample": self._anonymize_text(
                                str(example.input_data)[:100]
                            ),
                            "output_sample": self._anonymize_text(
                                str(getattr(example, "expected_output", ""))[:100]
                            ),
                            "has_metadata": bool(getattr(example, "metadata", {})),
                        }
                    )
                metadata["samples"] = samples

        return metadata

    def create_dataset_subset_indices(
        self, dataset: Dataset, subset_size: int, strategy: str = "diverse_sampling"
    ) -> list[int]:
        """Create dataset subset indices for privacy-preserving optimization.

        Args:
            dataset: SDK dataset
            subset_size: Size of subset
            strategy: Sampling strategy

        Returns:
            List of indices
        """
        logger.debug(
            f"Creating dataset subset indices: {subset_size} examples with {strategy}"
        )

        total_size = len(dataset.examples)
        subset_size = min(subset_size, total_size)

        if strategy == "diverse_sampling":
            # Select evenly spaced examples
            if subset_size == total_size:
                return list(range(total_size))

            step = total_size / subset_size
            indices = [int(i * step) for i in range(subset_size)]

        elif strategy == "random_sampling":
            import random

            indices = random.sample(range(total_size), subset_size)

        elif strategy == "first_n":
            indices = list(range(subset_size))

        else:
            # Default to diverse sampling
            indices = self.create_dataset_subset_indices(
                dataset, subset_size, "diverse_sampling"
            )

        return sorted(indices)

    # Internal Conversion Methods

    def _convert_evaluation_example_to_backend(
        self, example: EvaluationExample, index: int, privacy_mode: bool = False
    ) -> dict[str, Any]:
        """Convert SDK evaluation example to backend format."""
        backend_example: dict[str, Any] = {
            "example_id": f"EX{str(uuid.uuid4())[:6].upper()}",
            "input": self._serialize_input_data(example.input_data, privacy_mode),
            "output": (
                example.expected_output if hasattr(example, "expected_output") else None
            ),
            "explanation": "",
            "tags": [],
        }

        # Add metadata as tags if available and not in privacy mode
        if not privacy_mode and hasattr(example, "metadata") and example.metadata:
            backend_example["tags"] = [f"{k}:{v}" for k, v in example.metadata.items()]

        return backend_example

    def _convert_backend_example_to_evaluation_example(
        self, backend_example: dict[str, Any]
    ) -> EvaluationExample:
        """Convert backend example to SDK evaluation example."""
        input_data = self._deserialize_input_data(backend_example.get("input", ""))
        expected_output = backend_example.get("output")

        # Convert tags back to metadata
        metadata = {}
        if backend_example.get("tags"):
            metadata = self._deserialize_tags_to_metadata(backend_example["tags"])

        return EvaluationExample(
            input_data=input_data, expected_output=expected_output, metadata=metadata
        )

    def _serialize_input_data(self, input_data: Any, privacy_mode: bool = False) -> str:
        """Serialize input data to string format."""
        if privacy_mode:
            # In privacy mode, return placeholder or anonymized version
            if isinstance(input_data, dict):
                return json.dumps(
                    {k: f"<{type(v).__name__}>" for k, v in input_data.items()}
                )
            else:
                return f"<{type(input_data).__name__}>"

        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # Try to extract main content
            for key in ["query", "question", "input", "text", "content"]:
                if key in input_data and isinstance(input_data[key], str):
                    return cast(str, input_data[key])
            # Return JSON representation
            return json.dumps(input_data)
        else:
            return str(input_data)

    def _deserialize_input_data(self, input_str: str) -> Any:
        """Deserialize input data from string."""
        if not input_str:
            return ""

        # Try to parse as JSON first
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            # Return as string
            return input_str

    def _serialize_metadata_as_tags(
        self, metadata: dict[str, Any], privacy_mode: bool = False
    ) -> str:
        """Serialize metadata as comma-separated tags."""
        if privacy_mode or not metadata:
            return ""

        tags = []
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                tags.append(f"{key}:{value}")

        return ",".join(tags)

    def _deserialize_tags_to_metadata(self, tags: str | list[str]) -> dict[str, Any]:
        """Deserialize tags back to metadata dictionary."""
        metadata = {}

        if isinstance(tags, str):
            tag_list = tags.split(",") if tags else []
        else:
            tag_list = tags or []

        for tag in tag_list:
            if ":" in tag:
                key, value = tag.split(":", 1)
                metadata[key.strip()] = value.strip()

        return metadata

    def _determine_example_set_type(self, dataset: Dataset) -> str:
        """Determine backend example set type from SDK dataset."""
        if not dataset.examples:
            return "input-only"

        # Check if examples have outputs
        has_outputs = any(
            hasattr(example, "expected_output") and example.expected_output is not None
            for example in dataset.examples
        )

        return "input-output" if has_outputs else "input-only"

    def _analyze_input_structure(self, input_data: Any) -> dict[str, Any]:
        """Analyze input data structure without exposing content."""
        if isinstance(input_data, dict):
            return {
                "type": "dict",
                "keys": list(input_data.keys()),
                "key_count": len(input_data),
            }
        elif isinstance(input_data, list):
            return {
                "type": "list",
                "length": len(input_data),
                "item_types": list({type(item).__name__ for item in input_data[:5]}),
            }
        else:
            return {
                "type": type(input_data).__name__,
                "length": (
                    len(str(input_data)) if hasattr(input_data, "__len__") else None
                ),
            }

    def _anonymize_text(self, text: str) -> str:
        """Anonymize text by replacing content with placeholders."""
        if not text:
            return ""

        words = text.split()
        if len(words) <= 3:
            return f"<{len(words)} words>"
        else:
            return f"{words[0]} ... <{len(words) - 2} words> ... {words[-1]}"

    # Backend API Methods

    async def _create_backend_example_set(
        self, metadata: ExampleSetMetadata, agent_id: str | None = None
    ) -> str:
        """Create example set in backend via API."""
        if not self._session:
            raise RuntimeError("Session not initialized") from None

        url = f"{self.backend_base_url}/api/example-sets"
        payload = {
            "name": metadata.name,
            "type": metadata.type,
            "description": metadata.description,
            "agent_id": agent_id,
        }

        async with self._session.post(url, json=payload) as response:
            if response.status == 201:
                result = await response.json()
                return self._validate_example_set_id(result["example_set_id"])
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to create example set: {response.status} {error_text}"
                )

    async def _upload_examples_to_backend(
        self, example_set_id: str, examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Upload examples to backend example set."""
        if not self._session:
            raise RuntimeError("Session not initialized")

        example_set_id = self._validate_example_set_id(example_set_id)

        # Convert examples to CSV for upload
        output = io.StringIO()
        fieldnames = ["input", "output", "explanation", "tags"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for example in examples:
            writer.writerow(
                {
                    "input": example.get("input", ""),
                    "output": example.get("output", ""),
                    "explanation": example.get("explanation", ""),
                    "tags": ",".join(example.get("tags", [])),
                }
            )

        csv_content = output.getvalue()

        # Upload via API
        url = f"{self.backend_base_url}/api/example-sets/{example_set_id}/upload"

        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(csv_content.encode()),
            filename="examples.csv",
            content_type="text/csv",
        )

        async with self._session.post(url, data=data) as response:
            if response.status == 200:
                return cast(dict[str, Any], await response.json())
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to upload examples: {response.status} {error_text}"
                )

    async def _fetch_backend_example_set(self, example_set_id: str) -> dict[str, Any]:
        """Fetch example set metadata from backend."""
        if not self._session:
            raise RuntimeError("Session not initialized")

        example_set_id = self._validate_example_set_id(example_set_id)

        url = f"{self.backend_base_url}/api/example-sets/{example_set_id}"

        async with self._session.get(url) as response:
            if response.status == 200:
                return cast(dict[str, Any], await response.json())
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to fetch example set: {response.status} {error_text}"
                )

    async def _fetch_backend_examples(
        self, example_set_id: str
    ) -> list[dict[str, Any]]:
        """Fetch examples from backend example set."""
        if not self._session:
            raise RuntimeError("Session not initialized")

        example_set_id = self._validate_example_set_id(example_set_id)

        url = f"{self.backend_base_url}/api/example-sets/{example_set_id}/examples"

        async with self._session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                return cast(list[dict[str, Any]], result.get("examples", []))
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to fetch examples: {response.status} {error_text}"
                )


# Global converter instance
converter = DatasetConverter()
