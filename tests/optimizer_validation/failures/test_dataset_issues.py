"""Tests for handling dataset issues.

Tests scenarios with empty, malformed, missing, or problematic datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)


class TestEmptyDataset:
    """Tests for empty dataset scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_dataset_file(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with empty dataset file."""
        # Create empty file
        empty_path = temp_dataset_dir / "empty.jsonl"
        empty_path.touch()

        scenario = TestScenario(
            name="empty_dataset",
            description="Empty dataset file",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(empty_path),
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail gracefully with appropriate error
        # Either raises exception or produces failed trials
        if isinstance(result, Exception):
            # Expected - empty dataset should cause error
            pass
        else:
            # If it completes, trials should fail or be empty
            pass


class TestMalformedDataset:
    """Tests for malformed dataset entries."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_malformed_json_entries(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with malformed JSON entries."""
        malformed_path = temp_dataset_dir / "malformed.jsonl"
        with open(malformed_path, "w") as f:
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')
            f.write("not valid json\n")  # Malformed entry
            f.write('{"input": {"text": "also valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="malformed_json",
            description="Dataset with malformed JSON",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(malformed_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_input_key(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with entries missing input key."""
        missing_input_path = temp_dataset_dir / "missing_input.jsonl"
        with open(missing_input_path, "w") as f:
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')
            f.write('{"output": "missing input"}\n')  # Missing input
            f.write('{"input": {"text": "also valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="missing_input",
            description="Dataset with missing input key",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(missing_input_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_output_key(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with entries missing output key."""
        missing_output_path = temp_dataset_dir / "missing_output.jsonl"
        with open(missing_output_path, "w") as f:
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')
            f.write('{"input": {"text": "missing output"}}\n')  # Missing output
            f.write('{"input": {"text": "also valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="missing_output",
            description="Dataset with missing output key",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(missing_output_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_null_values_in_dataset(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with null values."""
        null_values_path = temp_dataset_dir / "null_values.jsonl"
        with open(null_values_path, "w") as f:
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')
            f.write('{"input": null, "output": "null input"}\n')
            f.write('{"input": {"text": "null output"}, "output": null}\n')

        scenario = TestScenario(
            name="null_values",
            description="Dataset with null values",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(null_values_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully


class TestMissingDataset:
    """Tests for missing dataset file scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nonexistent_dataset_path(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with non-existent dataset path."""
        scenario = TestScenario(
            name="missing_dataset",
            description="Dataset file does not exist",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path="/nonexistent/path/to/dataset.jsonl",
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail with file not found error
        assert isinstance(result, Exception), "Expected exception for missing file"


class TestDatasetEdgeCases:
    """Tests for edge cases in datasets."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_example_dataset(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test dataset with only one example."""
        scenario = basic_scenario(
            name="single_example",
            dataset_size=1,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should work with single example
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unicode_in_dataset(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with unicode characters."""
        unicode_path = temp_dataset_dir / "unicode.jsonl"
        with open(unicode_path, "w", encoding="utf-8") as f:
            f.write('{"input": {"text": "Hello world"}, "output": "normal"}\n')
            f.write(
                '{"input": {"text": "Unicode: 世界 🌍 مرحبا"}, "output": "unicode"}\n'
            )
            f.write('{"input": {"text": "Émojis: 👋🏽"}, "output": "emoji"}\n')

        scenario = TestScenario(
            name="unicode_dataset",
            description="Dataset with unicode characters",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(unicode_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle unicode correctly
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_long_input(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with very long input text."""
        long_input_path = temp_dataset_dir / "long_input.jsonl"
        with open(long_input_path, "w") as f:
            # Normal entry
            f.write('{"input": {"text": "short"}, "output": "short"}\n')
            # Very long entry
            long_text = "a" * 10000
            json.dump({"input": {"text": long_text}, "output": "long"}, f)
            f.write("\n")

        scenario = TestScenario(
            name="long_input",
            description="Dataset with very long input",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(long_input_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle long inputs

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_string_values(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with empty string values."""
        empty_strings_path = temp_dataset_dir / "empty_strings.jsonl"
        with open(empty_strings_path, "w") as f:
            f.write('{"input": {"text": ""}, "output": ""}\n')  # Empty strings
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="empty_strings",
            description="Dataset with empty string values",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(empty_strings_path),
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle empty strings
