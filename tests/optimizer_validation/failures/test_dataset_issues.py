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
            gist_template="empty-dataset -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should fail gracefully with appropriate error
        # Either raises exception or produces failed trials
        # Emit evidence regardless of outcome
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="malformed-json -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully - emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="missing-input -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully - emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="missing-output -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully - emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="null-values -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully - emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="missing-file -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should fail with file not found error
        assert isinstance(result, Exception), "Expected exception for missing file"

        # Emit evidence for expected failure
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="single-example -> {trial_count()} | {status()}",
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
            gist_template="unicode -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle unicode correctly
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="long-input -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle long inputs - emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="empty-strings -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should handle empty strings
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMalformedInputShapes:
    """Tests for various malformed input/output shapes and types."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_input_as_string_instead_of_dict(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test when input is a string instead of expected dict."""
        path = temp_dataset_dir / "input_string.jsonl"
        with open(path, "w") as f:
            f.write('{"input": "just a string", "output": "expected"}\n')
            f.write('{"input": {"text": "valid dict"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="input_string",
            description="Input is string instead of dict",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="input-string -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle gracefully - string input may be valid for some functions
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_input_as_array(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test when input is an array instead of dict."""
        path = temp_dataset_dir / "input_array.jsonl"
        with open(path, "w") as f:
            f.write('{"input": ["item1", "item2"], "output": "expected"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="input_array",
            description="Input is array instead of dict",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="input-array -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_input_as_number(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test when input is a number instead of dict."""
        path = temp_dataset_dir / "input_number.jsonl"
        with open(path, "w") as f:
            f.write('{"input": 12345, "output": "expected"}\n')
            f.write('{"input": 3.14159, "output": "float"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="input_number",
            description="Input is number instead of dict",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="input-number -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_input_as_boolean(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test when input is a boolean."""
        path = temp_dataset_dir / "input_boolean.jsonl"
        with open(path, "w") as f:
            f.write('{"input": true, "output": "expected"}\n')
            f.write('{"input": false, "output": "false"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="input_boolean",
            description="Input is boolean instead of dict",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="input-boolean -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nested_null_values(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test deeply nested null values in input."""
        path = temp_dataset_dir / "nested_nulls.jsonl"
        with open(path, "w") as f:
            f.write(
                '{"input": {"text": null, "meta": {"score": null}}, "output": "nested nulls"}\n'
            )
            f.write(
                '{"input": {"nested": {"deeply": {"value": null}}}, "output": "deep null"}\n'
            )
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="nested_nulls",
            description="Nested null values in input",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="nested-nulls -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle nested nulls gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_dict_input(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with empty dict as input."""
        path = temp_dataset_dir / "empty_dict.jsonl"
        with open(path, "w") as f:
            f.write('{"input": {}, "output": "empty input"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="empty_dict",
            description="Empty dict as input",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="empty-dict -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle empty dict input
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_array_input(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with empty array as input."""
        path = temp_dataset_dir / "empty_array.jsonl"
        with open(path, "w") as f:
            f.write('{"input": [], "output": "empty array"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="empty_array",
            description="Empty array as input",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="empty-array -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle empty array input
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_output_as_complex_object(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test when expected output is a complex object instead of string."""
        path = temp_dataset_dir / "output_complex.jsonl"
        with open(path, "w") as f:
            f.write(
                '{"input": {"text": "test1"}, "output": {"key": "value", "nested": {"a": 1}}}\n'
            )
            f.write('{"input": {"text": "test2"}, "output": ["item1", "item2"]}\n')
            f.write('{"input": {"text": "test3"}, "output": "simple string"}\n')

        scenario = TestScenario(
            name="output_complex",
            description="Output is complex object",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="output-complex -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle complex expected outputs
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_input_types_in_dataset(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test dataset with mixed input types across examples."""
        path = temp_dataset_dir / "mixed_types.jsonl"
        with open(path, "w") as f:
            f.write('{"input": {"text": "dict input"}, "output": "dict"}\n')
            f.write('{"input": "string input", "output": "string"}\n')
            f.write('{"input": 42, "output": "number"}\n')
            f.write('{"input": ["array", "input"], "output": "array"}\n')
            f.write('{"input": null, "output": "null"}\n')

        scenario = TestScenario(
            name="mixed_types",
            description="Mixed input types across examples",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="mixed-types -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle mixed types gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_characters_in_keys(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test input with special characters in dictionary keys."""
        path = temp_dataset_dir / "special_keys.jsonl"
        with open(path, "w") as f:
            f.write(
                '{"input": {"key with spaces": "value", "key-with-dashes": "value2"}, "output": "special"}\n'
            )
            f.write(
                '{"input": {"key.with.dots": "value", "key:with:colons": "value2"}, "output": "dots"}\n'
            )
            f.write('{"input": {"": "empty key"}, "output": "empty key"}\n')

        scenario = TestScenario(
            name="special_keys",
            description="Special characters in input keys",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="special-keys -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle special characters in keys
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deeply_nested_input(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with deeply nested input structure."""
        path = temp_dataset_dir / "deep_nested.jsonl"
        with open(path, "w") as f:
            deep_input = {
                "level1": {
                    "level2": {"level3": {"level4": {"level5": {"text": "deep"}}}}
                }
            }
            json.dump({"input": deep_input, "output": "deep"}, f)
            f.write("\n")
            f.write('{"input": {"text": "shallow"}, "output": "shallow"}\n')

        scenario = TestScenario(
            name="deep_nested",
            description="Deeply nested input structure",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="deep-nested -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle deeply nested structures
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_array_input(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with large array as input value."""
        path = temp_dataset_dir / "large_array.jsonl"
        with open(path, "w") as f:
            large_array = list(range(1000))
            json.dump({"input": {"items": large_array}, "output": "large"}, f)
            f.write("\n")
            f.write('{"input": {"text": "small"}, "output": "small"}\n')

        scenario = TestScenario(
            name="large_array",
            description="Large array in input",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="large-array -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle large arrays
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_binary_like_strings(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with strings that look like binary/encoded data."""
        path = temp_dataset_dir / "binary_strings.jsonl"
        with open(path, "w") as f:
            f.write('{"input": {"text": "SGVsbG8gV29ybGQ="}, "output": "base64"}\n')
            f.write('{"input": {"text": "\\x00\\x01\\x02"}, "output": "escape"}\n')
            f.write('{"input": {"text": "normal"}, "output": "normal"}\n')

        scenario = TestScenario(
            name="binary_strings",
            description="Binary-like string values",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            # Note: \x escape sequences are not valid JSON, so this file fails to parse
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="binary-strings -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle binary-like strings
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_numeric_string_keys(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with numeric strings as dictionary keys."""
        path = temp_dataset_dir / "numeric_keys.jsonl"
        with open(path, "w") as f:
            f.write(
                '{"input": {"0": "first", "1": "second", "2": "third"}, "output": "numeric keys"}\n'
            )
            f.write('{"input": {"123": "value"}, "output": "single numeric"}\n')

        scenario = TestScenario(
            name="numeric_keys",
            description="Numeric strings as keys",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="numeric-keys -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle numeric string keys
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_whitespace_only_values(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test with whitespace-only string values."""
        path = temp_dataset_dir / "whitespace.jsonl"
        with open(path, "w") as f:
            f.write('{"input": {"text": "   "}, "output": "spaces"}\n')
            f.write('{"input": {"text": "\\t\\n"}, "output": "tabs and newlines"}\n')
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="whitespace",
            description="Whitespace-only values",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="whitespace -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle whitespace-only values
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_duplicate_keys_in_entry(
        self,
        scenario_runner,
        result_validator,
        temp_dataset_dir: Path,
    ) -> None:
        """Test JSON with duplicate keys (last value wins per JSON spec)."""
        path = temp_dataset_dir / "duplicate_keys.jsonl"
        with open(path, "w") as f:
            # Note: JSON parsers typically take the last value for duplicate keys
            f.write(
                '{"input": {"text": "first"}, "input": {"text": "second"}, "output": "dup"}\n'
            )
            f.write('{"input": {"text": "valid"}, "output": "valid"}\n')

        scenario = TestScenario(
            name="duplicate_keys",
            description="Duplicate keys in JSON entry",
            config_space={"model": ["gpt-3.5-turbo"]},
            dataset_path=str(path),
            max_trials=2,
            gist_template="duplicate-keys -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)
        # Should handle duplicate keys (uses last value)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
