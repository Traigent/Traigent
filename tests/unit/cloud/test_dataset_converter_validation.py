"""Validation tests for DatasetConverter safety checks."""

import pytest

from traigent.cloud.dataset_converter import DatasetConverter


class TestDatasetConverterValidation:
    """Validate DatasetConverter input handling."""

    def test_backend_base_url_validation_rejects_invalid_urls(self):
        """Invalid backend URLs should raise ValueError."""
        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="ftp://example.com")

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="  ")

        with pytest.raises(ValueError):
            DatasetConverter(
                backend_base_url="https://api.example.com/base?query=1#fragment"
            )

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="https://user:pass@example.com")

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="https://api.example.com/../admin")

        with pytest.raises(ValueError):
            DatasetConverter(backend_base_url="https://api.example.com/\nadmin")

    def test_backend_base_url_normalization(self):
        """Valid backend URLs should be normalized."""
        converter = DatasetConverter(backend_base_url="https://api.example.com/base/")
        assert converter.backend_base_url == "https://api.example.com/base"

    def test_validate_example_set_id_allows_uuid(self):
        """Example set identifiers should accept UUIDs and trim whitespace."""
        converter = DatasetConverter()
        uuid_value = "123e4567-e89b-12d3-a456-426614174000"
        assert converter._validate_example_set_id(f" {uuid_value} ") == uuid_value

    def test_validate_example_set_id_rejects_path_characters(self):
        """Path traversal characters must be rejected."""
        converter = DatasetConverter()
        with pytest.raises(ValueError):
            converter._validate_example_set_id("../evil")

        with pytest.raises(ValueError):
            converter._validate_example_set_id("bad/id")

        with pytest.raises(ValueError):
            converter._validate_example_set_id("unsafe\nid")

    @pytest.mark.asyncio
    async def test_backend_example_set_to_sdk_dataset_rejects_invalid_id(self):
        """Public API should reject unsafe identifiers before network calls."""
        converter = DatasetConverter()
        with pytest.raises(ValueError):
            await converter.backend_example_set_to_sdk_dataset("??invalid??")


class TestDatasetConverterSecurePath:
    """Tests for secure path handling in DatasetConverter."""

    def test_backend_examples_to_jsonl_with_output_path(self, tmp_path):
        """Test that JSONL output uses secure path validation."""
        converter = DatasetConverter()

        # Create backend examples (dict format)
        examples = [
            {"input": '{"question": "What is 2+2?"}', "output": "4"},
            {
                "input": '{"question": "What is the capital of France?"}',
                "output": "Paris",
            },
        ]

        # Write to output path
        output_file = tmp_path / "output.jsonl"
        result = converter.backend_examples_to_jsonl(examples, output_path=output_file)

        # Verify file was written
        assert output_file.exists()
        assert len(result) > 0

        # Verify content
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_backend_examples_to_jsonl_with_absolute_output_path(self, tmp_path):
        """Test JSONL output with absolute path."""
        converter = DatasetConverter()

        examples = [
            {"input": '{"text": "Hello"}', "output": "Hi"},
        ]

        # Use absolute path
        output_file = tmp_path / "absolute_output.jsonl"
        abs_path = output_file.resolve()

        result = converter.backend_examples_to_jsonl(examples, output_path=abs_path)

        assert abs_path.exists()
        assert len(result) > 0

    def test_backend_examples_to_jsonl_without_output_path(self):
        """Test JSONL generation without writing to file."""
        converter = DatasetConverter()

        examples = [
            {"input": '{"prompt": "Test"}', "output": "Response"},
        ]

        # No output path - should return content without writing
        result = converter.backend_examples_to_jsonl(examples, output_path=None)

        assert len(result) > 0
        assert "Test" in result
        assert "Response" in result
