from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from traigent.evaluators.base import Dataset
from traigent.evaluators.dataset_registry import clear_dataset_registry_cache
from traigent.utils.exceptions import ValidationError


def _write_sample_dataset(path: Path) -> None:
    path.write_text(
        '{"input": {"text": "hello"}, "output": "world"}\n', encoding="utf-8"
    )


def test_dataset_from_jsonl_within_dataset_root(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_file = dataset_root / "sample.jsonl"
    _write_sample_dataset(dataset_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

    dataset_relative = Dataset.from_jsonl("sample.jsonl")
    assert len(dataset_relative) == 1

    dataset_absolute = Dataset.from_jsonl(str(dataset_file))
    assert len(dataset_absolute) == 1


def test_dataset_from_jsonl_blocks_directory_traversal(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    outside_file = tmp_path / "outside.jsonl"
    _write_sample_dataset(outside_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

    with pytest.raises(ValidationError):
        Dataset.from_jsonl("../outside.jsonl")

    with pytest.raises(ValidationError):
        Dataset.from_jsonl(str(outside_file))


def test_dataset_from_jsonl_blocks_symlink_escape(monkeypatch, tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("OS does not support symlinks")

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "dataset.jsonl"
    _write_sample_dataset(outside_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

    symlink_path = dataset_root / "link.jsonl"
    try:
        symlink_path.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted in this environment")

    with pytest.raises(ValidationError):
        Dataset.from_jsonl("link.jsonl")


def test_dataset_absolute_path_restricted_without_env(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_file = dataset_root / "in_root.jsonl"
    _write_sample_dataset(dataset_file)

    outside_file = tmp_path / "outside.jsonl"
    _write_sample_dataset(outside_file)

    monkeypatch.chdir(dataset_root)
    monkeypatch.delenv("TRAIGENT_DATASET_ROOT", raising=False)

    # Absolute path within the current dataset root is allowed
    dataset = Dataset.from_jsonl(str(dataset_file))
    assert len(dataset) == 1

    # Absolute path outside the dataset root is rejected
    with pytest.raises(ValidationError):
        Dataset.from_jsonl(str(outside_file))


def test_dataset_registry_lookup(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_file = dataset_root / "support.jsonl"
    _write_sample_dataset(dataset_file)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "datasets": {
                    "support": {
                        "path": "support.jsonl",
                        "name": "Customer Support",
                        "description": "Support dataset",
                        "metadata": {"owner": "qa"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("TRAIGENT_DATASET_REGISTRY", str(registry))
    clear_dataset_registry_cache()

    dataset = Dataset.from_jsonl("support")
    assert len(dataset) == 1
    assert dataset.name == "Customer Support"
    assert dataset.description == "Support dataset"
    assert dataset.metadata == {"owner": "qa"}


def test_dataset_registry_outside_root_rejected(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    outside_file = tmp_path / "outside.jsonl"
    _write_sample_dataset(outside_file)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"datasets": {"outside": {"path": str(outside_file)}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("TRAIGENT_DATASET_REGISTRY", str(registry))
    clear_dataset_registry_cache()

    with pytest.raises(ValidationError):
        Dataset.from_jsonl("outside")


class TestDatasetMetadata:
    """Tests for dataset metadata including source_path and hash."""

    def test_dataset_stores_source_path_in_metadata(self, monkeypatch, tmp_path):
        """Test that loading a dataset stores its source path in metadata."""
        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        dataset_file = dataset_root / "test_dataset.jsonl"
        _write_sample_dataset(dataset_file)

        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

        dataset = Dataset.from_jsonl(str(dataset_file))

        assert dataset.metadata is not None
        assert "source_path" in dataset.metadata
        assert dataset.metadata["source_path"] == str(dataset_file)

    def test_dataset_stores_hash_in_metadata(self, monkeypatch, tmp_path):
        """Test that loading a dataset stores a hash for cache invalidation."""
        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        dataset_file = dataset_root / "test_dataset.jsonl"
        _write_sample_dataset(dataset_file)

        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

        dataset = Dataset.from_jsonl(str(dataset_file))

        assert dataset.metadata is not None
        assert "dataset_hash" in dataset.metadata
        # Hash format is "size_mtime_ns" (nanosecond precision for rapid change detection)
        hash_value = dataset.metadata["dataset_hash"]
        assert "_" in hash_value
        parts = hash_value.split("_")
        assert len(parts) == 2
        assert parts[0].isdigit()  # size
        assert parts[1].isdigit()  # mtime_ns (nanoseconds)

    def test_dataset_hash_changes_with_content(self, monkeypatch, tmp_path):
        """Test that dataset hash changes when content changes."""
        import time

        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        dataset_file = dataset_root / "test_dataset.jsonl"
        _write_sample_dataset(dataset_file)

        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

        dataset1 = Dataset.from_jsonl(str(dataset_file))
        hash1 = dataset1.metadata["dataset_hash"]

        # Wait a bit and modify the file
        time.sleep(0.1)
        dataset_file.write_text(
            '{"input": {"text": "hello"}, "output": "world"}\n'
            '{"input": {"text": "foo"}, "output": "bar"}\n',
            encoding="utf-8",
        )

        dataset2 = Dataset.from_jsonl(str(dataset_file))
        hash2 = dataset2.metadata["dataset_hash"]

        # Hash should change because file size changed
        assert hash1 != hash2

    def test_dataset_preserves_existing_metadata(self, tmp_path):
        """Test that existing registry metadata is preserved along with source_path."""
        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        dataset_file = dataset_root / "test.jsonl"
        _write_sample_dataset(dataset_file)

        # Create a registry with custom metadata
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "datasets": {
                        "test_ds": {
                            "path": "test.jsonl",
                            "name": "Test Dataset",
                            "metadata": {
                                "custom_key": "custom_value",
                                "version": "1.0",
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # Set environment
        os.environ["TRAIGENT_DATASET_ROOT"] = str(dataset_root)
        os.environ["TRAIGENT_DATASET_REGISTRY"] = str(registry)
        clear_dataset_registry_cache()

        try:
            dataset = Dataset.from_jsonl("test_ds")

            # Check both registry metadata and source_path are present
            assert dataset.metadata is not None
            assert dataset.metadata.get("custom_key") == "custom_value"
            assert dataset.metadata.get("version") == "1.0"
            assert "source_path" in dataset.metadata
            assert "dataset_hash" in dataset.metadata
        finally:
            # Cleanup environment
            del os.environ["TRAIGENT_DATASET_ROOT"]
            del os.environ["TRAIGENT_DATASET_REGISTRY"]
            clear_dataset_registry_cache()
