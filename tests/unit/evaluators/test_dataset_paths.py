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


def test_dataset_root_error_includes_actionable_guidance(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    outside_file = tmp_path / "outside.jsonl"
    _write_sample_dataset(outside_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl(str(outside_file))

    message = str(exc_info.value)
    assert "Dataset path must reside under" in message
    assert "TRAIGENT_DATASET_ROOT" in message
    assert "current working directory" in message
    assert "Move the dataset under that root" in message


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
    # Registry metadata is preserved along with auto-added source_path and dataset_hash
    assert dataset.metadata["owner"] == "qa"
    assert "source_path" in dataset.metadata
    assert "dataset_hash" in dataset.metadata


def test_dataset_registry_relative_path_resolves_against_noncwd_root(
    monkeypatch, tmp_path
):
    """Registry-relative paths must join against dataset_root, not cwd.

    Regresses the "double-prefix" failure mode: cwd is set to an unrelated
    directory so any accidental cwd-based join (instead of dataset_root-based)
    would either miss the file or resolve to the wrong location.
    """
    dataset_root = tmp_path / "root"
    nested = dataset_root / "nested"
    nested.mkdir(parents=True)
    dataset_file = nested / "data.jsonl"
    _write_sample_dataset(dataset_file)

    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"datasets": {"my_ds": {"path": "nested/data.jsonl"}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("TRAIGENT_DATASET_REGISTRY", str(registry))
    monkeypatch.chdir(unrelated_cwd)
    clear_dataset_registry_cache()

    dataset = Dataset.from_jsonl("my_ds")
    assert len(dataset) == 1
    assert dataset.metadata["source_path"] == str(dataset_file.resolve())


def test_dataset_relative_path_error_names_resolution_rule_and_candidate(
    monkeypatch, tmp_path
):
    """A missing relative dataset must name the root, rule, and candidate tried."""
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()

    missing_relative = "data/math.jsonl"
    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl(missing_relative)

    message = str(exc_info.value)
    assert missing_relative in message
    assert "TRAIGENT_DATASET_ROOT" in message
    assert str(dataset_root) in message
    assert str((dataset_root / missing_relative).resolve()) in message


def test_dataset_absolute_path_error_names_used_as_is_rule_and_candidate(
    monkeypatch, tmp_path
):
    """A missing absolute dataset must state the used-as-is rule and the candidate.

    Exercises the ``is_absolute_path=True`` branch of
    ``_dataset_not_found_message``: no dataset root is prepended, and the
    message must say so and name the exact absolute path that was tried.
    """
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()

    missing_absolute = dataset_root / "nowhere" / "math.jsonl"
    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl(str(missing_absolute))

    message = str(exc_info.value)
    assert "absolute path, used as-is (no dataset root prepended)" in message
    assert f"Tried: {missing_absolute}" in message


def test_dataset_registry_missing_file_error_names_registry_resolution(
    monkeypatch, tmp_path
):
    """A missing registry-resolved dataset must name both the key and its path."""
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"datasets": {"ghost": {"path": "ghost.jsonl"}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("TRAIGENT_DATASET_REGISTRY", str(registry))
    clear_dataset_registry_cache()

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl("ghost")

    message = str(exc_info.value)
    assert "ghost" in message
    assert "ghost.jsonl" in message
    assert str(dataset_root) in message


def test_dataset_relative_path_matching_root_subdir_does_not_silently_double(
    monkeypatch, tmp_path
):
    """Reproduces the reported "doubled path" symptom with a self-explanatory error.

    When TRAIGENT_DATASET_ROOT already points at a "data" subdirectory and the
    caller also writes a redundant "data/" prefix (a natural mistake once the
    root already scopes to that directory), resolution deterministically joins
    dataset_root with the relative path exactly once. The file genuinely is not
    at that joined location, so the failure must explain the rule and name the
    exact (non-existent) candidate instead of a bare, confusing "not found".
    """
    project_root = tmp_path
    dataset_root = project_root / "data"
    dataset_root.mkdir()
    actual_file = dataset_root / "math.jsonl"
    _write_sample_dataset(actual_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl("data/math.jsonl")

    message = str(exc_info.value)
    doubled_candidate = dataset_root / "data" / "math.jsonl"
    assert str(doubled_candidate) in message
    assert "TRAIGENT_DATASET_ROOT" in message
    assert "relative" in message

    # The real file resolves cleanly once the redundant prefix is dropped.
    dataset = Dataset.from_jsonl("math.jsonl")
    assert len(dataset) == 1


def test_dataset_registry_path_matching_root_subdir_does_not_silently_double(
    monkeypatch, tmp_path
):
    """Registry-resolved paths must not silently double-prefix either.

    Same "doubled path" symptom as
    ``test_dataset_relative_path_matching_root_subdir_does_not_silently_double``,
    but reached through a registry entry: ``resolve_dataset_reference`` (base
    module 147-164) hands back a registry-relative path that is then joined
    against ``dataset_root`` exactly once by ``_resolve_dataset_source``
    (base.py 204-245). When the registry entry's path itself repeats the
    dataset root's own subdirectory name, the join is still performed exactly
    once, but the resulting candidate looks doubled to a human; the error must
    name the registry key, the resolved registry path, and the exact
    candidate tried instead of a bare file-not-found.
    """
    project_root = tmp_path
    dataset_root = project_root / "data"
    dataset_root.mkdir()
    actual_file = dataset_root / "math.jsonl"
    _write_sample_dataset(actual_file)

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"datasets": {"math": {"path": "data/math.jsonl"}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("TRAIGENT_DATASET_REGISTRY", str(registry))
    clear_dataset_registry_cache()

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl("math")

    message = str(exc_info.value)
    doubled_candidate = dataset_root / "data" / "math.jsonl"
    assert "math" in message
    assert "data/math.jsonl" in message
    assert str(doubled_candidate) in message
    assert "TRAIGENT_DATASET_ROOT" in message
    assert "relative" in message

    # Fixing the registry entry to drop the redundant prefix resolves cleanly.
    registry.write_text(
        json.dumps({"datasets": {"math": {"path": "math.jsonl"}}}),
        encoding="utf-8",
    )
    clear_dataset_registry_cache()

    dataset = Dataset.from_jsonl("math")
    assert len(dataset) == 1


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
