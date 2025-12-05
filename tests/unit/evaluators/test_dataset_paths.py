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
