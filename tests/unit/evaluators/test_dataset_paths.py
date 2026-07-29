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

    A caller-passed relative path is resolved in two steps: joined onto
    ``dataset_root`` exactly once, and - only if that misses - retried against
    the current working directory, accepted then only when the result is still
    contained by ``dataset_root`` (issue #2023). This test pins the case where
    *both* steps miss, so the diagnostic must still explain the rule and name
    the exact (non-existent) dataset-root candidate instead of a bare,
    confusing "not found".

    The launch directory is pinned to the dataset root itself, which makes the
    cwd retry run and genuinely miss ("data/math.jsonl" under
    ``<root>`` is ``<root>/data/math.jsonl``, which does not exist). Without
    that chdir the test would depend on wherever pytest happened to be invoked
    from - see ``test_dataset_cwd_relative_path_inside_root_is_accepted`` for
    the complementary launch directory, where the retry succeeds.
    """
    project_root = tmp_path
    dataset_root = project_root / "data"
    dataset_root.mkdir()
    actual_file = dataset_root / "math.jsonl"
    _write_sample_dataset(actual_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(dataset_root)

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
    """Registry paths keep the single deterministic join - no cwd retry.

    Same "doubled path" symptom as
    ``test_dataset_relative_path_matching_root_subdir_does_not_silently_double``,
    but reached through a registry entry: ``resolve_dataset_reference`` hands
    back a registry-relative path that ``_resolve_dataset_source`` joins against
    ``dataset_root`` exactly once. The cwd retry added for issue #2023 covers
    the path the *caller* typed; a registry is an admin-curated deployment
    artifact, so it is deliberately excluded - a key must name the same file no
    matter which directory the process was launched from.

    The launch directory is therefore varied on purpose. ``data/math.jsonl``
    interpreted against ``<tmp_path>`` would hit the real file and sit inside
    the dataset root, so a registry entry that participated in the cwd retry
    would resolve from ``<tmp_path>`` and fail from ``<dataset_root>``. Both
    launch directories must fail identically, and the error must name the
    registry key, the resolved registry path, and the exact candidate tried.
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

    doubled_candidate = dataset_root / "data" / "math.jsonl"
    # project_root is the launch directory where a cwd retry *would* succeed;
    # dataset_root is one where it would not. Resolution must not tell them
    # apart.
    for launch_dir in (project_root, dataset_root):
        clear_dataset_registry_cache()
        monkeypatch.chdir(launch_dir)

        with pytest.raises(ValidationError) as exc_info:
            Dataset.from_jsonl("math")

        message = str(exc_info.value)
        assert "math" in message
        assert "data/math.jsonl" in message
        assert str(doubled_candidate) in message
        assert "TRAIGENT_DATASET_ROOT" in message
        assert "relative" in message

    # Fixing the registry entry to drop the redundant prefix resolves cleanly,
    # and does so from either launch directory.
    registry.write_text(
        json.dumps({"datasets": {"math": {"path": "math.jsonl"}}}),
        encoding="utf-8",
    )
    for launch_dir in (project_root, dataset_root):
        clear_dataset_registry_cache()
        monkeypatch.chdir(launch_dir)

        dataset = Dataset.from_jsonl("math")
        assert len(dataset) == 1
        assert dataset.metadata["source_path"] == str(actual_file.resolve())


def test_dataset_cwd_relative_path_inside_root_is_accepted(monkeypatch, tmp_path):
    """Regression for issue #2023: the cwd-relative spelling must load.

    Layout from the filed repro: the run directory contains ``traigent-runs/``
    and TRAIGENT_DATASET_ROOT points at that subdirectory. Joining the natural
    cwd-relative spelling onto the root doubles the segment
    (``traigent-runs/traigent-runs/eval.jsonl``) and used to fail for a file
    that plainly exists. Resolution now retries against the cwd and accepts the
    result because it is still contained by the dataset root.
    """
    run_dir = tmp_path / "run"
    dataset_root = run_dir / "traigent-runs"
    dataset_root.mkdir(parents=True)
    dataset_file = dataset_root / "eval.jsonl"
    _write_sample_dataset(dataset_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(run_dir)

    dataset = Dataset.from_jsonl("traigent-runs/eval.jsonl")

    assert len(dataset) == 1
    assert dataset.metadata["source_path"] == str(dataset_file.resolve())


def test_dataset_cwd_fallback_still_rejects_paths_outside_root(monkeypatch, tmp_path):
    """The cwd fallback must not become an escape hatch out of the dataset root.

    Both spellings below resolve to a real file when interpreted against the
    cwd, but neither lands under TRAIGENT_DATASET_ROOT, so the fallback must
    refuse them and the original dataset-root diagnostic must survive.
    """
    run_dir = tmp_path / "run"
    dataset_root = run_dir / "traigent-runs"
    dataset_root.mkdir(parents=True)

    sibling_file = run_dir / "outside.jsonl"
    _write_sample_dataset(sibling_file)
    parent_file = tmp_path / "secret.jsonl"
    _write_sample_dataset(parent_file)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(run_dir)

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl("outside.jsonl")
    # The message still names the dataset-root candidate, proving the cwd hit
    # was discarded rather than silently loaded.
    assert str(dataset_root / "outside.jsonl") in str(exc_info.value)

    with pytest.raises(ValidationError):
        Dataset.from_jsonl("../secret.jsonl")


def test_dataset_cwd_fallback_rejects_symlink_escape(monkeypatch, tmp_path):
    """A symlink reached through the cwd fallback is still resolved and rejected.

    ``traigent-runs/link.jsonl`` misses under the root (doubled), so the cwd
    fallback finds the symlink inside the root - but containment is enforced on
    the symlink-resolved target, which lives outside it.
    """
    if not hasattr(os, "symlink"):
        pytest.skip("OS does not support symlinks")

    run_dir = tmp_path / "run"
    dataset_root = run_dir / "traigent-runs"
    dataset_root.mkdir(parents=True)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.jsonl"
    _write_sample_dataset(outside_file)

    symlink_path = dataset_root / "link.jsonl"
    try:
        symlink_path.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted in this environment")

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(run_dir)

    with pytest.raises(ValidationError):
        Dataset.from_jsonl("traigent-runs/link.jsonl")


def test_dataset_cwd_fallback_reports_enotdir_as_validation_error(
    monkeypatch, tmp_path
):
    """A non-ENOENT failure in the cwd tree must stay a ValidationError.

    The cwd retry resolves a path under a directory the caller did not choose,
    so ``resolve(strict=True)`` can fail with more than ENOENT. Here the first
    component of the cwd-relative spelling is a *file*, which raises
    NotADirectoryError (ENOTDIR). That is an OSError, not FileNotFoundError; if
    the retry let it escape, every caller that only handles ValidationError -
    the MCP ``validate_dataset`` tool and ``traigent validate-dataset`` - would
    see a raw OSError instead of a structured failure.
    """
    run_dir = tmp_path / "run"
    dataset_root = run_dir / "traigent-runs"
    dataset_root.mkdir(parents=True)

    # A plain file where the cwd-relative spelling expects a directory.
    _write_sample_dataset(run_dir / "notadir.jsonl")

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(run_dir)

    with pytest.raises(ValidationError) as exc_info:
        Dataset.from_jsonl("notadir.jsonl/eval.jsonl")

    # The unchanged dataset-root diagnostic, not an ENOTDIR leak.
    message = str(exc_info.value)
    assert "Dataset file not found" in message
    assert str(dataset_root / "notadir.jsonl" / "eval.jsonl") in message


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permission bits, so EACCES cannot be provoked",
)
def test_dataset_cwd_fallback_reports_eacces_as_validation_error(monkeypatch, tmp_path):
    """An unreadable component in the cwd tree must stay a ValidationError.

    Sibling of the ENOTDIR case above with the other common non-ENOENT errno:
    the directory exists but is mode 000, so ``resolve(strict=True)`` raises
    PermissionError (EACCES). Same contract - the retry absorbs it and the
    dataset-root diagnostic is what reaches the caller.
    """
    run_dir = tmp_path / "run"
    dataset_root = run_dir / "traigent-runs"
    dataset_root.mkdir(parents=True)

    locked = run_dir / "locked"
    locked.mkdir()
    _write_sample_dataset(locked / "eval.jsonl")
    locked.chmod(0o000)

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(run_dir)

    try:
        with pytest.raises(ValidationError) as exc_info:
            Dataset.from_jsonl("locked/eval.jsonl")
    finally:
        # Restore so pytest can clean the tmp tree up.
        locked.chmod(0o755)

    message = str(exc_info.value)
    assert "Dataset file not found" in message
    assert str(dataset_root / "locked" / "eval.jsonl") in message


def test_dataset_root_relative_spelling_wins_over_cwd_spelling(monkeypatch, tmp_path):
    """The fallback is additive: it never overrides a root-relative hit.

    With the cwd itself inside the dataset root, ``data.jsonl`` names a real
    file under both bases. The dataset root keeps winning, so no currently
    successful resolution changes meaning.
    """
    dataset_root = tmp_path / "root"
    nested = dataset_root / "nested"
    nested.mkdir(parents=True)

    _write_sample_dataset(dataset_root / "data.jsonl")
    (nested / "data.jsonl").write_text(
        '{"input": {"text": "a"}, "output": "1"}\n'
        '{"input": {"text": "b"}, "output": "2"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(dataset_root))
    monkeypatch.delenv("TRAIGENT_DATASET_REGISTRY", raising=False)
    clear_dataset_registry_cache()
    monkeypatch.chdir(nested)

    dataset = Dataset.from_jsonl("data.jsonl")

    assert len(dataset) == 1
    assert dataset.metadata["source_path"] == str(
        (dataset_root / "data.jsonl").resolve()
    )


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
