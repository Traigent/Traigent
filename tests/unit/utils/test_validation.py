"""Tests for validation utilities."""

import json
from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError, safe_open
from traigent.utils.validation import Validators


def _write_jsonl(path: Path, records: list[dict] | None = None) -> None:
    """Write a small valid JSONL dataset (each line carries an ``input`` field)."""
    records = records or [
        {"input": {"question": "What is 2+2?"}, "expected_output": "4"}
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class TestValidatePath:
    """Tests for Validators.validate_path security handling."""

    def test_validate_path_within_allowed_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()
        file_path = allowed_base / "data.txt"
        file_path.write_text("content")

        result = Validators.validate_path(
            file_path,
            "path",
            must_exist=True,
            must_be_file=True,
            allowed_base_dirs=[allowed_base],
        )

        assert result.is_valid

    def test_validate_path_outside_allowed_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        result = Validators.validate_path(
            outside_dir,
            "path",
            must_exist=True,
            must_be_dir=True,
            allowed_base_dirs=[allowed_base],
        )

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)


class TestValidateDataset:
    """Tests for Validators.validate_dataset."""

    def test_validate_dataset_accepts_jsonl_input_data_alias(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dataset_path = tmp_path / "inline_alias.jsonl"
        dataset_path.write_text(
            json.dumps(
                {"input_data": {"question": "What is 2+2?"}, "expected_output": "4"}
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

        result = Validators.validate_dataset(str(dataset_path))

        assert result.is_valid

    def test_validate_path_symlink_escaping_base(self, tmp_path: Path) -> None:
        allowed_base = tmp_path / "workspace"
        allowed_base.mkdir()

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        symlink_path = allowed_base / "escape"
        try:
            symlink_path.symlink_to(target_dir, target_is_directory=True)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")

        result = Validators.validate_path(
            symlink_path,
            "path",
            must_exist=True,
            must_be_dir=True,
            allowed_base_dirs=[allowed_base],
        )

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    # ----- Regression: issue #1983 (relative dataset paths) -----
    # Relative dataset paths are anchored to the invoking process's current working
    # directory. They must remain inside that base after resolving traversal and
    # symlinks, while the resolved absolute path is passed to ``safe_open`` so nested
    # relative paths are not joined a second time during the content read.

    def test_validate_dataset_accepts_nested_relative_str_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(a) A nested relative STR path is accepted (issue #1983)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("traigent-runs/dataset.jsonl")

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_nested_relative_path_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) A nested relative Path OBJECT is accepted (issue #1983)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset(Path("traigent-runs/dataset.jsonl"))

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_absolute_path(self, tmp_path: Path) -> None:
        """(c) An absolute path is still accepted (unchanged by the fix)."""
        nested = tmp_path / "traigent-runs"
        nested.mkdir()
        dataset_path = nested / "dataset.jsonl"
        _write_jsonl(dataset_path)

        result = Validators.validate_dataset(str(dataset_path.resolve()))

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_accepts_path_with_spaces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(d) A nested relative path whose dir and file contain spaces is accepted."""
        spaced_dir = tmp_path / "my data dir"
        spaced_dir.mkdir()
        _write_jsonl(spaced_dir / "my dataset.jsonl")
        monkeypatch.chdir(tmp_path)

        result = Validators.validate_dataset("my data dir/my dataset.jsonl")

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_uses_explicit_relative_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller can explicitly choose the allowed base for relative paths."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        nested = workspace / "datasets"
        nested.mkdir()
        _write_jsonl(nested / "dataset.jsonl")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        result = Validators.validate_dataset(
            "datasets/dataset.jsonl", base_dir=workspace
        )

        assert result.is_valid, [(e.error_code, e.message) for e in result.errors]

    def test_validate_dataset_rejects_existing_traversal_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing ``..`` target outside the CWD boundary is rejected."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _write_jsonl(outside / "dataset.jsonl")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("../outside/dataset.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_rejects_symlink_escape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink inside the CWD boundary cannot resolve outside it."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "dataset.jsonl"
        _write_jsonl(target)
        link = workspace / "dataset.jsonl"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("dataset.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_rejects_symlink_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A symlink loop inside the base returns a security error."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = workspace / "loop.jsonl"
        try:
            loop.symlink_to(loop.name)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("loop.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "SECURITY_ERROR" for error in result.errors)

    def test_validate_dataset_reports_missing_in_base_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing path inside the base retains the normal NOT_FOUND result."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        result = Validators.validate_dataset("missing.jsonl")

        assert not result.is_valid
        assert any(error.error_code == "NOT_FOUND" for error in result.errors)


class TestSafeOpenContainmentGuard:
    """The containment guard the issue #1983 fix relies on stays intact.

    ``validate_dataset`` derives its ``safe_open`` base from the dataset path
    itself, so it cannot, on its own, reject a ``../`` escape or an escaping
    symlink (empirically verified: an existing ``../outside/x.jsonl`` is accepted).
    The real defense is ``safe_open`` -> ``_resolve_path_in_base``: given a FIXED
    base directory, an escaping target still raises ``PathTraversalError``. The fix
    keeps passing an absolute path plus its own parent, so this guard is unweakened.
    """

    def test_safe_open_rejects_relative_traversal_escape(self, tmp_path: Path) -> None:
        """(e) A relative ``../`` target escaping a fixed base raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _write_jsonl(outside / "secret.jsonl")

        with pytest.raises(PathTraversalError):
            with safe_open("../outside/secret.jsonl", base, mode="r"):
                pass

    def test_safe_open_rejects_absolute_path_outside_base(self, tmp_path: Path) -> None:
        """An absolute target outside a fixed base raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.jsonl"
        _write_jsonl(secret)

        with pytest.raises(PathTraversalError):
            with safe_open(str(secret.resolve()), base, mode="r"):
                pass

    def test_safe_open_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """(f) A symlink inside a fixed base that points outside it raises."""
        base = tmp_path / "workspace"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.jsonl"
        _write_jsonl(secret)

        link = base / "link.jsonl"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("Symlink creation not supported on this platform")

        with pytest.raises(PathTraversalError):
            with safe_open("link.jsonl", base, mode="r"):
                pass


class TestDegenerateVariationDiagnostics:
    """Issue #2025: legal-but-useless configuration spaces are now reported.

    Every space below is structurally valid and runs to completion, so each
    diagnostic must arrive as a warning and never flip ``is_valid``.
    """

    @staticmethod
    def _warnings_for(config_space: dict, **kwargs) -> list[str]:
        result = Validators.validate_configuration_space(config_space, **kwargs)
        assert result.is_valid, "degenerate-variation diagnostics must stay non-fatal"
        return [f"{w.field}: {w.message}" for w in result.warnings]

    def test_values_too_close_to_differ_are_reported(self) -> None:
        messages = self._warnings_for({"temperature": [0.0001, 0.0002]})

        assert any("same configuration in practice" in m for m in messages), messages

    def test_values_within_the_resolution_floor_are_reported(self) -> None:
        messages = self._warnings_for({"temperature": [0.70, 0.71, 0.72]})

        assert any("0.7 and 0.71 differ by 0.01" in m for m in messages), messages

    def test_step_finer_than_the_resolution_floor_is_reported(self) -> None:
        messages = self._warnings_for(
            {
                "temperature": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.01},
                "model": ["gpt-4o-mini", "gpt-4o"],
            }
        )

        assert any("same configuration in practice" in m for m in messages), messages

    def test_duplicate_values_within_one_knob_are_reported(self) -> None:
        messages = self._warnings_for({"temperature": [0.7, 0.7, 0.9]})

        assert any(
            "3 values declared but only 2 are distinct" in m for m in messages
        ), messages

    def test_duplicates_do_not_count_toward_effective_variation(self) -> None:
        messages = self._warnings_for(
            {"temperature": [0.7, 0.7], "model": ["gpt-4o-mini"]}
        )

        assert any(
            "none of the 2 parameters has two or more distinct values" in m
            for m in messages
        ), messages

    def test_space_far_larger_than_the_trial_budget_is_reported(self) -> None:
        messages = self._warnings_for(
            {
                "model": ["a", "b", "c", "d", "e"],
                "prompt_style": ["p1", "p2", "p3", "p4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            max_trials=5,
        )

        assert any(
            "60 distinct configurations declared but max_trials=5" in m
            for m in messages
        ), messages

    def test_budget_covering_the_space_is_not_reported(self) -> None:
        messages = self._warnings_for(
            {"temperature": [0.0, 0.5, 1.0], "model": ["gpt-4o-mini", "gpt-4o"]},
            max_trials=6,
        )

        assert messages == []

    def test_single_varying_parameter_is_reported(self) -> None:
        messages = self._warnings_for(
            {"temperature": [0.0, 0.5, 1.0], "model": ["gpt-4o-mini"]}
        )

        assert any(
            "2 parameters declared but only one of them varies" in m for m in messages
        ), messages

    def test_deliberate_single_parameter_space_is_not_reported(self) -> None:
        messages = self._warnings_for({"temperature": [0.0, 0.5, 1.0]})

        assert messages == []

    def test_narrow_span_against_canonical_range_is_reported(self) -> None:
        messages = self._warnings_for(
            {"temperature": [0.1, 0.2], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert any(
            "sweeps 0.1-0.2, only 10% of temperature's usual 0-1 range" in m
            for m in messages
        ), messages

    def test_canonical_range_is_read_from_the_shipped_presets(self) -> None:
        """The ranges are reused from range_presets, not restated here."""
        from traigent.config_generator.presets.range_presets import get_preset_range

        preset = get_preset_range("max_tokens")
        assert preset is not None
        low = preset["kwargs"]["low"]
        high = preset["kwargs"]["high"]

        messages = self._warnings_for(
            {"max_tokens": [300, 400], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert any(f"usual {low}-{high} range" in m for m in messages), messages

    def test_full_range_sweep_is_not_reported(self) -> None:
        messages = self._warnings_for(
            {"temperature": (0.0, 1.0), "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert messages == []

    def test_diagnostics_are_skipped_for_a_structurally_invalid_space(self) -> None:
        result = Validators.validate_configuration_space({"temperature": []})

        assert not result.is_valid
        assert not result.warnings
