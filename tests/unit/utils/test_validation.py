"""Tests for validation utilities."""

import ast
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from traigent.utils.secure_path import PathTraversalError, safe_open
from traigent.utils.validation import Validators, _canonical_knob_ranges


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
        """The ranges are reused from range_presets, not restated here.

        Uses a continuous knob: an integer preset is a suggested sweep rather
        than the knob's whole domain, so it is deliberately never the yardstick
        a declared sweep is measured against.
        """
        from traigent.config_generator.presets.range_presets import get_preset_range

        preset = get_preset_range("frequency_penalty")
        assert preset is not None
        low = preset["kwargs"]["low"]
        high = preset["kwargs"]["high"]

        messages = self._warnings_for(
            {"frequency_penalty": [0.0, 0.2], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert any(f"usual {low:g}-{high:g} range" in m for m in messages), messages

    def test_full_range_sweep_is_not_reported(self) -> None:
        messages = self._warnings_for(
            {"temperature": (0.0, 1.0), "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert messages == []

    def test_diagnostics_are_skipped_for_a_structurally_invalid_space(self) -> None:
        result = Validators.validate_configuration_space({"temperature": []})

        assert not result.is_valid
        assert not result.warnings


class TestWholeNumberedKnobResolution:
    """A gap of 1 is the finest an integer knob can express, so it always counts.

    Scaling the closeness floor to the canonical range made the check
    unfalsifiable for integer knobs: no declaration of whole numbers could ever
    satisfy a floor of 1.98, so ``top_k: [1, 2, 3]`` was told its values were
    "the same configuration in practice" when ``top_k=1`` is greedy decoding
    and ``top_k=2`` is not.
    """

    @staticmethod
    def _warnings_for(config_space: dict, **kwargs) -> list[str]:
        result = Validators.validate_configuration_space(config_space, **kwargs)
        assert result.is_valid, "degenerate-variation diagnostics must stay non-fatal"
        return [f"{w.field}: {w.message}" for w in result.warnings]

    @pytest.mark.parametrize(
        ("param_name", "values"),
        [
            ("top_k", [1, 2, 3]),
            ("batch_size", [1, 2, 4, 8]),
            ("max_tokens", [80, 120]),
            ("chunk_size", [500, 512]),
            ("k", [1, 2]),
            ("few_shot_count", [0, 1, 2]),
        ],
    )
    def test_adjacent_whole_numbers_are_never_indistinguishable(
        self, param_name: str, values: list[int]
    ) -> None:
        messages = self._warnings_for(
            {param_name: values, "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert messages == [], messages

    def test_integer_preset_is_a_suggested_sweep_not_a_domain(self) -> None:
        """``max_tokens`` 80-120 is a short-answer task, not a 1%-covered range."""
        messages = self._warnings_for(
            {"max_tokens": [80, 120], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert not any("of max_tokens's usual" in m for m in messages), messages

    def test_fractional_values_on_a_whole_numbered_knob_are_reported(self) -> None:
        result = Validators.validate_configuration_space(
            {"max_tokens": [100, 100.5], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        messages = [w.message for w in result.warnings]
        assert any("same configuration in practice" in m for m in messages), messages
        assert any("a whole-number setting" in m for m in messages), messages

    def test_suggestion_for_a_whole_numbered_knob_stays_integral(self) -> None:
        """``e.g. [1, 50.5, 100]`` is not a sweep top_k can accept."""
        result = Validators.validate_configuration_space(
            {"top_k": [10, 10.5], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        suggestions = [s for w in result.warnings for s in w.suggestions]
        assert suggestions, [w.message for w in result.warnings]
        assert any("[1, 50, 100]" in s for s in suggestions), suggestions
        assert not any("50.5" in s for s in suggestions), suggestions

    def test_continuous_knobs_keep_their_resolution_floor(self) -> None:
        """Integer-awareness must not disarm the check where it is sound."""
        messages = self._warnings_for(
            {"temperature": [0.70, 0.71], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert any("same configuration in practice" in m for m in messages), messages


class TestSeedIsExcludedFromDistanceChecks:
    """Sweeping seeds measures run-to-run variance, not configuration quality.

    The distance between two seeds is meaningless by construction, so neither
    the closeness check nor the coverage check applies to them.
    """

    @staticmethod
    def _warnings_for(config_space: dict) -> list[str]:
        result = Validators.validate_configuration_space(config_space)
        assert result.is_valid
        return [f"{w.field}: {w.message}" for w in result.warnings]

    def test_adjacent_seeds_are_not_reported(self) -> None:
        messages = self._warnings_for(
            {"seed": [1, 2, 3], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert messages == [], messages

    def test_a_seed_sweep_is_not_measured_against_the_preset_range(self) -> None:
        messages = self._warnings_for(
            {"seed": [0, 1, 2, 3, 4], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert not any("usual" in m for m in messages), messages

    def test_repeated_seeds_are_still_reported(self) -> None:
        """A repeat really is the same run twice, whatever the knob means."""
        messages = self._warnings_for(
            {"seed": [1, 1, 2], "model": ["gpt-4o-mini", "gpt-4o"]}
        )

        assert any(
            "3 values declared but only 2 are distinct" in m for m in messages
        ), messages


class TestShippedExamplesAreNotFlagged:
    """A diagnostic that fires on the SDK's own examples is noise, not a signal.

    Every literal ``configuration_space`` under ``examples/`` is parsed
    statically (nothing is imported or executed) and run through the validator.
    No shipped example may be told that two of its values are indistinguishable
    or that its sweep is too narrow: those two claims rest on an assumption
    about what a knob's values mean, and the corpus is the evidence for whether
    that assumption holds.
    """

    #: Claims that must never fire on a shipped example.
    _VALUE_JUDGEMENT_MARKERS = (
        "same configuration in practice",
        "the run can tell you which of these settings won",
    )

    @staticmethod
    def _examples_root() -> Path:
        return Path(__file__).resolve().parents[3] / "examples"

    @classmethod
    def _literal_config_spaces(cls, path: Path) -> list[tuple[int, dict]]:
        """Every literal dict bound to a ``configuration_space`` name or kwarg."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return []

        found: list[tuple[int, dict]] = []

        def record(node: ast.expr) -> None:
            if not isinstance(node, ast.Dict):
                return
            try:
                value = ast.literal_eval(node)
            except (ValueError, SyntaxError, TypeError, MemoryError):
                return
            if not isinstance(value, dict) or not value:
                return
            # ``{"model": [...]}`` is a placeholder in a prose snippet, not a
            # space; literal_eval turns the ellipsis into Ellipsis.
            if any(
                Ellipsis in v for v in value.values() if isinstance(v, (list, tuple))
            ):
                return
            found.append((node.lineno, value))

        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "configuration_space":
                record(node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.lower() == "configuration_space"
                    ):
                        record(node.value)
        return found

    def test_no_shipped_example_is_told_its_values_are_meaningless(self) -> None:
        root = self._examples_root()
        if not root.is_dir():
            pytest.skip("examples/ is not present in this checkout")

        scanned = 0
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            for lineno, config_space in self._literal_config_spaces(path):
                scanned += 1
                result = Validators.validate_configuration_space(config_space)
                offenders.extend(
                    f"{path.relative_to(root)}:{lineno} [{w.field}] {w.message}"
                    for w in result.warnings
                    if any(m in w.message for m in self._VALUE_JUDGEMENT_MARKERS)
                )

        assert scanned > 50, f"expected the example corpus, scanned {scanned} spaces"
        assert offenders == [], "\n".join(offenders)


class TestSuggestedSpacingIsNeverFlagged:
    """Following the tool's own advice has to silence the tool.

    The thresholds are rounded heuristics and the quantities compared against
    them are binary floats, so an exact ``>=`` decided the outcome by which
    decimals the user happened to write: ``0.95 - 0.9`` is 0.04999999999999993
    and ``0.7 - 0.65`` is 0.04999999999999999, but ``0.75 - 0.7`` is
    0.050000000000000044. Ten of the twenty adjacent ``temperature`` pairs
    spaced at the recommended 0.05 were told they were "the same configuration
    in practice" while the other ten were silent, and ``top_p: [0.9, 0.95]`` -
    a mainstream sweep - was one of the ten.

    A diagnostic that keeps firing after the user does what it asked cannot be
    acted on, and it teaches users to ignore every warning on this channel,
    including the single-value warning that shares it.
    """

    #: The claim that must never survive a sweep at the recommended spacing.
    _CLOSENESS_MARKER = "same configuration in practice"

    #: The claim that must never survive a sweep covering the whole range.
    _NARROW_SPAN_MARKER = "the run can tell you which of these settings won"

    @staticmethod
    def _warnings_for(config_space: dict) -> list[str]:
        result = Validators.validate_configuration_space(config_space)
        assert result.is_valid, "degenerate-variation diagnostics must stay non-fatal"
        return [w.message for w in result.warnings]

    @staticmethod
    def _decimal_grid(low: float, high: float, spacing: float) -> list[float]:
        """The points a user would type to sweep *low*-*high* at *spacing*.

        Built with :class:`~decimal.Decimal` so each point is exactly the float
        a decimal literal produces: ``float(Decimal("0.65")) == 0.65``. Adding
        the step in binary floats instead would build a grid no user writes and
        would hide the representation problem this test exists for.
        """
        step = Decimal(str(spacing))
        assert step > 0
        point = Decimal(str(low))
        stop = Decimal(str(high))
        points: list[float] = []
        while point <= stop:
            points.append(float(point))
            point += step
        return points

    @staticmethod
    def _sampled_pairs(
        points: list[float], limit: int = 60
    ) -> list[tuple[float, float]]:
        """Adjacent pairs of *points*, thinned to *limit* for wide integer knobs."""
        pairs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        if len(pairs) <= limit:
            return pairs
        stride = len(pairs) // limit + 1
        return pairs[::stride] + pairs[-1:]

    def test_every_pair_at_the_suggested_spacing_is_silent(self) -> None:
        """Both knob kinds, every canonical knob, every adjacent pair.

        The spacing is the one the diagnostic itself prints: 1 for a
        whole-numbered knob, the resolution floor for a continuous one.
        """
        offenders: list[str] = []
        checked = 0

        for name, canonical in sorted(_canonical_knob_ranges().items()):
            spacing = Validators._resolution_floor(name)
            assert spacing is not None, name
            grid = self._decimal_grid(canonical.low, canonical.high, spacing)
            assert len(grid) >= 3, (name, grid)

            for lower, upper in self._sampled_pairs(grid):
                checked += 1
                pair: list[float] = [lower, upper]
                if canonical.is_integer:
                    pair = [int(lower), int(upper)]
                offenders.extend(
                    f"{name}={pair!r} (gap {upper - lower!r}): {message}"
                    for message in self._warnings_for(
                        {"model": ["gpt-4o-mini", "gpt-4o"], name: pair}
                    )
                    if self._CLOSENESS_MARKER in message
                )

        assert checked > 100, f"expected the whole knob catalog, checked {checked}"
        assert offenders == [], "\n".join(offenders)

    @pytest.mark.parametrize(
        ("param_name", "spacing"),
        [("temperature", 0.05), ("top_p", 0.05)],
    )
    def test_a_full_sweep_at_the_suggested_spacing_is_silent(
        self, param_name: str, spacing: float
    ) -> None:
        """The reviewer's repro: the whole canonical range, stepped as advised."""
        canonical = _canonical_knob_ranges()[param_name]
        grid = self._decimal_grid(canonical.low, canonical.high, spacing)

        messages = self._warnings_for(
            {"model": ["gpt-4o-mini", "gpt-4o"], param_name: grid}
        )

        assert messages == [], messages

    def test_the_spacing_the_tool_prints_is_the_spacing_it_accepts(self) -> None:
        """Take the number out of the printed advice and hand it back.

        ``_resolution_floor`` is not consulted here on purpose: the user only
        ever sees the rendered suggestion, so the rendered number is the one
        that has to work.
        """
        offenders: list[str] = []

        for name, canonical in sorted(_canonical_knob_ranges().items()):
            if name in Validators._VARIANCE_ONLY_KNOBS:
                continue  # No distance advice exists for these, by design.
            # Provoke the warning with a gap nothing could consider meaningful.
            provoking = [
                canonical.low,
                canonical.low + (canonical.high - canonical.low) / 1e6,
            ]
            result = Validators.validate_configuration_space(
                {"model": ["gpt-4o-mini", "gpt-4o"], name: provoking}
            )
            printed = [
                suggestion
                for warning in result.warnings
                for suggestion in warning.suggestions
                if "apart" in suggestion
            ]
            assert printed, f"{name}: expected spacing advice, got {result.warnings}"

            match = re.search(r"at least ([0-9]*\.?[0-9]+) apart", printed[0])
            assert match, printed[0]
            advised = float(match.group(1))

            grid = self._decimal_grid(canonical.low, canonical.high, advised)
            if canonical.is_integer:
                grid = [int(point) for point in grid]
            offenders.extend(
                f"{name}: advised {advised:g} apart, still warned: {message}"
                for message in self._warnings_for(
                    {"model": ["gpt-4o-mini", "gpt-4o"], name: grid}
                )
            )

        assert offenders == [], "\n".join(offenders)

    def test_the_mainstream_top_p_pair_is_not_called_one_configuration(self) -> None:
        """``top_p: [0.9, 0.95]`` sits exactly at the recommended 0.05."""
        messages = self._warnings_for(
            {"model": ["gpt-4o-mini", "gpt-4o"], "top_p": [0.9, 0.95]}
        )

        assert not any(self._CLOSENESS_MARKER in m for m in messages), messages

    @pytest.mark.parametrize(
        ("param_name", "values"),
        [
            # 0.3 - 0.1 is 0.19999999999999998, exactly the 20% coverage floor.
            ("temperature", [0.1, 0.2, 0.3]),
            # 0.7 - 0.3 is 0.39999999999999997, 20% of the 0-2 penalty range.
            ("frequency_penalty", [0.3, 0.5, 0.7]),
        ],
    )
    def test_a_sweep_exactly_at_the_coverage_floor_is_silent(
        self, param_name: str, values: list[float]
    ) -> None:
        """The same representation trap on the coverage threshold."""
        messages = self._warnings_for(
            {"model": ["gpt-4o-mini", "gpt-4o"], param_name: values}
        )

        assert not any(self._NARROW_SPAN_MARKER in m for m in messages), messages

    def test_values_below_the_suggested_spacing_are_still_reported(self) -> None:
        """The tolerance is an epsilon, not an amnesty."""
        messages = self._warnings_for(
            {"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.70, 0.71]}
        )

        assert any(self._CLOSENESS_MARKER in m for m in messages), messages


class TestTrialBudgetIsReadDefensively:
    """A budget this diagnostic cannot read is not this diagnostic's to report.

    ``traigent validate-config`` hands ``json.load(...)["max_trials"]`` to the
    validator untouched, so the value is whatever the config file contained.
    Comparing a string against a number raised ``TypeError``, which the CLI
    rendered as "Error reading config file" - a config that validated cleanly
    before this diagnostic existed suddenly failed to validate at all.
    """

    _CONFIG_SPACE = {
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5, 1.0],
    }

    @pytest.mark.parametrize(
        "max_trials",
        [
            "10",
            "",
            None,
            [10],
            {"n": 10},
            True,
            False,
            0,
            -5,
            float("nan"),
            float("inf"),
        ],
    )
    def test_an_unusable_budget_is_ignored_rather_than_raising(
        self, max_trials: object
    ) -> None:
        result = Validators.validate_configuration_space(
            self._CONFIG_SPACE, max_trials=max_trials
        )

        assert result.is_valid
        assert not any("max_trials=" in w.message for w in result.warnings)

    def test_a_usable_budget_is_still_reported(self) -> None:
        """The guard must not disarm the check for budgets it can read."""
        result = Validators.validate_configuration_space(
            {
                "model": ["a", "b", "c", "d", "e"],
                "prompt_style": ["p1", "p2", "p3", "p4"],
                "temperature": [0.0, 0.5, 1.0],
            },
            max_trials=5,
        )

        assert any(
            "60 distinct configurations declared but max_trials=5" in w.message
            for w in result.warnings
        ), [w.message for w in result.warnings]

    def test_validate_config_cli_accepts_a_non_numeric_max_trials(
        self, tmp_path: Path
    ) -> None:
        """End to end: the regression the reviewer reproduced from the CLI."""
        from click.testing import CliRunner

        from traigent.cli.main import cli

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "configuration_space": self._CONFIG_SPACE,
                    "objectives": ["accuracy"],
                    "max_trials": "10",
                }
            ),
            encoding="utf-8",
        )

        result = CliRunner().invoke(cli, ["validate-config", str(config_file)])

        assert result.exit_code == 0, result.output
        assert "Error reading config file" not in result.output, result.output
        assert "Configuration validation passed" in result.output, result.output
