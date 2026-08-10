"""The pricing linter must not mistake boolean flags for prices.

``isinstance(True, int)`` is True in Python, so a plain ``(int, float)`` check
counts ``True``/``False`` as numeric. Combined with P001's "input + output keys
together indicate pricing" heuristic, ANY dict pairing those two keys with
boolean flags read as a hardcoded pricing table.

That is not hypothetical: it fired on an evaluation-dataset row --
``{"input": ..., "output": ..., "holdout": False, "synthetic": True}`` -- where
``input``/``output`` are the documented JSONL dataset keys, not model pricing.
The tempting fix is to allowlist the offending file, which would blind the rule
to that file forever. Excluding bool fixes the rule instead.

These tests pin both halves: the false positive stays fixed, AND real pricing is
still caught. A precision fix that quietly cost us detection would be worse than
the false positive it removed.
"""

from __future__ import annotations

from pathlib import Path

from tests.optimizer_validation.tools.lint_pricing_consistency import lint_file


def _codes(tmp_path: Path, name: str, source: str) -> set[str]:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return {issue.code for issue in lint_file(path)}


class TestBooleansAreNotPrices:
    def test_dataset_row_with_input_output_and_bool_flags_is_clean(
        self, tmp_path: Path
    ) -> None:
        """The exact shape that misfired: JSONL row keys plus boolean flags."""
        source = (
            "import json\n"
            "def write(inputs, output):\n"
            "    return json.dumps(\n"
            '        {"input": dict(inputs), "output": output,'
            ' "holdout": False, "synthetic": True}\n'
            "    )\n"
        )
        assert "P001" not in _codes(tmp_path, "dataset_row.py", source)

    def test_a_bare_input_output_bool_pair_is_clean(self, tmp_path: Path) -> None:
        source = 'FLAGS = {"input": True, "output": False}\n'
        assert "P001" not in _codes(tmp_path, "flags.py", source)


class TestRealPricingIsStillCaught:
    """The precision fix must not have cost any detection."""

    def test_bare_input_output_price_pair_still_flagged(self, tmp_path: Path) -> None:
        source = 'TOKEN_COST = {"input": 0.0025, "output": 0.01}\n'
        assert "P001" in _codes(tmp_path, "bare_pair.py", source)

    def test_model_pricing_table_still_flagged(self, tmp_path: Path) -> None:
        source = (
            "MODEL_PRICING = {\n"
            '    "gpt-4o": {"input": 0.0025, "output": 0.01},\n'
            '    "claude-opus": {"input": 0.015, "output": 0.075},\n'
            "}\n"
        )
        assert "P002" in _codes(tmp_path, "model_table.py", source)

    def test_an_integer_price_is_still_flagged(self, tmp_path: Path) -> None:
        """Excluding bool must not have excluded plain ints."""
        source = 'COST = {"input": 1, "output": 3}\n'
        assert "P001" in _codes(tmp_path, "int_price.py", source)
