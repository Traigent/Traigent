"""Composite telemetry → measures channel adapter tests (RFC 0002 §3.10).

:func:`traigent.knobs.telemetry.composite_measures` flattens a composite run's
content-free §3.10 measures into a flat, identifier-safe, numeric-only dict
that satisfies the backend ``MeasuresDict`` contract (≤ 50 Python-identifier
keys, numeric values only). These tests pin:

- the flattening rules (scalars prefixed; the per-gate map flattened by index);
- ``MeasuresDict`` round-trip compliance of every emitted key/value;
- content-freedom: nothing output-derived can enter the output — the adapter
  reads ``run.measures`` only, never ``run.output``;
- the deterministic key cap (truncate + warn, NEVER raise mid-trial).

Stub stages are deterministic — NO LLM calls.
"""

from __future__ import annotations

import logging

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.knobs import composite_measures as composite_measures_reexport
from traigent.knobs.patterns import binary_cascade
from traigent.knobs.runtime import (
    CompositeRunResult,
    ResultKind,
    StageRunner,
    execute_composite,
)
from traigent.knobs.telemetry import MAX_COMPOSITE_MEASURES, composite_measures

GATE = "router_margin_threshold"


def _stage(outputs: list[str]) -> StageRunner:
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


def _binary() -> object:
    return binary_cascade(
        "answerer", base_stage="cheap", expert_stage="strong", threshold=GATE
    ).structure


# --------------------------------------------------------------------------- #
# Flattening rules over a REAL composite run                                  #
# --------------------------------------------------------------------------- #


class TestFlattenRealRun:
    def test_kept_at_cheap_arm_flattens_scalars_and_one_gate(self) -> None:
        # cheap unanimous: margin 1.0 >= theta 0.6 -> kept at arm 0.
        run = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "A"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={GATE: 0.6},
        )
        assert run.result_kind is ResultKind.OUTPUT

        flat = composite_measures(run)
        assert flat == {
            "composite_escalation_rate": 0.0,
            "composite_stage_selected": 0,
            "composite_gate_0_margin_pass_rate": 1.0,
        }

    def test_escalated_run_flattens_failing_gate(self) -> None:
        # cheap split 2/3 -> margin 0.666 < theta 0.9 -> escalate to arm 1.
        run = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={GATE: 0.9},
        )
        flat = composite_measures(run)
        assert flat["composite_escalation_rate"] == 1.0
        assert flat["composite_stage_selected"] == 1
        assert flat["composite_gate_0_margin_pass_rate"] == 0.0

    def test_custom_prefix_is_applied_to_every_key(self) -> None:
        run = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "A"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={GATE: 0.6},
        )
        flat = composite_measures(run, prefix="answerer")
        assert set(flat) == {
            "answerer_escalation_rate",
            "answerer_stage_selected",
            "answerer_gate_0_margin_pass_rate",
        }

    def test_multi_gate_map_flattens_by_ascending_index(self) -> None:
        # Synthetic measures mirroring a 3-arm cascade that escalated twice.
        run = CompositeRunResult(
            output=None,
            result_kind=ResultKind.OUTPUT,
            measures={
                "escalation_rate": 1.0,
                "stage_selected": 2,
                "gate_margin_pass_rate": {0: 0.0, 1: 0.0},
            },
        )
        flat = composite_measures(run)
        assert flat == {
            "composite_escalation_rate": 1.0,
            "composite_stage_selected": 2,
            "composite_gate_0_margin_pass_rate": 0.0,
            "composite_gate_1_margin_pass_rate": 0.0,
        }
        # Deterministic ordering: gate entries trail scalars, by index.
        keys = list(flat)
        assert keys.index("composite_gate_0_margin_pass_rate") < keys.index(
            "composite_gate_1_margin_pass_rate"
        )


# --------------------------------------------------------------------------- #
# MeasuresDict contract compliance (the wire channel's gate)                  #
# --------------------------------------------------------------------------- #


class TestMeasuresDictCompliance:
    def test_every_emitted_key_satisfies_measuresdict(self) -> None:
        run = execute_composite(
            _binary(),
            {"cheap": _stage(["A", "A", "B"]), "strong": _stage(["STRONG"])},
            config={},
            calibrated_values={GATE: 0.9},
        )
        flat = composite_measures(run)
        # The backend validates the trial's numeric metrics through MeasuresDict;
        # constructing one must NOT raise for any composite key/value.
        validated = MeasuresDict(flat)
        assert dict(validated) == flat

    def test_keys_are_python_identifiers(self) -> None:
        run = CompositeRunResult(
            output=None,
            result_kind=ResultKind.OUTPUT,
            measures={
                "escalation_rate": 0.0,
                "stage_selected": 0,
                "gate_margin_pass_rate": {0: 1.0},
            },
        )
        for key in composite_measures(run):
            assert key.isidentifier(), key


# --------------------------------------------------------------------------- #
# Content-freedom: nothing output-derived can enter the output                #
# --------------------------------------------------------------------------- #


class TestContentFree:
    def test_output_value_never_appears_in_measures(self) -> None:
        sentinel = "SECRET_MODEL_OUTPUT_DO_NOT_LEAK"
        run = execute_composite(
            _binary(),
            {"cheap": _stage([sentinel, sentinel, sentinel]), "strong": _stage(["x"])},
            config={},
            calibrated_values={GATE: 0.6},
        )
        assert run.output == sentinel  # the run really produced the sentinel
        flat = composite_measures(run)
        # No key OR value carries the produced content (§3.10 content-free).
        assert sentinel not in flat
        assert all(isinstance(v, (int, float)) for v in flat.values())
        assert sentinel not in {str(v) for v in flat.values()}

    def test_non_numeric_or_non_finite_values_are_dropped(self) -> None:
        run = CompositeRunResult(
            output=None,
            result_kind=ResultKind.OUTPUT,
            measures={
                "escalation_rate": float("nan"),  # non-finite -> dropped
                "stage_selected": 1,
                "leaked_text": "should never ride the channel",  # non-numeric
                "flag": True,  # bool is not a measure (MeasuresDict parity)
                "gate_margin_pass_rate": {0: float("inf"), 1: 0.0},
            },
        )
        flat = composite_measures(run)
        assert flat == {
            "composite_stage_selected": 1,
            "composite_gate_1_margin_pass_rate": 0.0,
        }


# --------------------------------------------------------------------------- #
# Deterministic cap: truncate + warn, never raise                            #
# --------------------------------------------------------------------------- #


class TestKeyCap:
    def test_overflow_truncates_deterministically_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A pathological per-gate map larger than the cap.
        big = dict.fromkeys(range(MAX_COMPOSITE_MEASURES + 10), 1.0)
        run = CompositeRunResult(
            output=None,
            result_kind=ResultKind.OUTPUT,
            measures={"stage_selected": 0, "gate_margin_pass_rate": big},
        )
        with caplog.at_level(logging.WARNING):
            flat = composite_measures(run)

        assert len(flat) == MAX_COMPOSITE_MEASURES  # capped, never raised
        assert "stage_selected" in {k.removeprefix("composite_") for k in flat} or any(
            k == "composite_stage_selected" for k in flat
        )
        # Deterministic: the scalar (priority 1) survives; lowest indices win.
        assert "composite_stage_selected" in flat
        assert "composite_gate_0_margin_pass_rate" in flat
        assert any("exceed the" in rec.message for rec in caplog.records)

    def test_repeated_calls_are_stable(self) -> None:
        big = dict.fromkeys(range(MAX_COMPOSITE_MEASURES + 5), 1.0)
        run = CompositeRunResult(
            output=None,
            result_kind=ResultKind.OUTPUT,
            measures={"escalation_rate": 1.0, "gate_margin_pass_rate": big},
        )
        assert composite_measures(run) == composite_measures(run)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_error_run_has_empty_measures_and_yields_empty_dict(self) -> None:
        # A fail-closed error run carries no measures (runtime contract).
        run = CompositeRunResult(
            output=None, result_kind=ResultKind.ERROR, measures={}, error="boom"
        )
        assert composite_measures(run) == {}

    def test_non_identifier_prefix_raises_value_error(self) -> None:
        run = CompositeRunResult(
            output=None, result_kind=ResultKind.OUTPUT, measures={"stage_selected": 0}
        )
        with pytest.raises(ValueError, match="identifier"):
            composite_measures(run, prefix="bad-prefix")

    def test_reexported_from_knobs_package(self) -> None:
        assert composite_measures_reexport is composite_measures


def _run_with_measures(measures):
    return CompositeRunResult(
        output=None, result_kind=ResultKind.OUTPUT, measures=measures
    )


class TestCodexRoundBlockers:
    def test_flatten_collision_raises(self):
        """Blocker 3: a scalar named gate_0_margin_pass_rate and the standard
        per-gate map both flatten to the same key — duplicates must RAISE,
        never silently last-write-win."""
        run = _run_with_measures(
            {
                "gate_0_margin_pass_rate": 0.5,
                "gate_margin_pass_rate": {0: 1.0},
            }
        )
        with pytest.raises(ValueError, match="collid|duplicate"):
            composite_measures(run)

    def test_merge_rejects_user_key_collisions(self):
        """Blocker 1: merging into a user metrics dict that already carries a
        composite_* key must fail LOUD (reserved namespace), never overwrite."""
        from traigent.knobs.telemetry import merge_composite_measures

        run = _run_with_measures({"escalation_rate": 1.0})
        user_metrics = {"accuracy": 0.9, "composite_escalation_rate": 123.0}
        with pytest.raises(ValueError, match="collid|reserved"):
            merge_composite_measures(user_metrics, run)

    def test_merge_enforces_the_total_measures_ceiling(self):
        """Blocker 2: the 50-key MeasuresDict cap applies to the WHOLE
        submitted metrics dict — the merge truncates composite keys
        deterministically (logged) so the TOTAL never exceeds 50."""
        from traigent.knobs.telemetry import merge_composite_measures

        run = _run_with_measures({"gate_margin_pass_rate": {i: 1.0 for i in range(30)}})
        user_metrics = {f"user_metric_{i}": float(i) for i in range(45)}
        merged = merge_composite_measures(dict(user_metrics), run)
        assert len(merged) <= 50
        # user metrics are NEVER dropped — only composite keys yield
        for k in user_metrics:
            assert k in merged

    def test_merge_happy_path(self):
        from traigent.knobs.telemetry import merge_composite_measures

        run = _run_with_measures({"escalation_rate": 1.0, "stage_selected": 1})
        merged = merge_composite_measures({"accuracy": 0.9}, run)
        assert merged["accuracy"] == 0.9
        assert merged["composite_escalation_rate"] == 1.0
        assert merged["composite_stage_selected"] == 1


class TestCodexConfirmResiduals:
    def test_any_prefixed_user_key_is_rejected(self):
        """The composite_* namespace is reserved WHOLESALE — not merely the
        keys this run happens to flatten."""
        from traigent.knobs.telemetry import merge_composite_measures

        run = _run_with_measures({"escalation_rate": 1.0})
        with pytest.raises(ValueError, match="reserved"):
            merge_composite_measures({"composite_user": 9.0}, run)

    def test_user_dict_already_over_ceiling_raises(self):
        """A user dict ALREADY beyond the 50-key ceiling cannot be made valid
        by truncating composite keys (user keys are never dropped) — fail
        LOUD instead of returning a contract-violating dict."""
        from traigent.knobs.telemetry import merge_composite_measures

        run = _run_with_measures({"escalation_rate": 1.0})
        over = {f"user_metric_{i}": float(i) for i in range(51)}
        with pytest.raises(ValueError, match="ceiling|50"):
            merge_composite_measures(over, run)
