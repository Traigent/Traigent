"""#304: the session-create wire carries ONE objective shape.

Canonical is ``{name, orientation, weight, band}``. The legacy
``{metric, direction, weight, band}`` spelling remains an accepted SDK *input* --
that is a deliberate convenience, per the owner's 2026-09-18 ruling -- but it is
translated before sending, so the API only ever sees canonical.

These tests pin both halves: legacy input keeps working, and no legacy key
survives onto the wire. Without the second half a partial migration passes
silently, which is the failure mode that made objectives ambiguous in the first
place.
"""

from __future__ import annotations

import pytest

from traigent.cloud.models import SessionObjectiveDefinition
from traigent.cloud.session_objectives import (
    normalize_typed_objectives,
    session_objective_to_wire,
)

# Legacy at the OBJECTIVE level. Note `test`/`alpha` are absent here on purpose:
# they are canonical INSIDE a band and legacy only as siblings of one, so a blanket
# ban would have failed a correct payload.
_LEGACY_OBJECTIVE_KEYS = {"metric", "direction", "test", "alpha", "bounds"}
# BandTarget is additionalProperties:false with exactly these properties.
_CANONICAL_BAND_KEYS = {"target", "test", "alpha"}


def _assert_no_legacy_keys(payload):
    """No legacy key on an emitted objective, and a band carrying only BandTarget keys."""
    if isinstance(payload, list):
        for item in payload:
            _assert_no_legacy_keys(item)
        return
    if not isinstance(payload, dict):
        return

    leaked = _LEGACY_OBJECTIVE_KEYS & set(payload)
    assert not leaked, (
        f"legacy keys leaked onto the wire: {sorted(leaked)} in {payload}"
    )

    band = payload.get("band")
    if isinstance(band, dict):
        extra = set(band) - _CANONICAL_BAND_KEYS
        assert not extra, (
            f"band carries keys BandTarget forbids (additionalProperties: false): "
            f"{sorted(extra)} in {band}"
        )


class TestLegacyInputStillAccepted:
    def test_session_objective_definition_emits_canonical(self):
        wire = session_objective_to_wire(
            SessionObjectiveDefinition(
                metric="accuracy", direction="maximize", weight=2.0
            )
        )
        assert wire == {"name": "accuracy", "orientation": "maximize", "weight": 2.0}
        _assert_no_legacy_keys(wire)

    def test_legacy_dict_is_translated_not_forwarded(self):
        wire = session_objective_to_wire(
            {"metric": "latency", "direction": "minimize", "weight": 1.0}
        )
        assert wire == {"name": "latency", "orientation": "minimize", "weight": 1.0}
        _assert_no_legacy_keys(wire)

    def test_an_already_canonical_dict_passes_through_unchanged(self):
        canonical = {"name": "accuracy", "orientation": "maximize", "weight": 1.0}
        assert session_objective_to_wire(dict(canonical)) == canonical

    def test_legacy_and_canonical_spellings_produce_the_same_wire(self):
        legacy = session_objective_to_wire({"metric": "f1", "direction": "maximize"})
        canonical = session_objective_to_wire({"name": "f1", "orientation": "maximize"})
        assert legacy == canonical


class TestBandTranslation:
    def test_low_high_becomes_a_canonical_band_target(self):
        wire = session_objective_to_wire(
            SessionObjectiveDefinition(
                metric="response_length",
                band={"low": 120, "high": 180},
                test="TOST",
                alpha=0.05,
                weight=2.0,
            )
        )
        assert wire == {
            "name": "response_length",
            "band": {"target": [120, 180], "test": "TOST", "alpha": 0.05},
            "orientation": "band",
            "weight": 2.0,
        }
        _assert_no_legacy_keys(wire)

    def test_center_tol_input_is_converted_and_never_forwarded(self):
        """The same defect fixed in #2385, guarded on this path too.

        BandTarget is additionalProperties:false, so center/tol on the wire is a
        payload the contract rejects.
        """
        wire = session_objective_to_wire(
            {"metric": "ratio", "band": {"center": 0.5, "tol": 0.1}}
        )
        assert wire["band"]["target"] == [pytest.approx(0.4), pytest.approx(0.6)]
        assert "center" not in wire["band"]
        assert "tol" not in wire["band"]
        _assert_no_legacy_keys(wire)

    def test_a_banded_objective_declares_orientation_band(self):
        """Legacy signalled banding structurally, by the band key being present;
        canonical says it outright, and the backend needs that to route it."""
        wire = session_objective_to_wire(
            {"metric": "ratio", "band": {"low": 1, "high": 2}}
        )
        assert wire["orientation"] == "band"

    def test_sibling_test_and_alpha_move_inside_the_band(self):
        wire = session_objective_to_wire(
            {
                "metric": "ratio",
                "band": {"low": 1, "high": 2},
                "test": "TOST",
                "alpha": 0.01,
            }
        )
        assert wire["band"]["test"] == "TOST"
        assert wire["band"]["alpha"] == 0.01
        assert "test" not in wire
        assert "alpha" not in wire


class TestNormalizeTypedObjectives:
    def test_bare_direction_shorthand_becomes_a_canonical_score_objective(self):
        assert normalize_typed_objectives(["maximize"]) == [
            {"name": "score", "orientation": "maximize"}
        ]

    def test_empty_objectives_default_to_score_maximize(self):
        assert normalize_typed_objectives(None) == [
            {"name": "score", "orientation": "maximize"}
        ]

    def test_duplicate_generated_score_objectives_are_deduped(self):
        assert normalize_typed_objectives(["maximize", "maximize"]) == [
            {"name": "score", "orientation": "maximize"}
        ]

    def test_opposite_score_directions_are_both_kept(self):
        assert normalize_typed_objectives(["maximize", "minimize"]) == [
            {"name": "score", "orientation": "maximize"},
            {"name": "score", "orientation": "minimize"},
        ]

    def test_a_mixed_batch_emits_only_canonical(self):
        wire = normalize_typed_objectives(
            [
                "maximize",
                {"metric": "latency", "direction": "minimize"},
                {"name": "f1", "orientation": "maximize"},
                SessionObjectiveDefinition(metric="cost", direction="minimize"),
            ]
        )
        _assert_no_legacy_keys(wire)
        assert [o["name"] for o in wire] == ["score", "latency", "f1", "cost"]

    def test_a_named_metric_string_stays_a_string(self):
        """A bare string that is NOT a direction word is a metric name and the
        backend accepts it as-is; rewriting it would change meaning."""
        assert normalize_typed_objectives(["accuracy"]) == ["accuracy"]


def test_an_unsupported_type_is_rejected():
    with pytest.raises(
        TypeError, match="strings, dicts, or SessionObjectiveDefinition"
    ):
        session_objective_to_wire(42)


class TestOrchestratorSessionObjectivePayload:
    """The decorator path builds its own session payload and used to drop the band.

    An ObjectiveDefinition with a band produced {name, orientation: "band", weight}
    with no band at all -- and the backend REJECTS an objective whose orientation is
    "band" with neither band nor bounds. So a banded objective declared through
    ObjectiveSchema failed at session create rather than optimizing wrongly.
    """

    @staticmethod
    def _orchestrator_cls():
        import traigent.core.orchestrator as module

        return next(
            getattr(module, name)
            for name in dir(module)
            if name.endswith("Orchestrator")
        )

    def test_a_banded_objective_carries_its_band(self):
        from traigent.core.objectives import ObjectiveDefinition
        from traigent.tvl.models import BandTarget

        payload = self._orchestrator_cls()._session_objective_to_wire(
            ObjectiveDefinition(
                name="accuracy",
                orientation="band",
                weight=1.0,
                band=BandTarget(center=0.5, tol=0.1),
            )
        )

        assert payload["orientation"] == "band"
        assert payload["band"]["target"] == [pytest.approx(0.4), pytest.approx(0.6)]
        # center/tol is an input convenience; BandTarget on the wire is
        # additionalProperties:false and admits only target/test/alpha
        assert set(payload["band"]) <= {"target", "test", "alpha"}

    def test_a_plain_objective_is_unchanged_apart_from_carried_fields(self):
        from traigent.core.objectives import ObjectiveDefinition

        payload = self._orchestrator_cls()._session_objective_to_wire(
            ObjectiveDefinition(name="f1", orientation="maximize", weight=2.0)
        )

        assert payload["name"] == "f1"
        assert payload["orientation"] == "maximize"
        assert payload["weight"] == 2.0
        assert "band" not in payload

    def test_bounds_are_not_sent_alongside_a_band(self):
        """They describe the same interval; sending both persists it twice."""
        from traigent.core.objectives import ObjectiveDefinition
        from traigent.tvl.models import BandTarget

        payload = self._orchestrator_cls()._session_objective_to_wire(
            ObjectiveDefinition(
                name="accuracy",
                orientation="band",
                weight=1.0,
                band=BandTarget(low=0.4, high=0.6),
                bounds=(0.0, 1.0),
            )
        )

        assert "band" in payload
        assert "bounds" not in payload
