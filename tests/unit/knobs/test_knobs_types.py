"""traigent.knobs type tests (SDK packet 6-B, RFC 0001).

Acceptance criteria from the program plan:
- ParameterRange -> Knob[Tuned] -> ParameterRange identity for all four
  range types, with the default carried as a CANDIDATE (never Fixed);
- certificates become stale when ANY freshness-core input changes
  (parametrized per-key flips) and cannot be forged across subjects,
  values, or contexts;
- KnobKind stays value-identical with the effectuation taxonomy;
- the persisted-signal shapes are CLOSED (P8).
"""

from __future__ import annotations

import dataclasses

import pytest

from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range
from traigent.knobs import (
    CTX_EXT_KEYS,
    Calibrated,
    CanonicalizationError,
    Certificate,
    CertificateDecision,
    Fixed,
    FreshnessContext,
    Knob,
    KnobKind,
    Ref,
    ResolutionNode,
    ResolutionRejection,
    SignalObservation,
    SignalSpec,
    TargetProperty,
    Tuned,
    canonical_hash,
    conformal_evidence_floor,
    issue_certificate,
    knob_to_parameter_range,
    parameter_range_to_knob,
)

# ---------------------------------------------------------------------------
# Kinds parity (effectuation reconciliation)
# ---------------------------------------------------------------------------


def test_knob_kind_parity_with_effectuation_taxonomy():
    """Pins member names AND serialized values to the effectuation enum
    (traigent/effectuation/contracts.py on its feature branch). If either
    side renames, this test trips."""
    assert {k.name for k in KnobKind} == {"VALUE", "CARDINALITY", "TOPOLOGY", "POLICY"}
    assert {k.value for k in KnobKind} == {"value", "cardinality", "topology", "policy"}


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


def _signal() -> SignalSpec:
    return SignalSpec(
        name="vote_margin",
        version="1",
        score_function="exact_match",
        score_function_version="1",
        comparator="key_eq",
        comparator_version="1",
    )


def _target() -> TargetProperty:
    return TargetProperty(name="margin_floor", mode="require_calibration")


def test_binding_discrimination_is_isinstance_based():
    tuned = Knob(name="k", binding=Tuned(range=IntRange(0, 8)))
    fixed = Knob(name="f", binding=Fixed(value=3))
    calibrated = Knob(
        name="theta",
        binding=Calibrated(signal=_signal(), target=_target()),
    )
    assert tuned.is_tuned() and not tuned.is_fixed() and not tuned.is_calibrated()
    assert fixed.is_fixed() and not fixed.is_tuned()
    assert calibrated.is_calibrated() and not calibrated.is_tuned()
    # P2: only Tuned is optimizer-visible
    assert tuned.optimizer_visible
    assert not fixed.optimizer_visible
    assert not calibrated.optimizer_visible


def test_knob_rejects_non_binding():
    with pytest.raises(TypeError):
        Knob(name="bad", binding="not-a-binding")  # type: ignore[arg-type]


def test_bindings_are_frozen():
    knob = Knob(name="k", binding=Fixed(value=1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        knob.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        knob.binding.value = 2  # type: ignore[misc]


def test_target_property_mode_registry():
    with pytest.raises(ValueError):
        TargetProperty(name="x", mode="vibes")


# ---------------------------------------------------------------------------
# Adapters — identity round-trip, default stays a candidate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameter_range",
    [
        Range(low=0.0, high=1.0, step=0.1, default=0.5),
        IntRange(low=0, high=8, default=2),
        LogRange(low=1e-5, high=1e-1),
        Choices(values=["a", "b", "c"], default="b"),
    ],
    ids=["Range", "IntRange", "LogRange", "Choices"],
)
def test_parameter_range_round_trip_identity(parameter_range):
    knob = parameter_range_to_knob("p", parameter_range)
    assert knob.is_tuned()
    # identity: the SAME object comes back — no copy, no coercion
    assert knob_to_parameter_range(knob) is parameter_range
    # the default is carried as a Tuned CANDIDATE, never a Fixed binding
    assert knob.binding.default == parameter_range.get_default()
    assert not knob.is_fixed()


def test_fixed_and_calibrated_are_not_projectable():
    assert knob_to_parameter_range(Knob(name="f", binding=Fixed(value=1))) is None
    assert (
        knob_to_parameter_range(
            Knob(name="c", binding=Calibrated(signal=_signal(), target=_target()))
        )
        is None
    )


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def test_canonical_hash_rules():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"x": -0.0}) == canonical_hash({"x": 0.0})
    assert canonical_hash("café") == canonical_hash("café")  # NFC
    with pytest.raises(CanonicalizationError):
        canonical_hash(float("nan"))
    with pytest.raises(CanonicalizationError):
        canonical_hash(float("inf"))
    with pytest.raises(CanonicalizationError):
        canonical_hash(object())


# ---------------------------------------------------------------------------
# Certificates — freshness, subject binding, fail-closed validity
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> FreshnessContext:
    base = dict(
        cvar_name="theta",
        tuned_parent_values=(("model", "m1"), ("retriever.k", 4)),
        calibration_source_id="pool_a",
        signal_spec_hash=_signal().spec_hash(),
        calibrator_id="budget_threshold",
        calibrator_version="1",
        calibrator_params_hash=canonical_hash({"budgets": [0.1, 0.2]}),
        dataset_hash="ds_v1",
        evidence_n=20,
        calibration_split="cal",
        eval_split="eval",
        target=_target(),
        extensions=(),
    )
    base.update(overrides)
    return FreshnessContext(**base)


CORE_FLIPS = [
    ("cvar_name", "theta2"),
    ("tuned_parent_values", (("model", "m2"), ("retriever.k", 4))),
    ("calibration_source_id", "pool_b"),
    ("signal_spec_hash", canonical_hash("other-signal")),
    ("calibrator_id", "other_calibrator"),
    ("calibrator_version", "2"),
    ("calibrator_params_hash", canonical_hash({"budgets": [0.3]})),
    ("dataset_hash", "ds_v2"),
    ("evidence_n", 21),
    ("calibration_split", "cal2"),
    ("eval_split", "eval2"),
    ("target", TargetProperty(name="other", mode="chance_constraint")),
]


def test_certificate_valid_for_same_context():
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    assert cert.valid_for("theta", "float", 0.5, ctx)


@pytest.mark.parametrize("field_name,new_value", CORE_FLIPS)
def test_every_core_key_flip_is_staleness_inducing(field_name, new_value):
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    flipped = dataclasses.replace(ctx, **{field_name: new_value})
    if field_name == "cvar_name":
        assert not cert.valid_for("theta2", "float", 0.5, flipped)
    assert not cert.valid_for("theta", "float", 0.5, flipped), field_name


def test_subject_binding_name_type_value():
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    assert not cert.valid_for("theta", "int", 0.5, ctx)  # type
    assert not cert.valid_for("theta", "float", 0.6, ctx)  # value


def test_forged_subject_cross_context_rejected():
    """A certificate issued against CVAR B's context with a forged subject
    naming A must not validate A (RFC v4 first conjunct)."""
    ctx_b = _ctx(cvar_name="theta_b")
    forged = dataclasses.replace(
        issue_certificate("theta_b", "float", 0.5, ctx_b), subject_cvar="theta_a"
    )
    assert not forged.valid_for("theta_a", "float", 0.5, ctx_b)
    assert not forged.valid_for("theta_a", "float", 0.5, _ctx(cvar_name="theta_a"))


def test_audit_copies_must_agree_with_live_context():
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    for field_name, bad in [
        ("target", TargetProperty(name="forged", mode="chance_constraint")),
        ("evidence", dataclasses.replace(cert.evidence, n=999)),
        ("evidence", dataclasses.replace(cert.evidence, pool_hash="forged")),
    ]:
        tampered = dataclasses.replace(cert, **{field_name: bad})
        assert not tampered.valid_for("theta", "float", 0.5, ctx)


def test_non_certified_decisions_never_valid():
    ctx = _ctx()
    for decision in (
        CertificateDecision.NO_DECISION,
        CertificateDecision.BEST_EFFORT_UNCERTIFIED,
    ):
        cert = issue_certificate("theta", "float", 0.5, ctx, decision=decision)
        assert not cert.valid_for("theta", "float", 0.5, ctx)


def test_issue_certificate_subject_must_match_context():
    with pytest.raises(ValueError):
        issue_certificate("theta_a", "float", 0.5, _ctx(cvar_name="theta_b"))


def test_parent_specificity_cannot_be_disabled_by_extensions():
    import itertools

    for r in range(len(CTX_EXT_KEYS) + 1):
        for subset in itertools.combinations(sorted(CTX_EXT_KEYS), r):
            ext = tuple((k, "v") for k in subset)
            a = _ctx(extensions=ext)
            b = _ctx(
                tuned_parent_values=(("model", "m2"), ("retriever.k", 4)),
                extensions=ext,
            )
            assert a.freshness_hash() != b.freshness_hash(), subset


def test_extensions_are_a_canonical_map():
    a = _ctx(extensions=(("model_versions", "mv1"), ("stage_versions", "sv1")))
    b = _ctx(extensions=(("stage_versions", "sv1"), ("model_versions", "mv1")))
    assert a.freshness_hash() == b.freshness_hash()
    with pytest.raises(ValueError, match="duplicate_calibration_context_key"):
        _ctx(
            extensions=(("model_versions", "a"), ("model_versions", "b"))
        ).freshness_hash()
    with pytest.raises(ValueError, match="invalid_calibration_context"):
        _ctx(extensions=(("tuned_parent_values", "x"),)).freshness_hash()


def test_parents_are_canonical_too():
    a = _ctx(tuned_parent_values=(("a", 1), ("b", 2)))
    b = _ctx(tuned_parent_values=(("b", 2), ("a", 1)))
    assert a.freshness_hash() == b.freshness_hash()
    with pytest.raises(ValueError, match="duplicate tuned parent"):
        _ctx(tuned_parent_values=(("a", 1), ("a", 2))).freshness_hash()


def test_conformal_evidence_floor():
    assert conformal_evidence_floor(0.1) == 9
    assert conformal_evidence_floor(0.01) == 99
    with pytest.raises(ValueError):
        conformal_evidence_floor(0.0)
    with pytest.raises(ValueError):
        conformal_evidence_floor(1.5)


# ---------------------------------------------------------------------------
# Signals — closed shape (P8)
# ---------------------------------------------------------------------------


def test_signal_observation_closed_shape():
    obs = SignalObservation(signal="vote_margin", value=0.7, n=10, split="cal")
    field_names = {f.name for f in dataclasses.fields(obs)}
    assert field_names == {"signal", "value", "n", "split"}  # NO metadata map
    with pytest.raises(ValueError):
        SignalObservation(signal="s", value=float("nan"), n=1, split="cal")
    with pytest.raises(ValueError):
        SignalObservation(signal="s", value=0.5, n=-1, split="cal")


def test_certificate_closed_shape():
    field_names = {f.name for f in dataclasses.fields(Certificate)}
    assert field_names == {
        "subject_cvar",
        "subject_type",
        "subject_value_hash",
        "target",
        "issued_hash",
        "decision",
        "evidence",
    }  # no payload / metadata field; target is the closed TargetProperty


# ---------------------------------------------------------------------------
# Resolution vocabulary
# ---------------------------------------------------------------------------


def test_resolution_vocabulary_matches_rfc():
    assert {r.value for r in ResolutionRejection} == {
        "cycle",
        "missing_ref",
        "duplicate_provider",
        "phase_mismatch",
        "infeasible_value",
        "stale_certificate",
        "evidence_leakage",
        "insufficient_evidence",
        "no_decision",
    }
    with pytest.raises(ValueError):
        ResolutionNode(knob="x", binding_kind="searched")
    with pytest.raises(ValueError):
        ResolutionNode(knob="x", binding_kind="tuned", phase="later")


def test_calibrated_refs_and_fallback_shape():
    calibrated = Calibrated(
        signal=_signal(),
        target=_target(),
        depends_on=(Ref(knob="model"), Ref(knob="retriever.k")),
        fallback=Fixed(value=0.5),
        require_calibration=True,
        target_epsilon=0.1,
    )
    assert calibrated.depends_on[0].knob == "model"
    assert calibrated.fallback.value == 0.5


def test_package_is_import_light():
    """traigent.knobs modules must not import the orchestrator or cloud
    client themselves. (Importing ANY traigent.* module triggers the
    top-level package __init__, so a sys.modules check would measure the
    package, not this code — a static source check states the actual design
    property.)"""
    import ast as ast_mod
    import pathlib

    import traigent.knobs

    package_dir = pathlib.Path(traigent.knobs.__file__).parent
    for module_path in package_dir.glob("*.py"):
        tree = ast_mod.parse(module_path.read_text(encoding="utf-8"))
        for node in ast_mod.walk(tree):
            names: list[str] = []
            if isinstance(node, ast_mod.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast_mod.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith("traigent.core"), module_path.name
                assert not name.startswith("traigent.cloud"), module_path.name
                if name.startswith("traigent.effectuation"):
                    # only the guarded kinds import may reference it
                    assert module_path.name == "kinds.py"


def test_jcs_number_rule():
    """RFC 8785: integral floats hash as the integer (1.0 == 1), while bool
    stays DISTINCT from int (checked before int)."""
    assert canonical_hash(1.0) == canonical_hash(1)
    assert canonical_hash({"n": 20.0}) == canonical_hash({"n": 20})
    assert canonical_hash(True) != canonical_hash(1)
    assert canonical_hash(0.5) != canonical_hash(0)


def test_bytes_like_rejected():
    for value in (b"x", bytearray(b"x"), memoryview(b"x")):
        with pytest.raises(CanonicalizationError):
            canonical_hash(value)


def test_ident_and_count_validation():
    with pytest.raises(ValueError):
        Knob(name="", binding=Fixed(value=1))
    with pytest.raises(ValueError):
        Knob(name="a..b", binding=Fixed(value=1))
    with pytest.raises(ValueError):
        Ref(knob="")
    with pytest.raises(ValueError):
        _ctx(evidence_n=-1)
    from traigent.knobs import EvidenceRef

    with pytest.raises(ValueError):
        EvidenceRef(n=-1, pool_hash="p")


def test_calibrated_self_ref_rejected():
    with pytest.raises(ValueError, match="cannot depend on itself"):
        Knob(
            name="theta",
            binding=Calibrated(
                signal=_signal(), target=_target(), depends_on=(Ref(knob="theta"),)
            ),
        )


def test_certificate_target_is_the_rfc_object():
    """RFC §3.5 fidelity: the certificate carries the TargetProperty itself
    (closed shape), and validity compares OBJECTS, not hashes."""
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    assert cert.target == _target()
    other_target_ctx = _ctx(target=TargetProperty(name="x", mode="chance_constraint"))
    assert not cert.valid_for("theta", "float", 0.5, other_target_ctx)


def test_counts_are_natural_numbers():
    """Round-2 finding: ℕ means strict non-negative INT — floats and bools
    are rejected as counts."""
    from traigent.knobs import EvidenceRef

    for bad in (1.5, True, False, -1):
        with pytest.raises(ValueError):
            _ctx(evidence_n=bad)
        with pytest.raises(ValueError):
            EvidenceRef(n=bad, pool_hash="p")
    EvidenceRef(n=0, pool_hash="p")  # zero is a natural number here
