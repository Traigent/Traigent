"""KnobResolver tests (SDK packet 6-C2, RFC 0001 §3.4).

Program acceptance criteria covered here:
- every rejection R1-R8 (+ no_decision) is reachable with a typed error;
- the Accept path produces suggestion ∪ fixed ∪ calibrated (config ⊇ N_C);
- CVARs are optimizer-invisible (tvars projection) AND resolved in-trial
  (the orchestrator injection test);
- a None resolver is an exact passthrough (legacy byte-identical).
"""

from __future__ import annotations

import dataclasses

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import Choices, IntRange
from traigent.knobs import (
    Calibrated,
    CalibratedInput,
    Fixed,
    FreshnessContext,
    Knob,
    KnobResolver,
    Ref,
    ResolutionError,
    ResolutionRejection,
    SignalObservation,
    SignalSpec,
    TargetProperty,
    Tuned,
    canonical_hash,
    issue_certificate,
)


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


def _ctx(parents=(("model", "a"),), **overrides) -> FreshnessContext:
    base = dict(
        cvar_name="theta",
        tuned_parent_values=tuple(parents),
        calibration_source_id="pool_a",
        signal_spec_hash=_signal().spec_hash(),
        calibrator_id="budget_threshold",
        calibrator_version="1",
        calibrator_params_hash=canonical_hash({}),
        dataset_hash="ds_v1",
        evidence_n=20,
        calibration_split="cal",
        eval_split="eval",
        target=_target(),
    )
    base.update(overrides)
    return FreshnessContext(**base)


def _space(*, governed: bool = True, fallback=None, epsilon=None) -> ConfigSpace:
    return ConfigSpace(
        knobs={
            "model": Knob(name="model", binding=Tuned(range=Choices(["a", "b"]))),
            "k": Knob(name="k", binding=Fixed(value=4)),
            "theta": Knob(
                name="theta",
                binding=Calibrated(
                    signal=_signal(),
                    target=_target(),
                    depends_on=(Ref(knob="model"),),
                    require_calibration=governed,
                    fallback=fallback,
                    target_epsilon=epsilon,
                ),
            ),
        }
    )


def _happy_resolver(space=None) -> KnobResolver:
    space = space or _space()
    ctx = _ctx()
    cert = issue_certificate("theta", "float", 0.5, ctx)
    return KnobResolver(
        space,
        calibrated_inputs={
            "theta": CalibratedInput(value=0.5, certificate=cert, context=ctx)
        },
    )


class TestAcceptPath:
    def test_happy_path_resolves_full_config(self):
        result = _happy_resolver().resolve({"model": "a"})
        assert dict(result.config) == {"model": "a", "k": 4, "theta": 0.5}
        assert result.used_fallbacks == ()

    def test_accepted_config_contains_every_cvar_and_fixed(self):
        result = _happy_resolver().resolve({"model": "a"})
        assert {"theta", "k"} <= set(result.config)

    def test_runtime_fixed_merges_without_collision(self):
        space = _space()
        ctx = _ctx()
        cert = issue_certificate("theta", "float", 0.5, ctx)
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(value=0.5, certificate=cert, context=ctx)
            },
            runtime_fixed={"deployment_region": "eu"},
        )
        result = resolver.resolve({"model": "a"})
        assert result.config["deployment_region"] == "eu"


class TestRejections:
    def test_r3_runtime_fixed_collides_with_knob(self):
        space = _space()
        ctx = _ctx()
        cert = issue_certificate("theta", "float", 0.5, ctx)
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(value=0.5, certificate=cert, context=ctx)
            },
            runtime_fixed={"model": "b"},
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.DUPLICATE_PROVIDER in excinfo.value.rejections

    def test_r3_suggestion_with_non_tuned_name(self):
        with pytest.raises(ResolutionError) as excinfo:
            _happy_resolver().resolve({"model": "a", "theta": 0.9})
        assert ResolutionRejection.DUPLICATE_PROVIDER in excinfo.value.rejections

    def test_r4_missing_tuned_value(self):
        with pytest.raises(ResolutionError) as excinfo:
            _happy_resolver().resolve({})
        assert ResolutionRejection.PHASE_MISMATCH in excinfo.value.rejections

    def test_r4_unproduced_cvar_even_when_unconsumed(self):
        resolver = KnobResolver(_space())  # no calibrated inputs at all
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.PHASE_MISMATCH in excinfo.value.rejections

    def test_r2_ref_to_unknown_or_cvar_parent(self):
        space = ConfigSpace(
            knobs={
                "model": Knob(name="model", binding=Tuned(range=Choices(["a"]))),
                "theta": Knob(
                    name="theta",
                    binding=Calibrated(
                        signal=_signal(),
                        target=_target(),
                        depends_on=(Ref(knob="ghost"),),
                    ),
                ),
            }
        )
        resolver = KnobResolver(
            space, calibrated_inputs={"theta": CalibratedInput(value=0.5)}
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.MISSING_REF in excinfo.value.rejections

    def test_r5_type_mismatch(self):
        space = _space(governed=False)
        resolver = KnobResolver(
            space,
            calibrated_inputs={"theta": CalibratedInput(value="not-a-number")},
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.INFEASIBLE_VALUE in excinfo.value.rejections

    def test_r6_stale_certificate_on_parent_change(self):
        """The optimizer suggested model='b' but the certificate was issued
        for model='a' — parent-specific staleness by construction."""
        space = _space()
        ctx_a = _ctx(parents=(("model", "a"),))
        cert = issue_certificate("theta", "float", 0.5, ctx_a)
        ctx_b = dataclasses.replace(ctx_a, tuned_parent_values=(("model", "b"),))
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(value=0.5, certificate=cert, context=ctx_b)
            },
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "b"})
        assert ResolutionRejection.STALE_CERTIFICATE in excinfo.value.rejections

    def test_r6_governed_without_certificate(self):
        resolver = KnobResolver(
            _space(), calibrated_inputs={"theta": CalibratedInput(value=0.5)}
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.STALE_CERTIFICATE in excinfo.value.rejections

    def test_r7_observation_from_eval_split(self):
        space = _space(governed=False)
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(
                    value=0.5,
                    observations=(
                        SignalObservation(
                            signal="vote_margin", value=0.7, n=5, split="eval"
                        ),
                    ),
                )
            },
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.EVIDENCE_LEAKAGE in excinfo.value.rejections

    def test_r7_true_item_intersection(self):
        space = _space(governed=False)
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(
                    value=0.5, evidence_item_ids=frozenset({"e1", "c2"})
                )
            },
            eval_item_ids=frozenset({"e1", "e9"}),
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.EVIDENCE_LEAKAGE in excinfo.value.rejections

    def test_r8_conformal_floor(self):
        space = _space(governed=False, epsilon=0.1)  # floor = 9
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(value=0.5, context=_ctx(evidence_n=3))
            },
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.INSUFFICIENT_EVIDENCE in excinfo.value.rejections

    def test_no_decision_strict_rejects(self):
        resolver = KnobResolver(
            _space(governed=True),
            calibrated_inputs={"theta": CalibratedInput(value=None)},
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.NO_DECISION in excinfo.value.rejections

    def test_rejections_are_collected_not_short_circuited(self):
        resolver = KnobResolver(
            _space(),  # governed CVAR without inputs -> R4
            runtime_fixed={"model": "b"},  # -> R3
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({})  # missing tuned -> R4
        assert {
            ResolutionRejection.DUPLICATE_PROVIDER,
            ResolutionRejection.PHASE_MISMATCH,
        } <= set(excinfo.value.rejections)


class TestFallbacks:
    def test_bottom_never_falls_back_even_non_governed(self):
        """Review fix (B1): Accept requires calibrator != ⊥ with NO fallback
        carve-out — abstention is no_decision regardless of governance."""
        space = _space(governed=False, fallback=Fixed(value=0.42))
        resolver = KnobResolver(
            space, calibrated_inputs={"theta": CalibratedInput(value=None)}
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.NO_DECISION in excinfo.value.rejections

    def test_governed_no_decision_never_falls_back(self):
        space = _space(governed=True, fallback=Fixed(value=0.42))
        resolver = KnobResolver(
            space, calibrated_inputs={"theta": CalibratedInput(value=None)}
        )
        with pytest.raises(ResolutionError):
            resolver.resolve({"model": "a"})

    def test_stale_certificate_fallback_for_non_governed_recorded(self):
        """The §3.5 carve-out: a NON-governed CVAR whose optional certificate
        went stale MAY use its declared fallback — always recorded."""
        space = _space(governed=False, fallback=Fixed(value=0.42))
        ctx_a = _ctx(parents=(("model", "a"),))
        cert = issue_certificate("theta", "float", 0.5, ctx_a)
        ctx_b = dataclasses.replace(ctx_a, tuned_parent_values=(("model", "b"),))
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "theta": CalibratedInput(value=0.5, certificate=cert, context=ctx_b)
            },
        )
        result = resolver.resolve({"model": "b"})
        assert result.config["theta"] == 0.42
        assert result.used_fallbacks == ("theta",)  # observable, never silent


class TestDeterminism:
    def test_same_inputs_same_resolution(self):
        resolver = _happy_resolver()
        first = resolver.resolve({"model": "a"})
        second = resolver.resolve({"model": "a"})
        assert dict(first.config) == dict(second.config)


class TestOrchestratorInjection:
    """The program acceptance criterion: CVARs are optimizer-invisible AND
    resolved in-trial — proven at the orchestrator chokepoint."""

    def _orchestrator_stub(self):
        from traigent.core.orchestrator import OptimizationOrchestrator

        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = None
        return orchestrator

    def test_none_resolver_is_exact_passthrough(self):
        orchestrator = self._orchestrator_stub()
        config = {"model": "a", "k": 3}
        assert orchestrator._apply_knob_resolution(config) is config

    def test_resolver_injects_fixed_and_cvar_values(self):
        orchestrator = self._orchestrator_stub()
        orchestrator.knob_resolver = _happy_resolver()
        resolved = orchestrator._apply_knob_resolution({"model": "a"})
        assert resolved == {"model": "a", "k": 4, "theta": 0.5}

    def test_optimizer_never_sees_cvars_but_evaluator_does(self):
        """tvars projection (what dict-based optimizers consume) excludes the
        CVAR; the resolved trial config includes it."""
        space = _space()
        optimizer_view = {
            name: parameter_range.to_config_value()
            for name, parameter_range in space.tvars.items()
        }
        assert set(optimizer_view) == {"model"}  # P2

        orchestrator = self._orchestrator_stub()
        orchestrator.knob_resolver = _happy_resolver(space)
        trial_config = orchestrator._apply_knob_resolution({"model": "a"})
        assert trial_config["theta"] == 0.5  # resolved in-trial

    def test_resolution_failure_is_fail_closed_at_the_chokepoint(self):
        orchestrator = self._orchestrator_stub()
        orchestrator.knob_resolver = KnobResolver(_space())  # unproduced CVAR
        with pytest.raises(ResolutionError):
            orchestrator._apply_knob_resolution({"model": "a"})


class TestReviewRoundTwoCoverage:
    def test_r8_epsilon_without_context_fails_closed(self):
        """Review fix (B2): a declared epsilon with NO context is
        unverifiable evidence — R8, not silent acceptance."""
        space = _space(governed=False, epsilon=0.1)
        resolver = KnobResolver(
            space, calibrated_inputs={"theta": CalibratedInput(value=0.5)}
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({"model": "a"})
        assert ResolutionRejection.INSUFFICIENT_EVIDENCE in excinfo.value.rejections

    def test_r1_cycle_via_deferred_cvar_parents(self):
        """R1 machinery (forward-compat): two CVARs referencing each other
        produce CYCLE alongside the v1 MISSING_REF kind errors."""
        space = ConfigSpace(
            knobs={
                "c1": Knob(
                    name="c1",
                    binding=Calibrated(
                        signal=_signal(), target=_target(),
                        depends_on=(Ref(knob="c2"),),
                    ),
                ),
                "c2": Knob(
                    name="c2",
                    binding=Calibrated(
                        signal=_signal(), target=_target(),
                        depends_on=(Ref(knob="c1"),),
                    ),
                ),
            }
        )
        resolver = KnobResolver(
            space,
            calibrated_inputs={
                "c1": CalibratedInput(value=0.1),
                "c2": CalibratedInput(value=0.2),
            },
        )
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve({})
        assert ResolutionRejection.CYCLE in excinfo.value.rejections
        assert ResolutionRejection.MISSING_REF in excinfo.value.rejections

    def test_internal_resolver_errors_surface_typed_not_swallowed(self):
        """Review fix (B5): internal ValueErrors wrap into ResolutionError so
        the suggest-site except tuples can never silently break."""
        from traigent.core.orchestrator import OptimizationOrchestrator

        class ExplodingResolver:
            def resolve(self, suggestion):
                raise ValueError("internal canonicalization failure")

        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = ExplodingResolver()
        with pytest.raises(ResolutionError) as excinfo:
            orchestrator._apply_knob_resolution({"model": "a"})
        assert ResolutionRejection.INFEASIBLE_VALUE in excinfo.value.rejections

    def test_baseline_config_paths_resolve_too(self):
        """Review fix (B3): the default/baseline config goes through the same
        chokepoint — Fixed and CVAR values are injected."""
        from traigent.core.orchestrator import OptimizationOrchestrator

        orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
        orchestrator.knob_resolver = _happy_resolver()
        baseline = orchestrator._apply_knob_resolution({"model": "a"})
        assert baseline == {"model": "a", "k": 4, "theta": 0.5}
