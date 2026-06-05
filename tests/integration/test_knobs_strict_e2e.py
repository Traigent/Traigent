"""End-to-end smoke: knobs resolution + fail-closed promotion through the
PUBLIC optimize() API (the consolidation release gate).

Plan obligation (TVL-first program, Verification): a spec with 1 TVAR + 1 CVAR
under require_calibration runs end-to-end — the CVAR resolves in-trial (the
evaluated function sees it; the optimizer never does), a stale certificate
fails closed with the typed error and no winner output, and a certified
promotion path produces a real winner.
"""

from __future__ import annotations

import pytest

import traigent
from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import Choices
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.knobs import (
    Calibrated,
    CalibratedInput,
    FreshnessContext,
    Knob,
    KnobResolver,
    ResolutionError,
    ResolutionRejection,
    SignalSpec,
    TargetProperty,
    Tuned,
    canonical_hash,
    issue_certificate,
)
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import register_optimizer
from traigent.tvl.models import PromotionPolicy, RequireCalibration
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionGate


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


def _ctx(parents: tuple = ()) -> FreshnessContext:
    return FreshnessContext(
        cvar_name="threshold",
        tuned_parent_values=parents,
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


def _space(depends_on: tuple = ()) -> ConfigSpace:
    return ConfigSpace(
        knobs={
            "variant": Knob(
                name="variant", binding=Tuned(range=Choices(["cheap", "strong"]))
            ),
            "threshold": Knob(
                name="threshold",
                binding=Calibrated(
                    signal=_signal(),
                    target=_target(),
                    depends_on=depends_on,
                    require_calibration=True,
                ),
            ),
        }
    )


def _fresh_resolver(space: ConfigSpace) -> KnobResolver:
    ctx = _ctx(parents=())
    cert = issue_certificate("threshold", "float", 0.42, ctx)
    return KnobResolver(
        space,
        calibrated_inputs={
            "threshold": CalibratedInput(value=0.42, certificate=cert, context=ctx)
        },
    )


def _dataset(size: int = 3) -> Dataset:
    return Dataset(
        [EvaluationExample({"text": f"q{i}"}, "strong:0.42") for i in range(size)],
        name="strict_e2e",
    )


def _strict_gate() -> PromotionGate:
    policy = PromotionPolicy(
        require_calibration=RequireCalibration(enabled=True),
    )
    return PromotionGate(policy, [ObjectiveSpec("accuracy", "maximize")])


def _make_wrapped(resolver: KnobResolver, gate: PromotionGate):
    """The public decorator path (context injection — the default lane); the
    trial function PROVES in-trial CVAR resolution by recording and emitting
    the resolved values it actually saw."""
    function_saw: list[dict] = []

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"variant": ["cheap", "strong"]},  # Tuned projection ONLY
        default_config={"variant": "cheap"},
    )
    def answer(text: str) -> str:
        cfg = traigent.get_config()
        # context injection hands the raw per-trial dict; TraigentConfig
        # surfaces the same keys under custom_params
        params = cfg.custom_params if hasattr(cfg, "custom_params") else dict(cfg)
        function_saw.append(dict(params))
        # the evaluator-visible answer encodes the RESOLVED configuration
        return f"{params.get('variant')}:{params.get('threshold')}"

    # the same attribute seam the TVL-spec path uses internally
    answer.promotion_gate = gate
    answer.knob_resolver = resolver
    return answer, function_saw


@pytest.mark.asyncio
@pytest.mark.usefixtures("cost_preflight_approved")
async def test_cvar_resolves_in_trial_and_strict_run_never_silently_promotes():
    """Fresh certificate: every trial executes with the injected CVAR; with
    one sample per config the strict gate cannot certify, so the run ends
    with the explicit no-winner shape — never a raw-score winner."""
    space = _space(depends_on=())
    wrapped, function_saw = _make_wrapped(_fresh_resolver(space), _strict_gate())

    result = await wrapped.optimize(algorithm="grid", max_trials=3)

    # the CVAR resolved IN-TRIAL: the evaluated FUNCTION saw threshold=0.42
    # on every call (not merely the recorded trial config)
    assert len(result.trials) == 3  # baseline + both grid configs
    for trial in result.trials:
        assert trial.config.get("threshold") == 0.42, trial.config
    assert function_saw and all(
        call.get("threshold") == 0.42 for call in function_saw
    ), function_saw[:3]
    assert {call.get("variant") for call in function_saw} == {"cheap", "strong"}
    # the optimizer's search space never contained the CVAR (P2)
    assert set(wrapped.configuration_space) == {"variant"}
    # strict + one-sample-per-config evidence => explicit no-winner, applied
    # nowhere (fail closed end-to-end)
    assert result.best_config == {}
    assert result.best_score is None
    assert wrapped.get_best_config() is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("cost_preflight_approved")
async def test_stale_certificate_blocks_the_run_with_typed_error():
    """Parent-specific certificate (depends_on=variant, issued for 'cheap'):
    the first suggestion for the other parent value is R6 stale — the run
    aborts with the typed fail-closed error and no winner is ever applied."""
    from traigent.knobs import Ref

    space = _space(depends_on=(Ref(knob="variant"),))
    ctx = _ctx(parents=(("variant", "cheap"),))
    cert = issue_certificate("threshold", "float", 0.42, ctx)
    resolver = KnobResolver(
        space,
        calibrated_inputs={
            "threshold": CalibratedInput(value=0.42, certificate=cert, context=ctx)
        },
    )
    wrapped, _ = _make_wrapped(resolver, _strict_gate())

    with pytest.raises(ResolutionError) as err:
        await wrapped.optimize(algorithm="grid", max_trials=3)

    assert ResolutionRejection.STALE_CERTIFICATE in err.value.rejections
    assert wrapped.get_best_config() is None


class _TwoPassScheduleOptimizer(RandomSearchOptimizer):
    """Public optimizer-plugin seam: revisits each config so the strict gate
    accumulates >= 2 metric samples per config (grid/random dedup discrete
    spaces by design; reps_per_trial is rejected in this version)."""

    _SCHEDULE = ({"variant": "strong"}, {"variant": "cheap"}, {"variant": "strong"})

    def suggest_next_trial(self, trials):
        # the consumed baseline default ({"variant": "cheap"}) is trial 0
        idx = len(trials) - 1
        if idx >= len(self._SCHEDULE):
            from traigent.utils.exceptions import OptimizationError

            raise OptimizationError("schedule exhausted")
        return dict(self._SCHEDULE[idx])


register_optimizer("strict-e2e-schedule", _TwoPassScheduleOptimizer)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cost_preflight_approved")
async def test_certified_promotion_produces_a_real_winner():
    """With two evaluations per config the strict gate accumulates real
    evidence and certifies a promotion — the winner is the gate's certified
    incumbent, not a raw re-derivation."""
    space = _space(depends_on=())
    wrapped, function_saw = _make_wrapped(_fresh_resolver(space), _strict_gate())

    result = await wrapped.optimize(
        algorithm="strict-e2e-schedule",
        max_trials=4,
    )

    samples_per_config: dict[str, int] = {}
    for trial in result.trials:
        key = str(trial.config.get("variant"))
        samples_per_config[key] = samples_per_config.get(key, 0) + 1
    assert samples_per_config == {"cheap": 2, "strong": 2}

    # a certified winner exists and carries the resolved CVAR
    assert result.best_config.get("variant") == "strong"
    assert result.best_score is not None
    assert wrapped.get_best_config() is not None
