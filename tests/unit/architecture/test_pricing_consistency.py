"""Architecture tests: LLM pricing consistency across the codebase.

These tests verify that all cost-calculation paths produce consistent
results using traigent.utils.cost_calculator as the single source of truth.
Prevents silent pricing drift that can cause 2-33x cost mis-tracking.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from traigent.utils.cost_calculator import (
    FALLBACK_MODEL_PRICING,
    CostCalculator,
    _fallback_cost_from_tokens,
)

# Tolerance for pricing comparisons (20% — tight enough to catch real drift,
# loose enough for rounding differences between per-1K and per-token math).
TOLERANCE = 0.20

# Standard token counts for comparison tests
INPUT_TOKENS = 1000
OUTPUT_TOKENS = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_error(actual: float, expected: float) -> float:
    """Compute relative error, handling zero expected gracefully."""
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / expected


def _canonical_cost(model: str) -> float:
    """Compute canonical cost for model using CostCalculator."""
    calc = CostCalculator()
    input_cost, output_cost = calc._calculate_from_tokens(
        INPUT_TOKENS, OUTPUT_TOKENS, model
    )
    return float(input_cost + output_cost)


# ---------------------------------------------------------------------------
# Test 1: platforms.py _calculate_cost vs CostCalculator
# ---------------------------------------------------------------------------


class TestPlatformsCostMatchesCalculator:
    """Verify OpenAIAgentExecutor._calculate_cost agrees with CostCalculator."""

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
    def test_cost_within_tolerance(self, model: str) -> None:
        from traigent.agents.platforms import OpenAIAgentExecutor

        executor = OpenAIAgentExecutor()

        # Build a mock usage object
        usage = MagicMock()
        usage.prompt_tokens = INPUT_TOKENS
        usage.completion_tokens = OUTPUT_TOKENS

        platform_cost = executor._calculate_cost(model, usage)
        canonical = _canonical_cost(model)

        assert canonical > 0, f"Canonical cost for {model} should be > 0"
        error = _relative_error(platform_cost, canonical)
        assert error <= TOLERANCE, (
            f"platforms.py cost for {model} deviates {error:.0%} from canonical "
            f"(platform={platform_cost:.8f}, canonical={canonical:.8f})"
        )


# ---------------------------------------------------------------------------
# Test 2: validator.py MODEL_COST_PER_1K models are all resolvable
# ---------------------------------------------------------------------------


class TestValidatorModelsResolvable:
    """Every model in hooks/validator.MODEL_COST_PER_1K must trace back to
    FALLBACK_MODEL_PRICING — either directly or as an alias whose cost
    matches a canonical entry.
    """

    @pytest.mark.unit
    def test_all_validator_models_resolvable(self) -> None:
        from traigent.hooks.validator import MODEL_COST_PER_1K

        # Build set of valid per-1K costs derived from FALLBACK_MODEL_PRICING.
        # Any alias in MODEL_COST_PER_1K must have a cost matching one of these.
        canonical_per_1k = set()
        for p in FALLBACK_MODEL_PRICING.values():
            avg = (p["input_cost_per_token"] + p["output_cost_per_token"]) / 2
            canonical_per_1k.add(round(avg * 1000, 12))

        unresolvable = []
        for model, cost in MODEL_COST_PER_1K.items():
            # Direct entry in FALLBACK_MODEL_PRICING
            if model in FALLBACK_MODEL_PRICING:
                continue
            # Alias — cost must match a canonical entry
            if round(cost, 12) in canonical_per_1k:
                continue
            unresolvable.append(model)

        assert not unresolvable, (
            f"These validator.py models are not traceable to "
            f"FALLBACK_MODEL_PRICING: {unresolvable}. "
            f"Add them to FALLBACK_MODEL_PRICING or as aliases in "
            f"_build_model_cost_per_1k()."
        )


# ---------------------------------------------------------------------------
# Test 3: validator.py costs match CostCalculator
# ---------------------------------------------------------------------------


class TestValidatorCostMatchesCalculator:
    """For models present in both validator and canonical, costs should agree.

    The validator uses a blended (input+output)/2 average per-1K rate applied
    to total tokens, while canonical uses separate input/output per-token rates.
    With asymmetric pricing (e.g. output 4x input), the blended approach
    inherently deviates by ~25-30% depending on input/output ratio.

    We use a 35% tolerance to allow for this structural mismatch, while still
    catching actual pricing errors (>2x drift).
    """

    BLENDED_TOLERANCE = 0.35

    @pytest.mark.unit
    def test_overlapping_models_within_tolerance(self) -> None:
        from traigent.hooks.validator import MODEL_COST_PER_1K

        mismatches = []

        for model, per_1k_cost in MODEL_COST_PER_1K.items():
            # Only compare models directly in FALLBACK_MODEL_PRICING.
            # Aliases (gpt-4, gpt-4-32k, etc.) intentionally map to different
            # canonical models for cost estimation, so they'll diverge from
            # what CostCalculator resolves for the alias name.
            if model not in FALLBACK_MODEL_PRICING:
                continue

            canonical = _canonical_cost(model)
            if canonical == 0:
                continue

            # Validator applies blended avg to total tokens
            validator_cost = per_1k_cost * (INPUT_TOKENS + OUTPUT_TOKENS) / 1000

            error = _relative_error(validator_cost, canonical)
            if error > self.BLENDED_TOLERANCE:
                mismatches.append(
                    f"{model}: validator={validator_cost:.8f}, "
                    f"canonical={canonical:.8f}, error={error:.0%}"
                )

        assert not mismatches, (
            f"Validator costs deviate >{self.BLENDED_TOLERANCE:.0%} from canonical:\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Test 4: handler.py fallback costs match CostCalculator
# ---------------------------------------------------------------------------


class TestHandlerFallbackMatchesCalculator:
    """Verify langchain handler._fallback_cost_estimate agrees with canonical."""

    @pytest.mark.unit
    def test_handler_fallback_within_tolerance(self) -> None:
        from traigent.integrations.langchain.handler import TraigentHandler

        handler = TraigentHandler.__new__(TraigentHandler)

        models_to_test = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-haiku",
            "claude-3-5-sonnet",
        ]

        mismatches = []
        for model in models_to_test:
            handler_cost = handler._fallback_cost_estimate(
                model, INPUT_TOKENS, OUTPUT_TOKENS
            )
            canonical = _canonical_cost(model)
            if canonical == 0:
                continue

            error = _relative_error(handler_cost, canonical)
            if error > TOLERANCE:
                mismatches.append(
                    f"{model}: handler={handler_cost:.8f}, "
                    f"canonical={canonical:.8f}, error={error:.0%}"
                )

        assert not mismatches, (
            f"Handler fallback costs deviate >{TOLERANCE:.0%} from canonical:\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Test 5: AST scanner finds zero P001/P002 errors (post-fix)
# ---------------------------------------------------------------------------


class TestNoOrphanPricingTables:
    """Run the AST pricing scanner and assert no errors remain."""

    @pytest.mark.unit
    def test_no_pricing_lint_errors(self) -> None:
        from tests.optimizer_validation.tools.lint_pricing_consistency import (
            Severity,
            lint_directory,
        )

        traigent_dir = Path(__file__).resolve().parents[3] / "traigent"
        issues = lint_directory(traigent_dir)
        errors = [i for i in issues if i.severity == Severity.ERROR]

        if errors:
            report = "\n".join(str(e) for e in errors)
            pytest.fail(
                f"Found {len(errors)} pricing lint errors:\n{report}\n\n"
                "Fix: Delegate to traigent.utils.cost_calculator instead of "
                "maintaining hardcoded pricing tables."
            )


# ---------------------------------------------------------------------------
# Test 6: intelligence.py fallback matches canonical
# ---------------------------------------------------------------------------


class TestIntelligenceFallbackMatchesCalculator:
    """Verify analytics/intelligence.py fallback pricing path is consistent.

    Exercises the ImportError fallback in fetch_current_pricing() by
    temporarily removing get_model_pricing_per_1k from the cost_calculator
    module namespace. The fallback derives per-1K rates from
    FALLBACK_MODEL_PRICING, so the test verifies those match canonical.
    """

    @pytest.mark.unit
    def test_intelligence_fallback_within_tolerance(self) -> None:
        import traigent.utils.cost_calculator as _cc_mod
        from traigent.analytics.intelligence import CostOptimizationAI

        ai = CostOptimizationAI()

        # Temporarily remove get_model_pricing_per_1k so the
        # `from ... import get_model_pricing_per_1k` inside
        # fetch_current_pricing raises ImportError, triggering fallback.
        saved = _cc_mod.get_model_pricing_per_1k
        del _cc_mod.get_model_pricing_per_1k
        try:
            pricing = ai.fetch_current_pricing(providers=["openai", "anthropic"])
        finally:
            _cc_mod.get_model_pricing_per_1k = saved

        mismatches = []
        for provider in ("openai", "anthropic"):
            if provider not in pricing:
                continue
            for model, rates in pricing[provider].items():
                # Per-1K input/output rates from the fallback path
                intel_cost = (
                    rates["input"] * INPUT_TOKENS / 1000
                    + rates["output"] * OUTPUT_TOKENS / 1000
                )
                canonical = _canonical_cost(model)
                if canonical == 0:
                    continue

                error = _relative_error(intel_cost, canonical)
                if error > TOLERANCE:
                    mismatches.append(
                        f"{model}: intelligence={intel_cost:.8f}, "
                        f"canonical={canonical:.8f}, error={error:.0%}"
                    )

        assert not mismatches, (
            f"Intelligence fallback costs deviate >{TOLERANCE:.0%} "
            f"from canonical:\n" + "\n".join(mismatches)
        )

    @pytest.mark.unit
    def test_intelligence_normal_path_returns_per_1k_rates(self) -> None:
        """Verify the normal path returns per-1K rates (same unit as fallback)."""
        from traigent.analytics.intelligence import CostOptimizationAI

        ai = CostOptimizationAI()
        pricing = ai.fetch_current_pricing(providers=["openai"])

        if "openai" not in pricing or "gpt-4o" not in pricing.get("openai", {}):
            pytest.skip("gpt-4o not available in normal pricing path")

        rates = pricing["openai"]["gpt-4o"]
        # Per-1K rates for gpt-4o should be in the range $0.001-$0.05
        # (not micro-dollar sample costs like 2e-05)
        assert rates["input"] > 0.0001, (
            f"Normal path input rate {rates['input']} looks like a sample cost, "
            "not a per-1K rate"
        )
        assert rates["output"] > 0.0001, (
            f"Normal path output rate {rates['output']} looks like a sample cost, "
            "not a per-1K rate"
        )
