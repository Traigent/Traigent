"""Config generator pipeline.

Orchestrates all 6 subsystems in dependency order to produce
an ``AutoConfigResult`` from a ``DetectionResult``.
"""

from __future__ import annotations

from traigent.config_generator.agent_classifier import (
    ClassificationResult,
    classify_agent,
)
from traigent.config_generator.llm_backend import ConfigGenLLM
from traigent.config_generator.subsystems.benchmarks import generate_benchmarks
from traigent.config_generator.subsystems.objectives import generate_objectives
from traigent.config_generator.subsystems.safety_constraints import (
    generate_safety_constraints,
)
from traigent.config_generator.subsystems.structural_constraints import (
    generate_structural_constraints,
)
from traigent.config_generator.subsystems.tvar_ranges import generate_tvar_specs
from traigent.config_generator.subsystems.tvar_recommendations import (
    generate_recommendations,
)
from traigent.config_generator.types import AutoConfigResult, SafetySpec
from traigent.tuned_variables.detection_types import DetectionResult

# All subsystem names that can be selectively enabled.
ALL_SUBSYSTEMS = frozenset(
    {
        "tvars",
        "objectives",
        "safety",
        "structural",
        "benchmarks",
        "recommendations",
    }
)


class ConfigGeneratorPipeline:
    """Orchestrate auto-config generation from detection results.

    Parameters
    ----------
    llm:
        Optional LLM backend for enrichment.  ``None`` = preset-only mode.
    subsystems:
        Which subsystems to run.  ``None`` means all.
    """

    def __init__(
        self,
        llm: ConfigGenLLM | None = None,
        subsystems: frozenset[str] | None = None,
    ) -> None:
        self._llm = llm
        self._subsystems = subsystems or ALL_SUBSYSTEMS

    def generate(
        self,
        detection_results: list[DetectionResult],
        source_code: str = "",
    ) -> AutoConfigResult:
        """Run the pipeline and return the complete config result.

        Parameters
        ----------
        detection_results:
            One or more ``DetectionResult`` from ``TunedVariableDetector``.
        source_code:
            Original function source code (for LLM context).
        """
        warnings: list[str] = []

        # 1. TVars with ranges (foundation)
        tvars = []
        if "tvars" in self._subsystems:
            tvars = generate_tvar_specs(
                detection_results,
                llm=self._llm,
                source_code=source_code,
            )

        # 2. Agent classification (shared — used by objectives, safety, benchmarks, recommendations)
        _CLASSIFICATION_CONSUMERS = {
            "objectives",
            "safety",
            "benchmarks",
            "recommendations",
        }
        classification: ClassificationResult | None = None
        if source_code and self._subsystems & _CLASSIFICATION_CONSUMERS:
            classification = classify_agent(source_code, llm=self._llm)

        # 3. Objectives
        objectives = []
        if "objectives" in self._subsystems:
            objectives = generate_objectives(
                source_code,
                tvars,
                llm=self._llm,
                classification=classification,
            )

        # 4. Safety constraints
        safety: list[SafetySpec] = []
        if "safety" in self._subsystems:
            safety_result = generate_safety_constraints(
                source_code,
                llm=self._llm,
                classification=classification,
            )
            safety = safety_result[0]

        # 5. Structural constraints
        structural = []
        if "structural" in self._subsystems:
            structural = generate_structural_constraints(
                tvars,
                llm=self._llm,
                source_code=source_code,
            )

        # 6. Benchmarks
        benchmarks = []
        if "benchmarks" in self._subsystems:
            benchmarks = generate_benchmarks(
                source_code,
                llm=self._llm,
                classification=classification,
            )

        # 7. TVAR recommendations
        recommendations = []
        if "recommendations" in self._subsystems:
            recommendations = generate_recommendations(
                tvars,
                llm=self._llm,
                source_code=source_code,
                classification=classification,
            )

        # Collect LLM stats if available
        llm_calls = 0
        llm_cost = 0.0
        if self._llm is not None and hasattr(self._llm, "calls_made"):
            llm_calls = self._llm.calls_made
        if self._llm is not None and hasattr(self._llm, "_spent_usd"):
            llm_cost = self._llm._spent_usd

        return AutoConfigResult(
            tvars=tuple(tvars),
            objectives=tuple(objectives),
            benchmarks=tuple(benchmarks),
            safety_constraints=tuple(safety),
            structural_constraints=tuple(structural),
            recommendations=tuple(recommendations),
            agent_type=classification.agent_type if classification else None,
            warnings=tuple(warnings),
            llm_calls_made=llm_calls,
            llm_cost_usd=llm_cost,
        )
