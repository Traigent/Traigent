"""Subsystem 3: Benchmark matching and generation.

Matches the classified agent type to a benchmark template from the
catalog and optionally uses LLM to generate sample evaluation examples.
"""

from __future__ import annotations

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.benchmark_catalog import (
    get_benchmark_for_agent_type,
)
from traigent.config_generator.types import BenchmarkSpec


def generate_benchmarks(
    source_code: str,
    *,
    llm: ConfigGenLLM | None = None,
    classification: ClassificationResult | None = None,
) -> list[BenchmarkSpec]:
    """Generate benchmark suggestions based on agent classification.

    Parameters
    ----------
    source_code:
        The function source code (used for LLM context).
    llm:
        Optional LLM for generating sample examples.
    classification:
        Pre-computed agent classification.
    """
    agent_type = classification.agent_type if classification else "general_llm"

    # Catalog match
    benchmark = get_benchmark_for_agent_type(agent_type)
    if benchmark is None:
        benchmark = get_benchmark_for_agent_type("general_llm")
    if benchmark is None:
        return []

    # LLM enrichment: generate sample examples
    if llm is not None:
        enriched = _llm_generate_examples(benchmark, source_code, llm)
        if enriched is not None:
            return [enriched]

    return [benchmark]


def _llm_generate_examples(
    benchmark: BenchmarkSpec,
    source_code: str,
    llm: ConfigGenLLM,
) -> BenchmarkSpec | None:
    """Ask LLM to generate sample evaluation examples."""
    import json

    prompt = (
        f"Generate 3 sample evaluation examples for a '{benchmark.name}' benchmark.\n\n"
        f"Expected format: {json.dumps(benchmark.example_schema, indent=2)}\n\n"
        "Source code being evaluated:\n"
        f"```python\n{source_code[:2000]}\n```\n\n"
        "Reply with ONLY a JSON array of 3 examples, each with 'input' and 'output' keys.\n"
    )

    try:
        response = llm.complete(prompt, max_tokens=1024)
    except BudgetExhausted:
        return None

    text = response.strip()
    if "```" in text:
        for part in text.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("["):
                text = stripped
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    return BenchmarkSpec(
        name=benchmark.name,
        description=benchmark.description,
        example_schema=benchmark.example_schema,
        source="llm",
        sample_examples=tuple(data),
    )
