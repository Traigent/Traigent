"""Auto-optimization config generator.

Public API:

- ``generate_config()`` — one-shot config generation from a Python file
- ``ConfigGeneratorPipeline`` — fine-grained control over subsystems
- ``AutoConfigResult`` — the complete output type
"""

from __future__ import annotations

import ast
from pathlib import Path

from traigent.config_generator.llm_backend import ConfigGenLLM, LiteLLMBackend
from traigent.config_generator.pipeline import ConfigGeneratorPipeline
from traigent.config_generator.types import AutoConfigResult
from traigent.tuned_variables.detector import TunedVariableDetector

__all__ = [
    "AutoConfigResult",
    "ConfigGeneratorPipeline",
    "generate_config",
]


def generate_config(
    file_path: str | Path,
    *,
    function_name: str | None = None,
    enrich: bool = False,
    model: str = "gpt-4o-mini",
    budget_usd: float = 0.10,
    subsystems: frozenset[str] | None = None,
) -> AutoConfigResult:
    """Generate a complete optimization config from a Python file.

    Parameters
    ----------
    file_path:
        Path to the Python source file.
    function_name:
        Specific function to analyze (``None`` = all top-level).
    enrich:
        If ``True``, use LLM for richer results.
    model:
        LLM model name for enrichment.
    budget_usd:
        Maximum LLM spend for enrichment.
    subsystems:
        Which subsystems to run (``None`` = all).
    """
    file_path = Path(file_path)
    source_code = file_path.read_text()

    # Scope source to the selected function (if specified) so that
    # classification/objectives/safety are not influenced by unrelated
    # functions in the same file.
    scoped_source = source_code
    if function_name:
        scoped_source = _extract_function_source(source_code, function_name)

    # Run detection
    detector = TunedVariableDetector()
    detection_results = detector.detect_from_file(file_path, function_name)

    # Build LLM backend
    llm: ConfigGenLLM | None = None
    if enrich:
        llm = LiteLLMBackend(model=model, budget_usd=budget_usd)

    # Run pipeline
    pipeline = ConfigGeneratorPipeline(llm=llm, subsystems=subsystems)
    return pipeline.generate(detection_results, source_code=scoped_source)


def _extract_function_source(source: str, function_name: str) -> str:
    """Extract the source of a specific function from a module source string.

    Raises
    ------
    ValueError
        If the named function is not found in the source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return ast.get_source_segment(source, node) or source

    raise ValueError(
        f"Function '{function_name}' not found in source. "
        f"Check the function name and try again."
    )
