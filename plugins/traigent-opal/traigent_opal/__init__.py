"""Traigent OPAL plugin.

Provides a pragmatic bridge from OPAL-style declarations to Traigent's
`@traigent.optimize(...)` decorator kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from traigent.plugins import FEATURE_TVL, FeaturePlugin
from traigent.utils.logging import get_logger

from .api import (
    CallableTemplate,
    ChanceConstraintSpec,
    ConstraintSpec,
    DomainChoices,
    DomainRange,
    ObjectiveSpec,
    ProgramBuilder,
    ProgramSpec,
    SymbolRef,
    TunedVariable,
    TVarSpec,
    callable_template,
    chance_constraint,
    choices,
    constraint,
    frange,
    maximize,
    minimize,
    opal_program,
    program,
    tv,
    tvar,
    when,
)
from .compiler import (
    ObjectiveDecl,
    OpalCompilationArtifact,
    OpalCompileError,
    compile_opal_file,
    compile_opal_source,
    compile_opal_spec,
)

logger = get_logger(__name__)

__version__ = "0.3.0"

__all__ = [
    "OpalPlugin",
    "ProgramSpec",
    "ProgramBuilder",
    "TVarSpec",
    "SymbolRef",
    "TunedVariable",
    "DomainChoices",
    "DomainRange",
    "CallableTemplate",
    "ObjectiveSpec",
    "ConstraintSpec",
    "ChanceConstraintSpec",
    "ObjectiveDecl",
    "OpalCompilationArtifact",
    "OpalCompileError",
    "compile_opal_source",
    "compile_opal_file",
    "compile_opal_spec",
    "program",
    "tvar",
    "choices",
    "frange",
    "tv",
    "opal_program",
    "callable_template",
    "maximize",
    "minimize",
    "constraint",
    "when",
    "chance_constraint",
    "opal_optimize",
]


class OpalPlugin(FeaturePlugin):
    """Plugin that exposes OPAL compilation helpers for Traigent users."""

    @property
    def name(self) -> str:
        return "traigent-opal"

    @property
    def version(self) -> str:
        return __version__

    @property
    def description(self) -> str:
        return "Compile OPAL-style declarations into Traigent optimization kwargs"

    @property
    def author(self) -> str:
        return "Traigent Team"

    @property
    def dependencies(self) -> list[str]:
        return ["traigent"]

    def provides_features(self) -> list[str]:
        """Expose TVL/OPAL support as a feature capability."""
        return [FEATURE_TVL]

    def initialize(self) -> None:
        logger.info("Traigent OPAL plugin initialized")

    def get_feature_impl(self, feature: str) -> Any | None:
        if feature == FEATURE_TVL:
            return {
                "compile_opal_source": compile_opal_source,
                "compile_opal_file": compile_opal_file,
                "compile_opal_spec": compile_opal_spec,
                "opal_optimize": opal_optimize,
            }
        return None


def opal_optimize(
    spec_or_source: ProgramSpec | str | Path,
    **override_kwargs: Any,
) -> Callable[[Callable[..., Any]], Any]:
    """Build a Traigent optimize decorator from OPAL declarations.

    Args:
        spec_or_source: ProgramSpec object, OPAL source text, or a file path.
        **override_kwargs: Explicit overrides merged into compiled kwargs.

    Returns:
        A decorator equivalent to `@traigent.optimize(**compiled_kwargs)`.
    """
    import traigent

    if isinstance(spec_or_source, ProgramSpec):
        artifact = compile_opal_spec(spec_or_source)
    else:
        artifact = (
            compile_opal_file(spec_or_source)
            if _looks_like_path(spec_or_source)
            else compile_opal_source(str(spec_or_source))
        )
    kwargs = artifact.to_optimize_kwargs()
    kwargs.update(override_kwargs)
    return traigent.optimize(**kwargs)


def _looks_like_path(value: str | Path) -> bool:
    """Heuristic path detection for API ergonomics."""
    if isinstance(value, Path):
        return True
    if "\n" in value:
        return False
    if value.endswith(".opal"):
        return True
    if "/" in value or "\\" in value:
        return True
    # Avoid filesystem probing on arbitrary bare strings (e.g. "model").
    # Bare non-path strings are treated as source text unless the caller uses Path.
    return False
