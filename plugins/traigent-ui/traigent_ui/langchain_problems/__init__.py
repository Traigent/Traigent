"""
LangChain optimization problems for Traigent SDK demonstrations.

This package provides a collection of realistic, challenging optimization problems
that demonstrate Traigent's capabilities across different domains and use cases.
"""

import logging

from .base import BaseLangChainProblem, ProblemConfig, ProblemDefinition, ProblemMetric

logger = logging.getLogger(__name__)

# Problem registry - will be populated as problems are imported
_PROBLEM_REGISTRY: dict[str, type[ProblemDefinition]] = {}

# Registrations made by problem modules at import time, keyed by the module's
# full name. A cached module does not re-run its register_problem() call, so
# load_all_problems() replays these to keep repeated loads identical.
_MODULE_REGISTRATIONS: dict[str, dict[str, type[ProblemDefinition]]] = {}


def register_problem(name: str, problem_class: type[ProblemDefinition]):
    """Register a problem class with the global registry."""
    _PROBLEM_REGISTRY[name] = problem_class
    _MODULE_REGISTRATIONS.setdefault(problem_class.__module__, {})[name] = problem_class


def get_available_problems() -> list[str]:
    """Get list of available problem names."""
    return list(_PROBLEM_REGISTRY.keys())


def get_problem_class(name: str) -> type[ProblemDefinition]:
    """Get problem class by name."""
    if name not in _PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem: {name}. Available problems: {get_available_problems()}"
        )
    return _PROBLEM_REGISTRY[name]


def create_problem(name: str, **kwargs) -> ProblemDefinition:
    """Create problem instance by name."""
    problem_class = get_problem_class(name)
    return problem_class(**kwargs)


def load_all_problems():
    """Load all available problems into the registry.

    A problem module that fails to import is logged as a warning naming the
    module; it is never skipped silently.
    """
    import importlib
    from pathlib import Path

    # Clear registry to ensure fresh load
    _PROBLEM_REGISTRY.clear()

    module_dir = Path(__file__).parent

    for file_path in sorted(module_dir.glob("*.py")):
        # Skip __init__.py and base.py
        if file_path.name in ["__init__.py", "base.py"]:
            continue

        # Skip test files
        if file_path.name.startswith("test_"):
            continue

        module_name = file_path.stem

        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
        except (Exception, SystemExit) as e:
            # SystemExit: several modules call sys.exit(1) when langchain is
            # missing; one such module must not abort loading the rest.
            logger.warning(
                "Failed to load problem module %s: %s: %s",
                module_name,
                type(e).__name__,
                e,
            )
            continue

        # Replay registrations a cached module made on its first import.
        _PROBLEM_REGISTRY.update(_MODULE_REGISTRATIONS.get(module.__name__, {}))

        # Check if it has already registered itself
        # If not, try to find and register problem classes manually
        if module_name not in [
            k.replace("_", "").lower() for k in _PROBLEM_REGISTRY.keys()
        ]:
            # Look for classes that inherit from ProblemDefinition
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "create_function")
                    and hasattr(attr, "create_dataset")
                    and attr.__name__
                    not in ["ProblemDefinition", "BaseLangChainProblem"]
                ):
                    # Found a problem class, register it
                    register_problem(module_name, attr)
                    break


def print_available_problems():
    """Print information about all available problems."""
    load_all_problems()

    if not _PROBLEM_REGISTRY:
        print("❌ No problems available. Check problem implementations.")
        return

    print("🎯 Available LangChain Optimization Problems:")
    print("=" * 50)

    for name, problem_class in _PROBLEM_REGISTRY.items():
        # Create temporary instance to get info (use default config)
        try:
            # Most problems should have a default config class method
            if hasattr(problem_class, "get_default_config"):
                temp_config = problem_class.get_default_config()
                temp_problem = problem_class(temp_config)
                print(f"\n📋 {name}")
                print(f"   Description: {temp_problem.description}")
                print(f"   Difficulty: {temp_problem.config.difficulty_level}")
                print(
                    f"   Metrics: {', '.join(m.name for m in temp_problem.config.metrics)}"
                )
            else:
                print(f"\n📋 {name}")
                print(f"   Class: {problem_class.__name__}")
        except Exception as e:
            print(f"\n📋 {name}")
            print(f"   Class: {problem_class.__name__}")
            print(f"   ⚠️  Could not load details: {e}")


# Export main classes and functions
__all__ = [
    "ProblemDefinition",
    "ProblemConfig",
    "ProblemMetric",
    "BaseLangChainProblem",
    "register_problem",
    "get_available_problems",
    "get_problem_class",
    "create_problem",
    "load_all_problems",
    "print_available_problems",
]
