"""
LangChain optimization problems for Traigent SDK demonstrations.

This package provides a collection of realistic, challenging optimization problems
that demonstrate Traigent's capabilities across different domains and use cases.
"""

from typing import Dict, List, Type

from .base import BaseLangChainProblem, ProblemConfig, ProblemDefinition, ProblemMetric

# Problem registry - will be populated as problems are imported
_PROBLEM_REGISTRY: Dict[str, Type[ProblemDefinition]] = {}


def register_problem(name: str, problem_class: Type[ProblemDefinition]):
    """Register a problem class with the global registry."""
    _PROBLEM_REGISTRY[name] = problem_class


def get_available_problems() -> List[str]:
    """Get list of available problem names."""
    return list(_PROBLEM_REGISTRY.keys())


def get_problem_class(name: str) -> Type[ProblemDefinition]:
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
    """Load all available problems into the registry."""
    import importlib
    import sys
    from pathlib import Path

    # Clear registry to ensure fresh load
    _PROBLEM_REGISTRY.clear()

    # Get the directory of this module
    module_dir = Path(__file__).parent

    # Temporarily add parent directory to path for imports from examples dir
    old_path = sys.path.copy()
    sys.path.insert(0, str(module_dir.parent))

    try:
        # Find all Python files in the directory
        for file_path in module_dir.glob("*.py"):
            # Skip __init__.py and base.py
            if file_path.name in ["__init__.py", "base.py"]:
                continue

            # Skip test files
            if file_path.name.startswith("test_"):
                continue

            # Get module name (without .py extension)
            module_name = file_path.stem

            try:
                # Try both import paths to handle different execution contexts
                module = None
                for import_path in [
                    f"examples.langchain_problems.{module_name}",  # From project root
                    f"langchain_problems.{module_name}",  # From examples dir
                    f".{module_name}",  # Relative import
                ]:
                    try:
                        if import_path.startswith("."):
                            module = importlib.import_module(
                                import_path, package=__name__
                            )
                        else:
                            module = importlib.import_module(import_path)
                        break  # Successfully imported
                    except ImportError:
                        continue

                if module is None:
                    # Could not import module - skip silently
                    pass
                else:
                    # Module imported successfully

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
                                problem_name = module_name
                                register_problem(problem_name, attr)
                                break

            except Exception as e:
                # Log other errors but continue
                print(f"Warning: Failed to load problem module {module_name}: {e}")
    finally:
        # Restore original path
        sys.path = old_path


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
