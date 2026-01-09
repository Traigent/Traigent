"""Auto-discovery of callables from modules for TunedCallable definitions (P-5).

This module provides utilities to automatically discover callable functions
from Python modules based on patterns, signatures, or decorators.

Example:
    ```python
    import my_retrieval_module
    from traigent.tuned_variables.discovery import discover_callables

    # Discover all public functions
    callables = discover_callables(my_retrieval_module)

    # Discover functions matching a pattern
    callables = discover_callables(
        my_retrieval_module,
        pattern=r"^retrieve_"
    )

    # Discover functions with a specific signature
    from inspect import Parameter, signature
    callables = discover_callables(
        my_retrieval_module,
        required_params=["query", "k"]
    )
    ```
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class CallableInfo:
    """Information about a discovered callable.

    Attributes:
        name: The callable's name (function name or __name__).
        callable: The callable object itself.
        signature: The function's signature.
        module: The module the callable was discovered in.
        docstring: The callable's docstring, if any.
        tags: Extracted tags from docstring or attributes.
    """

    name: str
    callable: Callable[..., Any]
    signature: inspect.Signature
    module: str
    docstring: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def matches_params(self, required_params: list[str]) -> bool:
        """Check if this callable has all required parameters.

        Args:
            required_params: List of parameter names that must be present.

        Returns:
            True if all required parameters are present.
        """
        param_names = set(self.signature.parameters.keys())
        return all(p in param_names for p in required_params)

    def matches_return_type(self, expected_type: type) -> bool:
        """Check if this callable's return annotation matches expected type.

        Args:
            expected_type: The expected return type.

        Returns:
            True if return annotation matches or is Any/missing.
        """
        return_annotation = self.signature.return_annotation
        if return_annotation is inspect.Parameter.empty:
            return True  # No annotation, assume compatible
        # Check exact match or subclass
        try:
            return return_annotation is expected_type or issubclass(
                return_annotation, expected_type
            )
        except TypeError:
            # Can't check subclass (e.g., for generics)
            return True


def discover_callables(
    module: ModuleType,
    *,
    pattern: str | None = None,
    include_private: bool = False,
    required_params: list[str] | None = None,
    return_type: type | None = None,
) -> dict[str, CallableInfo]:
    """Auto-discover callables from a module.

    Scans a module for callable functions matching optional filters.

    Args:
        module: The module to scan for callables.
        pattern: Optional regex pattern to match function names against.
            Only functions whose names match the pattern are included.
        include_private: Whether to include private functions (starting with _).
            Defaults to False.
        required_params: Optional list of parameter names that must be present.
            Functions missing any of these parameters are excluded.
        return_type: Optional expected return type. Functions with incompatible
            return annotations are excluded.

    Returns:
        Dictionary mapping function names to CallableInfo objects.

    Example:
        ```python
        # Discover all retrieval functions
        retrievers = discover_callables(
            my_module,
            pattern=r"^(retrieve|search)_",
            required_params=["query"]
        )

        # Use in TunedCallable
        from traigent.api.parameter_ranges import Choices
        retriever_choices = Choices(list(retrievers.keys()))
        ```
    """
    compiled_pattern = re.compile(pattern) if pattern else None
    result: dict[str, CallableInfo] = {}

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip private functions unless explicitly included
        if not include_private and name.startswith("_"):
            continue

        # Skip functions defined in other modules (imports)
        if obj.__module__ != module.__name__:
            continue

        # Apply name pattern filter
        if compiled_pattern and not compiled_pattern.match(name):
            continue

        # Get signature
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue  # Skip functions with invalid signatures

        # Build CallableInfo
        info = CallableInfo(
            name=name,
            callable=obj,
            signature=sig,
            module=module.__name__,
            docstring=obj.__doc__,
            tags=_extract_tags(obj),
        )

        # Apply required_params filter
        if required_params and not info.matches_params(required_params):
            continue

        # Apply return_type filter
        if return_type and not info.matches_return_type(return_type):
            continue

        result[name] = info

    return result


def discover_callables_by_decorator(
    module: ModuleType,
    decorator_attr: str = "__traigent_callable__",
    *,
    include_private: bool = False,
) -> dict[str, CallableInfo]:
    """Discover callables that have been marked with a specific decorator attribute.

    Useful for discovering functions that have been explicitly registered
    as tunable callables using a decorator.

    Args:
        module: The module to scan.
        decorator_attr: The attribute name set by the decorator.
            Defaults to "__traigent_callable__".
        include_private: Whether to include private functions.

    Returns:
        Dictionary mapping function names to CallableInfo objects.

    Example:
        ```python
        # In module my_retrievers.py:
        def traigent_callable(func):
            func.__traigent_callable__ = True
            return func

        @traigent_callable
        def similarity_search(query: str, k: int = 5) -> list:
            ...

        # Discovery:
        callables = discover_callables_by_decorator(my_retrievers)
        ```
    """
    result: dict[str, CallableInfo] = {}

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not include_private and name.startswith("_"):
            continue

        if obj.__module__ != module.__name__:
            continue

        # Check for decorator attribute
        if not hasattr(obj, decorator_attr):
            continue

        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue

        info = CallableInfo(
            name=name,
            callable=obj,
            signature=sig,
            module=module.__name__,
            docstring=obj.__doc__,
            tags=_extract_tags(obj),
        )

        result[name] = info

    return result


def filter_by_signature(
    callables: dict[str, CallableInfo],
    target_signature: inspect.Signature,
    *,
    strict: bool = False,
) -> dict[str, CallableInfo]:
    """Filter callables to those matching a target signature.

    Args:
        callables: Dictionary of CallableInfo objects to filter.
        target_signature: The signature to match against.
        strict: If True, require exact parameter match. If False,
            allow callables with extra parameters that have defaults.

    Returns:
        Filtered dictionary of compatible callables.
    """
    target_params = list(target_signature.parameters.items())
    result: dict[str, CallableInfo] = {}

    for name, info in callables.items():
        if _signature_compatible(info.signature, target_params, strict):
            result[name] = info

    return result


def _signature_compatible(
    sig: inspect.Signature,
    target_params: list[tuple[str, inspect.Parameter]],
    strict: bool,
) -> bool:
    """Check if a signature is compatible with target parameters.

    Args:
        sig: The signature to check.
        target_params: List of (name, Parameter) tuples from target signature.
        strict: Whether to require exact match.

    Returns:
        True if compatible.
    """
    sig_params = dict(sig.parameters)

    for name, target_param in target_params:
        if name not in sig_params:
            return False

        if strict:
            # Check kind compatibility
            sig_param = sig_params[name]
            if sig_param.kind != target_param.kind:
                return False

            # Check annotation compatibility (if both have annotations)
            if (
                target_param.annotation is not inspect.Parameter.empty
                and sig_param.annotation is not inspect.Parameter.empty
            ):
                if target_param.annotation != sig_param.annotation:
                    return False

    if strict:
        # In strict mode, extra params without defaults are not allowed
        for name, param in sig_params.items():
            if name not in dict(target_params):
                if param.default is inspect.Parameter.empty:
                    return False

    return True


def _extract_tags(func: Callable[..., Any]) -> tuple[str, ...]:
    """Extract tags from a function's attributes or docstring.

    Looks for:
    - __tags__ attribute (list/tuple of strings)
    - Tags in docstring: "Tags: tag1, tag2, tag3"

    Args:
        func: The function to extract tags from.

    Returns:
        Tuple of tag strings.
    """
    tags: list[str] = []

    # Check __tags__ attribute
    if hasattr(func, "__tags__"):
        attr_tags = func.__tags__
        if isinstance(attr_tags, (list, tuple)):
            tags.extend(str(t) for t in attr_tags)

    # Check docstring for Tags: line
    if func.__doc__:
        for line in func.__doc__.split("\n"):
            line = line.strip()
            if line.lower().startswith("tags:"):
                tag_str = line[5:].strip()
                tags.extend(t.strip() for t in tag_str.split(",") if t.strip())
                break

    return tuple(tags)
