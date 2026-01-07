"""TunedCallable - composition pattern for function-valued variables.

TunedCallable is NOT a ParameterRange subclass. It's a container for
callable functions with optional per-callable parameters. Use .as_choices()
to convert to a Choices for inclusion in configuration space.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices, ParameterRange


@dataclass
class TunedCallable:
    """A callable component with its own tunable parameters.

    NOT a ParameterRange subclass. Use Choices when you want to
    select between callables as a configuration value.

    Example:
        ```python
        retrievers = TunedCallable(
            name="retriever",
            callables={
                "similarity": similarity_fn,
                "mmr": mmr_fn,
            },
            parameters={
                "mmr": {"lambda_mult": Range(0.0, 1.0)},
            },
        )

        # Use in optimization
        @traigent.optimize(
            retriever=retrievers.as_choices(),
        )
        def my_agent(...):
            config = traigent.get_config()
            fn = retrievers.get_callable(config["retriever"])
            result = fn(...)
        ```
    """

    name: str
    callables: dict[str, Callable] = field(default_factory=dict)
    parameters: dict[str, dict[str, ParameterRange]] | None = None
    description: str | None = None

    def as_choices(self) -> Choices:
        """Convert to Choices for use in configuration space.

        Returns:
            Choices instance with callable names as options
        """
        from traigent.api.parameter_ranges import Choices

        return Choices(list(self.callables.keys()), name=self.name)

    def get_callable(self, name: str) -> Callable:
        """Get callable by name.

        Args:
            name: Name of the callable

        Returns:
            The callable function

        Raises:
            KeyError: If name not found
        """
        if name not in self.callables:
            raise KeyError(
                f"Callable '{name}' not found. "
                f"Available: {list(self.callables.keys())}"
            )
        return self.callables[name]

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a callable by name.

        Args:
            name: Name of the callable
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            Result of the callable invocation
        """
        fn = self.get_callable(name)
        return fn(*args, **kwargs)

    def get_parameters(self, name: str) -> dict[str, ParameterRange]:
        """Get parameters specific to a callable.

        Args:
            name: Name of the callable

        Returns:
            Dictionary of parameter names to ParameterRange
        """
        if self.parameters is None:
            return {}
        return self.parameters.get(name, {})

    def register(
        self,
        name: str,
        callable: Callable,
        *,
        parameters: dict[str, ParameterRange] | None = None,
    ) -> TunedCallable:
        """Register a new callable.

        Args:
            name: Name for the callable
            callable: The function to register
            parameters: Optional per-callable parameters

        Returns:
            Self for chaining
        """
        self.callables[name] = callable
        if parameters:
            if self.parameters is None:
                self.parameters = {}
            self.parameters[name] = parameters
        return self

    def __contains__(self, name: str) -> bool:
        """Check if a callable is registered."""
        return name in self.callables

    def __len__(self) -> int:
        """Return number of registered callables."""
        return len(self.callables)

    def __iter__(self):
        """Iterate over callable names."""
        return iter(self.callables)
