"""@dataclass and @overload - should produce WARNINGs, not ERRORs.

We cannot statically infer the generated __init__ for @dataclass,
and @overload stubs don't represent actual implementations.
"""

from dataclasses import dataclass
from typing import overload


@dataclass
class Point:
    """Dataclass - __init__ is generated from fields."""

    x: float
    y: float


@dataclass
class Config:
    """Dataclass with defaults."""

    name: str
    value: int = 0
    enabled: bool = True


@dataclass
class Empty:
    """Empty dataclass - no fields, no __init__ args."""

    pass


class Converter:
    """Class with overloaded methods."""

    @overload
    def convert(self, value: int) -> str: ...

    @overload
    def convert(self, value: str) -> int: ...

    def convert(self, value):
        """Actual implementation."""
        if isinstance(value, int):
            return str(value)
        return int(value)


@overload
def parse(value: str) -> int: ...


@overload
def parse(value: bytes) -> str: ...


def parse(value):
    """Module-level overloaded function."""
    if isinstance(value, str):
        return int(value)
    return value.decode()


# === WARNING: @dataclass - cannot verify generated __init__ ===

# WARNING: Point has @dataclass, signature may differ
point1 = Point(1.0, 2.0)
point2 = Point(x=1.0, y=2.0)

# WARNING: Config has @dataclass, signature may differ
config1 = Config("test")
config2 = Config("test", 42)
config3 = Config(name="test", value=42, enabled=False)

# WARNING: Empty has @dataclass, signature may differ
empty = Empty()


# === WARNING: @overload - cannot verify overloaded signatures ===

# WARNING: parse has @overload, signature may differ
result1 = parse("42")
result2 = parse(b"hello")

# These calls might be wrong, but we can't verify due to overload
maybe_wrong1 = parse(42)  # No overload for int input
maybe_wrong2 = parse("a", "b")  # Too many args


# Note: Instance method overloads (Converter.convert) are skipped in V1
# because they require self.method() tracking
