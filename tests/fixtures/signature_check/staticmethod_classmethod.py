"""@staticmethod and @classmethod handling.

- @staticmethod: No self, count all params as required
- @classmethod: Has cls, but should skip it when counting required args
- Regular method: Has self, should skip it when counting required args
"""


class Calculator:
    """Class with various method types."""

    def __init__(self, value):
        """Constructor requires 1 arg (value), self is implicit."""
        self.value = value

    def add(self, x):
        """Instance method - self is implicit, needs 1 arg."""
        return self.value + x

    def multiply(self, x, y):
        """Instance method - self is implicit, needs 2 args."""
        return self.value * x * y

    @staticmethod
    def static_add(a, b):
        """Static method - no self/cls, needs exactly 2 args."""
        return a + b

    @staticmethod
    def static_triple(a, b, c):
        """Static method - needs exactly 3 args."""
        return a + b + c

    @classmethod
    def from_string(cls, s):
        """Class method - cls is implicit, needs 1 arg."""
        return cls(int(s))

    @classmethod
    def create_pair(cls, a, b):
        """Class method - cls is implicit, needs 2 args."""
        return cls(a), cls(b)


# === CORRECT CALLS ===

# Constructor
obj = Calculator(10)
obj2 = Calculator(value=20)

# Instance methods (called on class, would need self at runtime)
# Note: We skip self.method() calls in V1, so these are for class-level calls

# Static methods - no self/cls
ok1 = Calculator.static_add(1, 2)
ok2 = Calculator.static_triple(1, 2, 3)

# Class methods - cls is implicit
ok3 = Calculator.from_string("42")
ok4 = Calculator.create_pair(1, 2)


# === ERRORS ===

# ERROR: Calculator() missing 1 required positional arg(s)
err1 = Calculator()

# ERROR: static_add() takes 2 positional args, got 3
err2 = Calculator.static_add(1, 2, 3)

# ERROR: static_add() missing 1 required positional arg(s)
err3 = Calculator.static_add(1)

# ERROR: static_triple() missing 2 required positional arg(s)
err4 = Calculator.static_triple(1)

# ERROR: from_string() missing 1 required positional arg(s)
err5 = Calculator.from_string()

# ERROR: create_pair() missing 1 required positional arg(s)
err6 = Calculator.create_pair(1)

# ERROR: create_pair() takes 2 positional args, got 3
err7 = Calculator.create_pair(1, 2, 3)
