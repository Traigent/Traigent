"""Class inheritance - handling missing __init__.

If a child class has no explicit __init__, we cannot easily determine
the parent's __init__ signature without MRO resolution, so we WARNING.
"""


class BaseClass:
    """Base class with explicit __init__."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class ChildWithInit(BaseClass):
    """Child with its own __init__ - can verify."""

    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z


class ChildWithoutInit(BaseClass):
    """Child without __init__ - inherits from parent.

    We cannot easily resolve this statically without MRO,
    so we should WARNING.
    """

    def extra_method(self):
        return self.x + self.y


class GrandchildWithInit(ChildWithoutInit):
    """Grandchild with its own __init__."""

    def __init__(self, x, y, w):
        super().__init__(x, y)
        self.w = w


class GrandchildWithoutInit(ChildWithInit):
    """Grandchild without __init__ - inherits from ChildWithInit."""

    pass


class MultipleInheritance(BaseClass, dict):
    """Multiple inheritance - complex MRO."""

    pass


# === CORRECT: Explicit __init__ can be verified ===

# BaseClass requires 2 args
ok1 = BaseClass(1, 2)

# ChildWithInit requires 3 args
ok2 = ChildWithInit(1, 2, 3)

# GrandchildWithInit requires 3 args
ok3 = GrandchildWithInit(1, 2, 3)


# === ERRORS: Explicit __init__ - definite mismatches ===

# ERROR: BaseClass() missing 1 required positional arg(s)
err1 = BaseClass(1)

# ERROR: ChildWithInit() missing 2 required positional arg(s)
err2 = ChildWithInit(1)

# ERROR: ChildWithInit() takes 3 positional args, got 4
err3 = ChildWithInit(1, 2, 3, 4)


# === WARNING: No explicit __init__ - cannot verify ===

# WARNING: ChildWithoutInit has no explicit __init__, inherits from parent
# We can't easily verify these without MRO resolution
w1 = ChildWithoutInit(1, 2)  # Probably correct (inherits 2-arg __init__)
w2 = ChildWithoutInit(1)  # Probably wrong
w3 = ChildWithoutInit(1, 2, 3)  # Probably wrong

# WARNING: GrandchildWithoutInit has no explicit __init__
w4 = GrandchildWithoutInit(1, 2, 3)  # Inherits 3-arg from ChildWithInit
w5 = GrandchildWithoutInit(1, 2)  # Probably wrong

# WARNING: MultipleInheritance - complex inheritance
w6 = MultipleInheritance(1, 2)
