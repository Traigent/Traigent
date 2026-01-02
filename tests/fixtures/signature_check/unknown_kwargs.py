"""Unknown keyword arguments - should raise ERRORs."""


def accepts_a_b(a, b):
    """Only accepts a and b."""
    return a + b


def accepts_x_y_z(x, y, z=10):
    """Accepts x, y, and z."""
    return x + y + z


def accepts_kwargs(a, **kwargs):
    """Accepts a and any kwargs (should NOT error on unknown kwargs)."""
    return a, kwargs


class StrictInit:
    """Class with strict __init__ params."""

    def __init__(self, name, value):
        self.name = name
        self.value = value


# === ERRORS: Unknown keyword arguments ===

# ERROR: accepts_a_b() got unexpected keyword arg(s): {'c'}
err1 = accepts_a_b(a=1, b=2, c=3)

# ERROR: accepts_a_b() got unexpected keyword arg(s): {'x', 'y'}
err2 = accepts_a_b(1, 2, x=1, y=2)

# ERROR: accepts_x_y_z() got unexpected keyword arg(s): {'w'}
err3 = accepts_x_y_z(1, 2, w=100)

# ERROR: StrictInit() got unexpected keyword arg(s): {'extra'}
err4 = StrictInit(name="test", value=42, extra="bad")


# === CORRECT: **kwargs accepts anything ===

# These should NOT error
ok1 = accepts_kwargs(1, b=2, c=3, d=4)
ok2 = accepts_kwargs(a=1, unknown_param=999)
