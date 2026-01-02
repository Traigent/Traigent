"""Wrong argument count - should raise ERRORs."""


def requires_three(a, b, c):
    """Requires exactly 3 args."""
    return a + b + c


def requires_two(a, b):
    """Requires exactly 2 args."""
    return a + b


def one_required_two_optional(a, b=1, c=2):
    """One required, two optional."""
    return a + b + c


class NeedsInit:
    """Class requiring 2 args for __init__."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


# === ERRORS: Too few arguments ===

# ERROR: requires_three() missing 2 required positional arg(s)
# But this one is suppressed with noqa
err1 = requires_three(1)  # noqa: sigcheck

# ERROR: requires_three() missing 1 required positional arg(s)
err2 = requires_three(1, 2)

# ERROR: requires_two() missing 1 required positional arg(s)
err3 = requires_two(1)

# ERROR: one_required_two_optional() missing 1 required positional arg(s)
err4 = one_required_two_optional()

# ERROR: NeedsInit() missing 2 required positional arg(s)
err5 = NeedsInit()

# ERROR: NeedsInit() missing 1 required positional arg(s)
err6 = NeedsInit(1)


# === ERRORS: Too many arguments ===

# ERROR: requires_two() takes 2 positional args, got 3
err7 = requires_two(1, 2, 3)

# ERROR: requires_three() takes 3 positional args, got 5
err8 = requires_three(1, 2, 3, 4, 5)

# ERROR: NeedsInit() takes 2 positional args, got 4
err9 = NeedsInit(1, 2, 3, 4)
