"""*args and **kwargs at both definition and call sites.

When a CALL uses *args or **kwargs, we cannot statically verify
what values are passed -> downgrade to INFO.
"""


def strict_func(a, b, c):
    """Strict function - no *args/**kwargs."""
    return a + b + c


def accepts_varargs(a, *args):
    """Accepts *args."""
    return a, args


def accepts_varkw(a, **kwargs):
    """Accepts **kwargs."""
    return a, kwargs


def accepts_both(a, *args, **kwargs):
    """Accepts both."""
    return a, args, kwargs


# === CORRECT: Definition has *args, any positional count OK ===

ok1 = accepts_varargs(1)
ok2 = accepts_varargs(1, 2, 3, 4, 5)

# === CORRECT: Definition has **kwargs, any keyword OK ===

ok3 = accepts_varkw(1, x=1, y=2, z=3)
ok4 = accepts_both(1, 2, 3, a=4, b=5)


# === INFO: Call uses *args - cannot verify ===

my_args = [1, 2, 3]

# INFO: strict_func() uses *args, cannot verify
info1 = strict_func(*my_args)

# INFO: strict_func() uses *args, cannot verify
info2 = strict_func(1, *my_args)


# === INFO: Call uses **kwargs - cannot verify ===

my_kwargs = {"b": 2, "c": 3}

# INFO: strict_func() uses **kwargs, cannot verify
info3 = strict_func(a=1, **my_kwargs)

# INFO: strict_func() uses *args/**kwargs, cannot verify
info4 = strict_func(*my_args, **my_kwargs)


# === ERRORS: Even with *args in def, still need minimum required ===

# ERROR: accepts_varargs() missing 1 required positional arg(s)
err1 = accepts_varargs()  # needs at least 'a'
