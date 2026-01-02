"""Positional-only, keyword-only, and defaults - complex parameter patterns."""


def posonly_only(a, b, /):
    """Positional-only args (Python 3.8+)."""
    return a + b


def posonly_with_regular(a, /, b, c):
    """Mix of posonly and regular."""
    return a + b + c


def kwonly_only(*, a, b):
    """Keyword-only args."""
    return a + b


def kwonly_with_defaults(*, a, b=10, c=20):
    """Keyword-only with defaults."""
    return a + b + c


def full_signature(a, b, /, c, d, *args, e, f=100, **kwargs):
    """Full complex signature.

    - a, b: positional-only (2 required)
    - c, d: regular positional (2 required)
    - *args: var positional
    - e: keyword-only required
    - f: keyword-only with default
    - **kwargs: var keyword
    """
    return (a, b, c, d, args, e, f, kwargs)


# === CORRECT CALLS ===

# Positional-only
ok1 = posonly_only(1, 2)
ok2 = posonly_with_regular(1, 2, 3)
ok3 = posonly_with_regular(1, 2, c=3)  # c can be keyword

# Keyword-only
ok4 = kwonly_only(a=1, b=2)
ok5 = kwonly_with_defaults(a=1)  # b and c have defaults
ok6 = kwonly_with_defaults(a=1, b=5, c=10)

# Full signature
ok7 = full_signature(1, 2, 3, 4, e=5)
ok8 = full_signature(1, 2, 3, 4, 5, 6, 7, e=8, f=9, extra=10)


# === ERRORS ===

# ERROR: posonly_only() missing 1 required positional arg(s)
err1 = posonly_only(1)

# ERROR: posonly_with_regular() missing 2 required positional arg(s)
err2 = posonly_with_regular(1)

# ERROR: kwonly_only() missing required keyword-only arg(s)
# Note: This is a special case - missing kwonly args
err3 = kwonly_only(a=1)  # missing b

# ERROR: kwonly_with_defaults() missing required keyword-only arg(s)
err4 = kwonly_with_defaults()  # missing a (required kwonly)

# ERROR: full_signature() missing 4 required positional arg(s)
err5 = full_signature(1)

# ERROR: posonly_only() takes 2 positional args, got 3
err6 = posonly_only(1, 2, 3)

# ERROR: posonly_only() got positional-only arg(s) as keyword: {'a'}
# This is invalid Python - posonly args cannot be passed by keyword
err7 = posonly_only(1, a=2)  # 'a' is posonly, can't be keyword

# ERROR: posonly_with_regular() got positional-only arg(s) as keyword: {'a'}
err8 = posonly_with_regular(a=1, b=2, c=3)  # 'a' is posonly

# ERROR: full_signature() got positional-only arg(s) as keyword: {'a', 'b'}
err9 = full_signature(a=1, b=2, c=3, d=4, e=5)  # 'a' and 'b' are posonly
