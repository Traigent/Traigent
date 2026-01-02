"""Custom decorators and @functools.wraps.

Custom decorators may alter signatures, so we downgrade to WARNING.
@functools.wraps preserves the wrapped signature, but we still warn
because we can't verify the decorator implementation.
"""

import functools


def simple_decorator(func):
    """Simple decorator that doesn't preserve signature."""

    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def wraps_decorator(func):
    """Decorator using @functools.wraps - preserves signature metadata."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def signature_changing_decorator(func):
    """Decorator that changes the signature (adds a required arg)."""

    def wrapper(extra_arg, *args, **kwargs):
        print(f"Extra: {extra_arg}")
        return func(*args, **kwargs)

    return wrapper


@simple_decorator
def decorated_simple(a, b, c):
    """Decorated function - signature may be altered."""
    return a + b + c


@wraps_decorator
def decorated_wraps(a, b, c):
    """Decorated with @wraps - signature should be preserved."""
    return a + b + c


@signature_changing_decorator
def decorated_changed(a, b):
    """Decorated - actual signature at runtime is (extra_arg, a, b)."""
    return a + b


@simple_decorator
@wraps_decorator
def multi_decorated(x, y):
    """Multiple decorators - even harder to verify."""
    return x * y


# === WARNING: Decorated functions - cannot verify signature ===

# WARNING: decorated_simple has decorator(s), signature may be altered
w1 = decorated_simple(1, 2, 3)
w2 = decorated_simple(1, 2)  # Might be wrong, but we can't verify

# WARNING: decorated_wraps has decorator(s), signature may be altered
w3 = decorated_wraps(1, 2, 3)
w4 = decorated_wraps(1)  # Might be wrong

# WARNING: decorated_changed has decorator(s), signature may be altered
# The actual runtime signature is different!
w5 = decorated_changed(1, 2)  # At runtime needs 3 args due to decorator
w6 = decorated_changed("extra", 1, 2)  # This is actually correct at runtime

# WARNING: multi_decorated has decorator(s), signature may be altered
w7 = multi_decorated(1, 2)
w8 = multi_decorated(1, 2, 3, 4)  # Probably wrong, but can't verify


class DecoratedClass:
    """Class with decorated methods."""

    @simple_decorator
    def decorated_method(self, a, b):
        return a + b

    @staticmethod
    @simple_decorator
    def decorated_static(a, b):
        return a * b


# WARNING: decorated_static has decorator(s), signature may be altered
w9 = DecoratedClass.decorated_static(1, 2)
