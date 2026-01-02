"""Correct calls - all should pass with no errors."""


def simple_func(a, b, c):
    """Function with 3 required args."""
    return a + b + c


def func_with_defaults(a, b, c=10):
    """Function with defaults."""
    return a + b + c


def func_with_kwargs(a, b, **kwargs):
    """Function accepting **kwargs."""
    return {"a": a, "b": b, **kwargs}


def func_with_args(*args):
    """Function accepting *args."""
    return sum(args)


def func_with_both(*args, **kwargs):
    """Function accepting both."""
    return args, kwargs


class MyClass:
    """Class with various methods."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def method(self, a, b):
        return a + b

    @staticmethod
    def static_method(a, b):
        return a * b

    @classmethod
    def class_method(cls, a):
        return a


# === CORRECT CALLS (should not raise errors) ===

# Basic calls
result1 = simple_func(1, 2, 3)
result2 = func_with_defaults(1, 2)
result3 = func_with_defaults(1, 2, 3)
result4 = func_with_defaults(1, 2, c=5)

# Kwargs
result5 = func_with_kwargs(1, 2)
result6 = func_with_kwargs(1, 2, extra="value")
result7 = func_with_kwargs(a=1, b=2, c=3, d=4)

# Args
result8 = func_with_args()
result9 = func_with_args(1, 2, 3, 4, 5)

# Both
result10 = func_with_both()
result11 = func_with_both(1, 2, 3, x=1, y=2)

# Class instantiation
obj = MyClass(1, 2)
obj2 = MyClass(x=1, y=2)

# Static/class methods
result12 = MyClass.static_method(1, 2)
result13 = MyClass.class_method(5)
