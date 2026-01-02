"""Relative imports - from . import, from .. import.

Tests symbol resolution through relative import statements.
This file is in tests/fixtures/signature_check/ package.
"""

# Relative imports within the same package
from . import correct_calls
from .correct_calls import func_with_defaults, simple_func
from .wrong_arg_count import requires_three, requires_two

# Note: We can't actually do parent-relative imports here
# because tests/fixtures/signature_check is the deepest level.
# In real code, you'd see patterns like:
# from ..other_package import something
# from ...top_level import config


# === CORRECT: Calls via relative imports ===

# Direct function import
ok1 = simple_func(1, 2, 3)
ok2 = func_with_defaults(1, 2)
ok3 = func_with_defaults(1, 2, c=10)

# Module.function pattern
ok4 = correct_calls.simple_func(1, 2, 3)
ok5 = correct_calls.func_with_defaults(1, 2)
ok6 = correct_calls.MyClass(1, 2)


# === ERRORS: Wrong args via relative imports ===

# ERROR: simple_func() missing 2 required positional arg(s)
err1 = simple_func(1)

# ERROR: func_with_defaults() missing 1 required positional arg(s)
err2 = func_with_defaults()

# ERROR: requires_three() takes 3 positional args, got 4
err3 = requires_three(1, 2, 3, 4)

# ERROR: requires_two() missing 1 required positional arg(s)
err4 = requires_two(1)

# ERROR via module.function
err5 = correct_calls.simple_func(1)  # missing 2 args
err6 = correct_calls.MyClass()  # missing 2 args
