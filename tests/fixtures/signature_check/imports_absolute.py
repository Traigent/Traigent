"""Absolute imports - import x, from x import y.

Tests symbol resolution through absolute import statements.
"""

import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

# Import from sibling fixture modules
from tests.fixtures.signature_check.correct_calls import MyClass, simple_func
from tests.fixtures.signature_check.staticmethod_classmethod import Calculator
from tests.fixtures.signature_check.wrong_arg_count import requires_three

# === CORRECT: Calls via imported names ===

# os.path.join - external, should be INFO (unresolvable)
ok1 = os.path.join("a", "b", "c")

# json.dumps - external, INFO
ok2 = json.dumps({"key": "value"})

# Path - external, INFO
ok3 = Path("/tmp/test")

# OrderedDict - external, INFO
ok4 = OrderedDict()

# defaultdict - external, INFO
ok5 = defaultdict(list)

# simple_func - from our fixtures, should resolve
ok6 = simple_func(1, 2, 3)

# MyClass - from our fixtures, should resolve
ok7 = MyClass(1, 2)


# === ERRORS: Imported functions with wrong args ===

# ERROR: simple_func() missing 2 required positional arg(s)
err1 = simple_func(1)

# ERROR: MyClass() missing 1 required positional arg(s)
err2 = MyClass(1)

# ERROR: requires_three() takes 3 positional args, got 4
err3 = requires_three(1, 2, 3, 4)


# === INFO: External modules - cannot resolve ===

# INFO: os.getcwd() - external module
info1 = os.getcwd()

# INFO: json.loads() - external module
info2 = json.loads('{"a": 1}')

# These might have wrong args, but we can't verify external modules
maybe1 = json.dumps()  # Actually wrong, but INFO since external
maybe2 = Path()  # Actually wrong, but INFO since external


# === ERRORS: Imported class methods (staticmethod/classmethod) ===

# Correct calls via imported class
ok_static = Calculator.static_add(1, 2)
ok_classmethod = Calculator.from_string("42")

# ERROR: Calculator.static_add() takes 2 positional arg(s), got 3
err_static1 = Calculator.static_add(1, 2, 3)

# ERROR: Calculator.static_add() missing required arg(s)
err_static2 = Calculator.static_add(1)

# ERROR: Calculator.from_string() missing required arg(s)
err_classmethod = Calculator.from_string()
