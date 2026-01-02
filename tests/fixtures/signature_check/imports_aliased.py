"""Aliased imports - import x as y, from x import y as z.

Tests symbol resolution through aliased import statements.
"""

import json as j
import os.path as osp
from collections import OrderedDict as OD
from pathlib import Path as P
from typing import Dict as D
from typing import List as L

from tests.fixtures.signature_check import correct_calls as cc

# Aliased imports from our fixtures
from tests.fixtures.signature_check.correct_calls import MyClass as MC
from tests.fixtures.signature_check.correct_calls import func_with_defaults as fwd
from tests.fixtures.signature_check.correct_calls import simple_func as sf
from tests.fixtures.signature_check.wrong_arg_count import requires_three as r3

# === CORRECT: Calls via aliased names ===

# Aliased stdlib (external - INFO)
ok1 = osp.join("a", "b")
ok2 = j.dumps({"key": "value"})
ok3 = P("/tmp/test")
ok4 = OD()

# Aliased internal functions - should resolve
ok5 = sf(1, 2, 3)  # simple_func
ok6 = fwd(1, 2)  # func_with_defaults
ok7 = fwd(1, 2, c=5)

# Aliased class
ok8 = MC(1, 2)  # MyClass

# Module alias + function
ok9 = cc.simple_func(1, 2, 3)
ok10 = cc.MyClass(1, 2)


# === ERRORS: Wrong args via aliased names ===

# ERROR: sf() missing 2 required positional arg(s)
err1 = sf(1)

# ERROR: fwd() missing 1 required positional arg(s)
err2 = fwd()

# ERROR: MC() missing 1 required positional arg(s)
err3 = MC(1)

# ERROR: r3() takes 3 positional args, got 4
err4 = r3(1, 2, 3, 4)

# Via module alias
err5 = cc.simple_func(1)  # missing 2 args
err6 = cc.MyClass()  # missing 2 args


# === INFO: External modules via alias - cannot resolve ===

info1 = osp.exists("/tmp")  # os.path.exists
info2 = j.loads('{"a": 1}')  # json.loads

# These might be wrong, but we can't verify external
maybe1 = j.dumps()  # Actually wrong
maybe2 = P()  # Actually wrong
