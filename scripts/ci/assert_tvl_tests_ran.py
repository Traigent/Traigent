#!/usr/bin/env python3
"""Fail if the runtime-tvl compiler tests were silently skipped.

The tests in tests/unit/runtime guard on ``pytest.importorskip("tvl.runtime_compiler")``
so they skip on a machine without the ``runtime-tvl`` extra. In the
``runtime-tvl-compiler`` PR-gate job that skip would be a silent false green, so
this reads pytest's JUnit XML and fails on any skipped test except the one
documented, environment-bound skip (the real-Backend acceptance check, which needs
a Backend checkout CI does not have). It also fails when nothing passed.

Usage: assert_tvl_tests_ran.py <junit.xml>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET  # noqa: S405 - parsing pytest's own output

# Skip reasons that are expected in CI. Anything else is a failure.
ALLOWED_SKIP_MARKERS = ("TRAIGENT_BACKEND_SRC",)


def main(path: str) -> int:
    cases = ET.parse(path).getroot().iter("testcase")  # noqa: S314
    passed = 0
    bad: list[str] = []
    for case in cases:
        name = f"{case.get('classname')}::{case.get('name')}"
        skipped = case.find("skipped")
        if skipped is not None:
            reason = (skipped.get("message") or "") + (skipped.text or "")
            if not any(marker in reason for marker in ALLOWED_SKIP_MARKERS):
                bad.append(f"{name}: {reason.strip()[:200]}")
        elif case.find("failure") is None and case.find("error") is None:
            passed += 1
    if bad:
        print("::error::runtime-tvl tests were skipped (extra not effective?):")
        for line in bad:
            print(f"::error::  {line}")
        return 1
    if passed == 0:
        print("::error::no runtime-tvl test passed; nothing was exercised")
        return 1
    print(f"runtime-tvl tests: {passed} passed, no unexpected skips")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
