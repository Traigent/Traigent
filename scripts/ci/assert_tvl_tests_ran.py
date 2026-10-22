#!/usr/bin/env python3
"""Fail if the runtime-tvl compiler tests were silently skipped.

The tests in tests/unit/runtime guard on ``pytest.importorskip("tvl.runtime_compiler")``
so they skip on a machine without the ``runtime-tvl`` extra. In the
``runtime-tvl-compiler`` PR-gate job that skip would be a silent false green, so
this reads pytest's JUnit XML and fails on any skipped test except the one
documented, environment-bound skip (the real-Backend acceptance test, matched by its
exact testcase id and reason, which needs a Backend checkout CI does not have). It
also fails unless every named compiler testcase is present and passed.

Usage: assert_tvl_tests_ran.py <junit.xml>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET  # noqa: S405 - parsing pytest's own output

CLASS_PREFIX = "tests.unit.runtime."

# The only skip allowed in CI: the real-Backend acceptance test, whose reason names
# the environment variable that points at a Backend checkout CI does not have.
ALLOWED_SKIPS = {
    (
        "test_canary",
        "test_tvl_route_is_accepted_by_the_real_backend_parser",
    ): "TRAIGENT_BACKEND_SRC",
}

# Compiler testcases that must be present AND passed (parametrised cases match by
# prefix: ``name`` or ``name[...]``). No total count: adding tests must not break CI.
REQUIRED = (
    ("test_real_compiler", "test_token_map_format_is_the_compilers_contract"),
    ("test_real_compiler", "test_real_compiler_resolves_tokens"),
    ("test_hosted_subset", "test_compiled_contract_matches_checked_in_fixture"),
    ("test_hosted_subset", "test_full_policy_lints_with_only_the_liveness_warning"),
    ("test_canary", "test_tvl_route_sends_only_refs_and_the_environment_label"),
)


def _key(case: ET.Element) -> tuple[str, str]:
    classname = case.get("classname") or ""
    module = classname.removeprefix(CLASS_PREFIX)
    return module, case.get("name") or ""


def _matches(case_name: str, wanted: str) -> bool:
    return case_name == wanted or case_name.startswith(wanted + "[")


def main(path: str) -> int:
    cases = list(ET.parse(path).getroot().iter("testcase"))  # noqa: S314
    problems: list[str] = []
    passed_keys: list[tuple[str, str]] = []
    for case in cases:
        key = _key(case)
        label = f"{key[0]}::{key[1]}"
        skipped = case.find("skipped")
        if skipped is not None:
            reason = (skipped.get("message") or "") + (skipped.text or "")
            marker = ALLOWED_SKIPS.get(key)
            if marker is None or marker not in reason:
                problems.append(f"unexpected skip {label}: {reason.strip()[:200]}")
        elif case.find("failure") is not None or case.find("error") is not None:
            problems.append(f"failed {label}")
        else:
            passed_keys.append(key)
    for module, wanted in REQUIRED:
        if not any(m == module and _matches(n, wanted) for m, n in passed_keys):
            problems.append(
                f"required compiler test missing or not passed: {module}::{wanted}"
            )
    if problems:
        print("::error::runtime-tvl guard failed (extra not effective?):")
        for line in problems:
            print(f"::error::  {line}")
        return 1
    print(f"runtime-tvl tests: {len(passed_keys)} passed, all required cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
