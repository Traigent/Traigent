"""Acceptance check of the hosted-contract fixture against the REAL Backend importer.

Not collected by pytest.  Run it with a Python that can import the Backend's
``src`` package (a TraigentBackend worktree venv), from the Backend checkout::

    cd <TraigentBackend worktree> && \
      <backend-venv>/bin/python <this file> tests/unit/runtime/fixtures/hosted_contract.json

It runs ``parse_import`` (schema, versions, refs, declaration fields, profile
consistency, environment binding) on the fixture and on mutated copies, which
must each be refused with a closed code.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from src.services.analytics.production_requirements.requirements import (
    ProductionRequirementError,
)
from src.services.analytics.production_requirements.tvl_import import parse_import


def _refused(contract: dict, binding: dict) -> str | None:
    try:
        parse_import(contract, binding)
    except ProductionRequirementError as exc:
        return exc.code
    return None


def main(path: str) -> int:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    contract, binding = fixture["contract"], fixture["environment_binding"]
    parsed = parse_import(contract, binding)
    print("accepted payload:", sorted(parsed.payload))

    def mutate(fn):
        c, b = copy.deepcopy(contract), copy.deepcopy(binding)
        fn(c, b)
        return c, b

    profile = lambda c: c["backend_profiles"][0]["backend_profile"]  # noqa: E731
    cases = {
        "profile threshold drift": mutate(
            lambda c, b: profile(c).update(latency_threshold_ms=1999)
        ),
        "profile rate drift": mutate(
            lambda c, b: profile(c).update(max_violation_rate="0.06")
        ),
        "environment ref drift": mutate(
            lambda c, b: profile(c).update(environment_ref="tother")
        ),
        "binding ref mismatch": mutate(lambda c, b: b.update(ref="tother")),
        "binding extra key": mutate(lambda c, b: b.update(extra="x")),
        "bad ref characters": mutate(
            lambda c, b: c["declarations"]["tvl"].update(module="bad ref!")
        ),
        "responses present": mutate(
            lambda c, b: c["declarations"].update(
                responses=[{"name": "r", "rules": []}]
            )
        ),
        "second monitor": mutate(
            lambda c, b: c["declarations"]["monitors"].append(
                c["declarations"]["monitors"][0]
            )
        ),
        "toolchain version": mutate(
            lambda c, b: c["toolchain"].update(version="9.9.9")
        ),
    }
    failed = 0
    for label, (c, b) in cases.items():
        code = _refused(c, b)
        print(f"{'ok  ' if code else 'FAIL'} {label}: {code}")
        failed += 0 if code else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
