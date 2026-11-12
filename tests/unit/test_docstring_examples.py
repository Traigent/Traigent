"""Curated doctest gate: ``>>>`` examples in these public modules must stay true.

Some examples omit imports of public names (``Range``, ``Choices``, or a module's own classes), so the
examples run with the ``traigent`` public namespace injected as globals,
matching what a reader would have after ``from traigent import ...``.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

import traigent

CURATED_MODULES = [
    "traigent.api.constraint_builders",
    "traigent.api.constraints",
    "traigent.core.namespace",
    "traigent.utils.example_id",
    "traigent.metrics.agent_metrics",
    "traigent.hybrid.discovery",
    "traigent.api.safety",
    "traigent.providers.validation",
    "traigent.plugins.registry",
    "traigent.utils.validation",
]


@pytest.mark.parametrize("module_name", CURATED_MODULES)
def test_module_doctests_pass(module_name: str) -> None:
    module = importlib.import_module(module_name)
    public = {name: getattr(traigent, name) for name in dir(traigent)}
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    tests = [
        t
        for t in finder.find(
            module, globs={**vars(module), **public, "traigent": traigent}
        )
        if t.examples
    ]
    assert tests, f"no doctest examples found in {module_name}"
    failures: list[str] = []
    attempted = 0
    for test in tests:
        output: list[str] = []
        result = runner.run(test, out=output.append, clear_globs=False)
        attempted += result.attempted
        if result.failed:
            failures.append("".join(output))
    assert attempted > 0
    assert not failures, "\n".join(failures)
