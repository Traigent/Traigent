"""The ``@traigent.optimize`` docstring must document every keyword option (#3192)."""

from __future__ import annotations

import inspect
import re

import traigent


def _documented_names(doc: str) -> set[str]:
    """Names introduced as ``name:`` / ``**name:`` entries in the docstring."""
    return set(re.findall(r"^\s+\*{0,2}(\w+):\s", doc, flags=re.MULTILINE))


def test_optimize_docstring_documents_every_signature_option() -> None:
    doc = inspect.getdoc(traigent.optimize)
    assert doc is not None
    documented = _documented_names(doc)
    params = [
        name
        for name, param in inspect.signature(traigent.optimize).parameters.items()
        if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
    ]
    missing = [name for name in params if name not in documented]
    assert not missing, f"optimize() options missing from its docstring: {missing}"


def test_optimize_docstring_describes_skill_train_options() -> None:
    doc = inspect.getdoc(traigent.optimize) or ""
    entry = re.search(r"^\s+skill_train:\s(.*?)(?=^\s+\w+:\s)", doc, re.M | re.S)
    assert entry is not None
    assert "SkillTrainOptions" in entry.group(1)
    assert "train_skill" in entry.group(1)
