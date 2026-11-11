"""Doc claim matches code (#2971)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _raises_names(source: str, func: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            doc = ast.get_docstring(node) or ""
            block = doc.split("Raises:", 1)[1].split("\n\n", 1)[0]
            return re.findall(r"^\s*(\w+):", block, re.M)
    raise AssertionError(f"{func} not found")


def test_optimize_docstring_raises_only_real_exceptions_2971() -> None:
    import traigent.utils.exceptions as exc_mod

    names = _raises_names(_read("traigent/api/decorators.py"), "optimize")
    assert names
    unresolved = [n for n in names if not hasattr(exc_mod, n)]
    assert not unresolved, unresolved


def test_missing_dataset_raises_a_documented_exception_2971(monkeypatch) -> None:
    """Running the issue's reproduction raises a class the Raises section lists."""
    import traigent
    import traigent.utils.exceptions as exc_mod

    # The SDK skips the dataset-path check while PYTEST_CURRENT_TEST is set.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    raised: BaseException | None = None
    try:

        @traigent.optimize(
            eval_dataset="does_not_exist_2971.jsonl",
            objectives=["accuracy"],
            configuration_space={"temperature": [0.1, 0.2]},
        )
        def f(question: str) -> str:
            return question

    except Exception as exc:  # noqa: BLE001 - we assert on the concrete class below
        raised = exc

    assert raised is not None, "a missing dataset must fail loudly"
    names = _raises_names(_read("traigent/api/decorators.py"), "optimize")
    documented = tuple(getattr(exc_mod, n) for n in names)
    assert isinstance(raised, documented), (type(raised).__name__, names)
