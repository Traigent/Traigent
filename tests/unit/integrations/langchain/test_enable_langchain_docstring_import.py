"""The import shown in the enable_langchain_optimization docstring must resolve."""

import re

from traigent.integrations.llms.langchain import base as langchain_base

_FROM_TRAIGENT_IMPORT = re.compile(r"^\s*(from traigent[\w.]* import .+)$", re.MULTILINE)


def test_docstring_import_of_enable_langchain_optimization_resolves() -> None:
    doc = langchain_base.enable_langchain_optimization.__doc__ or ""
    statements = _FROM_TRAIGENT_IMPORT.findall(doc)

    assert statements, "docstring example no longer shows a traigent import"
    namespace: dict = {}
    for statement in statements:
        exec(statement, namespace)  # noqa: S102 - runs the documented import line only
    assert namespace["enable_langchain_optimization"] is (
        langchain_base.enable_langchain_optimization
    )
