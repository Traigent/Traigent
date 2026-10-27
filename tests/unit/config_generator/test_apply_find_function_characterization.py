"""Characterization of config_generator.apply._find_function."""

from __future__ import annotations

import ast

import pytest

from traigent.config_generator.apply import _find_function

SRC = """
def top(): pass
async def atop(): pass
class A:
    x = 1
    def meth(self): pass
    async def ameth(self): pass
    class Inner:
        def deep(self): pass
if True:
    def nested_in_if(): pass
def dup(): pass
class B:
    def dup(self): pass
class C:
    def top(self): pass
"""


@pytest.mark.parametrize(
    ("name", "kind", "line"),
    [
        ("top", ast.FunctionDef, 2),
        ("atop", ast.AsyncFunctionDef, 3),
        ("meth", ast.FunctionDef, 6),
        ("ameth", ast.AsyncFunctionDef, 7),
        ("dup", ast.FunctionDef, 12),
    ],
)
def test_found(name, kind, line):
    node = _find_function(ast.parse(SRC), name)
    assert isinstance(node, kind)
    assert node.lineno == line


@pytest.mark.parametrize("name", ["deep", "nested_in_if", "missing", "A", "x"])
def test_not_found(name):
    assert _find_function(ast.parse(SRC), name) is None


def test_source_order_first_match_wins_across_class_and_module():
    tree = ast.parse("class A:\n    def f(self): pass\ndef f(): pass\n")
    assert _find_function(tree, "f").lineno == 2
