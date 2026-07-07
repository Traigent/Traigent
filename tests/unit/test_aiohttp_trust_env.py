"""Ensure aiohttp backend egress honors proxy and netrc environment settings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIGENT_ROOT = REPO_ROOT / "traigent"


@dataclass(frozen=True)
class AiohttpClientSessionSite:
    path: str
    lineno: int


def _is_aiohttp_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "aiohttp"


def _is_cast_aiohttp(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and len(node.args) >= 2
        and _is_aiohttp_name(node.args[1])
    )


def _is_aiohttp_client_session_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ClientSession"
        and (_is_aiohttp_name(node.func.value) or _is_cast_aiohttp(node.func.value))
    )


def _trust_env_is_true(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "trust_env":
            return (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            )
    return False


def _iter_aiohttp_client_session_sites() -> list[
    tuple[AiohttpClientSessionSite, ast.Call]
]:
    sites: list[tuple[AiohttpClientSessionSite, ast.Call]] = []
    for path in sorted(TRAIGENT_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relpath = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if _is_aiohttp_client_session_call(node):
                assert isinstance(node, ast.Call)
                sites.append((AiohttpClientSessionSite(relpath, node.lineno), node))
    return sites


def test_all_aiohttp_client_sessions_trust_env() -> None:
    """aiohttp defaults trust_env=False, so every SDK egress session opts in."""

    sites = _iter_aiohttp_client_session_sites()
    assert sites, "Expected at least one aiohttp.ClientSession construction site"

    known_paths = {site.path for site, _ in sites}
    assert "traigent/integrations/langfuse/client.py" in known_paths
    assert "traigent/integrations/observability/workflow_traces.py" in known_paths
    assert "traigent/cloud/backend_client.py" in known_paths

    missing = [
        f"{site.path}:{site.lineno}"
        for site, node in sites
        if not _trust_env_is_true(node)
    ]
    assert not missing, "aiohttp.ClientSession missing trust_env=True: " + ", ".join(
        missing
    )
