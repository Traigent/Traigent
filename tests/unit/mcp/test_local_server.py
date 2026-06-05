"""Tests for the local stdio MCP server surface."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from traigent.mcp.server import create_server
from traigent.mcp.tools import V1_TOOL_NAMES


@pytest.fixture(autouse=True)
def reset_traigent_mock_mode() -> None:
    yield
    from traigent.testing import _reset_for_tests

    _reset_for_tests()


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    server = create_server()
    async with create_connected_server_and_client_session(server) as session:
        yield session


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = await session.call_tool(name, arguments or {})
    assert result.structuredContent is not None
    return dict(result.structuredContent)


def write_fixture_agent(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import traigent",
                "",
                "@traigent.optimize(",
                "    eval_dataset=[",
                "        {'input': {'query': 'a'}, 'expected': 'A'},",
                "        {'input': {'query': 'b'}, 'expected': 'B'},",
                "    ],",
                "    objectives=['accuracy'],",
                "    configuration_space={'temperature': [0.0, 1.0]},",
                ")",
                "def answer(query: str) -> str:",
                "    temperature = 0.0",
                "    return {'a': 'A', 'b': 'B'}[query]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_fixture_dataset(path: Path) -> None:
    rows = [
        {"input": {"query": "a"}, "output": "A"},
        {"input": {"query": "b"}, "output": "B"},
        {"input": {"query": "c"}, "output": "C"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


async def test_tool_listing_matches_v1_set() -> None:
    async with mcp_session() as session:
        result = await session.list_tools()

    assert [tool.name for tool in result.tools] == list(V1_TOOL_NAMES)
    assert "scaffold_eval" not in {tool.name for tool in result.tools}
    assert "export_evidence" not in {tool.name for tool in result.tools}
    assert "significant_variables" not in {tool.name for tool in result.tools}
    assert "generate_examples" not in {tool.name for tool in result.tools}


async def test_catalog_and_recommend_happy_path(
) -> None:
    async with mcp_session() as session:
        types_payload = await call_tool(session, "list_recommendation_agent_types")
        recommendation = await call_tool(
            session,
            "recommend_configuration_space",
            {"agent_type": "rag", "min_impact": "low", "min_confidence": "low"},
        )

    assert types_payload["ok"] is True
    assert "rag" in types_payload["agent_types"]
    assert recommendation["ok"] is True
    assert recommendation["recommendation"]["agent_type"] == "rag"
    assert "configuration_space" in recommendation["recommendation"]


async def test_detect_validate_and_estimate_happy_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    agent_file = tmp_path / "agent.py"
    dataset = tmp_path / "eval.jsonl"
    write_fixture_agent(agent_file)
    write_fixture_dataset(dataset)
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

    async with mcp_session() as session:
        detected = await call_tool(
            session,
            "detect_tvars",
            {"file_path": "agent.py", "function_name": "answer"},
        )
        validation = await call_tool(session, "validate_dataset", {"path": "eval.jsonl"})
        estimate = await call_tool(
            session,
            "estimate_cost",
            {
                "dataset_path": "eval.jsonl",
                "max_trials": 4,
                "model": "gpt-4o-mini",
            },
        )

    assert detected["ok"] is True
    assert detected["results"][0]["function_name"] == "answer"
    assert any(
        candidate["name"] == "temperature"
        for candidate in detected["results"][0]["candidates"]
    )

    assert validation["ok"] is True
    assert validation["errors"] == []

    assert estimate["ok"] is True
    assert estimate["dataset_examples"] == 3
    assert estimate["llm_calls_upper_bound"] == 12
    assert estimate["estimated_total_cost_usd"] is not None
    assert any("max_trials * dataset" in item for item in estimate["assumptions"])


async def test_run_optimization_refusal_matrix_and_mock_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    agent_file = tmp_path / "agent.py"
    write_fixture_agent(agent_file)

    async with mcp_session() as session:
        no_confirm = await call_tool(
            session,
            "run_optimization",
            {"script_path": "agent.py", "mode": "real"},
        )
        no_cost_limit = await call_tool(
            session,
            "run_optimization",
            {"script_path": "agent.py", "mode": "real", "confirm": True},
        )
        mock_run = await call_tool(
            session,
            "run_optimization",
            {"script_path": "agent.py", "max_trials": 2, "algorithm": "random"},
        )
        listed = await call_tool(session, "get_results")
        shown = await call_tool(
            session,
            "get_results",
            {"result_name": mock_run["results"][0]["result_name"]},
        )

    assert no_confirm["ok"] is False
    assert no_confirm["refused"] is True
    assert "confirm=true" in no_confirm["message"]

    assert no_cost_limit["ok"] is False
    assert no_cost_limit["refused"] is True
    assert "cost_limit" in no_cost_limit["message"]

    assert mock_run["ok"] is True
    assert mock_run["mode"] == "mock"
    assert mock_run["results"][0]["total_trials"] == 2
    assert mock_run["results"][0]["result_name"]
    from traigent.testing import is_mock_mode_enabled

    assert is_mock_mode_enabled() is False

    result_names = {item["name"] for item in listed["results"]}
    assert mock_run["results"][0]["result_name"] in result_names

    assert shown["ok"] is True
    assert shown["result"]["best_config"] is not None


async def test_auth_status_masks_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_key = "sk_test_secret_1234567890"
    monkeypatch.setattr(
        "traigent.mcp.tools.CredentialManager.get_credentials",
        lambda: {
            "api_key": full_key,
            "backend_url": "https://backend.example.test",
            "tenant_id": "tenant_123",
            "project_id": "project_456",
            "source": "test",
        },
    )

    async with mcp_session() as session:
        payload = await call_tool(session, "auth_status")
    serialized = json.dumps(payload)

    assert payload["ok"] is True
    assert payload["api_key"]["present"] is True
    assert payload["api_key"]["prefix"] == "sk_t"
    assert payload["api_key"]["last4"] == "7890"
    assert full_key not in serialized


async def test_path_containment_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "agent.py").write_text("def answer(): pass\n", encoding="utf-8")
    (outside / "eval.jsonl").write_text(
        '{"input": "x", "output": "y"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(root))

    async with mcp_session() as session:
        detect = await call_tool(
            session,
            "detect_tvars",
            {"file_path": str(outside / "agent.py")},
        )
        validate = await call_tool(
            session,
            "validate_dataset",
            {"path": str(outside / "eval.jsonl")},
        )
        run = await call_tool(
            session,
            "run_optimization",
            {"script_path": str(outside / "agent.py")},
        )

    assert detect["ok"] is False
    assert detect["code"] == "path_rejected"

    assert validate["ok"] is False
    assert validate["code"] == "path_rejected"

    assert run["ok"] is False
    assert run["code"] == "path_rejected"
