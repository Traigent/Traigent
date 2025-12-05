"""Basic functionality test to verify MCP test framework is working."""

import sys
from pathlib import Path

import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all required imports work."""
    # Core imports
    # Test utility functions (not fixtures)
    from tests.mcp.conftest import generate_test_agent_spec, generate_test_dataset
    from traigent.cloud.models import AgentSpecification

    assert AgentSpecification is not None
    assert generate_test_dataset is not None
    assert generate_test_agent_spec is not None


@pytest.mark.asyncio
async def test_mock_service_creation(mock_mcp_service):
    """Test that mock MCP service can be created."""
    service = mock_mcp_service
    assert service is not None
    assert hasattr(service, "create_agent")
    assert hasattr(service, "execute_agent")
    assert len(service.agents) == 0
    assert len(service.sessions) == 0


@pytest.mark.asyncio
async def test_basic_agent_creation(mock_mcp_service):
    """Test basic agent creation flow."""
    from tests.mcp.conftest import generate_test_agent_spec

    service = mock_mcp_service
    agent_spec = generate_test_agent_spec(name="Test Agent")

    result = await service.create_agent(agent_spec)

    assert result["status"] == "created"
    assert "agent_id" in result
    assert len(service.agents) == 1


@pytest.mark.asyncio
async def test_llm_interpreter(mock_llm_interpreter):
    """Test LLM interpreter functionality."""
    interpreter = mock_llm_interpreter

    # Test agent creation interpretation
    result = await interpreter.interpret_task(
        "Create an agent that answers questions about Python"
    )

    assert result["action"] == "create_agent"
    assert result["topic"] == "Python programming"


def test_fixture_files_exist():
    """Test that fixture files are present."""
    fixture_dir = Path(__file__).parent / "fixtures"

    expected_files = [
        "sample_agents.json",
        "sample_datasets.jsonl",
        "task_interpretations.json",
    ]

    for filename in expected_files:
        filepath = fixture_dir / filename
        assert filepath.exists(), f"Missing fixture file: {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
