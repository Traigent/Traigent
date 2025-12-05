"""Tests for framework override platform functionality."""

from unittest.mock import MagicMock

import pytest

from traigent.agents.platforms import (
    LangChainAgentExecutor,
    OpenAIAgentExecutor,
    PlatformRegistry,
)
from traigent.cloud.models import AgentSpecification


class TestPlatformRegistry:
    """Test the platform registry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        # Reset the registry state
        PlatformRegistry._executors = {
            "langchain": LangChainAgentExecutor,
            "openai": OpenAIAgentExecutor,
        }
        return PlatformRegistry

    def test_get_executor_langchain(self, registry):
        """Test getting LangChain executor from registry."""
        executor_class = registry.get_executor("langchain")
        assert executor_class == LangChainAgentExecutor

    def test_get_executor_openai(self, registry):
        """Test getting OpenAI executor from registry."""
        executor_class = registry.get_executor("openai")
        assert executor_class == OpenAIAgentExecutor

    def test_get_executor_unknown(self, registry):
        """Test getting unknown executor raises error."""
        with pytest.raises(ValueError, match="Unknown platform: unknown"):
            registry.get_executor("unknown")

    def test_register_executor(self, registry):
        """Test registering a new executor."""
        from traigent.agents.executor import AgentExecutor

        class CustomExecutor(AgentExecutor):
            async def _platform_initialize(self):
                pass

            async def _execute_agent(self, agent_spec, input_data, config):
                return {"output": "custom"}

            def _validate_platform_spec(self, agent_spec):
                pass

            async def _validate_platform_config(self, config):
                return {"valid": True}

            def _get_platform_capabilities(self):
                return ["custom"]

        registry.register_executor("custom", CustomExecutor)

        assert registry.get_executor("custom") == CustomExecutor

    def test_list_platforms(self, registry):
        """Test listing available platforms."""
        platforms = registry.list_platforms()

        assert "langchain" in platforms
        assert "openai" in platforms
        assert len(platforms) >= 2

    def test_register_duplicate_platform(self, registry):
        """Test that registering duplicate platform overwrites."""
        from traigent.agents.executor import AgentExecutor

        # Save original executor for cleanup
        original_executor = registry.get_executor("langchain")

        class CustomLangChain(AgentExecutor):
            async def _platform_initialize(self):
                pass

            async def _execute_agent(self, agent_spec, input_data, config):
                return {"output": "custom"}

            def _validate_platform_spec(self, agent_spec):
                pass

            async def _validate_platform_config(self, config):
                return {"errors": [], "warnings": []}

            def _get_platform_capabilities(self):
                return ["custom"]

        # Should not raise error, just overwrite
        registry.register_executor("langchain", CustomLangChain)

        assert registry.get_executor("langchain") == CustomLangChain

        # Restore original executor to avoid test contamination
        registry.register_executor("langchain", original_executor)


class TestFrameworkOverride:
    """Test framework override functionality."""

    @pytest.mark.asyncio
    async def test_custom_platform_integration(self):
        """Test integrating a completely custom platform."""
        from traigent.agents.executor import AgentExecutor

        class VertexAIExecutor(AgentExecutor):
            """Custom executor for Google Vertex AI."""

            async def _platform_initialize(self):
                self._vertex_available = True
                self.client = MagicMock()  # Mock Vertex AI client

            async def _execute_agent(self, agent_spec, input_data, config):
                # Simulate Vertex AI execution
                prompt = self._format_prompt(agent_spec.prompt_template, input_data)

                return {
                    "output": f"Vertex AI response to: {prompt}",
                    "tokens_used": 120,
                    "metadata": {
                        "model": config.get("model", "text-bison"),
                        "location": config.get("location", "us-central1"),
                        "project": config.get("project_id"),
                    },
                }

            def _validate_platform_spec(self, agent_spec):
                if agent_spec.agent_platform != "vertex-ai":
                    raise ValueError("Invalid platform")

            async def _validate_platform_config(self, config):
                errors = []
                if not config.get("project_id"):
                    errors.append("project_id is required for Vertex AI")

                return {"valid": len(errors) == 0, "errors": errors, "warnings": []}

            def _get_platform_capabilities(self):
                return ["task", "conversational", "embeddings"]

            def _format_prompt(self, template, data):
                prompt = template
                for key, value in data.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))
                return prompt

        # Register the custom executor
        PlatformRegistry.register_executor("vertex-ai", VertexAIExecutor)

        # Use the custom executor
        executor = PlatformRegistry.get_executor("vertex-ai")()
        await executor.initialize()

        spec = AgentSpecification(
            name="vertex_test",
            agent_platform="vertex-ai",
            agent_type="task",
            description="Vertex AI test",
            prompt_template="Generate code for: {task}",
            model_parameters={"model": "code-bison"},
        )

        result = await executor.execute(
            agent_spec=spec,
            input_data={"task": "sorting algorithm"},
            config_overrides={
                "model": "code-bison",
                "project_id": "my-gcp-project",
                "location": "us-east1",
            },
        )

        assert "Vertex AI response" in result.output
        assert result.metadata["model"] == "code-bison"
        assert result.metadata["location"] == "us-east1"
        assert result.metadata["project"] == "my-gcp-project"

    def test_platform_capability_checking(self):
        """Test checking platform capabilities before execution."""
        executor = LangChainAgentExecutor()

        capabilities = executor._get_platform_capabilities()

        # Verify expected capabilities
        assert "conversational" in capabilities
        assert "task" in capabilities
        assert "tools" in capabilities
        assert "async" in capabilities

        # Test capability checking logic
        spec = AgentSpecification(
            name="test",
            agent_platform="langchain",
            agent_type="conversational",  # Should be supported
            description="Test",
            prompt_template="Test",
            model_parameters={"model": "gpt-4"},
        )

        # This should not raise an error
        executor._validate_platform_spec(spec)
