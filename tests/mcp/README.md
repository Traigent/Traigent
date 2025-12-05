# MCP (Model Context Protocol) Testing Framework

This directory contains a comprehensive testing framework for MCP integration in the TraiGent SDK. The framework verifies that natural language task descriptions can be correctly interpreted and mapped to appropriate MCP API calls.

## Overview

The MCP testing framework provides:
- **Unit tests** for individual MCP endpoints
- **Integration tests** for complete end-to-end scenarios
- **LLM task interpretation tests** for natural language understanding
- **Automatic cleanup** and resource management
- **Complete coverage** of all MCP functionality

## Directory Structure

```
tests/mcp/
├── __init__.py                      # Package initialization
├── conftest.py                      # Pytest fixtures and utilities
├── test_mcp_unit.py                # Unit tests for individual endpoints
├── test_mcp_integration.py         # End-to-end integration tests
├── test_llm_task_interpretation.py # Natural language interpretation tests
├── run_all_tests.py               # Comprehensive test runner
├── README.md                      # This documentation
└── fixtures/                      # Test data and examples
    ├── sample_agents.json         # Pre-defined agent specifications
    ├── sample_datasets.jsonl      # Test datasets
    └── task_interpretations.json  # Task interpretation examples
```

## Core Components

### 1. MCP Endpoints Tested

The framework tests the following MCP endpoints:

#### Agent Management
- `create_agent` - Create new AI agents with specifications
- `execute_agent` - Execute agents with input data
- `update_agent` - Modify existing agent configurations
- `list_agents` - Retrieve available agents

#### Optimization
- `create_optimization_session` - Start optimization process
- `get_next_trial` - Get next configuration to test
- `submit_trial_results` - Submit evaluation results
- `finalize_optimization` - Complete optimization and get best config

#### Agent-Specific Optimization
- `start_agent_optimization` - Optimize agent configurations
- `get_optimization_status` - Check optimization progress

### 2. Natural Language Task Interpretation

The framework verifies that LLMs can interpret task descriptions like:

```python
# Input: "Create an agent that answers questions about Python"
# Output: {
#     "action": "create_agent",
#     "agent_type": "conversational",
#     "topic": "Python programming"
# }

# Input: "Optimize my chatbot to reduce costs by 50%"
# Output: {
#     "action": "optimize_agent",
#     "optimization_goal": "cost_reduction",
#     "target_reduction": 0.5
# }
```

### 3. Test Fixtures

#### Mock MCP Service
Provides a complete mock implementation of MCP endpoints for testing without external dependencies:

```python
@pytest.fixture
def mock_mcp_service():
    """Mock MCP service endpoints for testing."""
    return MockMCPService()
```

#### Sample Data
Pre-defined test data in `fixtures/`:
- Agent specifications for different use cases
- Evaluation datasets in JSONL format
- Task interpretation examples

#### Test Context
Tracks created resources for automatic cleanup:

```python
@pytest.fixture
def mcp_test_context():
    """Provide test context with automatic cleanup."""
    context = MCPTestContext([], [], [], {})
    yield context
    # Automatic cleanup of all created resources
```

## Running Tests

### Run All Tests
```bash
python tests/mcp/run_all_tests.py
```

### Run Specific Test Suite
```bash
# Unit tests only
pytest tests/mcp/test_mcp_unit.py -v

# Integration tests only
pytest tests/mcp/test_mcp_integration.py -v

# LLM interpretation tests only
pytest tests/mcp/test_llm_task_interpretation.py -v
```

### Run with Coverage
```bash
pytest tests/mcp/ --cov=traigent --cov-report=html
```

## Test Categories

### Unit Tests (`test_mcp_unit.py`)

Tests individual MCP endpoints in isolation:

1. **Agent Creation**
   - Basic agent creation
   - Agents with custom tools
   - Validation of agent specifications
   - Multiple agent creation

2. **Agent Execution**
   - Basic execution
   - Execution with configuration overrides
   - Error handling
   - Execution with context

3. **Optimization Sessions**
   - Session creation
   - Trial suggestions
   - Trial progression
   - Invalid session handling

4. **Call History**
   - Call tracking
   - History management

### Integration Tests (`test_mcp_integration.py`)

Tests complete workflows involving multiple endpoints:

1. **Complete Optimization Flow**
   - Create → Execute → Optimize → Analyze

2. **Iterative Optimization**
   - Session creation
   - Multiple trial execution
   - Result submission
   - Progress tracking

3. **Multi-Agent Optimization**
   - Parallel optimization of multiple agents

4. **Error Recovery**
   - Handling trial failures
   - Resource cleanup on errors

5. **Cost Optimization**
   - Cost-aware model selection
   - Budget constraints

### LLM Task Interpretation Tests (`test_llm_task_interpretation.py`)

Tests natural language understanding:

1. **Task Interpretation**
   - Agent creation requests
   - Optimization requests
   - Execution requests
   - Ambiguous task handling

2. **End-to-End Execution**
   - From description to MCP calls
   - Complex multi-step tasks
   - Conditional tasks

3. **Validation**
   - Parameter validation
   - Error messages
   - Helpful suggestions

## Writing New Tests

### Adding a Unit Test

```python
class TestNewFeature:
    @pytest.mark.asyncio
    async def test_new_endpoint(self, mock_mcp_service):
        # Arrange
        request = NewEndpointRequest(...)

        # Act
        response = await mock_mcp_service.new_endpoint(request)

        # Assert
        assert response.status == "success"
```

### Adding an Integration Test

```python
@pytest.mark.asyncio
async def test_new_workflow(self, mock_cloud_client, mcp_test_context):
    # Step 1: Setup
    agent = await create_test_agent(mock_cloud_client)
    mcp_test_context.created_agents.append(agent.id)

    # Step 2: Execute workflow
    result = await execute_workflow(agent)

    # Step 3: Verify results
    assert result.success
    # Resources automatically cleaned up via context
```

### Adding Task Interpretation

```python
test_cases = [
    {
        "description": "Natural language task",
        "expected": {
            "action": "expected_action",
            "parameters": {...}
        }
    }
]
```

## Best Practices

1. **Resource Cleanup**
   - Always use `mcp_test_context` for resource tracking
   - Resources are automatically cleaned up after tests

2. **Async Testing**
   - Use `@pytest.mark.asyncio` for async tests
   - Use `await` for all async operations

3. **Mock Services**
   - Use provided mock services instead of real APIs
   - Mock services track all calls for verification

4. **Test Data**
   - Use fixtures for reusable test data
   - Keep test data in `fixtures/` directory

5. **Error Testing**
   - Test both success and failure scenarios
   - Verify error messages are helpful

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/TraigentSDK
   python -m pytest tests/mcp/
   ```

2. **Async Warnings**
   ```bash
   # Install pytest-asyncio
   pip install pytest-asyncio
   ```

3. **Coverage Not Working**
   ```bash
   # Install coverage
   pip install coverage pytest-cov
   ```

## Future Enhancements

1. **Performance Testing**
   - Benchmark MCP endpoint response times
   - Load testing for concurrent requests

2. **Security Testing**
   - Authentication/authorization tests
   - Input validation and sanitization

3. **Extended LLM Testing**
   - More complex task interpretations
   - Multi-language support
   - Context-aware interpretations

4. **Visualization**
   - Test result dashboards
   - Coverage heatmaps
   - Performance graphs

## Contributing

When adding new MCP functionality:

1. Add unit tests in `test_mcp_unit.py`
2. Add integration tests in `test_mcp_integration.py`
3. Add interpretation examples in `test_llm_task_interpretation.py`
4. Update fixtures if needed
5. Run full test suite before submitting

## License

This testing framework is part of the TraiGent SDK and follows the same MIT license.
