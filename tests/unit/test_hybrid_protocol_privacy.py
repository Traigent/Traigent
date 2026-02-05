"""Unit tests for Hybrid Mode protocol DTOs with privacy-preserving fields.

Tests that the protocol DTOs correctly handle:
- InputItem with optional data field
- OutputItem with optional output and output_id fields
- EvaluationItem with optional output, output_id, target, target_id fields
"""

import pytest

from traigent.hybrid.protocol import (
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
)


class TestInputItemPrivacy:
    """Tests for InputItem privacy-preserving support."""

    def test_input_with_data(self) -> None:
        """Test standard mode: input with data field."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"model": "fast"},
            inputs=[
                {"input_id": "ex_1", "data": {"query": "What is AI?"}},
            ],
        )

        request_dict = request.to_dict()
        assert request_dict["inputs"][0]["input_id"] == "ex_1"
        assert request_dict["inputs"][0]["data"] == {"query": "What is AI?"}

    def test_input_without_data_privacy_mode(self) -> None:
        """Test privacy-preserving mode: input with only input_id."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"model": "fast"},
            inputs=[
                {"input_id": "ex_1"},  # No data field
                {"input_id": "ex_2"},  # No data field
            ],
        )

        request_dict = request.to_dict()
        assert request_dict["inputs"][0]["input_id"] == "ex_1"
        assert "data" not in request_dict["inputs"][0]
        assert request_dict["inputs"][1]["input_id"] == "ex_2"
        assert "data" not in request_dict["inputs"][1]

    def test_mixed_inputs(self) -> None:
        """Test mixed mode: some inputs with data, some without."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"model": "balanced"},
            inputs=[
                {"input_id": "ex_1", "data": {"query": "test"}},  # With data
                {"input_id": "ex_2"},  # Without data
            ],
        )

        request_dict = request.to_dict()
        assert "data" in request_dict["inputs"][0]
        assert "data" not in request_dict["inputs"][1]


class TestOutputItemPrivacy:
    """Tests for OutputItem privacy-preserving support."""

    def test_output_with_content(self) -> None:
        """Test standard mode: output with full content."""
        response_data = {
            "request_id": "req_123",
            "execution_id": "exec_456",
            "status": "completed",
            "outputs": [
                {
                    "input_id": "ex_1",
                    "output": {"response": "AI is..."},
                    "cost_usd": 0.001,
                },
            ],
            "operational_metrics": {"total_cost_usd": 0.001},
        }

        response = HybridExecuteResponse.from_dict(response_data)
        assert response.outputs[0]["input_id"] == "ex_1"
        assert response.outputs[0]["output"]["response"] == "AI is..."
        assert "output_id" not in response.outputs[0]

    def test_output_with_output_id(self) -> None:
        """Test privacy-preserving mode: output with output_id instead of content."""
        response_data = {
            "request_id": "req_123",
            "execution_id": "exec_456",
            "status": "completed",
            "outputs": [
                {
                    "input_id": "ex_1",
                    "output_id": "out_ex_1_session_abc",
                    "cost_usd": 0.001,
                },
            ],
            "operational_metrics": {"total_cost_usd": 0.001},
        }

        response = HybridExecuteResponse.from_dict(response_data)
        assert response.outputs[0]["input_id"] == "ex_1"
        assert response.outputs[0]["output_id"] == "out_ex_1_session_abc"
        assert "output" not in response.outputs[0]

    def test_output_metrics_preserved(self) -> None:
        """Test that cost and latency are preserved in both modes."""
        response_data = {
            "request_id": "req_123",
            "execution_id": "exec_456",
            "status": "completed",
            "outputs": [
                {
                    "input_id": "ex_1",
                    "output_id": "out_ex_1",
                    "cost_usd": 0.005,
                    "latency_ms": 150.0,
                },
            ],
            "operational_metrics": {"total_cost_usd": 0.005},
        }

        response = HybridExecuteResponse.from_dict(response_data)
        assert response.outputs[0]["cost_usd"] == 0.005
        assert response.outputs[0]["latency_ms"] == 150.0


class TestEvaluationItemPrivacy:
    """Tests for EvaluationItem privacy-preserving support."""

    def test_evaluation_with_content(self) -> None:
        """Test standard mode: evaluation with full output and target content."""
        request = HybridEvaluateRequest(
            capability_id="test_agent",
            evaluations=[
                {
                    "input_id": "ex_1",
                    "output": {"response": "AI is..."},
                    "target": {"expected": "AI is artificial..."},
                },
            ],
        )

        request_dict = request.to_dict()
        eval_item = request_dict["evaluations"][0]
        assert eval_item["output"]["response"] == "AI is..."
        assert eval_item["target"]["expected"] == "AI is artificial..."

    def test_evaluation_with_ids(self) -> None:
        """Test privacy-preserving mode: evaluation with output_id and target_id."""
        request = HybridEvaluateRequest(
            capability_id="test_agent",
            evaluations=[
                {
                    "input_id": "ex_1",
                    "output_id": "out_ex_1_session_abc",
                    "target_id": "target_ex_1",
                },
            ],
        )

        request_dict = request.to_dict()
        eval_item = request_dict["evaluations"][0]
        assert eval_item["output_id"] == "out_ex_1_session_abc"
        assert eval_item["target_id"] == "target_ex_1"
        assert "output" not in eval_item
        assert "target" not in eval_item

    def test_evaluation_mixed_mode(self) -> None:
        """Test mixed mode: output content with target_id."""
        request = HybridEvaluateRequest(
            capability_id="test_agent",
            evaluations=[
                {
                    "input_id": "ex_1",
                    "output": {"response": "AI is..."},  # Full content
                    "target_id": "target_ex_1",  # ID reference
                },
            ],
        )

        request_dict = request.to_dict()
        eval_item = request_dict["evaluations"][0]
        assert "output" in eval_item
        assert "target_id" in eval_item
        assert "output_id" not in eval_item
        assert "target" not in eval_item


class TestEvaluateResponsePrivacy:
    """Tests for EvaluateResponse result handling."""

    def test_results_with_metrics(self) -> None:
        """Test that evaluation results contain metrics regardless of privacy mode."""
        response_data = {
            "request_id": "req_123",
            "status": "completed",
            "results": [
                {
                    "input_id": "ex_1",
                    "metrics": {
                        "accuracy": 0.92,
                        "relevance": 0.88,
                        "fluency": 0.95,
                    },
                },
            ],
            "aggregate_metrics": {
                "accuracy": {"mean": 0.92, "std": 0.05, "n": 1},
            },
        }

        response = HybridEvaluateResponse.from_dict(response_data)
        assert response.results[0]["metrics"]["accuracy"] == 0.92
        assert response.aggregate_metrics["accuracy"]["mean"] == 0.92


class TestSessionIdScoping:
    """Tests for session_id handling in requests."""

    def test_execute_with_session_id(self) -> None:
        """Test that session_id is included in execute request."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"model": "fast"},
            inputs=[{"input_id": "ex_1"}],
            session_id="session_abc123",
        )

        request_dict = request.to_dict()
        assert request_dict["session_id"] == "session_abc123"

    def test_evaluate_with_session_id(self) -> None:
        """Test that session_id is included in evaluate request."""
        request = HybridEvaluateRequest(
            capability_id="test_agent",
            evaluations=[
                {"input_id": "ex_1", "output_id": "out_1", "target_id": "t_1"}
            ],
            session_id="session_abc123",
        )

        request_dict = request.to_dict()
        assert request_dict["session_id"] == "session_abc123"

    def test_execute_response_with_session_id(self) -> None:
        """Test that session_id is parsed from execute response."""
        response_data = {
            "request_id": "req_123",
            "execution_id": "exec_456",
            "status": "completed",
            "outputs": [],
            "operational_metrics": {},
            "session_id": "session_abc123",
        }

        response = HybridExecuteResponse.from_dict(response_data)
        assert response.session_id == "session_abc123"


class TestPrivacyModeDocumentation:
    """Tests that verify privacy-preserving mode contract."""

    def test_input_only_requires_input_id(self) -> None:
        """Verify that input_id is the only required field for InputItem."""
        # This should work without data field
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={},
            inputs=[{"input_id": "required_only"}],
        )

        assert len(request.inputs) == 1
        assert request.inputs[0]["input_id"] == "required_only"

    def test_output_item_flexible_fields(self) -> None:
        """Verify OutputItem accepts both output and output_id."""
        # With output content
        response1 = HybridExecuteResponse.from_dict(
            {
                "request_id": "r1",
                "execution_id": "e1",
                "status": "completed",
                "outputs": [{"input_id": "ex_1", "output": {"data": "content"}}],
                "operational_metrics": {},
            }
        )
        assert "output" in response1.outputs[0]

        # With output_id
        response2 = HybridExecuteResponse.from_dict(
            {
                "request_id": "r2",
                "execution_id": "e2",
                "status": "completed",
                "outputs": [{"input_id": "ex_1", "output_id": "out_123"}],
                "operational_metrics": {},
            }
        )
        assert "output_id" in response2.outputs[0]

    def test_evaluation_item_flexible_fields(self) -> None:
        """Verify EvaluationItem accepts both content and IDs."""
        request = HybridEvaluateRequest(
            capability_id="test_agent",
            evaluations=[
                # All content
                {"input_id": "ex_1", "output": {}, "target": {}},
                # All IDs
                {"input_id": "ex_2", "output_id": "o2", "target_id": "t2"},
                # Mixed
                {"input_id": "ex_3", "output": {}, "target_id": "t3"},
                {"input_id": "ex_4", "output_id": "o4", "target": {}},
            ],
        )

        assert len(request.evaluations) == 4
