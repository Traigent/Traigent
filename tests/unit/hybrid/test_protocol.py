"""Unit tests for hybrid mode protocol DTOs."""

from traigent.hybrid.protocol import (
    BatchOptions,
    ConfigSpaceResponse,
    HealthCheckResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
    ServiceCapabilities,
    TVARDefinition,
)


class TestBatchOptions:
    """Tests for BatchOptions dataclass."""

    def test_default_values(self) -> None:
        """Test default batch options."""
        opts = BatchOptions()
        assert opts.parallelism == 1
        assert opts.fail_fast is False
        assert opts.timeout_per_item_ms == 0

    def test_custom_values(self) -> None:
        """Test custom batch options."""
        opts = BatchOptions(parallelism=5, fail_fast=True, timeout_per_item_ms=1000)
        assert opts.parallelism == 5
        assert opts.fail_fast is True
        assert opts.timeout_per_item_ms == 1000


class TestHybridExecuteRequest:
    """Tests for HybridExecuteRequest."""

    def test_minimal_request(self) -> None:
        """Test creating minimal execute request."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"temperature": 0.7},
            inputs=[{"input_id": "ex_1", "data": {"query": "test"}}],
        )

        assert request.capability_id == "test_agent"
        assert request.config == {"temperature": 0.7}
        assert len(request.inputs) == 1
        assert request.request_id  # Auto-generated
        assert request.session_id is None
        assert request.batch_options is None
        assert request.timeout_ms == 30000

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        request = HybridExecuteRequest(
            capability_id="test_agent",
            config={"model": "gpt-4"},
            inputs=[{"input_id": "1", "data": {}}],
            session_id="sess_123",
            batch_options=BatchOptions(parallelism=3),
            timeout_ms=60000,
        )

        d = request.to_dict()
        assert d["capability_id"] == "test_agent"
        assert d["config"] == {"model": "gpt-4"}
        assert d["inputs"] == [{"input_id": "1", "data": {}}]
        assert d["session_id"] == "sess_123"
        assert d["batch_options"]["parallelism"] == 3
        assert d["timeout_ms"] == 60000

    def test_to_dict_minimal(self) -> None:
        """Test minimal serialization excludes None fields."""
        request = HybridExecuteRequest(
            capability_id="test",
            config={},
            inputs=[],
        )

        d = request.to_dict()
        assert "session_id" not in d
        assert "batch_options" not in d


class TestHybridExecuteResponse:
    """Tests for HybridExecuteResponse."""

    def test_from_dict(self) -> None:
        """Test creating response from dictionary."""
        data = {
            "request_id": "req_123",
            "execution_id": "exec_456",
            "status": "completed",
            "outputs": [{"input_id": "1", "output": "result"}],
            "operational_metrics": {"cost_usd": 0.002, "latency_ms": 150},
            "session_id": "sess_789",
        }

        response = HybridExecuteResponse.from_dict(data)

        assert response.request_id == "req_123"
        assert response.execution_id == "exec_456"
        assert response.status == "completed"
        assert len(response.outputs) == 1
        assert response.operational_metrics["cost_usd"] == 0.002
        assert response.session_id == "sess_789"
        assert response.quality_metrics is None
        assert response.error is None

    def test_get_total_cost(self) -> None:
        """Test extracting total cost."""
        response = HybridExecuteResponse(
            request_id="r1",
            execution_id="e1",
            status="completed",
            outputs=[],
            operational_metrics={"total_cost_usd": 0.05},
        )
        assert response.get_total_cost() == 0.05

    def test_get_total_cost_fallback(self) -> None:
        """Test cost fallback to cost_usd."""
        response = HybridExecuteResponse(
            request_id="r1",
            execution_id="e1",
            status="completed",
            outputs=[],
            operational_metrics={"cost_usd": 0.03},
        )
        assert response.get_total_cost() == 0.03


class TestHybridEvaluateRequest:
    """Tests for HybridEvaluateRequest."""

    def test_minimal_request(self) -> None:
        """Test creating minimal evaluate request."""
        request = HybridEvaluateRequest(capability_id="test")
        assert request.capability_id == "test"
        assert request.request_id  # Auto-generated
        assert request.execution_id is None
        assert request.evaluations is None

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        request = HybridEvaluateRequest(
            capability_id="test",
            execution_id="exec_123",
            evaluations=[{"input_id": "1", "output": {}, "target": {}}],
            config={"setting": "value"},
            session_id="sess_456",
        )

        d = request.to_dict()
        assert d["capability_id"] == "test"
        assert d["execution_id"] == "exec_123"
        assert len(d["evaluations"]) == 1
        assert d["config"] == {"setting": "value"}
        assert d["session_id"] == "sess_456"


class TestHybridEvaluateResponse:
    """Tests for HybridEvaluateResponse."""

    def test_from_dict(self) -> None:
        """Test creating response from dictionary."""
        data = {
            "request_id": "req_123",
            "status": "completed",
            "results": [{"input_id": "1", "metrics": {"accuracy": 0.95}}],
            "aggregate_metrics": {"accuracy": {"mean": 0.95, "std": 0.02, "n": 10}},
        }

        response = HybridEvaluateResponse.from_dict(data)

        assert response.request_id == "req_123"
        assert response.status == "completed"
        assert len(response.results) == 1
        assert response.results[0]["metrics"]["accuracy"] == 0.95
        assert response.aggregate_metrics["accuracy"]["mean"] == 0.95


class TestServiceCapabilities:
    """Tests for ServiceCapabilities."""

    def test_default_values(self) -> None:
        """Test default capabilities."""
        caps = ServiceCapabilities(version="1.0")
        assert caps.version == "1.0"
        assert caps.supports_evaluate is True
        assert caps.supports_keep_alive is False
        assert caps.supports_streaming is False
        assert caps.max_batch_size == 100
        assert caps.max_payload_bytes is None

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "version": "2.0",
            "supports_evaluate": False,
            "supports_keep_alive": True,
            "max_batch_size": 50,
        }

        caps = ServiceCapabilities.from_dict(data)
        assert caps.version == "2.0"
        assert caps.supports_evaluate is False
        assert caps.supports_keep_alive is True
        assert caps.max_batch_size == 50


class TestTVARDefinition:
    """Tests for TVARDefinition."""

    def test_enum_tvar(self) -> None:
        """Test enum type TVAR."""
        tvar = TVARDefinition(
            name="model",
            type="enum",
            domain={"values": ["gpt-4", "claude-3"]},
            default="gpt-4",
        )

        assert tvar.name == "model"
        assert tvar.type == "enum"
        assert tvar.to_traigent_config_space() == ["gpt-4", "claude-3"]

    def test_bool_tvar(self) -> None:
        """Test bool type TVAR."""
        tvar = TVARDefinition(
            name="use_cache",
            type="bool",
            domain={},
        )

        assert tvar.to_traigent_config_space() == [True, False]

    def test_int_tvar(self) -> None:
        """Test int type TVAR."""
        tvar = TVARDefinition(
            name="max_tokens",
            type="int",
            domain={"range": [100, 4096]},
        )

        config = tvar.to_traigent_config_space()
        assert config["low"] == 100
        assert config["high"] == 4096
        assert config["type"] == "int"

    def test_float_tvar(self) -> None:
        """Test float type TVAR with resolution."""
        tvar = TVARDefinition(
            name="temperature",
            type="float",
            domain={"range": [0.0, 2.0], "resolution": 0.1},
        )

        config = tvar.to_traigent_config_space()
        assert config["low"] == 0.0
        assert config["high"] == 2.0
        assert config["step"] == 0.1

    def test_from_dict(self) -> None:
        """Test creating TVAR from dictionary."""
        data = {
            "name": "test_var",
            "type": "enum",
            "domain": {"values": ["a", "b"]},
            "is_tool": True,
            "constraints": ["requires model == gpt-4"],
        }

        tvar = TVARDefinition.from_dict(data)
        assert tvar.name == "test_var"
        assert tvar.is_tool is True
        assert len(tvar.constraints) == 1


class TestConfigSpaceResponse:
    """Tests for ConfigSpaceResponse."""

    def test_from_dict(self) -> None:
        """Test creating config space from dictionary."""
        data = {
            "schema_version": "0.9",
            "capability_id": "test_agent",
            "tvars": [
                {"name": "model", "type": "enum", "domain": {"values": ["gpt-4"]}},
                {"name": "temp", "type": "float", "domain": {"range": [0.0, 1.0]}},
            ],
        }

        response = ConfigSpaceResponse.from_dict(data)
        assert response.schema_version == "0.9"
        assert response.capability_id == "test_agent"
        assert len(response.tvars) == 2

    def test_to_traigent_config_space(self) -> None:
        """Test converting to Traigent config space format."""
        response = ConfigSpaceResponse(
            schema_version="0.9",
            capability_id="test",
            tvars=[
                TVARDefinition(
                    name="model",
                    type="enum",
                    domain={"values": ["gpt-4", "gpt-3.5"]},
                ),
                TVARDefinition(
                    name="temperature",
                    type="float",
                    domain={"range": [0.0, 2.0]},
                ),
            ],
        )

        config_space = response.to_traigent_config_space()
        assert config_space["model"] == ["gpt-4", "gpt-3.5"]
        assert config_space["temperature"]["low"] == 0.0
        assert config_space["temperature"]["high"] == 2.0


class TestHealthCheckResponse:
    """Tests for HealthCheckResponse."""

    def test_from_dict(self) -> None:
        """Test creating health response from dictionary."""
        data = {
            "status": "healthy",
            "version": "1.2.3",
            "uptime_seconds": 3600.0,
        }

        response = HealthCheckResponse.from_dict(data)
        assert response.status == "healthy"
        assert response.version == "1.2.3"
        assert response.uptime_seconds == 3600.0

    def test_default_status(self) -> None:
        """Test default status when not provided."""
        response = HealthCheckResponse.from_dict({})
        assert response.status == "unhealthy"
