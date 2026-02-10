"""Unit tests for TraigentService wrapper.

Tests for the TraigentService class, ServiceConfig, Session,
and all decorator/handler functionality.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from traigent.wrapper.service import ServiceConfig, Session, TraigentService


# ---------------------------------------------------------------------------
# ServiceConfig tests
# ---------------------------------------------------------------------------
class TestServiceConfig:
    """Tests for ServiceConfig dataclass."""

    def test_defaults(self) -> None:
        """Test ServiceConfig default values."""
        cfg = ServiceConfig()
        assert cfg.capability_id == "default"
        assert cfg.version == "1.0"
        assert cfg.supports_keep_alive is False
        assert cfg.supports_streaming is False
        assert cfg.max_batch_size == 100
        assert cfg.schema_version == "0.9"
        assert cfg.constraints is None
        assert cfg.objectives is None
        assert cfg.exploration is None
        assert cfg.promotion_policy is None
        assert cfg.defaults is None
        assert cfg.measures is None

    def test_custom_values(self) -> None:
        """Test ServiceConfig with custom values."""
        cfg = ServiceConfig(
            capability_id="qa_agent",
            version="2.5",
            supports_keep_alive=True,
            supports_streaming=True,
            max_batch_size=50,
            schema_version="1.0",
            constraints={"structural": [{"expr": "params.temperature <= 1.0"}]},
            objectives=[{"name": "accuracy", "direction": "maximize"}],
            exploration={"strategy": "nsga2"},
            promotion_policy={"dominance": "epsilon_pareto"},
            defaults={"model": "gpt-4"},
            measures=["accuracy"],
        )
        assert cfg.capability_id == "qa_agent"
        assert cfg.version == "2.5"
        assert cfg.supports_keep_alive is True
        assert cfg.supports_streaming is True
        assert cfg.max_batch_size == 50
        assert cfg.schema_version == "1.0"
        assert cfg.constraints is not None
        assert cfg.objectives is not None
        assert cfg.exploration is not None
        assert cfg.promotion_policy is not None
        assert cfg.defaults == {"model": "gpt-4"}
        assert cfg.measures == ["accuracy"]


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------
class TestSession:
    """Tests for Session dataclass."""

    def test_creation(self) -> None:
        """Test Session creation with default fields."""
        session = Session(session_id="abc-123")
        assert session.session_id == "abc-123"
        assert isinstance(session.created_at, float)
        assert isinstance(session.last_activity, float)
        assert session.state == {}

    def test_creation_with_state(self) -> None:
        """Test Session creation with custom state."""
        session = Session(session_id="s1", state={"key": "value"})
        assert session.state == {"key": "value"}

    def test_touch_updates_last_activity(self) -> None:
        """Test that touch() updates last_activity timestamp."""
        session = Session(session_id="s1")
        old_activity = session.last_activity
        # Small sleep to ensure timestamp differs
        time.sleep(0.01)
        session.touch()
        assert session.last_activity > old_activity


# ---------------------------------------------------------------------------
# TraigentService initialisation tests
# ---------------------------------------------------------------------------
class TestTraigentServiceInit:
    """Tests for TraigentService initialization."""

    def test_defaults(self) -> None:
        """Test TraigentService default initialization."""
        svc = TraigentService()
        assert svc.config.capability_id == "default"
        assert svc.config.version == "1.0"
        assert svc.config.supports_keep_alive is False
        assert svc.config.supports_streaming is False
        assert svc.config.max_batch_size == 100
        assert svc._tvars_handler is None
        assert svc._execute_handler is None
        assert svc._evaluate_handler is None
        assert svc._sessions == {}
        assert svc._cached_tvars is None

    def test_custom_init(self) -> None:
        """Test TraigentService with custom parameters."""
        svc = TraigentService(
            capability_id="my_agent",
            version="3.0",
            supports_keep_alive=True,
            supports_streaming=True,
            max_batch_size=25,
            objectives=[{"name": "accuracy", "direction": "maximize"}],
            exploration={"strategy": "nsga2"},
            promotion_policy={"dominance": "epsilon_pareto"},
            defaults={"model": "gpt-4"},
            measures=["accuracy"],
        )
        assert svc.config.capability_id == "my_agent"
        assert svc.config.version == "3.0"
        assert svc.config.supports_keep_alive is True
        assert svc.config.supports_streaming is True
        assert svc.config.max_batch_size == 25
        assert svc.config.objectives == [{"name": "accuracy", "direction": "maximize"}]
        assert svc.config.exploration == {"strategy": "nsga2"}
        assert svc.config.promotion_policy == {"dominance": "epsilon_pareto"}
        assert svc.config.defaults == {"model": "gpt-4"}
        assert svc.config.measures == ["accuracy"]


# ---------------------------------------------------------------------------
# Decorator registration tests
# ---------------------------------------------------------------------------
class TestDecorators:
    """Tests for tvars, tunables, execute, evaluate decorators."""

    def test_tvars_registers_handler(self) -> None:
        """Test that @tvars registers the function and invalidates cache."""
        svc = TraigentService()
        # Set a cached value to verify invalidation
        svc._cached_tvars = {"old": "data"}

        @svc.tvars
        def config_space():
            return {"model": {"type": "enum", "values": ["gpt-4"]}}

        assert svc._tvars_handler is config_space
        assert svc._cached_tvars is None  # cache invalidated

    def test_tvars_returns_original_function(self) -> None:
        """Test that @tvars returns the original function unchanged."""
        svc = TraigentService()

        @svc.tvars
        def my_func():
            return {}

        assert my_func() == {}

    def test_tunables_is_alias_for_tvars(self) -> None:
        """Test that @tunables delegates to tvars."""
        svc = TraigentService()

        @svc.tunables
        def config_space():
            return {"temp": {"type": "float"}}

        assert svc._tvars_handler is config_space

    def test_execute_registers_handler(self) -> None:
        """Test that @execute registers the execution handler."""
        svc = TraigentService()

        @svc.execute
        async def run(input_id, data, config):
            return {"output": "ok"}

        assert svc._execute_handler is run

    def test_execute_returns_original_function(self) -> None:
        """Test that @execute returns the original function unchanged."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": "result"}

        assert run("id1", {}, {}) == {"output": "result"}

    def test_evaluate_registers_handler(self) -> None:
        """Test that @evaluate registers the evaluation handler."""
        svc = TraigentService()

        @svc.evaluate
        async def score(output, target, config):
            return {"accuracy": 0.9}

        assert svc._evaluate_handler is score

    def test_evaluate_returns_original_function(self) -> None:
        """Test that @evaluate returns the original function unchanged."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        assert score(None, None, {}) == {"accuracy": 1.0}

    def test_declarative_spec_decorators_register_handlers(self) -> None:
        """Objective/exploration/policy/defaults/measures decorators register."""
        svc = TraigentService()

        @svc.objectives
        def objectives():
            return [{"name": "accuracy", "direction": "maximize"}]

        @svc.exploration
        def exploration():
            return {"strategy": "nsga2"}

        @svc.promotion_policy
        def promotion_policy():
            return {"dominance": "epsilon_pareto"}

        @svc.defaults
        def defaults():
            return {"model": "gpt-4"}

        @svc.measures
        def measures():
            return ["accuracy"]

        assert svc._objectives_handler is objectives
        assert svc._exploration_handler is exploration
        assert svc._promotion_policy_handler is promotion_policy
        assert svc._defaults_handler is defaults
        assert svc._measures_handler is measures

    def test_constraints_decorator_registers_handler(self) -> None:
        """@constraints registers the constraints declaration handler."""
        svc = TraigentService()

        @svc.constraints
        def constraints():
            return {"structural": [{"expr": "params.temperature <= 1.0"}]}

        assert svc._constraints_handler is constraints


# ---------------------------------------------------------------------------
# get_config_space / _normalize_tvars tests
# ---------------------------------------------------------------------------
class TestGetConfigSpace:
    """Tests for get_config_space and _normalize_tvars."""

    def test_no_handler_returns_empty_tvars(self) -> None:
        """Test that config space with no handler returns empty tvars."""
        svc = TraigentService(capability_id="test")
        result = svc.get_config_space()
        assert result["schema_version"] == "0.9"
        assert result["capability_id"] == "test"
        assert result["tvars"] == []
        assert result["constraints"] == {}

    def test_dict_tvars_pass_through(self) -> None:
        """Test that dict-type TVAR specs pass through unchanged."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"model": {"type": "enum", "values": ["gpt-4"]}}

        result = svc.get_config_space()
        assert len(result["tvars"]) == 1
        tvar = result["tvars"][0]
        assert tvar["name"] == "model"
        assert tvar["type"] == "enum"
        assert tvar["domain"]["values"] == ["gpt-4"]

    def test_dict_tvars_without_type_preserves_domain_fields(self) -> None:
        """Top-level domain keys should not be dropped when type is omitted."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"model": {"values": ["gpt-4", "claude-3"]}}

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert tvar["name"] == "model"
        assert tvar["domain"]["values"] == ["gpt-4", "claude-3"]

    def test_empty_domain_is_preserved(self) -> None:
        """Explicit empty domain should not be dropped for bool-like TVARs."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"use_cache": {"type": "bool", "domain": {}}}

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert tvar["name"] == "use_cache"
        assert "domain" in tvar
        assert tvar["domain"] == {}

    def test_async_declaration_handler_rejected_without_coroutine_leak(self) -> None:
        """Async declaration handlers should fail before a coroutine object is created."""
        svc = TraigentService()

        @svc.objectives
        async def declared_objectives():
            return [{"name": "accuracy", "direction": "maximize"}]

        with pytest.raises(
            ValueError, match="declaration handlers must be synchronous functions"
        ):
            svc.get_config_space()

    def test_list_tvars_converted_to_enum(self) -> None:
        """Test that list-type TVAR specs are converted to enum."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"model": ["gpt-4", "claude-3"]}

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert tvar["name"] == "model"
        assert tvar["type"] == "enum"
        assert tvar["domain"] == {"values": ["gpt-4", "claude-3"]}

    def test_scalar_tvars_converted_to_str(self) -> None:
        """Test that scalar TVAR specs become str type with default."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"greeting": "hello"}

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert tvar["name"] == "greeting"
        assert tvar["type"] == "str"
        assert tvar["default"] == "hello"

    def test_caching(self) -> None:
        """Test that config space is cached after first call."""
        svc = TraigentService()
        call_count = 0

        @svc.tvars
        def cfg():
            nonlocal call_count
            call_count += 1
            return {"a": {"type": "int"}}

        svc.get_config_space()
        svc.get_config_space()
        assert call_count == 1  # Handler called only once

    def test_cache_invalidated_on_re_register(self) -> None:
        """Test that cache is invalidated when tvars is re-registered."""
        svc = TraigentService()

        @svc.tvars
        def cfg_v1():
            return {"a": {"type": "int"}}

        svc.get_config_space()
        assert svc._cached_tvars is not None

        @svc.tvars
        def cfg_v2():
            return {"b": {"type": "float"}}

        assert svc._cached_tvars is None  # Invalidated

    def test_mixed_tvar_types(self) -> None:
        """Test config space with mixed TVAR definition types."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {
                "model": {"type": "enum", "values": ["gpt-4"]},
                "options": ["a", "b", "c"],
                "label": "default_label",
            }

        result = svc.get_config_space()
        tvars = {t["name"]: t for t in result["tvars"]}
        assert tvars["model"]["type"] == "enum"
        assert tvars["options"]["type"] == "enum"
        assert tvars["label"]["type"] == "str"

    def test_config_space_includes_optional_optimization_sections(self) -> None:
        """Config-space response includes optional declarative optimization sections."""
        svc = TraigentService(
            capability_id="financial_qa",
            constraints={"structural": [{"expr": "params.temperature <= 1.0"}]},
            objectives=[{"name": "accuracy", "direction": "maximize"}],
            exploration={"strategy": "nsga2"},
            promotion_policy={"dominance": "epsilon_pareto"},
            defaults={"model": "gpt-4"},
            measures=["accuracy", "cost"],
        )
        result = svc.get_config_space()
        assert result["constraints"] == {
            "structural": [{"expr": "params.temperature <= 1.0"}]
        }
        assert result["objectives"] == [{"name": "accuracy", "direction": "maximize"}]
        assert result["exploration"] == {"strategy": "nsga2"}
        assert result["promotion_policy"] == {"dominance": "epsilon_pareto"}
        assert result["defaults"] == {"model": "gpt-4"}
        assert result["measures"] == ["accuracy", "cost"]

    def test_config_space_decorator_sections_override_init_values(self) -> None:
        """Decorator-declared sections override constructor-provided values."""
        svc = TraigentService(
            objectives=[{"name": "old_objective", "direction": "maximize"}],
            defaults={"model": "old"},
        )

        @svc.objectives
        def objectives():
            return [{"name": "accuracy", "direction": "maximize"}]

        @svc.defaults
        def defaults():
            return {"model": "gpt-4"}

        result = svc.get_config_space()
        assert result["objectives"] == [{"name": "accuracy", "direction": "maximize"}]
        assert result["defaults"] == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# get_capabilities tests
# ---------------------------------------------------------------------------
class TestGetCapabilities:
    """Tests for get_capabilities."""

    def test_capabilities_without_evaluate(self) -> None:
        """Test capabilities when no evaluate handler is registered."""
        svc = TraigentService(
            capability_id="test",
            version="2.0",
            supports_keep_alive=True,
            supports_streaming=False,
            max_batch_size=50,
        )
        caps = svc.get_capabilities()
        assert caps["version"] == "2.0"
        assert caps["supports_evaluate"] is False
        assert caps["supports_keep_alive"] is True
        assert caps["supports_streaming"] is False
        assert caps["max_batch_size"] == 50
        assert caps["max_payload_bytes"] is None

    def test_capabilities_with_evaluate(self) -> None:
        """Test capabilities when evaluate handler is registered."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        caps = svc.get_capabilities()
        assert caps["supports_evaluate"] is True


# ---------------------------------------------------------------------------
# get_health tests
# ---------------------------------------------------------------------------
class TestGetHealth:
    """Tests for get_health."""

    def test_health_no_sessions(self) -> None:
        """Test health status with no active sessions."""
        svc = TraigentService(capability_id="agent_x", version="1.2")
        health = svc.get_health()
        assert health["status"] == "healthy"
        assert health["version"] == "1.2"
        assert isinstance(health["uptime_seconds"], float)
        assert health["details"]["capability_id"] == "agent_x"
        assert health["details"]["active_sessions"] == 0

    def test_health_with_sessions(self) -> None:
        """Test health status reflecting active sessions."""
        svc = TraigentService()
        svc.create_session()
        svc.create_session()
        health = svc.get_health()
        assert health["details"]["active_sessions"] == 2


# ---------------------------------------------------------------------------
# handle_execute tests
# ---------------------------------------------------------------------------
class TestHandleExecute:
    """Tests for handle_execute."""

    @pytest.mark.asyncio
    async def test_no_handler_raises_value_error(self) -> None:
        """Test that execute without a handler raises ValueError."""
        svc = TraigentService()
        with pytest.raises(ValueError, match="No execute handler registered"):
            await svc.handle_execute({})

    @pytest.mark.asyncio
    async def test_sync_handler_dict_result(self) -> None:
        """Test execute with a synchronous handler returning a dict."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": f"processed-{input_id}", "cost_usd": 0.01}

        request = {
            "request_id": "req-1",
            "config": {"temperature": 0.5},
            "inputs": [
                {"input_id": "i1", "data": {"query": "hello"}},
                {"input_id": "i2", "data": {"query": "world"}},
            ],
        }
        resp = await svc.handle_execute(request)
        assert resp["request_id"] == "req-1"
        assert resp["status"] == "completed"
        assert len(resp["outputs"]) == 2
        assert resp["outputs"][0]["input_id"] == "i1"
        assert resp["outputs"][0]["output"] == "processed-i1"
        assert resp["outputs"][0]["cost_usd"] == 0.01
        assert resp["operational_metrics"]["total_cost_usd"] == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        """Test execute with an async handler."""
        svc = TraigentService()

        @svc.execute
        async def run(input_id, data, config):
            return {"output": "async_result", "cost_usd": 0.05}

        resp = await svc.handle_execute({"inputs": [{"input_id": "a1", "data": {}}]})
        assert resp["status"] == "completed"
        assert resp["outputs"][0]["output"] == "async_result"
        assert resp["outputs"][0]["cost_usd"] == 0.05

    @pytest.mark.asyncio
    async def test_non_dict_result(self) -> None:
        """Test execute where handler returns a non-dict result."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return "plain_string_result"

        resp = await svc.handle_execute({"inputs": [{"input_id": "x1", "data": {}}]})
        assert resp["outputs"][0]["output"] == "plain_string_result"
        assert resp["outputs"][0]["cost_usd"] == 0.0
        assert "metrics" not in resp["outputs"][0]

    @pytest.mark.asyncio
    async def test_handler_exception_captured(self) -> None:
        """Test that handler exceptions are captured per-input."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            raise RuntimeError("boom")

        resp = await svc.handle_execute({"inputs": [{"input_id": "e1", "data": {}}]})
        assert resp["status"] == "failed"
        assert resp["outputs"][0]["input_id"] == "e1"
        assert "boom" in resp["outputs"][0]["error"]

    @pytest.mark.asyncio
    async def test_default_request_id_generated(self) -> None:
        """Test that a request_id is generated when not provided."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": "ok"}

        resp = await svc.handle_execute({"inputs": [{"input_id": "i1", "data": {}}]})
        assert "request_id" in resp
        assert len(resp["request_id"]) > 0

    @pytest.mark.asyncio
    async def test_execute_request_id_is_idempotent(self) -> None:
        """Same request_id and payload should return cached execute response."""
        svc = TraigentService()
        call_count = {"n": 0}

        @svc.execute
        def run(input_id, data, config):
            call_count["n"] += 1
            return {"output": f"ok-{call_count['n']}"}

        request = {
            "request_id": "req-idem-1",
            "capability_id": "default",
            "config": {"temperature": 0.2},
            "inputs": [{"input_id": "i1", "data": {}}],
        }
        first = await svc.handle_execute(request)
        second = await svc.handle_execute(request)

        assert call_count["n"] == 1
        assert second == first

    @pytest.mark.asyncio
    async def test_execute_request_id_payload_mismatch_raises(self) -> None:
        """Reusing request_id with changed execute payload should fail fast."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": "ok"}

        base_request = {
            "request_id": "req-idem-2",
            "capability_id": "default",
            "config": {"temperature": 0.2},
            "inputs": [{"input_id": "i1", "data": {}}],
        }
        await svc.handle_execute(base_request)

        with pytest.raises(ValueError, match="request_id reuse"):
            await svc.handle_execute(
                {
                    "request_id": "req-idem-2",
                    "capability_id": "default",
                    "config": {"temperature": 0.7},
                    "inputs": [{"input_id": "i1", "data": {}}],
                }
            )

    @pytest.mark.asyncio
    async def test_session_touch_on_execute(self) -> None:
        """Test that session is touched when session_id is provided."""
        svc = TraigentService()
        sid = svc.create_session()
        old_activity = svc._sessions[sid].last_activity

        @svc.execute
        def run(input_id, data, config):
            return {"output": "ok"}

        time.sleep(0.01)
        resp = await svc.handle_execute(
            {"session_id": sid, "inputs": [{"input_id": "i1", "data": {}}]}
        )
        assert resp["session_id"] == sid
        assert svc._sessions[sid].last_activity > old_activity

    @pytest.mark.asyncio
    async def test_unknown_session_id_does_not_error(self) -> None:
        """Test that an unknown session_id does not cause an error."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": "ok"}

        resp = await svc.handle_execute(
            {"session_id": "nonexistent", "inputs": [{"input_id": "i1", "data": {}}]}
        )
        assert resp["status"] == "completed"
        assert resp["session_id"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_quality_metrics_aggregated(self) -> None:
        """Test that quality_metrics are included when handler returns metrics."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {
                "output": "ok",
                "cost_usd": 0.0,
                "metrics": {"accuracy": 0.8},
            }

        resp = await svc.handle_execute(
            {
                "inputs": [
                    {"input_id": "i1", "data": {}},
                    {"input_id": "i2", "data": {}},
                ]
            }
        )
        assert "quality_metrics" in resp
        assert resp["quality_metrics"]["accuracy"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_empty_inputs_list(self) -> None:
        """Test execute with an empty inputs list."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": "ok"}

        with pytest.raises(ValueError, match="inputs must be a non-empty list"):
            await svc.handle_execute({"inputs": []})

    @pytest.mark.asyncio
    async def test_input_without_input_id_gets_uuid(self) -> None:
        """Test that inputs without input_id get a generated UUID."""
        svc = TraigentService()

        @svc.execute
        def run(input_id, data, config):
            return {"output": input_id}

        resp = await svc.handle_execute({"inputs": [{"data": {"q": "test"}}]})
        assert len(resp["outputs"][0]["input_id"]) > 0  # UUID generated

    @pytest.mark.asyncio
    async def test_input_data_defaults_to_input_dict(self) -> None:
        """Test that data defaults to the full input dict when 'data' key is absent."""
        svc = TraigentService()
        received_data = {}

        @svc.execute
        def run(input_id, data, config):
            received_data["data"] = data
            return {"output": "ok"}

        inp = {"input_id": "i1", "query": "hello"}
        await svc.handle_execute({"inputs": [inp]})
        # When 'data' key is missing, the entire input dict is passed
        assert received_data["data"] == inp


# ---------------------------------------------------------------------------
# handle_evaluate tests
# ---------------------------------------------------------------------------
class TestHandleEvaluate:
    """Tests for handle_evaluate."""

    @pytest.mark.asyncio
    async def test_no_handler_raises_value_error(self) -> None:
        """Test that evaluate without handler raises ValueError."""
        svc = TraigentService()
        with pytest.raises(ValueError, match="No evaluate handler registered"):
            await svc.handle_evaluate({})

    @pytest.mark.asyncio
    async def test_sync_handler(self) -> None:
        """Test evaluate with synchronous handler."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0 if output == target else 0.0}

        resp = await svc.handle_evaluate(
            {
                "request_id": "eval-1",
                "config": {},
                "evaluations": [
                    {"input_id": "e1", "output": "a", "target": "a"},
                    {"input_id": "e2", "output": "a", "target": "b"},
                ],
            }
        )
        assert resp["request_id"] == "eval-1"
        assert resp["status"] == "completed"
        assert len(resp["results"]) == 2
        assert resp["results"][0]["metrics"]["accuracy"] == 1.0
        assert resp["results"][1]["metrics"]["accuracy"] == 0.0
        # Aggregate metrics
        agg = resp["aggregate_metrics"]["accuracy"]
        assert agg["mean"] == pytest.approx(0.5)
        assert agg["n"] == 2.0

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        """Test evaluate with async handler."""
        svc = TraigentService()

        @svc.evaluate
        async def score(output, target, config):
            return {"f1": 0.9}

        resp = await svc.handle_evaluate(
            {"evaluations": [{"input_id": "a1", "output": "x", "target": "y"}]}
        )
        assert resp["results"][0]["metrics"]["f1"] == 0.9

    @pytest.mark.asyncio
    async def test_non_dict_result_wrapped(self) -> None:
        """Test that non-dict evaluate result is wrapped in {'score': value}."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return 0.75

        resp = await svc.handle_evaluate(
            {"evaluations": [{"input_id": "a1", "output": "x", "target": "y"}]}
        )
        assert resp["results"][0]["metrics"] == {"score": 0.75}

    @pytest.mark.asyncio
    async def test_handler_exception_captured(self) -> None:
        """Test that evaluate handler exceptions are captured per-evaluation."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            raise ValueError("scoring failed")

        resp = await svc.handle_evaluate(
            {"evaluations": [{"input_id": "e1", "output": "x", "target": "y"}]}
        )
        assert resp["status"] == "failed"
        assert resp["results"][0]["metrics"] == {}
        assert "scoring failed" in resp["results"][0]["error"]

    @pytest.mark.asyncio
    async def test_session_touch_on_evaluate(self) -> None:
        """Test that session is touched when session_id is provided."""
        svc = TraigentService()
        sid = svc.create_session()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        old_activity = svc._sessions[sid].last_activity
        time.sleep(0.01)
        await svc.handle_evaluate(
            {
                "session_id": sid,
                "evaluations": [{"input_id": "e1", "output": "a", "target": "a"}],
            }
        )
        assert svc._sessions[sid].last_activity > old_activity

    @pytest.mark.asyncio
    async def test_empty_evaluations_list(self) -> None:
        """Test evaluate with empty evaluations list."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        resp = await svc.handle_evaluate({"evaluations": []})
        assert resp["results"] == []
        assert resp["aggregate_metrics"] == {}

    @pytest.mark.asyncio
    async def test_evaluate_request_id_is_idempotent(self) -> None:
        """Same request_id and payload should return cached evaluate response."""
        svc = TraigentService()
        call_count = {"n": 0}

        @svc.evaluate
        def score(output, target, config):
            call_count["n"] += 1
            return {"accuracy": 0.9}

        request = {
            "request_id": "eval-idem-1",
            "capability_id": "default",
            "evaluations": [{"input_id": "e1", "output": "a", "target": "a"}],
        }
        first = await svc.handle_evaluate(request)
        second = await svc.handle_evaluate(request)

        assert call_count["n"] == 1
        assert second == first

    @pytest.mark.asyncio
    async def test_evaluate_request_id_payload_mismatch_raises(self) -> None:
        """Reusing request_id with changed evaluate payload should fail fast."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        await svc.handle_evaluate(
            {
                "request_id": "eval-idem-2",
                "capability_id": "default",
                "evaluations": [{"input_id": "e1", "output": "a", "target": "a"}],
            }
        )

        with pytest.raises(ValueError, match="request_id reuse"):
            await svc.handle_evaluate(
                {
                    "request_id": "eval-idem-2",
                    "capability_id": "default",
                    "evaluations": [{"input_id": "e1", "output": "a", "target": "b"}],
                }
            )


# ---------------------------------------------------------------------------
# _aggregate_metrics tests
# ---------------------------------------------------------------------------
class TestAggregateMetrics:
    """Tests for _aggregate_metrics."""

    def test_empty_list(self) -> None:
        """Test aggregation of empty metrics list."""
        svc = TraigentService()
        assert svc._aggregate_metrics([]) == {}

    def test_single_metric(self) -> None:
        """Test aggregation with single metric set."""
        svc = TraigentService()
        result = svc._aggregate_metrics([{"accuracy": 0.9}])
        assert result == {"accuracy": pytest.approx(0.9)}

    def test_multiple_metrics_averaged(self) -> None:
        """Test that multiple metrics are averaged correctly."""
        svc = TraigentService()
        result = svc._aggregate_metrics(
            [{"accuracy": 0.8, "f1": 0.7}, {"accuracy": 1.0, "f1": 0.9}]
        )
        assert result["accuracy"] == pytest.approx(0.9)
        assert result["f1"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _compute_aggregate_metrics tests
# ---------------------------------------------------------------------------
class TestComputeAggregateMetrics:
    """Tests for _compute_aggregate_metrics."""

    def test_empty_list(self) -> None:
        """Test aggregation of empty list returns empty dict."""
        svc = TraigentService()
        assert svc._compute_aggregate_metrics([]) == {}

    def test_single_metric_zero_std(self) -> None:
        """Test that a single metric has std=0."""
        svc = TraigentService()
        result = svc._compute_aggregate_metrics([{"accuracy": 0.9}])
        assert result["accuracy"]["mean"] == pytest.approx(0.9)
        assert result["accuracy"]["std"] == pytest.approx(0.0)
        assert result["accuracy"]["n"] == 1

    def test_multiple_values(self) -> None:
        """Test aggregate stats with multiple values."""
        svc = TraigentService()
        result = svc._compute_aggregate_metrics([{"accuracy": 0.8}, {"accuracy": 1.0}])
        assert result["accuracy"]["mean"] == pytest.approx(0.9)
        assert result["accuracy"]["n"] == 2
        assert result["accuracy"]["std"] > 0.0

    def test_multiple_metric_names(self) -> None:
        """Test aggregate with different metric names across evaluations."""
        svc = TraigentService()
        result = svc._compute_aggregate_metrics(
            [
                {"accuracy": 0.8, "f1": 0.7},
                {"accuracy": 1.0, "f1": 0.9},
            ]
        )
        assert "accuracy" in result
        assert "f1" in result
        assert result["accuracy"]["n"] == 2
        assert result["f1"]["n"] == 2


# ---------------------------------------------------------------------------
# Session management tests
# ---------------------------------------------------------------------------
class TestSessionManagement:
    """Tests for create_session and handle_keep_alive."""

    def test_create_session(self) -> None:
        """Test creating a new session."""
        svc = TraigentService()
        sid = svc.create_session()
        assert sid in svc._sessions
        assert svc._sessions[sid].session_id == sid

    def test_handle_keep_alive_existing_session(self) -> None:
        """Test keep alive with existing session."""
        svc = TraigentService(supports_keep_alive=True)
        sid = svc.create_session()
        old_activity = svc._sessions[sid].last_activity
        time.sleep(0.01)
        result = svc.handle_keep_alive(sid)
        assert result is True
        assert svc._sessions[sid].last_activity > old_activity

    def test_handle_keep_alive_missing_session(self) -> None:
        """Test keep alive with non-existent session."""
        svc = TraigentService()
        result = svc.handle_keep_alive("nonexistent-id")
        assert result is False

    def test_handle_keep_alive_missing_session_autocreates_when_enabled(self) -> None:
        """When keep-alive is enabled, unknown sessions are auto-created."""
        svc = TraigentService(supports_keep_alive=True)
        result = svc.handle_keep_alive("nonexistent-id")
        assert result is True
        assert "nonexistent-id" in svc._sessions


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------
class TestRun:
    """Tests for TraigentService.run method."""

    def test_run_delegates_to_server(self) -> None:
        """Test that run() creates app and calls run_server."""
        svc = TraigentService()
        mock_app = MagicMock()
        mock_create_app = MagicMock(return_value=mock_app)
        mock_run_server = MagicMock()

        mock_server_module = MagicMock(
            create_app=mock_create_app,
            run_server=mock_run_server,
        )
        with patch.dict("sys.modules", {"traigent.wrapper.server": mock_server_module}):
            svc.run(host="127.0.0.1", port=9090, server="uvicorn")

        mock_create_app.assert_called_once_with(svc)
        mock_run_server.assert_called_once_with(
            mock_app, host="127.0.0.1", port=9090, server="uvicorn"
        )


# ---------------------------------------------------------------------------
# __init__.py exports tests
# ---------------------------------------------------------------------------
class TestWrapperExports:
    """Tests for traigent.wrapper package exports."""

    def test_exports_traigent_service(self) -> None:
        """Test that TraigentService is exported."""
        from traigent.wrapper import TraigentService as TS

        assert TS is TraigentService

    def test_exports_service_config(self) -> None:
        """Test that ServiceConfig is exported."""
        from traigent.wrapper import ServiceConfig as SC

        assert SC is ServiceConfig

    def test_exports_session(self) -> None:
        """Test that Session is exported."""
        from traigent.wrapper import Session as S

        assert S is Session

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected names."""
        import traigent.wrapper as wrapper

        assert "TraigentService" in wrapper.__all__
        assert "ServiceConfig" in wrapper.__all__
        assert "Session" in wrapper.__all__


# ---------------------------------------------------------------------------
# Validation error paths in get_config_space
# ---------------------------------------------------------------------------
class TestGetConfigSpaceValidation:
    """Tests for validation errors in get_config_space."""

    def test_tvars_handler_returns_non_dict_raises(self) -> None:
        """Tvars handler returning non-dict should raise ValueError."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return ["not", "a", "dict"]

        with pytest.raises(ValueError, match="tunables handler must return a dict"):
            svc.get_config_space()

    def test_constraints_not_dict_or_list_raises(self) -> None:
        """Constraints that are not dict or list should raise ValueError."""
        svc = TraigentService(constraints="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="constraints must be a dict or list"):
            svc.get_config_space()

    def test_objectives_not_list_raises(self) -> None:
        """Objectives that are not a list should raise ValueError."""
        svc = TraigentService(objectives="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="objectives must be a list"):
            svc.get_config_space()

    def test_exploration_not_dict_raises(self) -> None:
        """Exploration that is not a dict should raise ValueError."""
        svc = TraigentService(exploration="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exploration must be a dict"):
            svc.get_config_space()

    def test_promotion_policy_not_dict_raises(self) -> None:
        """Promotion policy that is not a dict should raise ValueError."""
        svc = TraigentService(promotion_policy="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="promotion_policy must be a dict"):
            svc.get_config_space()

    def test_defaults_not_dict_raises(self) -> None:
        """Defaults that is not a dict should raise ValueError."""
        svc = TraigentService(defaults="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="defaults must be a dict"):
            svc.get_config_space()

    def test_measures_not_list_of_strings_raises(self) -> None:
        """Measures that is not a list of strings should raise ValueError."""
        svc = TraigentService(measures=[1, 2, 3])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="measures must be a list of strings"):
            svc.get_config_space()

    def test_sync_handler_returning_awaitable_raises(self) -> None:
        """Sync handler that returns an awaitable should fail with coroutine cleanup."""
        svc = TraigentService()

        @svc.objectives
        def objectives():
            # Return a coroutine without awaiting — simulates accidental async usage
            async def inner():
                return [{"name": "accuracy", "direction": "maximize"}]
            return inner()

        with pytest.raises(
            ValueError, match="declaration handlers must be synchronous functions"
        ):
            svc.get_config_space()

    def test_normalize_tvars_range_and_resolution(self) -> None:
        """Top-level range/resolution keys should be moved into domain."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {
                "temperature": {
                    "type": "float",
                    "range": [0.0, 1.5],
                    "resolution": 0.1,
                },
            }

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert tvar["domain"]["range"] == [0.0, 1.5]
        assert tvar["domain"]["resolution"] == 0.1

    def test_normalize_tvars_non_dict_domain_replaced(self) -> None:
        """Non-dict domain should be replaced with empty dict."""
        svc = TraigentService()

        @svc.tvars
        def cfg():
            return {"flag": {"type": "bool", "domain": "invalid"}}

        result = svc.get_config_space()
        tvar = result["tvars"][0]
        assert isinstance(tvar["domain"], dict)


# ---------------------------------------------------------------------------
# handle_execute edge cases
# ---------------------------------------------------------------------------
class TestHandleExecuteEdgeCases:
    """Tests for edge cases in handle_execute."""

    @pytest.mark.asyncio
    async def test_non_dict_input_gets_uuid_and_input_as_data(self) -> None:
        """Non-dict input should get a UUID and be passed as data."""
        svc = TraigentService()
        received = {}

        @svc.execute
        def run(input_id, data, config):
            received["input_id"] = input_id
            received["data"] = data
            return {"output": "ok"}

        resp = await svc.handle_execute({"inputs": ["raw_string"]})
        assert resp["status"] == "completed"
        assert len(received["input_id"]) > 0  # UUID generated

    @pytest.mark.asyncio
    async def test_partial_status_on_mixed_success_and_failure(self) -> None:
        """Partial status when some inputs succeed and some fail."""
        svc = TraigentService()
        call_count = 0

        @svc.execute
        def run(input_id, data, config):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail on second")
            return {"output": f"ok-{input_id}"}

        resp = await svc.handle_execute(
            {
                "inputs": [
                    {"input_id": "i1", "data": {}},
                    {"input_id": "i2", "data": {}},
                    {"input_id": "i3", "data": {}},
                ]
            }
        )
        assert resp["status"] == "partial"


# ---------------------------------------------------------------------------
# handle_evaluate edge cases
# ---------------------------------------------------------------------------
class TestHandleEvaluateEdgeCases:
    """Tests for edge cases in handle_evaluate."""

    @pytest.mark.asyncio
    async def test_evaluations_not_list_raises(self) -> None:
        """Evaluations must be a list."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        with pytest.raises(ValueError, match="evaluations must be a list"):
            await svc.handle_evaluate({"evaluations": "not_a_list"})

    @pytest.mark.asyncio
    async def test_evaluation_with_output_id(self) -> None:
        """Evaluation with output_id instead of output."""
        svc = TraigentService()
        received_output = {}

        @svc.evaluate
        def score(output, target, config):
            received_output["output"] = output
            return {"accuracy": 1.0}

        await svc.handle_evaluate(
            {
                "evaluations": [
                    {
                        "input_id": "e1",
                        "output_id": "out-123",
                        "target": "expected",
                    }
                ]
            }
        )
        assert received_output["output"] == {"output_id": "out-123"}

    @pytest.mark.asyncio
    async def test_bool_metrics_excluded_from_aggregation(self) -> None:
        """Bool-type metric values should be excluded from aggregation."""
        svc = TraigentService()

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 0.9, "is_valid": True}

        resp = await svc.handle_evaluate(
            {
                "evaluations": [
                    {"input_id": "e1", "output": "a", "target": "a"},
                ]
            }
        )
        agg = resp["aggregate_metrics"]
        assert "accuracy" in agg
        assert "is_valid" not in agg  # Bool excluded
