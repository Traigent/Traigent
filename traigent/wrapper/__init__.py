"""Traigent API Wrapper - Framework for external agentic services.

This package provides a minimal framework for exposing agentic services
via the Traigent hybrid API protocol. External services can use decorators
to define their configuration space, execution logic, and evaluation logic.

Example:
    from traigent.wrapper import TraigentService

    app = TraigentService(tunable_id="my_agent")

    @app.tvars
    def config_space():
        return {
            "model": {"type": "enum", "values": ["gpt-4o", "claude-3"]},
            "temperature": {"type": "float", "range": [0.0, 2.0]},
        }

    @app.execute
    async def run_agent(example_id: str, data: dict, config: dict) -> dict:
        # Your agent logic here
        return {"output": result, "cost_usd": 0.002, "latency_ms": 150}

    @app.evaluate
    async def score(output: dict, target: dict, config: dict) -> dict:
        return {"accuracy": 0.95, "safety": 1.0}

    if __name__ == "__main__":
        app.run(port=8080)

The service automatically exposes endpoints at:
    - GET  /traigent/v1/capabilities - Service capabilities
    - GET  /traigent/v1/config-space - TVAR definitions
    - POST /traigent/v1/execute - Execute agent
    - POST /traigent/v1/evaluate - Evaluate outputs
    - GET  /traigent/v1/health - Health check
    - POST /traigent/v1/keep-alive - Session keep-alive
"""

# Traceability: HYBRID-MODE-OPTIMIZATION CLIENT-WRAPPER-SDK

from traigent.wrapper.errors import (
    BadRequestError,
    HybridAPIError,
    RateLimitError,
    RequestTimeoutError,
    ServiceUnavailableError,
    UnauthorizedError,
)
from traigent.wrapper.service import ServiceConfig, Session, TraigentService

__all__ = [
    "TraigentService",
    "ServiceConfig",
    "Session",
    "HybridAPIError",
    "BadRequestError",
    "UnauthorizedError",
    "RequestTimeoutError",
    "RateLimitError",
    "ServiceUnavailableError",
]
