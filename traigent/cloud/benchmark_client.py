"""Sync-friendly benchmark generation client.

Provides a one-step flow for generating evaluation benchmarks:
describe a task, optionally provide seed examples, get a Dataset back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request

from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env, scope_api_path
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)


@dataclass
class BenchmarkClientConfig:
    """Configuration for the benchmark generation client."""

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    api_path: str = "/api/v1"
    request_timeout: float = 120.0  # generation can be slow

    def __post_init__(self) -> None:
        self.backend_origin = self.backend_origin.rstrip("/")
        self.tenant_id = (
            self.tenant_id.strip() or None if self.tenant_id is not None else None
        )
        self.project_id = (
            self.project_id.strip() or None if self.project_id is not None else None
        )
        self.api_path = scope_api_path(self.api_path, self.project_id)

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-benchmark/0.1",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers


class BenchmarkClient:
    """Generate evaluation benchmarks in one step.

    Usage::

        from traigent import BenchmarkClient

        client = BenchmarkClient()

        # Description only
        dataset = client.generate_sync(
            description="Customer support Q&A for e-commerce",
            count=20,
            use_case="question-answering",
        )

        # With seed examples
        dataset = client.generate_sync(
            description="Customer support Q&A for e-commerce",
            count=20,
            use_case="question-answering",
            seed_examples=[
                {"input": "How do I return an item?", "output": "Go to Orders..."},
            ],
        )
    """

    def __init__(
        self,
        config: BenchmarkClientConfig | None = None,
    ) -> None:
        self.config = config or BenchmarkClientConfig()

    def generate_sync(
        self,
        description: str,
        count: int,
        use_case: str,
        *,
        name: str | None = None,
        seed_examples: list[dict[str, str]] | None = None,
    ) -> Dataset:
        """Generate a benchmark with examples and return as a Dataset.

        Args:
            description: Task description for example generation.
            count: Number of examples to generate (1-100).
            use_case: One of: question-answering, summarization, classification,
                code-generation, chat, extraction, generation, translation, other.
            name: Optional benchmark name (auto-derived from description if omitted).
            seed_examples: Optional list of 1-10 seed examples, each a dict with
                ``input`` and ``output`` keys, used as few-shot context.

        Returns:
            A :class:`Dataset` containing the generated examples, ready for use
            with ``@traigent.optimize(eval_dataset=dataset)``.
        """
        payload: dict[str, Any] = {
            "description": description,
            "count": count,
            "use_case": use_case,
        }
        if name:
            payload["name"] = name
        if seed_examples:
            payload["seed_examples"] = seed_examples

        response = self._post("/datasets/generate", payload)
        return self._response_to_dataset(response, description)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        full_url = f"{self.config.backend_origin}{self.config.api_path}{path}"
        encoded = json.dumps(payload).encode("utf-8")
        http_req = request.Request(
            full_url,
            data=encoded,
            headers=self.config.build_headers(),
            method="POST",
        )
        try:
            with request.urlopen(  # nosec B310
                http_req, timeout=self.config.request_timeout
            ) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                return parsed
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code == 401:
                raise AuthenticationError(
                    f"Authentication failed ({e.code}): {body}"
                ) from e
            raise ClientError(
                f"Benchmark generation failed ({e.code}): {body}"
            ) from e
        except error.URLError as e:
            raise TraigentConnectionError(
                f"Cannot reach backend at {self.config.backend_origin}: {e.reason}"
            ) from e

    @staticmethod
    def _response_to_dataset(
        response: dict[str, Any], description: str
    ) -> Dataset:
        data = response.get("data", response)
        examples_raw = data.get("examples", [])
        benchmark = data.get("benchmark", {})

        examples = []
        for ex in examples_raw:
            input_text = ex.get("input_text", "")
            expected_output = ex.get("expected_output")
            examples.append(
                EvaluationExample(
                    input_data={"input": input_text},
                    expected_output=expected_output,
                )
            )

        return Dataset(
            examples=examples,
            name=benchmark.get("name", "generated_benchmark"),
            description=description,
            metadata={
                "benchmark_id": benchmark.get("id"),
                "generation_status": data.get("status"),
            },
        )
