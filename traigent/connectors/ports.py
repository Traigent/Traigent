"""Platform-neutral connector ports; implementations stay customer-side."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from .models import DatasetItem, Observation, Score

T = TypeVar("T")


class Page(Protocol[T]):
    items: list[T]
    next_cursor: str | None


class ObservationQuery(Protocol):
    """Closed query objects are supplied by use-case code."""


class TraceSource(Protocol):
    def query(
        self, query: ObservationQuery, cursor: str | None = None
    ) -> Page[Observation]: ...
    def get(self, ref: Any) -> Observation: ...


class ScoreSource(Protocol):
    def query(self, query: Any, cursor: str | None = None) -> Page[Score]: ...


class ScoreSink(Protocol):
    def write(self, batch: list[Score], idem_key: str) -> Any: ...


class DatasetSource(Protocol):
    def list(self, query: Any = None, cursor: str | None = None) -> Page[Any]: ...
    def read(
        self, dataset: Any, revision: str, cursor: str | None = None
    ) -> Page[DatasetItem]: ...


class DatasetSink(Protocol):
    def upsert(self, dataset: Any, items: list[DatasetItem], idem_key: str) -> Any: ...


class ExperimentSink(Protocol):
    def open(self, run: Any, idem_key: str) -> Any: ...
    def append(self, handle: Any, item_results: list[Any]) -> Any: ...
    def finalize(self, handle: Any, status: str) -> Any: ...


class ArtifactStore(Protocol):
    def put(self, artifact: Any, idem_key: str) -> Any: ...
    def get(self, version_ref: Any) -> Any: ...


class DeploymentTarget(Protocol):
    def read_active(self) -> tuple[Any, str]: ...
    def set_active(self, version_ref: Any, expect_revision: str) -> Any: ...


class EventSource(Protocol):
    def poll(self, cursor: str | None = None) -> Page[Any]: ...
