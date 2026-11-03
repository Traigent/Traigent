"""Characterization tests pinning BackendSessionManager.attach_session_metadata.

Collaborators and the manager's own attributes are traced into one shared event
list, so the order and number of reads and calls are pinned together with the
metadata that ends up on the result.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping
from typing import Any

import pytest

from traigent.core.backend_session_manager import BackendSessionManager

TRACED = {"_backend_client", "_session_owning_context", "_egress_disabled"}


class _Manager(BackendSessionManager):
    """Manager recording reads of the attributes attach_session_metadata uses."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name in TRACED:
            trace = self.__dict__.get("_trace")
            if trace is not None:
                trace.append(("read", name))
        return value


class _Metadata(dict[str, Any]):
    """Result metadata recording update calls into the shared trace."""

    def __init__(self, trace: list[Any], data: dict[str, Any] | None = None) -> None:
        super().__init__(data or {})
        self.trace = trace

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.trace.append(("metadata_update", dict(args[0])))
        super().update(*args, **kwargs)


class _Result:
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata


def _manager(
    trace: list[Any],
    *,
    client: Any = None,
    egress_disabled: bool = False,
    owning: dict[str, Any] | None = None,
) -> BackendSessionManager:
    manager = _Manager.__new__(_Manager)
    manager.__dict__["_trace"] = trace
    manager.__dict__["_backend_client"] = client

    def egress() -> bool:
        trace.append(("egress_check",))
        return egress_disabled

    manager.__dict__["_egress_disabled"] = egress
    manager.__dict__["_session_owning_context"] = {} if owning is None else owning

    def warn(metadata: dict[str, Any]) -> None:
        trace.append(("warn", metadata))

    manager.__dict__["_warn_on_refused_warm_start"] = warn
    return manager


class _Client:
    def __init__(
        self, trace: list[Any], mapping: Any = None, error: Any = None
    ) -> None:
        self.trace = trace
        self.mapping = mapping
        self.error = error

    def get_session_mapping(self, session_id: str) -> Any:
        self.trace.append(("get_session_mapping", session_id))
        if self.error is not None:
            raise self.error
        return self.mapping


class _Mapping:
    def __init__(self, trace: list[Any], experiment_id: Any, run_id: Any) -> None:
        self.trace = trace
        self._experiment_id = experiment_id
        self._run_id = run_id

    @property
    def experiment_id(self) -> Any:
        self.trace.append(("experiment_id",))
        if isinstance(self._experiment_id, BaseException):
            raise self._experiment_id
        return self._experiment_id

    @property
    def experiment_run_id(self) -> Any:
        self.trace.append(("experiment_run_id",))
        if isinstance(self._run_id, BaseException):
            raise self._run_id
        return self._run_id


def _attach(
    manager: BackendSessionManager,
    result: Any,
    session_id: Any = "s-1",
    summary: Any = None,
) -> None:
    manager.attach_session_metadata(result, session_id, summary)


# ------------------------------------------------------------------- gating


def test_missing_session_id_returns_before_any_other_read() -> None:
    trace: list[Any] = []
    result = _Result({"keep": 1})
    _attach(_manager(trace), result, session_id=None, summary={"a": 1})
    assert result.metadata == {"keep": 1}
    assert trace == []


def test_empty_string_session_id_is_not_missing() -> None:
    trace: list[Any] = []
    result = _Result({})
    _attach(_manager(trace), result, session_id="")
    assert result.metadata == {"local_session_id": ""}


def test_disabled_egress_returns_without_touching_the_result() -> None:
    trace: list[Any] = []
    metadata: dict[str, Any] = {}
    result = _Result(metadata)
    _attach(
        _manager(trace, egress_disabled=True, client=_Client(trace)),
        result,
        summary={"a": 1},
    )
    assert result.metadata is metadata
    assert metadata == {}
    assert trace == [("read", "_egress_disabled"), ("egress_check",)]


def test_result_without_metadata_attribute_returns_after_egress_check() -> None:
    trace: list[Any] = []
    result = object()
    _attach(_manager(trace), result, summary={"a": 1})
    assert not hasattr(result, "metadata")
    assert trace == [("read", "_egress_disabled"), ("egress_check",)]


def test_metadata_accessor_error_other_than_attribute_error_propagates() -> None:
    class Result:
        @property
        def metadata(self) -> Any:
            raise RuntimeError("metadata failed")

    with pytest.raises(RuntimeError, match="metadata failed"):
        _attach(_manager([]), Result())


def test_metadata_attribute_error_counts_as_missing() -> None:
    class Result:
        @property
        def metadata(self) -> Any:
            raise AttributeError("nope")

    trace: list[Any] = []
    _attach(_manager(trace), Result())
    assert trace == [("read", "_egress_disabled"), ("egress_check",)]


def test_egress_check_error_propagates() -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    def boom() -> bool:
        raise RuntimeError("egress failed")

    manager.__dict__["_egress_disabled"] = boom
    result = _Result({})
    with pytest.raises(RuntimeError, match="egress failed"):
        _attach(manager, result)
    assert result.metadata == {}


# ------------------------------------------------------------ metadata object


@pytest.mark.parametrize("empty", [{}, None, [], 0, ""])
def test_falsy_metadata_is_replaced_by_a_fresh_dict(empty: Any) -> None:
    result = _Result(empty)
    _attach(_manager([]), result, session_id="s-9")
    assert result.metadata == {"local_session_id": "s-9"}
    assert result.metadata is not empty
    assert isinstance(result.metadata, dict)


def test_empty_dict_metadata_object_is_not_mutated_when_replaced() -> None:
    original: dict[str, Any] = {}
    result = _Result(original)
    _attach(_manager([]), result)
    assert original == {}
    assert result.metadata == {"local_session_id": "s-1"}


def test_truthy_metadata_object_is_updated_in_place() -> None:
    original = {"keep": 1, "local_session_id": "old"}
    result = _Result(original)
    _attach(_manager([]), result, session_id="s-2")
    assert result.metadata is original
    assert original == {"keep": 1, "local_session_id": "s-2"}


# ------------------------------------------------------------ summary handling


def test_none_summary_adds_no_summary_key() -> None:
    result = _Result({})
    _attach(_manager([]), result, summary=None)
    assert result.metadata == {"local_session_id": "s-1"}


@pytest.mark.parametrize("summary", [{}, 0, "", [], False])
def test_non_none_falsy_summary_is_recorded_by_identity(summary: Any) -> None:
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert list(result.metadata) == ["local_session_id", "local_session_summary"]
    assert result.metadata["local_session_summary"] is summary


def test_dict_summary_copies_the_warm_start_transfer() -> None:
    transfer = {"transfer_mode": "accepted", "n_seed_configs_applied": 2}
    summary = {"metadata": {"warm_start_transfer": transfer, "other": 1}}
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert result.metadata == {
        "local_session_id": "s-1",
        "local_session_summary": summary,
        "warm_start_transfer": transfer,
    }
    assert result.metadata["warm_start_transfer"] is not transfer
    assert result.metadata["local_session_summary"] is summary
    assert list(result.metadata) == [
        "local_session_id",
        "local_session_summary",
        "warm_start_transfer",
    ]


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"metadata": None},
        {"metadata": []},
        {"metadata": "text"},
        {"metadata": {}},
        {"metadata": {"warm_start_transfer": None}},
        {"metadata": {"warm_start_transfer": []}},
        {"metadata": {"warm_start_transfer": "x"}},
        {"metadata": {"warm_start_transfer": 5}},
    ],
)
def test_summaries_without_a_dict_transfer_add_nothing(summary: dict[str, Any]) -> None:
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert "warm_start_transfer" not in result.metadata
    assert result.metadata["local_session_summary"] is summary


def test_empty_dict_transfer_is_still_copied() -> None:
    summary: dict[str, Any] = {"metadata": {"warm_start_transfer": {}}}
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert result.metadata["warm_start_transfer"] == {}


def test_object_summary_metadata_is_read_through_the_attribute() -> None:
    class Summary:
        metadata = {"warm_start_transfer": {"mode": "x"}}

        def get(self, key: str) -> Any:
            raise AssertionError("dataclass-like summaries have no get()")

    summary = Summary()
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert result.metadata["warm_start_transfer"] == {"mode": "x"}
    assert result.metadata["local_session_summary"] is summary


def test_object_summary_without_metadata_adds_nothing() -> None:
    result = _Result({})
    _attach(_manager([]), result, summary=object())
    assert "warm_start_transfer" not in result.metadata


def test_non_dict_mapping_summary_uses_the_attribute_path() -> None:
    class Summary(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise AssertionError("must not be indexed")

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

    result = _Result({})
    _attach(_manager([]), result, summary=Summary())
    assert "warm_start_transfer" not in result.metadata


def test_dict_subclass_summary_get_is_used_once_and_copy_is_taken_once() -> None:
    trace: list[Any] = []

    class Inner(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            trace.append(("inner_get", key))
            return super().get(key, default)

    class Summary(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            trace.append(("summary_get", key))
            return super().get(key, default)

    summary = Summary(metadata=Inner(warm_start_transfer={"a": 1}))
    result = _Result({})
    _attach(_manager([]), result, summary=summary)
    assert trace == [("summary_get", "metadata"), ("inner_get", "warm_start_transfer")]
    assert result.metadata["warm_start_transfer"] == {"a": 1}


def test_summary_get_error_propagates_before_metadata_is_modified() -> None:
    class Summary(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get failed")

    metadata: dict[str, Any] = {"keep": 1}
    result = _Result(metadata)
    with pytest.raises(RuntimeError, match="get failed"):
        _attach(_manager([]), result, summary=Summary())
    assert metadata == {"keep": 1}


def test_summary_attribute_error_other_than_attribute_error_propagates() -> None:
    class Summary:
        @property
        def metadata(self) -> Any:
            raise RuntimeError("summary failed")

    with pytest.raises(RuntimeError, match="summary failed"):
        _attach(_manager([]), _Result({}), summary=Summary())


def test_summary_attribute_error_counts_as_absent() -> None:
    class Summary:
        @property
        def metadata(self) -> Any:
            raise AttributeError("gone")

    result = _Result({})
    _attach(_manager([]), result, summary=Summary())
    assert "warm_start_transfer" not in result.metadata


# -------------------------------------------------------------- owning context


def test_owning_context_is_merged_after_the_summary_fields() -> None:
    trace: list[Any] = []
    owning = {"s-1": {"project": "p", "local_session_id": "override"}}
    result = _Result({})
    _attach(_manager(trace, owning=owning), result, summary={"a": 1})
    assert result.metadata == {
        "local_session_id": "override",
        "local_session_summary": {"a": 1},
        "project": "p",
    }


@pytest.mark.parametrize("context", [{}, None])
def test_falsy_or_absent_owning_context_adds_nothing(context: Any) -> None:
    owning = {} if context is None else {"s-1": context}
    result = _Result({})
    _attach(_manager([], owning=owning), result)
    assert result.metadata == {"local_session_id": "s-1"}


def test_owning_context_for_other_sessions_is_ignored() -> None:
    result = _Result({})
    _attach(_manager([], owning={"other": {"project": "p"}}), result)
    assert result.metadata == {"local_session_id": "s-1"}


def test_owning_context_truth_is_tested_once_then_keys_then_items_are_read() -> None:
    trace: list[Any] = []

    class Context(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            trace.append(("getitem", key))
            return "v-" + key

        def __iter__(self) -> Iterator[str]:
            trace.append(("iter",))
            return iter(("a", "b"))

        def __len__(self) -> int:
            trace.append(("len",))
            return 2

        def keys(self) -> KeysView[str]:
            trace.append(("keys",))
            return {"a": "", "b": ""}.keys()

        def items(self) -> ItemsView[str, str]:
            raise AssertionError("update must use keys() and indexing")

    result = _Result({})
    _attach(_manager([], owning={"s-1": Context()}), result)
    assert result.metadata == {"local_session_id": "s-1", "a": "v-a", "b": "v-b"}
    assert trace == [("len",), ("keys",), ("getitem", "a"), ("getitem", "b")]


def test_owning_context_lookup_error_propagates() -> None:
    class Owning(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("lookup failed")

    metadata: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="lookup failed"):
        _attach(_manager([], owning=Owning()), _Result(metadata))
    assert metadata == {}


# ------------------------------------------------------------- experiment ids


def test_no_client_leaves_experiment_fields_out() -> None:
    trace: list[Any] = []
    result = _Result({})
    _attach(_manager(trace, client=None), result)
    assert [event for event in trace if event[0] == "read"] == [
        ("read", "_egress_disabled"),
        ("read", "_session_owning_context"),
        ("read", "_backend_client"),
    ]
    assert result.metadata == {"local_session_id": "s-1"}


def test_client_mapping_adds_experiment_fields_and_reads_the_client_twice() -> None:
    trace: list[Any] = []
    mapping = _Mapping(trace, "exp-1", "run-1")
    metadata = _Metadata(trace, {"keep": 1})
    result = _Result(metadata)
    _attach(_manager(trace, client=_Client(trace, mapping)), result, session_id="s-7")
    expected = {
        "local_session_id": "s-7",
        "experiment_id": "exp-1",
        "experiment_run_id": "run-1",
    }
    assert dict(result.metadata) == {"keep": 1, **expected}
    assert trace == [
        ("read", "_egress_disabled"),
        ("egress_check",),
        ("read", "_session_owning_context"),
        ("read", "_backend_client"),
        ("read", "_backend_client"),
        ("get_session_mapping", "s-7"),
        ("experiment_id",),
        ("experiment_run_id",),
        ("metadata_update", expected),
        ("warn", {"keep": 1, **expected}),
    ]


def test_experiment_fields_override_owning_context_keys() -> None:
    trace: list[Any] = []
    mapping = _Mapping(trace, "exp-1", None)
    owning = {"s-1": {"experiment_id": "ctx", "extra": "e"}}
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, mapping), owning=owning), result)
    assert result.metadata == {
        "local_session_id": "s-1",
        "extra": "e",
        "experiment_id": "exp-1",
        "experiment_run_id": None,
    }


def test_none_mapping_adds_nothing() -> None:
    trace: list[Any] = []
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, None)), result)
    assert result.metadata == {"local_session_id": "s-1"}
    assert ("get_session_mapping", "s-1") in trace


@pytest.mark.parametrize("error", [RuntimeError("x"), KeyError("k"), ValueError("v")])
def test_client_error_is_swallowed(error: Exception) -> None:
    trace: list[Any] = []
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, error=error)), result)
    assert result.metadata == {"local_session_id": "s-1"}
    assert trace[-1][0] == "warn"


def test_base_exception_from_the_client_is_not_swallowed() -> None:
    trace: list[Any] = []
    result = _Result({})
    with pytest.raises(KeyboardInterrupt):
        _attach(
            _manager(trace, client=_Client(trace, error=KeyboardInterrupt())), result
        )
    assert result.metadata == {}
    assert not any(event[0] == "warn" for event in trace)


def test_first_accessor_error_is_swallowed_and_nothing_is_written() -> None:
    trace: list[Any] = []
    mapping = _Mapping(trace, RuntimeError("boom"), "run-1")
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, mapping)), result)
    assert result.metadata == {"local_session_id": "s-1"}
    assert ("experiment_run_id",) not in trace


def test_second_accessor_error_keeps_the_first_field() -> None:
    trace: list[Any] = []
    mapping = _Mapping(trace, "exp-1", RuntimeError("boom"))
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, mapping)), result)
    assert result.metadata == {"local_session_id": "s-1", "experiment_id": "exp-1"}
    assert trace[-1] == (
        "warn",
        {"local_session_id": "s-1", "experiment_id": "exp-1"},
    )


def test_mapping_truthiness_is_not_tested() -> None:
    class Mapping_:
        experiment_id = "e"
        experiment_run_id = "r"

        def __bool__(self) -> bool:
            raise AssertionError("must compare with None only")

    trace: list[Any] = []
    result = _Result({})
    _attach(_manager(trace, client=_Client(trace, Mapping_())), result)
    assert result.metadata["experiment_id"] == "e"


# ----------------------------------------------------------------- final steps


def test_metadata_is_updated_once_then_warning_check_runs_on_the_same_object() -> None:
    trace: list[Any] = []
    metadata = _Metadata(trace, {"warm_start_from": "exp-0"})
    result = _Result(metadata)
    _attach(_manager(trace), result, summary={"a": 1})
    assert trace[-2] == (
        "metadata_update",
        {"local_session_id": "s-1", "local_session_summary": {"a": 1}},
    )
    assert trace[-1][0] == "warn" and trace[-1][1] is metadata
    assert [event[0] for event in trace].count("metadata_update") == 1
    assert [event[0] for event in trace].count("warn") == 1


def test_update_error_prevents_the_warning_check() -> None:
    trace: list[Any] = []

    class Metadata(_Metadata):
        def update(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("update failed")

    result = _Result(Metadata(trace, {"keep": 1}))
    with pytest.raises(RuntimeError, match="update failed"):
        _attach(_manager(trace), result)
    assert not any(event[0] == "warn" for event in trace)


def test_warning_check_error_propagates_after_the_update() -> None:
    trace: list[Any] = []
    manager = _manager(trace)

    def boom(metadata: dict[str, Any]) -> None:
        raise RuntimeError("warn failed")

    manager.__dict__["_warn_on_refused_warm_start"] = boom
    result = _Result({"keep": 1})
    with pytest.raises(RuntimeError, match="warn failed"):
        _attach(manager, result)
    assert result.metadata == {"keep": 1, "local_session_id": "s-1"}


def test_real_warning_check_logs_a_refused_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.core import backend_session_manager as module

    records: list[tuple[Any, ...]] = []

    class Logger:
        def warning(self, *args: Any) -> None:
            records.append(args)

    monkeypatch.setattr(module, "logger", Logger())
    manager = _manager([])
    del manager.__dict__["_warn_on_refused_warm_start"]
    result = _Result({"warm_start_from": "exp-0"})
    summary = {"metadata": {"warm_start_transfer": {"refused_reason": "mismatch"}}}
    _attach(manager, result, summary=summary)
    assert len(records) == 1
    assert records[0][1:] == ("exp-0", "mismatch", None)


@pytest.mark.parametrize("kind", ["falsy", "raising"])
def test_client_truthiness_is_never_evaluated(kind: str) -> None:
    trace: list[Any] = []

    class Client(_Client):
        def __bool__(self) -> bool:
            trace.append(("client_bool",))
            if kind == "raising":
                raise RuntimeError("client truth failed")
            return False

    mapping = _Mapping(trace, "exp-1", "run-1")
    result = _Result({})
    _attach(_manager(trace, client=Client(trace, mapping)), result)
    assert result.metadata == {
        "local_session_id": "s-1",
        "experiment_id": "exp-1",
        "experiment_run_id": "run-1",
    }
    assert ("client_bool",) not in trace
