"""Every public Traigent exception must survive pickle/copy (#2969).

Worker processes (ProcessPoolExecutor, joblib, Celery, dask) and ``copy`` rebuild
exceptions from their constructor call. Exceptions that format a message and
call ``super().__init__(message)`` while requiring other arguments used to raise
``TypeError`` on unpickle or re-wrap the message.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

from traigent.utils.exceptions import CostLimitExceeded

# Modules that define exception classes (discovered by importing the package tree
# and walking BaseException subclasses; keep in sync when a module adds one).
EXCEPTION_MODULES = [
    "traigent.analytics.render",
    "traigent.analytics_mcp.tools",
    "traigent.api.constraints",
    "traigent.api.safety",
    "traigent.certification.bind",
    "traigent.certification.client_evidence",
    "traigent.certification.client_keys",
    "traigent.certification.pins",
    "traigent.certification.transport",
    "traigent.cloud.analytics_client",
    "traigent.cloud.auth",
    "traigent.cloud.backend_client",
    "traigent.cloud.client",
    "traigent.cloud.content_less",
    "traigent.cloud.process_record_linkage",
    "traigent.cloud.production_mcp_client",
    "traigent.cloud.token_manager",
    "traigent.cloud.url_security",
    "traigent.config.backend_config",
    "traigent.config.providers",
    "traigent.config_generator.llm_backend",
    "traigent.config_generator.tvl_constraints",
    "traigent.connectors.http",
    "traigent.connectors.privacy",
    "traigent.core.backend_session_manager",
    "traigent.core.best_config_diagnostics",
    "traigent.core.best_config_runtime",
    "traigent.core.config_state_manager",
    "traigent.core.cost_enforcement",
    "traigent.core.execution_budget",
    "traigent.core.execution_policy_runtime",
    "traigent.core.license",
    "traigent.core.resume",
    "traigent.core.session_types",
    "traigent.economics.errors",
    "traigent.generation.backend_provider",
    "traigent.generation.llm_provider",
    "traigent.hybrid.transport",
    "traigent.identity.agent_build",
    "traigent.identity.content_identity",
    "traigent.identity.workload",
    "traigent.integrations.langfuse.client",
    "traigent.integrations.llms.nous_auth",
    "traigent.knobs.canonical",
    "traigent.knobs.resolution",
    "traigent.knobs.runtime",
    "traigent.labels._codec",
    "traigent.labels.content_classes",
    "traigent.labels.token_store",
    "traigent.labels.transport",
    "traigent.labels.transport_gate",
    "traigent.labels.trust_anchor",
    "traigent.labels.types",
    "traigent.mcp.tools",
    "traigent.metrics.ragas_metrics",
    "traigent.observability.client",
    "traigent.optimizers.interactive_optimizer",
    "traigent.runtime.apply",
    "traigent.runtime.deploy",
    "traigent.runtime.guards",
    "traigent.runtime.hosted",
    "traigent.runtime.model",
    "traigent.runtime.respond",
    "traigent.security.audit",
    "traigent.security.crypto_utils",
    "traigent.security.jwt_validator",
    "traigent.snapshot.errors",
    "traigent.snapshot.locator",
    "traigent.utils.cost_calculator",
    "traigent.utils.error_handler",
    "traigent.utils.exceptions",
    "traigent.utils.fp2",
    "traigent.utils.langchain_interceptor",
    "traigent.utils.litellm_interceptor",
    "traigent.utils.mock_transport_guard",
    "traigent.utils.module_loading",
    "traigent.utils.optimization_only_egress",
    "traigent.utils.outcome_signals",
    "traigent.utils.regression_dataset",
    "traigent.utils.retry",
    "traigent.utils.secure_path",
    "traigent.utils.url_security",
    "traigent.wrapper.errors",
]


def _discover() -> list[type[BaseException]]:
    loaded = set()
    for name in EXCEPTION_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:  # optional dependency missing in this environment
            continue
        loaded.add(name)

    found: dict[tuple[str, str], type[BaseException]] = {}
    stack = list(BaseException.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__ in loaded and "<locals>" not in cls.__qualname__:
            found[(cls.__module__, cls.__qualname__)] = cls
    return [found[key] for key in sorted(found)]


def _sample_overrides() -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    from traigent.core.best_config_runtime import CloudPublishUnavailableReason
    from traigent.core.session_types import SessionCreationFailureDetail
    from traigent.knobs.resolution import ResolutionRejection
    from traigent.snapshot.errors import SnapshotErrorCode

    return {
        # A code that is not the constructor's fallback: a generic sample would
        # normalise to the fallback and hide a lossy round trip.
        "CommitmentLinkageClientError": (("COMMITMENT_LINKAGE_DISABLED",), {}),
        "CostLimitExceeded": ((3.0, 2.0), {"estimated": 1.5}),
        "CloudPublishUnavailable": (
            (CloudPublishUnavailableReason.DISABLED_BY_CONFIG, "msg"),
            {},
        ),
        "SessionCreationHTTPError": (
            (SessionCreationFailureDetail(status_code=500),),
            {},
        ),
        "ResolutionError": ((tuple(ResolutionRejection)[:1], "detail"), {}),
        "TransportGateRefusal": (("route", {"a": 1}), {"cause_type": "ValueError"}),
        "SnapshotStoreError": ((SnapshotErrorCode.NOT_FOUND, "part", "backend"), {}),
    }


def _build(cls: type[BaseException]) -> BaseException:
    overrides = _sample_overrides()
    if cls.__name__ in overrides:
        args, kwargs = overrides[cls.__name__]
        return cls(*args, **kwargs)
    codes = getattr(cls, "CODES", None)
    if codes:  # closed-code exceptions validate their code token
        return cls(sorted(codes)[0])
    try:
        params = inspect.signature(cls).parameters.values()
    except ValueError:  # builtin __init__: takes free-form args
        return cls("sample")
    args = []
    kwargs = {}
    for param in params:
        if param.default is not param.empty:
            continue
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args.append("sample")
        elif param.kind is param.KEYWORD_ONLY:
            kwargs[param.name] = "sample"
    return cls(*args, **kwargs)


EXCEPTION_CLASSES = _discover()


def _same(original: BaseException, rebuilt: BaseException) -> None:
    assert type(rebuilt) is type(original)
    assert str(rebuilt) == str(original)
    assert rebuilt.args == original.args
    assert rebuilt.__dict__.keys() == original.__dict__.keys()
    for key, value in original.__dict__.items():
        if key.startswith("_ctor_"):
            continue
        assert repr(rebuilt.__dict__[key]) == repr(value), key


def test_discovery_finds_the_known_problem_classes() -> None:
    names = {cls.__name__ for cls in EXCEPTION_CLASSES}
    assert {
        "CostLimitExceeded",
        "MetricExtractionError",
        "DTOSerializationError",
        "PluginVersionError",
        "HybridAPIError",
        "SessionBackendRefusedError",
        "SessionQuotaRefusedError",
        "LocalGuardError",
        "FeatureNotAvailableError",
        "DeployHold",
    } <= names


@pytest.mark.parametrize(
    "cls", EXCEPTION_CLASSES, ids=lambda c: f"{c.__module__}.{c.__qualname__}"
)
def test_public_exception_round_trips_through_pickle_and_copy(
    cls: type[BaseException],
) -> None:
    original = _build(cls)
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        _same(original, pickle.loads(pickle.dumps(original, protocol)))
    _same(original, copy.copy(original))
    _same(original, copy.deepcopy(original))


def test_attributes_set_after_construction_survive() -> None:
    err = CostLimitExceeded(3.0, 2.0, estimated=1.0)
    err.extra_marker = "kept"  # type: ignore[attr-defined]
    rebuilt = pickle.loads(pickle.dumps(err))
    assert rebuilt.extra_marker == "kept"
    assert (rebuilt.accumulated, rebuilt.limit) == (3.0, 2.0)


def _raise_cost_limit() -> None:
    raise CostLimitExceeded(3.0, 2.0)


def test_cost_limit_exceeded_crosses_a_process_pool() -> None:
    with ProcessPoolExecutor(1) as pool:
        with pytest.raises(CostLimitExceeded) as info:
            pool.submit(_raise_cost_limit).result()
    assert info.value.accumulated == 3.0
    assert info.value.limit == 2.0


def test_replay_is_not_skewed_by_later_mutation_of_constructor_arguments() -> None:
    from traigent.utils.regression_dataset import UnresolvedExampleIdError

    unresolved = ["a"]
    err = UnresolvedExampleIdError(unresolved, [])
    unresolved.append("b")
    for rebuilt in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert str(rebuilt) == str(err)
        assert list(rebuilt.unresolved) == ["a"]


def test_user_list_argument_mutation_does_not_skew_replay() -> None:
    from collections import UserList

    from traigent.utils.regression_dataset import UnresolvedExampleIdError

    ids = UserList(["a"])
    err = UnresolvedExampleIdError(ids, [])
    ids.append("b")
    assert str(pickle.loads(pickle.dumps(err))) == str(err)


def test_closed_code_error_does_not_retain_or_serialize_unknown_input() -> None:
    from traigent.cloud.process_record_linkage import CommitmentLinkageClientError

    err = CommitmentLinkageClientError("sensitive sentinel")
    assert err.code == "BACKEND_REFUSED"
    assert b"sensitive sentinel" not in pickle.dumps(err)
    assert pickle.loads(pickle.dumps(err)).code == "BACKEND_REFUSED"


def test_api_key_error_does_not_serialize_a_dropped_key_name() -> None:
    from traigent.utils.error_handler import APIKeyError

    err = APIKeyError(
        "sk-canary-2969", message="Credential rejected.", fix="Rotate it."
    )
    assert "sk-canary-2969" not in str(err)
    for rebuilt in (pickle.loads(pickle.dumps(err)), copy.deepcopy(err)):
        assert str(rebuilt) == str(err)
        assert rebuilt.fix == "Rotate it."
    assert b"sk-canary-2969" not in pickle.dumps(err)
    # The classic key_name form (name is part of the message) still round-trips.
    classic = APIKeyError("OPENAI_API_KEY")
    assert str(pickle.loads(pickle.dumps(classic))) == str(classic)


def test_unresolved_example_id_error_does_not_serialize_sequence_extras() -> None:
    from collections import UserList

    from traigent.utils.regression_dataset import UnresolvedExampleIdError

    ids = UserList(["id-1"])
    ids.note = "sk-canary-2969"  # type: ignore[attr-defined]
    err = UnresolvedExampleIdError(ids, [])
    assert b"sk-canary-2969" not in pickle.dumps(err)
    assert str(pickle.loads(pickle.dumps(err))) == str(err)
