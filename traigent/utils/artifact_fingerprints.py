"""Content-free artifact fingerprint helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

FP_ALGORITHM = "fp1"
FP_PREFIX = f"{FP_ALGORITHM}:"
FP_WIRE_RE = re.compile(r"^fp1:[0-9a-f]{64}$")

_FINGERPRINT_KEYS = ("dataset", "agent", "evaluator", "config_space")
_EXPECTED_OUTPUT_FIELDS = (
    "expected_output",
    "expected",
    "output",
    "answer",
    "target",
    "label",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _fingerprint_bytes(value: bytes) -> str:
    return f"{FP_PREFIX}{hashlib.sha256(value).hexdigest()}"


def _fingerprint_text(value: str) -> str:
    return _fingerprint_bytes(value.encode("utf-8"))


def _extract_examples(dataset_or_examples: Any) -> list[Any]:
    if dataset_or_examples is None or isinstance(
        dataset_or_examples, (str, bytes, bytearray)
    ):
        return []

    if isinstance(dataset_or_examples, Mapping):
        examples = dataset_or_examples.get("examples")
        if examples is None:
            return [dataset_or_examples]
        if isinstance(examples, Iterable) and not isinstance(
            examples, (str, bytes, bytearray)
        ):
            return list(examples)
        return []

    examples = getattr(dataset_or_examples, "examples", None)
    if examples is not None:
        return list(examples)

    if isinstance(dataset_or_examples, Iterable):
        return list(dataset_or_examples)

    return []


def is_dataset_materialized(dataset_or_examples: Any) -> bool:
    """Return True when the dataset's example content is already realized in
    memory (a list, or a Dataset-like object/mapping whose ``examples`` is
    already a list) -- i.e. safe to fingerprint without draining a
    single-use iterator the caller may still need to consume itself.

    A bare ``Iterable`` with no realized ``examples`` attribute (a
    generator, a database cursor, ...) returns False: fingerprinting it
    would silently consume it via :func:`_extract_examples`'s
    ``list(...)`` fallback. Callers that only have such an iterable MUST
    send ``None`` for the dataset fingerprint rather than draining it.
    """
    if dataset_or_examples is None or isinstance(
        dataset_or_examples, (str, bytes, bytearray)
    ):
        return False

    if isinstance(dataset_or_examples, list):
        return True

    if isinstance(dataset_or_examples, Mapping):
        examples = dataset_or_examples.get("examples")
        if examples is None:
            # Falls through to `_extract_examples`'s "wrap self in a list"
            # branch -- no iteration, so nothing to drain.
            return True
        return isinstance(examples, list)

    examples = getattr(dataset_or_examples, "examples", None)
    if examples is not None:
        return isinstance(examples, list)

    return False


def _example_input_expected(example: Any) -> tuple[Any, Any]:
    if isinstance(example, Mapping):
        input_data = example.get("input", example.get("input_data"))
        expected_output = None
        for field in _EXPECTED_OUTPUT_FIELDS:
            if field in example:
                expected_output = example[field]
                break
        return input_data, expected_output

    return (
        example.input_data,
        getattr(example, "expected_output", None),
    )


def _compute_dataset_fingerprint_and_count(
    dataset_or_examples: Any,
) -> tuple[str | None, int]:
    try:
        examples = _extract_examples(dataset_or_examples)
        if not examples:
            return None, 0

        example_digests = []
        for example in examples:
            input_data, expected_output = _example_input_expected(example)
            canonical = _canonical_json_bytes(
                {"input": input_data, "expected": expected_output}
            )
            example_digests.append(hashlib.sha256(canonical).hexdigest())

        canonical_set_digest = "\n".join(sorted(example_digests)).encode("utf-8")
        return _fingerprint_bytes(canonical_set_digest), len(examples)
    except Exception:
        return None, 0


def compute_dataset_fingerprint(dataset_or_examples: Any) -> str | None:
    """Return the fp1 dataset fingerprint, or None when unavailable."""

    return _compute_dataset_fingerprint_and_count(dataset_or_examples)[0]


def _has_example_content(dataset_or_examples: Any) -> bool:
    """Return True when at least one extracted example carries real content.

    "Content" is an input or an expected output -- the only two fields
    :func:`_example_input_expected` canonicalizes, and therefore the only two
    the dataset digest is computed over. An example that yields ``(None,
    None)`` contributes nothing that distinguishes one dataset from another,
    so a set consisting entirely of such examples is not fingerprintable
    content and must produce no fingerprint at all.

    Safe to call only after :func:`is_dataset_materialized` returns True --
    it extracts examples, which would drain a single-use iterator.
    """
    for example in _extract_examples(dataset_or_examples):
        input_data, expected_output = _example_input_expected(example)
        if input_data is not None or expected_output is not None:
            return True
    return False


def build_dataset_only_fingerprint_payload(
    dataset_or_examples: Any,
) -> dict[str, dict[str, Any]] | None:
    """Build the additive session-create fingerprint payload for callers
    that have ONLY a (possibly materialized) dataset and no func/evaluator/
    configuration_space to fingerprint -- e.g. the hybrid/privacy
    create-session path and the direct ``InteractiveOptimizer`` path, where
    the SDK never receives the decorated function or evaluator at all.

    Unlike :func:`build_artifact_fingerprints`, this never fills the
    ``evaluator`` slot with the "no evaluator provided" sentinel fingerprint
    (``compute_evaluator_fingerprint``'s fallback to ``fingerprint_text(
    "none")``) -- callers here have no evaluator information one way or the
    other, so that slot stays genuinely absent rather than a fixed
    placeholder value.

    Returns ``None`` (send nothing) when `dataset_or_examples` is not
    already materialized (see :func:`is_dataset_materialized`), carries no
    example CONTENT, or produces no fingerprint -- "missing is absence":
    never invent a fingerprint from metadata, and never drain a single-use
    iterator to get one.

    The content check is load-bearing, not defensive padding. A bare mapping
    with no ``examples`` key -- a metadata-only descriptor such as
    ``{"name": ..., "size": ...}``, which is exactly the shape the typed
    session already sends as ``dataset_metadata`` -- falls through
    :func:`_extract_examples`'s "wrap self as one example" branch and
    fingerprints as a single example whose input and expected output are both
    ``None``. Every such descriptor therefore hashes to the SAME digest, so
    unrelated datasets would collide onto one identity. Since the backend
    keys dataset identity off this digest, that would silently merge distinct
    datasets into a single optimization history -- the precise defect this
    fingerprint exists to prevent. No content, no fingerprint.
    """
    if not is_dataset_materialized(dataset_or_examples):
        return None

    if not _has_example_content(dataset_or_examples):
        return None

    dataset_fingerprint, dataset_example_count = _compute_dataset_fingerprint_and_count(
        dataset_or_examples
    )
    if dataset_fingerprint is None:
        return None

    return {
        "artifact_fingerprints": {
            "dataset": dataset_fingerprint,
            "agent": None,
            "evaluator": None,
            "config_space": None,
        },
        "fingerprint_meta": {
            "algorithm": FP_ALGORITHM,
            "dataset_example_count": dataset_example_count,
            "source_available": False,
        },
    }


def _compute_agent_fingerprint_and_source(
    func: Callable[..., Any] | None,
) -> tuple[str | None, bool]:
    try:
        if func is None:
            return None, False
        try:
            return _fingerprint_text(inspect.getsource(func)), True
        except (OSError, TypeError):
            module = getattr(func, "__module__", None)
            qualname = getattr(func, "__qualname__", None) or getattr(
                func, "__name__", None
            )
            if module is None and qualname is None:
                raise
            return _fingerprint_text(f"{module}:{qualname}"), False
    except Exception:
        return None, False


def compute_agent_fingerprint(func: Callable[..., Any] | None) -> str | None:
    """Return the fp1 agent fingerprint, falling back to module:qualname."""

    return _compute_agent_fingerprint_and_source(func)[0]


def _callable_source(func: Callable[..., Any]) -> str:
    return inspect.getsource(func)


def _sanitize_endpoint(endpoint: Any) -> str:
    if endpoint is None:
        return ""

    value = str(endpoint).strip()
    if not value:
        return ""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return value.split("?", 1)[0].split("#", 1)[0]

    if parsed.scheme and parsed.netloc:
        hostname = parsed.hostname or ""
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))

    return value.split("?", 1)[0].split("#", 1)[0]


def _external_identity(external: Any) -> str | None:
    if external is None:
        return None

    if isinstance(external, str):
        return f"external:{_sanitize_endpoint(external)}"

    if isinstance(external, Mapping):
        kind = external.get("kind") or external.get("type") or "external"
        endpoint = (
            external.get("endpoint")
            or external.get("api_endpoint")
            or external.get("url")
            or external.get("base_url")
        )
        return f"{kind}:{_sanitize_endpoint(endpoint)}"

    kind = (
        getattr(external, "kind", None)
        or getattr(external, "type", None)
        or external.__class__.__name__
    )
    endpoint = (
        getattr(external, "endpoint", None)
        or getattr(external, "api_endpoint", None)
        or getattr(external, "_api_endpoint", None)
        or getattr(external, "base_url", None)
    )
    return f"{kind}:{_sanitize_endpoint(endpoint)}"


def compute_evaluator_fingerprint(
    custom_evaluator: Callable[..., Any] | None = None,
    scoring_function: Callable[..., Any] | None = None,
    metric_functions: Mapping[str, Callable[..., Any]] | None = None,
    external: Any = None,
) -> str | None:
    """Return the fp1 evaluator fingerprint, or None on failure."""

    try:
        if custom_evaluator is not None:
            return _fingerprint_text(_callable_source(custom_evaluator))
        if scoring_function is not None:
            return _fingerprint_text(_callable_source(scoring_function))
        if metric_functions:
            sources = [
                _callable_source(metric_functions[name])
                for name in sorted(metric_functions)
            ]
            return _fingerprint_text("".join(sources))

        external_identity = _external_identity(external)
        if external_identity is not None:
            return _fingerprint_text(external_identity)

        return _fingerprint_text("none")
    except Exception:
        return None


def compute_surrogate_fingerprint(
    surrogate_evaluator: Callable[..., Any] | None,
) -> str | None:
    """Return the fp1 fingerprint of a surrogate evaluator's source, or None.

    The surrogate (pre-screen) evaluator is a cheap second scorer over already
    captured outputs. Its descriptor carries ``fingerprint_source`` so the
    backend can key a per-evaluator score tensor to the exact scorer source.
    Content-free: hashes only the callable's source, never any output content.
    Mirrors :func:`compute_evaluator_fingerprint`'s source-hashing path.
    """

    try:
        if surrogate_evaluator is None:
            return None
        return _fingerprint_text(_callable_source(surrogate_evaluator))
    except Exception:
        return None


def compute_config_space_fingerprint(configuration_space: Any) -> str | None:
    """Return the fp1 configuration-space fingerprint, or None on failure."""

    try:
        if configuration_space is None:
            return None
        return _fingerprint_bytes(_canonical_json_bytes(configuration_space))
    except Exception:
        return None


def build_artifact_fingerprints(
    *,
    dataset: Any = None,
    examples: Any = None,
    func: Callable[..., Any] | None = None,
    agent: Callable[..., Any] | None = None,
    custom_evaluator: Callable[..., Any] | None = None,
    scoring_function: Callable[..., Any] | None = None,
    metric_functions: Mapping[str, Callable[..., Any]] | None = None,
    external: Any = None,
    configuration_space: Any = None,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build the additive session-create fingerprint payload.

    Each individual fingerprint computation (`_compute_dataset_fingerprint_and_count`,
    `_compute_agent_fingerprint_and_source`, `compute_evaluator_fingerprint`,
    `compute_config_space_fingerprint`) already fails open to `None`
    internally, so one bad artifact never blocks session creation. The
    per-stage guards below only trip on a genuinely unexpected error that
    bypasses those inner catches, and each stage degrades independently so
    one failing stage never blanks the other three fingerprints.

    fail-open(intentional): mirrors `traigent.core.best_config_runtime`'s
    strict convention. Non-strict callers (the default) get a degraded
    payload explicitly marked ``fingerprint_meta["is_fallback"] = True``;
    strict callers re-raise instead of silently receiving a synthetic
    payload with unexplained ``None`` fields indistinguishable from a
    legitimate empty result.
    """

    is_fallback = False

    def _stage_failed(stage: str, exc: Exception) -> None:
        nonlocal is_fallback
        is_fallback = True
        logger.warning(
            "Falling back to degraded %s artifact fingerprint after error: %s",
            stage,
            exc,
        )

    try:
        dataset_source = dataset if dataset is not None else examples
        dataset_fingerprint, dataset_example_count = (
            _compute_dataset_fingerprint_and_count(dataset_source)
        )
    except Exception as exc:
        if strict:
            raise
        _stage_failed("dataset", exc)
        dataset_fingerprint, dataset_example_count = None, 0

    try:
        agent_fingerprint, source_available = _compute_agent_fingerprint_and_source(
            func or agent
        )
    except Exception as exc:
        if strict:
            raise
        _stage_failed("agent", exc)
        agent_fingerprint, source_available = None, False

    try:
        evaluator_fingerprint = compute_evaluator_fingerprint(
            custom_evaluator=custom_evaluator,
            scoring_function=scoring_function,
            metric_functions=metric_functions,
            external=external,
        )
    except Exception as exc:
        if strict:
            raise
        _stage_failed("evaluator", exc)
        evaluator_fingerprint = None

    try:
        config_space_fingerprint = compute_config_space_fingerprint(configuration_space)
    except Exception as exc:
        if strict:
            raise
        _stage_failed("config_space", exc)
        config_space_fingerprint = None

    fingerprint_meta: dict[str, Any] = {
        "algorithm": FP_ALGORITHM,
        "dataset_example_count": dataset_example_count,
        "source_available": source_available,
    }
    if is_fallback:
        fingerprint_meta["is_fallback"] = True

    return {
        "artifact_fingerprints": {
            "dataset": dataset_fingerprint,
            "agent": agent_fingerprint,
            "evaluator": evaluator_fingerprint,
            "config_space": config_space_fingerprint,
        },
        "fingerprint_meta": fingerprint_meta,
    }


def _fingerprint_value_to_wire(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and FP_WIRE_RE.fullmatch(value):
        return value
    return None


def artifact_fingerprints_to_wire(value: Any) -> dict[str, str | None] | None:
    """Return the strict wire shape for artifact_fingerprints."""

    if not isinstance(value, Mapping):
        return None
    return {
        key: _fingerprint_value_to_wire(value.get(key)) for key in _FINGERPRINT_KEYS
    }


def fingerprint_meta_to_wire(value: Any) -> dict[str, Any] | None:
    """Return the strict wire shape for fingerprint_meta."""

    if not isinstance(value, Mapping):
        return None

    count = value.get("dataset_example_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = 0

    source_available = value.get("source_available")
    if not isinstance(source_available, bool):
        source_available = False

    return {
        "algorithm": FP_ALGORITHM,
        "dataset_example_count": count,
        "source_available": source_available,
    }
