"""Compute a pillar fingerprint locally, reusing ``traigent.identity`` -- never re-deriving it.

Each ``*Source`` dataclass says what the caller hands over for one pillar kind.
:func:`compute_pillar` returns a :class:`Pillar`: the fingerprint, its scheme and
key id, plus the canonical representations the recovery-manifest builder exports.
Names, paths and content never enter a :class:`Pillar`'s wire fields.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from traigent.identity import agent_build as _ab
from traigent.identity.content_identity import (
    ContentIdentityError,
    canonical_bytes,
    compute_agent_build_digest,
    compute_evaluator_version_digest,
)
from traigent.identity.evaluator_version import build_evaluator_binding
from traigent.identity.examples import identify_dataset
from traigent.identity.keys import ContentIdentityKeys
from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.utils import fp2


@dataclass(frozen=True)
class AgentSource:
    """An agent callable plus its declared ``agent_id`` and applied configuration."""

    func: Callable[..., Any]
    agent_id: str
    config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AgentManifestSource:
    """A ready ``AgentBuildManifestV1`` dict (digest recomputed by the identity code)."""

    manifest: dict[str, Any]


@dataclass(frozen=True)
class ConfigurationSource:
    """An applied configuration object (``{}`` is the base default)."""

    config: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetSource:
    """An evaluation dataset (object with ``.examples``); keys default to the installed grant."""

    dataset: Any
    keys: ContentIdentityKeys | None = None


@dataclass(frozen=True)
class EvaluatorSource:
    """An evaluator plus its objectives; the manifest comes from the identity code."""

    evaluator: Any
    objectives: Any = None
    evaluator_id: str | None = None


@dataclass(frozen=True)
class EvaluatorManifestSource:
    """A ready ``EvaluatorVersionManifestV1`` dict."""

    manifest: dict[str, Any]


@dataclass(frozen=True)
class SearchSpaceSource:
    """A search space whose ``ssp1:`` id was minted by the Backend (``K_search_space`` is Backend-only).

    The SDK cannot derive the id, so the caller supplies it; the identity is
    then ``client_declared``, never ``sdk_fingerprinted``.
    """

    fingerprint: str
    definition: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Representation:
    """One canonical byte export of part of a pillar."""

    kind: str
    data: bytes


@dataclass(frozen=True)
class Pillar:
    kind: str
    fingerprint_scheme: str
    key_id: str | None
    fingerprint: str
    identity_source: str
    representations: tuple[Representation, ...] = field(default=())
    covered: tuple[str, ...] = field(default=())
    ordering: str = c.ORDERING_NOT_APPLICABLE
    dependencies: tuple[tuple[str, str], ...] = field(default=())

    def wire_identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fingerprint_scheme": self.fingerprint_scheme,
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
        }


def _refuse(code: str = "invalid_request") -> LegendError:
    return LegendError(code, f"pillar could not be fingerprinted ({code})")


def _agent_source_text(func: Callable[..., Any]) -> bytes | None:
    try:
        text = inspect.getsource(_ab.innermost_callable(func))
    except (OSError, TypeError):
        return None
    normalized = _ab.normalize_source(text)
    return normalized.encode("utf-8") if normalized is not None else None


def _agent(source: AgentSource | AgentManifestSource) -> Pillar:
    reps: list[Representation] = []
    covered: list[str] = []
    if isinstance(source, AgentManifestSource):
        try:
            digest = compute_agent_build_digest(source.manifest)
        except ContentIdentityError:
            raise _refuse() from None
        reps.append(
            Representation(c.REPR_CONFIG_DOCUMENT, canonical_bytes(source.manifest))
        )
    else:
        base = _ab.collect_agent_build_base(source.func, agent_id=source.agent_id)
        version = _ab.candidate_agent_version(base, source.config)
        if version is None:
            raise _refuse()
        digest = version["build_digest"]
        text = _agent_source_text(source.func)
        if text is not None:
            reps.append(Representation(c.REPR_SOURCE_BUNDLE, text))
            covered.append("code")
        reps.append(
            Representation(
                c.REPR_CONFIG_DOCUMENT, canonical_bytes(dict(source.config or {}))
            )
        )
        covered.append("applied_config")
    return Pillar(
        c.KIND_AGENT,
        c.SCHEME_BY_KIND[c.KIND_AGENT],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        tuple(reps),
        tuple(covered),
    )


def _configuration(source: ConfigurationSource) -> Pillar:
    try:
        digest = str(fp2.digest(dict(source.config)))
        data = canonical_bytes(dict(source.config))
    except (fp2.Fp2UnsupportedValue, ContentIdentityError, TypeError, ValueError):
        raise _refuse() from None
    return Pillar(
        c.KIND_CONFIGURATION,
        c.SCHEME_BY_KIND[c.KIND_CONFIGURATION],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        (Representation(c.REPR_CONFIG_DOCUMENT, data),),
        ("applied_config",),
    )


def _dataset(source: DatasetSource) -> Pillar:
    identity = identify_dataset(source.dataset, source.keys)
    if identity is None:
        raise _refuse("store_not_configured")
    rows = []
    for example in list(getattr(source.dataset, "examples", None) or []):
        metadata = getattr(example, "metadata", None)
        rows.append(
            {
                "input": getattr(example, "input_data", None),
                "expected": getattr(example, "expected_output", None),
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else None,
            }
        )
    try:
        data = canonical_bytes(rows)  # a list: order and multiplicity are preserved
    except ContentIdentityError:
        raise _refuse() from None
    return Pillar(
        c.KIND_DATASET,
        c.SCHEME_BY_KIND[c.KIND_DATASET],
        identity.key_id,
        identity.dataset_root,
        c.IDENTITY_SDK_FINGERPRINTED,
        (Representation(c.REPR_DATASET_ROWS, data),),
        ("dataset_members", "dataset_order"),
        c.ORDERING_PRESERVED,
    )


def _evaluator(source: EvaluatorSource | EvaluatorManifestSource) -> Pillar:
    if isinstance(source, EvaluatorManifestSource):
        manifest = source.manifest
        try:
            digest = compute_evaluator_version_digest(manifest)
        except ContentIdentityError:
            raise _refuse() from None
    else:
        binding, _src, _reason = build_evaluator_binding(
            source.evaluator,
            objectives=source.objectives,
            evaluator_id=source.evaluator_id,
        )
        if binding is None:
            raise _refuse()
        manifest, digest = binding["manifest"], binding["version_digest"]
    covered = ["judge_config", "objectives"]
    return Pillar(
        c.KIND_EVALUATOR,
        c.SCHEME_BY_KIND[c.KIND_EVALUATOR],
        None,
        digest,
        c.IDENTITY_SDK_FINGERPRINTED,
        (Representation(c.REPR_EVALUATOR_BUNDLE, canonical_bytes(manifest)),),
        tuple(covered),
    )


def _search_space(source: SearchSpaceSource) -> Pillar:
    match = c.SEARCH_SPACE_ID_RE.fullmatch(source.fingerprint or "")
    if match is None:
        raise _refuse()
    reps: tuple[Representation, ...] = ()
    covered: tuple[str, ...] = ()
    if source.definition is not None:
        try:
            reps = (
                Representation(
                    c.REPR_SEARCH_SPACE, canonical_bytes(dict(source.definition))
                ),
            )
        except ContentIdentityError:
            raise _refuse() from None
        covered = ("search_space_definition",)
    return Pillar(
        c.KIND_SEARCH_SPACE,
        c.SCHEME_BY_KIND[c.KIND_SEARCH_SPACE],
        match.group(1),
        source.fingerprint,
        c.IDENTITY_CLIENT_DECLARED,
        reps,
        covered,
    )


def compute_pillar(kind: str, source: Any) -> Pillar:
    """Fingerprint ``source`` as pillar ``kind`` with the existing identity code."""
    if kind not in c.ENTITY_KINDS:
        raise LegendError("invalid_request", "unknown pillar kind")
    allowed: dict[str, tuple[type, ...]] = {
        c.KIND_AGENT: (AgentSource, AgentManifestSource),
        c.KIND_CONFIGURATION: (ConfigurationSource,),
        c.KIND_DATASET: (DatasetSource,),
        c.KIND_EVALUATOR: (EvaluatorSource, EvaluatorManifestSource),
        c.KIND_SEARCH_SPACE: (SearchSpaceSource,),
    }
    if not isinstance(source, allowed[kind]):
        raise LegendError(
            "invalid_request", "source type does not match the pillar kind"
        )
    src: Any = source
    if kind == c.KIND_AGENT:
        return _agent(src)
    if kind == c.KIND_CONFIGURATION:
        return _configuration(src)
    if kind == c.KIND_DATASET:
        return _dataset(src)
    if kind == c.KIND_EVALUATOR:
        return _evaluator(src)
    return _search_space(src)
