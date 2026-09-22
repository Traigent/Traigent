"""Per-run content identity: the session-create and per-trial wire objects.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` sections 5, 8, 10
and 11. The wire shape mirrors the JS SDK (traigent-js
``src/identity/run-identity.ts``) field for field, so both SDKs send the same
thing until milestone M3 gives it a typed home in TraigentSchema:

Session create carries a top-level ``content_identity``::

    {"scheme", "provenance": "declared",
     "key_status": "available" | "unavailable", "key_id"?,
     "agent": {"agent_id_source": "declared" | "fallback",
               "agent_id", "build_digest", "manifest"} | null,
     "dataset": DatasetIdentityV1 (record_state "draft") | null,
     "unavailable": {"agent"?: reason, "dataset"?: reason}}

Each trial result carries ``metadata.content_identity``::

    {"scheme", "provenance": "declared", "trial_id",
     "candidate": AgentVersionV1 | null,
     "evaluated": EvaluatedSetV1 | null,
     "observed_provider_versions": [...],
     "unavailable": {"candidate"?: reason, "evaluated"?: reason}}

Everything here is a CLIENT DECLARATION (``provenance: "declared"``): the
Backend recomputes and witnesses it in M3; until then no certificate may treat
it as server-recorded. Member lists above :data:`MAX_INLINE_MEMBERS` distinct
members are replaced by ``members_unavailable`` because the Backend's typed
session routes cap a request at 1 MiB and 50,000 JSON values
(TraigentBackend ``src/routes/traigent_session_routes.py`` ``_MAX_TRAIGENT_JSON_*``);
root and counts are still sent.

**Keys fail closed.** Without a Backend purpose-key grant no example id,
dataset root or evaluated root is minted, and the wire says so
(``purpose_keys_unavailable``).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from traigent.identity.agent_build import (
    AgentBuildBase,
    collect_agent_build_base,
    innermost_callable,
)
from traigent.identity.content_identity import (
    SCHEME,
    ConflictingExampleVersionsWarning,
    ContentIdentityError,
    MultisetRoot,
    compute_agent_build_digest,
    compute_multiset_root,
    key_id_of,
)
from traigent.identity.examples import DatasetContentIdentity, identify_dataset
from traigent.identity.keys import get_content_identity_keys
from traigent.utils import fp2
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CONTENT_IDENTITY_METADATA_KEY",
    "MAX_INLINE_MEMBERS",
    "ContentIdentityRun",
    "prepare_content_identity_run",
]

#: Key of the block in session-create bodies and trial ``metadata``.
CONTENT_IDENTITY_METADATA_KEY = "content_identity"
#: Largest member list sent inline (same cap as the JS SDK).
MAX_INLINE_MEMBERS = 2_000

_FOREIGN_KEY_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_RUN_CACHE_ATTR = "_traigent_content_identity_run"
_warned_agent_id_fallback = False


def _members_field(multiset: MultisetRoot) -> dict[str, Any]:
    if multiset.distinct_count <= MAX_INLINE_MEMBERS:
        return {
            "members": [
                {
                    "example_id": m.example_id,
                    "example_version": m.example_version,
                    "count": m.count,
                }
                for m in multiset.members
            ]
        }
    return {"members_unavailable": "members_exceed_inline_cap"}


def _result_field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _resolve_agent_id(
    agent_key: str | None, func: Any
) -> tuple[str | None, str | None]:
    """``(agent_id, source)``: the declared agent key, else the function name."""
    global _warned_agent_id_fallback
    declared = agent_key.strip() if isinstance(agent_key, str) else ""
    if declared:
        if _FOREIGN_KEY_ID.fullmatch(declared):
            return declared, "declared"
        return None, None
    if not _warned_agent_id_fallback:
        _warned_agent_id_fallback = True
        warnings.warn(
            "optimize() has no agent name: the agent build manifest uses the "
            "function name as a fallback agent id. Declare an agent name so "
            "agent history does not split.",
            UserWarning,
            stacklevel=3,
        )
    name = getattr(innermost_callable(func), "__name__", None)
    if isinstance(name, str) and _FOREIGN_KEY_ID.fullmatch(name):
        return name, "fallback"
    return None, None


@dataclass
class ContentIdentityRun:
    """Everything one optimization run needs to emit content identity."""

    key_id: str | None
    dataset: DatasetContentIdentity | None
    dataset_reason: str | None
    agent_id: str | None
    agent_id_source: str | None
    agent_base: AgentBuildBase | None

    def _version_for(
        self, config: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        if self.agent_id is None:
            return None, "agent_id_unavailable"
        if self.agent_base is None:
            return None, "agent_manifest_unavailable"
        try:
            applied = str(fp2.digest(dict(config or {})))
        except (fp2.Fp2UnsupportedValue, TypeError, ValueError):
            return None, "config_not_canonicalizable"
        try:
            manifest = self.agent_base.manifest(applied)
            build_digest = compute_agent_build_digest(manifest)
        except ContentIdentityError:
            return None, "agent_manifest_unavailable"
        return {
            "agent_id": self.agent_id,
            "build_digest": build_digest,
            "manifest": manifest,
        }, None

    def _dataset_wire(self) -> dict[str, Any] | None:
        if self.dataset is None:
            return None
        multiset = self.dataset.multiset
        return {
            "scheme": SCHEME,
            # draft: members are client-declared until the Backend stores and
            # recomputes them (M3); only a server-completed record is certifiable.
            "record_state": "draft",
            "key_id": self.dataset.key_id,
            "dataset_root": multiset.root,
            "distinct_count": multiset.distinct_count,
            "total_count": multiset.total_count,
            "conflicting_example_ids": list(multiset.conflicting_example_ids),
            **_members_field(multiset),
        }

    def session_wire(self, default_config: Mapping[str, Any] | None) -> dict[str, Any]:
        """The session-create ``content_identity`` object (base agent + dataset)."""
        base, agent_reason = self._version_for(default_config)
        dataset_wire = self._dataset_wire()
        unavailable: dict[str, str] = {}
        if agent_reason is not None:
            unavailable["agent"] = agent_reason
        if dataset_wire is None:
            unavailable["dataset"] = self.dataset_reason or "purpose_keys_unavailable"
        wire: dict[str, Any] = {
            "scheme": SCHEME,
            "provenance": "declared",
            "key_status": "unavailable" if self.key_id is None else "available",
        }
        if self.key_id is not None:
            wire["key_id"] = self.key_id
        wire["agent"] = (
            None
            if base is None
            else {"agent_id_source": self.agent_id_source or "declared", **base}
        )
        wire["dataset"] = dataset_wire
        wire["unavailable"] = unavailable
        return wire

    def _evaluated_for(
        self, trial_id: str, example_results: Iterable[Any] | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        if self.dataset is None:
            return None, self.dataset_reason or "purpose_keys_unavailable"
        if example_results is None:
            # The trial ended without per-example results (e.g. it raised):
            # what it attempted is unknown, which is not the empty set.
            return None, "evaluation_results_unavailable"
        pairs: list[tuple[str, str]] = []
        for result in example_results:
            example_id = _result_field(result, "example_id")
            example_version = _result_field(result, "example_version")
            try:
                if (
                    key_id_of(example_id) != self.dataset.key_id
                    or key_id_of(example_version) != self.dataset.key_id
                ):
                    return None, "row_outside_dataset"
            except ContentIdentityError:
                return None, "row_outside_dataset"
            pairs.append((example_id, example_version))
        with warnings.catch_warnings():
            # Conflicts were already reported once for the dataset.
            warnings.simplefilter("ignore", ConflictingExampleVersionsWarning)
            evaluated = compute_multiset_root(pairs, key_id=self.dataset.key_id)
        wire: dict[str, Any] = {"scheme": SCHEME}
        if isinstance(trial_id, str) and _FOREIGN_KEY_ID.fullmatch(trial_id):
            wire["trial_id"] = trial_id
        wire.update(
            {
                "key_id": self.dataset.key_id,
                "dataset_root": self.dataset.dataset_root,
                "evaluated_root": evaluated.root,
                "distinct_count": evaluated.distinct_count,
                "total_count": evaluated.total_count,
                **_members_field(evaluated),
            }
        )
        return wire, None

    def trial_wire(
        self,
        trial_id: str,
        config: Mapping[str, Any] | None,
        example_results: Iterable[Any] | None,
        observed_provider_versions: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """One trial's ``metadata.content_identity`` object.

        ``example_results`` is what the trial attempted (failed examples
        included): each must carry a content identity under the run's key, or
        no evaluated set is claimed (``row_outside_dataset``).
        """
        candidate, candidate_reason = self._version_for(config)
        evaluated, evaluated_reason = self._evaluated_for(trial_id, example_results)
        unavailable: dict[str, str] = {}
        if candidate_reason is not None:
            unavailable["candidate"] = candidate_reason
        if evaluated_reason is not None:
            unavailable["evaluated"] = evaluated_reason
        return {
            "scheme": SCHEME,
            "provenance": "declared",
            "trial_id": trial_id,
            "candidate": candidate,
            "evaluated": evaluated,
            # Empty means nothing was observed (e.g. a mocked run): an honest unknown.
            "observed_provider_versions": list(observed_provider_versions or []),
            "unavailable": unavailable,
        }


def prepare_content_identity_run(
    func: Callable[..., Any], dataset: Any, *, agent_key: str | None
) -> ContentIdentityRun:
    """Compute the run's content identity once (cached on the dataset object).

    Identifies and stamps every example (only with a key grant) and collects
    the run-constant part of the agent build manifest. Never raises for data it
    cannot represent: that slot becomes ``None`` with a content-free reason.
    """
    keys = get_content_identity_keys()
    cache_key = (id(func), agent_key, keys.kid if keys is not None else None)
    cache = getattr(dataset, "__dict__", None)
    if isinstance(cache, dict):
        cached = cache.get(_RUN_CACHE_ATTR)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
            run: ContentIdentityRun = cached[1]
            # Re-stamp examples (idempotent and cached) in case they changed.
            identify_dataset(dataset, keys)
            return run

    dataset_identity: DatasetContentIdentity | None = None
    dataset_reason: str | None = "purpose_keys_unavailable"
    if keys is not None:
        dataset_identity = identify_dataset(dataset, keys)
        dataset_reason = (
            None if dataset_identity is not None else "row_not_canonicalizable"
        )

    agent_id, agent_source = _resolve_agent_id(agent_key, func)
    agent_base: AgentBuildBase | None = None
    if agent_id is not None:
        try:
            agent_base = collect_agent_build_base(func, agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001 - identity must never fail a run
            logger.debug("Agent build manifest unavailable: %s", type(exc).__name__)

    run = ContentIdentityRun(
        key_id=keys.kid if keys is not None else None,
        dataset=dataset_identity,
        dataset_reason=dataset_reason,
        agent_id=agent_id,
        agent_id_source=agent_source,
        agent_base=agent_base,
    )
    if isinstance(cache, dict):
        cache[_RUN_CACHE_ATTR] = (cache_key, run)
    return run


def _reset_warnings_for_tests() -> None:
    global _warned_agent_id_fallback
    _warned_agent_id_fallback = False
