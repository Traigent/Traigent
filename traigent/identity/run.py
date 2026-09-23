"""Per-run content identity: the session-create and per-trial wire objects.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` sections 5, 8-11.
Both SDKs send the same shape (traigent-js ``src/identity/run-identity.ts``)
until milestone M3 gives it a typed home in TraigentSchema.

Session create carries a top-level ``content_identity``::

    {"scheme", "provenance": "declared", "key_status": "available", "key_id",
     "agent_id_source": "declared" | "fallback" | null,
     "agent": AgentVersionV1 | null,
     "evaluator_id_source": "declared" | "fallback" | null,
     "evaluator": EvaluatorVersionBindingV1 | null,
     "dataset": DatasetIdentityV1 (record_state "draft") | null,
     "unavailable": {slot: reason}}

Each trial result carries ``metadata.content_identity``::

    {"scheme", "provenance": "declared", "trial_id",
     "candidate": AgentVersionV1 | null,
     "evaluated": EvaluatedSetV1 | null,
     "observed_provider_versions": [ObservedProviderVersionV1, ...],
     "unavailable": {slot: reason}}

Every slot object is schema-valid on its own; a slot the SDK cannot state
honestly is ``null`` and its reason (one of :data:`UNAVAILABLE_REASONS`, the
union shared with the JS SDK) goes in ``unavailable``. Everything is a CLIENT
DECLARATION (``provenance: "declared"``) until the Backend witnesses it (M3).

**No grant, nothing new.** Without a Backend purpose-key grant there is no run
(:func:`prepare_content_identity_run` returns ``None``) and neither object is
emitted: session-create and trial payloads stay byte-identical to an SDK
without content identity (owner-delegated decision on the M2 review).

**One snapshot per run.** The run is computed once when ``optimize()`` starts,
from the rows' content at that moment, and owned by that run; the next run
recomputes everything (dataset identity and build evidence).
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
from traigent.identity.evaluator_version import build_evaluator_binding
from traigent.identity.examples import DatasetContentIdentity, identify_dataset
from traigent.identity.keys import get_content_identity_keys
from traigent.utils import fp2
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CONTENT_IDENTITY_METADATA_KEY",
    "MAX_CONFLICTING_EXAMPLE_IDS",
    "MAX_INLINE_MEMBERS",
    "UNAVAILABLE_REASONS",
    "ContentIdentityRun",
    "prepare_content_identity_run",
]

#: Key of the block in session-create bodies and trial ``metadata``.
CONTENT_IDENTITY_METADATA_KEY = "content_identity"
#: Largest member list sent inline. The Backend's typed session routes cap a
#: request at 1 MiB and 50,000 JSON values (TraigentBackend
#: ``src/routes/traigent_session_routes.py`` ``_MAX_TRAIGENT_JSON_*``); above
#: this the whole slot is withheld (``members_exceed_inline_cap``) until the
#: Backend's member-list store (``members_ref``, M3) exists.
MAX_INLINE_MEMBERS = 2_000
#: Largest ``conflicting_example_ids`` list sent; longer lists are truncated
#: (sorted prefix) and flagged ``conflicting_example_ids_truncated``.
MAX_CONFLICTING_EXAMPLE_IDS = 1_000

#: Every reason either SDK may put in ``unavailable`` (shared with traigent-js).
UNAVAILABLE_REASONS = frozenset(
    {
        "purpose_keys_unavailable",
        "purpose_key_provider_failed",
        "row_not_canonicalizable",
        "row_outside_dataset",
        "evaluation_results_unavailable",
        "members_exceed_inline_cap",
        "conflicting_example_ids_truncated",
        "agent_id_unavailable",
        "agent_manifest_unavailable",
        "config_not_canonicalizable",
        "evaluator_id_unavailable",
        "evaluator_manifest_unavailable",
    }
)

_FOREIGN_KEY_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_warned_agent_id_fallback = False


def _members(multiset: MultisetRoot) -> list[dict[str, Any]] | None:
    """Inline members, or ``None`` when above :data:`MAX_INLINE_MEMBERS`."""
    if multiset.distinct_count > MAX_INLINE_MEMBERS:
        return None
    return [
        {
            "example_id": m.example_id,
            "example_version": m.example_version,
            "count": m.count,
        }
        for m in multiset.members
    ]


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
    """One optimization run's content identity snapshot (only exists with a grant)."""

    key_id: str
    dataset: DatasetContentIdentity | None
    dataset_reason: str | None
    agent_id: str | None
    agent_id_source: str | None
    agent_base: AgentBuildBase | None
    evaluator: dict[str, Any] | None
    evaluator_id_source: str | None
    evaluator_reason: str | None

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

    def _dataset_slot(self, unavailable: dict[str, str]) -> dict[str, Any] | None:
        if self.dataset is None:
            unavailable["dataset"] = self.dataset_reason or "row_not_canonicalizable"
            return None
        multiset = self.dataset.multiset
        members = _members(multiset)
        if members is None:
            unavailable["dataset"] = "members_exceed_inline_cap"
            return None
        conflicts = list(multiset.conflicting_example_ids)
        if len(conflicts) > MAX_CONFLICTING_EXAMPLE_IDS:
            conflicts = conflicts[:MAX_CONFLICTING_EXAMPLE_IDS]
            unavailable["conflicting_example_ids"] = "conflicting_example_ids_truncated"
        return {
            "scheme": SCHEME,
            # draft: members are client-declared until the Backend stores and
            # recomputes them (M3); only a server-completed record is certifiable.
            "record_state": "draft",
            "key_id": self.dataset.key_id,
            "dataset_root": multiset.root,
            "distinct_count": multiset.distinct_count,
            "total_count": multiset.total_count,
            "conflicting_example_ids": conflicts,
            "members": members,
        }

    def _evaluator_slot(self, unavailable: dict[str, str]) -> dict[str, Any] | None:
        if self.evaluator is None:
            unavailable["evaluator"] = (
                self.evaluator_reason or "evaluator_manifest_unavailable"
            )
        return self.evaluator

    def session_wire(self, default_config: Mapping[str, Any] | None) -> dict[str, Any]:
        """The session-create ``content_identity`` object."""
        unavailable: dict[str, str] = {}
        agent, agent_reason = self._version_for(default_config)
        if agent_reason is not None:
            unavailable["agent"] = agent_reason
        evaluator = self._evaluator_slot(unavailable)
        dataset = self._dataset_slot(unavailable)
        return {
            "scheme": SCHEME,
            "provenance": "declared",
            "key_status": "available",
            "key_id": self.key_id,
            "agent_id_source": self.agent_id_source,
            "agent": agent,
            "evaluator_id_source": self.evaluator_id_source,
            "evaluator": evaluator,
            "dataset": dataset,
            "unavailable": unavailable,
        }

    def _evaluated_slot(
        self,
        trial_id: str,
        example_results: Iterable[Any] | None,
        unavailable: dict[str, str],
    ) -> dict[str, Any] | None:
        if self.dataset is None:
            unavailable["evaluated"] = self.dataset_reason or "row_not_canonicalizable"
            return None
        if example_results is None:
            # The trial ended without per-example results (e.g. it raised):
            # what it attempted is unknown, which is not the empty set.
            unavailable["evaluated"] = "evaluation_results_unavailable"
            return None
        pairs: list[tuple[str, str]] = []
        for result in example_results:
            example_id = _result_field(result, "example_id")
            example_version = _result_field(result, "example_version")
            try:
                if (
                    key_id_of(example_id) != self.dataset.key_id
                    or key_id_of(example_version) != self.dataset.key_id
                ):
                    unavailable["evaluated"] = "row_outside_dataset"
                    return None
            except ContentIdentityError:
                unavailable["evaluated"] = "row_outside_dataset"
                return None
            pairs.append((example_id, example_version))
        with warnings.catch_warnings():
            # Conflicts were already reported once for the dataset.
            warnings.simplefilter("ignore", ConflictingExampleVersionsWarning)
            evaluated = compute_multiset_root(pairs, key_id=self.dataset.key_id)
        members = _members(evaluated)
        if members is None:
            unavailable["evaluated"] = "members_exceed_inline_cap"
            return None
        slot: dict[str, Any] = {"scheme": SCHEME}
        if isinstance(trial_id, str) and _FOREIGN_KEY_ID.fullmatch(trial_id):
            slot["trial_id"] = trial_id
        slot.update(
            {
                "key_id": self.dataset.key_id,
                "dataset_root": self.dataset.dataset_root,
                "evaluated_root": evaluated.root,
                "distinct_count": evaluated.distinct_count,
                "total_count": evaluated.total_count,
                "members": members,
            }
        )
        return slot

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
        unavailable: dict[str, str] = {}
        candidate, candidate_reason = self._version_for(config)
        if candidate_reason is not None:
            unavailable["candidate"] = candidate_reason
        evaluated = self._evaluated_slot(trial_id, example_results, unavailable)
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
    func: Callable[..., Any],
    dataset: Any,
    *,
    agent_key: str | None,
    evaluator: Any = None,
    objectives: Iterable[Any] | None = None,
    evaluator_id: str | None = None,
) -> ContentIdentityRun | None:
    """Take this run's content identity snapshot; ``None`` without a key grant.

    Identifies and stamps every example from its current content, collects the
    agent build evidence and the evaluator version. Never raises for data it
    cannot represent: that slot becomes ``None`` with a reason.
    """
    keys = get_content_identity_keys()
    if keys is None:
        return None
    dataset_identity = identify_dataset(dataset, keys)
    agent_id, agent_source = _resolve_agent_id(agent_key, func)
    agent_base: AgentBuildBase | None = None
    if agent_id is not None:
        try:
            agent_base = collect_agent_build_base(func, agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001 - identity must never fail a run
            logger.debug("Agent build manifest unavailable: %s", type(exc).__name__)
    binding, evaluator_source, evaluator_reason = build_evaluator_binding(
        evaluator, objectives=objectives, evaluator_id=evaluator_id
    )
    return ContentIdentityRun(
        key_id=keys.kid,
        dataset=dataset_identity,
        dataset_reason=None
        if dataset_identity is not None
        else "row_not_canonicalizable",
        agent_id=agent_id,
        agent_id_source=agent_source,
        agent_base=agent_base,
        evaluator=binding,
        evaluator_id_source=evaluator_source,
        evaluator_reason=evaluator_reason,
    )


def _reset_warnings_for_tests() -> None:
    global _warned_agent_id_fallback
    _warned_agent_id_fallback = False
