"""Per-trial content-identity block, shaped after ``TrialIdentityV1``.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` section 10 and
``schemas/execution/run_identity_binding_v1_schema.json``. Each trial result
carries, under ``metadata["content_identity"]``::

    {
      "scheme": "traigent.content_identity.v1",
      "dataset":   {key_id, dataset_root, distinct_count, total_count,
                    conflicting_example_ids}                    # keyed; needs a grant
      "evaluated": EvaluatedSetV1 (evaluated_root + members)     # keyed; needs a grant
      "candidate": AgentVersionV1 {agent_id, build_digest, manifest}   # unkeyed
      "observed_provider_versions": [ObservedProviderVersionV1, ...]
    }

The SDK sends it inside the configuration-run result ``metadata`` (an open
object in ``evaluation/configuration_run_schema.json``): no Backend endpoint or
typed field exists for it yet. The Backend's run identity binding
(``RunIdentityBindingV1``, milestone M3) is what turns these declared facts
into server-recorded ones; until then every fact here is a CLIENT DECLARATION
(spec section 11).

Keys that cannot be stated honestly are omitted, never filled with a
placeholder: no grant -> no ``dataset``/``evaluated``; a result without an
identity -> no ``evaluated``; no agent id or unpinnable code -> no
``candidate``. ``observed_provider_versions`` may be empty, which the schema
defines as "nothing observed" (for example a mocked run), not "no providers".
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from traigent.identity.content_identity import SCHEME
from traigent.identity.examples import DatasetContentIdentity, build_evaluated_set

__all__ = ["CONTENT_IDENTITY_METADATA_KEY", "build_trial_content_identity"]

#: Key of the block in ``TrialResult.metadata`` and in the submitted trial metadata.
CONTENT_IDENTITY_METADATA_KEY = "content_identity"


def build_trial_content_identity(
    *,
    dataset_identity: DatasetContentIdentity | None,
    example_results: Iterable[Any] | None,
    candidate: dict[str, Any] | None,
    observed_provider_versions: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Assemble one trial's block; ``None`` when there is nothing to state."""
    block: dict[str, Any] = {"scheme": SCHEME}
    if dataset_identity is not None:
        block["dataset"] = dataset_identity.summary()
        evaluated = build_evaluated_set(example_results, dataset_identity)
        if evaluated is not None:
            block["evaluated"] = evaluated
    if candidate is not None:
        block["candidate"] = candidate
    observed = list(observed_provider_versions or [])
    if len(block) == 1 and not observed:
        return None
    block["observed_provider_versions"] = observed
    return block
