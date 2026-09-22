"""Content identity v1: keyed example ids, multiset dataset roots, agent build versions.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md``
(``traigent.content_identity.v1``).

* :mod:`traigent.identity.content_identity` -- the spec primitives (port of the
  TraigentSchema reference implementation).
* :mod:`traigent.identity.keys` -- the Backend-granted purpose keys. Without a
  grant no content id is ever minted (fail closed).
* :mod:`traigent.identity.examples` -- ``EvaluationExample`` projection,
  ``dataset_root`` and per-trial ``evaluated_root``.
* :mod:`traigent.identity.agent_build` -- the agent build manifest and
  ``build_digest``.
* :mod:`traigent.identity.provider_versions` -- provider/model versions
  observed from responses.
"""

from traigent.identity.content_identity import SCHEME, ContentIdentityError
from traigent.identity.keys import (
    ContentIdentityKeys,
    clear_content_identity_keys,
    get_content_identity_keys,
    set_content_identity_keys,
)

__all__ = [
    "SCHEME",
    "ContentIdentityError",
    "ContentIdentityKeys",
    "clear_content_identity_keys",
    "get_content_identity_keys",
    "set_content_identity_keys",
]
