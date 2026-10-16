# Vendored content-identity v1 conformance vectors

`content_identity_v1_vectors.json` in this directory is vendored **byte-for-byte**
(not hand-edited) from TraigentSchema:

- Source path: `traigent_schema/data/content_identity_v1_vectors.json`
- Source commit: `024d55b0514c8c961d9496c34e8c5fc3548e1eb1` (TraigentSchema `develop`, the merge of PR #515).
  Byte-identical at the PR's final head `86a3bac861884d7b57de1b1253a932b743ab2be7`
  and at `002ff566` (the M1-approved head).
- SHA-256 of the vendored file:
  `7a75e6ab1316a015d1be91211b23e1ac3a0ce169ba29d6913e877856ce34b66e`

The tenant master secrets in the file are PUBLIC TEST VALUES published by the
spec. They are used only by the conformance tests; the SDK never holds a tenant
master in production (spec section 12, ruling D1).

The algorithm port lives at `traigent/identity/content_identity.py` (same
source commit; see its module docstring).

`tests/unit/identity/test_content_identity_vectors.py` checks, on every run:

1. the vendored file's SHA-256 equals the value above (no hand edits);
2. when an installed `traigent_schema` ships this data file (the SDK's pinned
   build in `scripts/ci/schema-pin.txt` does not yet -- it predates PR #515),
   the vendored file is byte-identical to it, so a schema-pin bump that
   changes the vectors fails until they are re-vendored.

To re-vendor after the Schema changes:

```bash
git -C <TraigentSchema checkout> show <commit>:traigent_schema/data/content_identity_v1_vectors.json \
  > tests/unit/identity/fixtures/content_identity_v1_vectors.json
sha256sum tests/unit/identity/fixtures/content_identity_v1_vectors.json
```

then update the commit and SHA-256 above and `EXPECTED_SHA256` in the test.

## identity_vectors_v1.json (agent identity labels)

`identity_vectors_v1.json` is vendored **byte-for-byte** from the TraigentSchema
local branch `feat/identity-vectors` (commit a6bbbaa)
(`traigent_schema/data/identity/identity_vectors_v1.json`).

- SHA-256: `294814bec6a947b3c32cc4ec1c438b16f2c3fdcca815adb4883258ac7f0ff4d4`
  (pinned in `tests/unit/identity/test_agent_identity_labels.py`).
- Replace this copy with the TraigentSchema package data once that package is
  published; until then the sha256 pin is what keeps the three consumers on the
  same vectors.
