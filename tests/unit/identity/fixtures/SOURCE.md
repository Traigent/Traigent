# Vendored content-identity v1 conformance vectors

`content_identity_v1_vectors.json` in this directory is vendored **byte-for-byte**
(not hand-edited) from TraigentSchema:

- Source path: `traigent_schema/data/content_identity_v1_vectors.json`
- Source commit: `86a3bac861884d7b57de1b1253a932b743ab2be7` (branch
  `feat/content-identity-v1`, TraigentSchema PR #515). The file is byte-identical
  at `002ff566` (the M1-approved head).
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
