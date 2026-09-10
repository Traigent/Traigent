# Evaluator-quality fixture provenance

`evaluator_quality_bundle.json`, `evaluator_quality_context.json`,
`evaluator_quality_anchor.json`, `evaluator_quality_trust_anchor.json`,
`evaluator_quality_trust_status_active.json`, and
`evaluator_quality_trust_status_revoked.json` are generated, not hand-written.

## Source

`TraigentSchema@4b3373925cee6bd57071980285c58044165c90a4`
`tests/test_evaluator_quality_verifier.py::build_bundle` (and the same
module's `_build_evq_trust_status`), imported against the **installed**
`traigent_schema` package at the same pin (see `scripts/ci/schema-pin.txt`).

## Test keys (not secrets)

- Issuer signing key: `Ed25519PrivateKey.from_private_bytes(bytes(range(32)))`
- Trust-anchor signing key: `Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))`

Both are fixed, publicly-derivable byte sequences used only to produce
deterministic, real ed25519 signatures for test fixtures. They are not used
anywhere outside this test fixture set and grant no access to anything.

## `evaluator_quality_bundle.json` sha256

```
d4ccc09b9099df5d520d3b0475bc31b2eb2097d0a2bcffe11702beef260a118e
```

Verified deterministic: generated twice from a clean checkout of the pinned
Schema commit, byte-identical both times (same sha256 above on each run).

## Regeneration

```bash
export PATH="$PWD/.venv/bin:$PATH"
git -C <TraigentSchema checkout> show \
  4b3373925cee6bd57071980285c58044165c90a4:tests/test_evaluator_quality_verifier.py \
  > /tmp/evq_builder.py
python - <<'PY'
import json, sys
sys.path.insert(0, "/tmp")
import evq_builder as B

F = "tests/fixtures/certification"

def dump(obj, name):
    with open(f"{F}/{name}", "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")

dump(B.build_bundle(), "evaluator_quality_bundle.json")
dump({
    "expected_project_ref": B.PROJECT_REF,
    "expected_evaluator_commitment_ref": B.COMMITMENT_REF,
    "allow_unchecked_trust_status": True,
    "verification_time": "2026-09-10T00:00:00Z",
}, "evaluator_quality_context.json")
dump({
    "key_ref": B.KEY_REF, "algorithm": "ed25519",
    "public_key_der_b64": B.PUBLIC_B64, "public_key_digest": B.PUBLIC_KEY_DIGEST,
}, "evaluator_quality_anchor.json")

env_revoked, anchor = B._build_evq_trust_status(key_status="revoked")
dump(env_revoked, "evaluator_quality_trust_status_revoked.json")
dump({
    "key_ref": anchor.key_ref, "algorithm": anchor.algorithm,
    "public_key_der_b64": anchor.public_key_der_b64,
    "public_key_digest": anchor.public_key_digest,
}, "evaluator_quality_trust_anchor.json")

env_ok, _ = B._build_evq_trust_status()
dump(env_ok, "evaluator_quality_trust_status_active.json")
PY
```

## Note on `--anchor`

`evaluator_quality_anchor.json` carries the *issuer* signing key
(`B.KEY_REF`/`B.PUBLIC_B64`), not the trust anchor. It is used for tests that
pass `allow_unchecked_trust_status: true`, where `context.trust_anchor` is
`None` and the `--anchor` value is not consulted by the verifier at all — any
well-formed `TrustAnchorKeyV1` payload satisfies the CLI's required flag.
Tests that exercise the *checked* trust-status path (`KEY_REVOKED`, the
checked happy path) must instead pass `--anchor
evaluator_quality_trust_anchor.json`, which is the public key that actually
signs `evaluator_quality_trust_status_{active,revoked}.json`.
