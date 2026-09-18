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

# Agent-quality (pillar 1) fixture provenance

All `agent_quality_*.json` files below are generated, not hand-written, by a
single script pasted in full at the bottom of this section.

## Source

`TraigentSchema@321d98133645d343297545a155b9549650e3152d`
(`tests/test_agent_quality_verifier.py`'s `build_agent_quality_bundle`,
`build_abstained_agent_quality_bundle`, `build_agent_quality_context`,
`_GV_PROCESS_RECORD_BUNDLE`/`_GV_PROCESS_RECORD_CONTEXT`, and
`tests/test_process_record_verifier.py`'s `_build_bundle`/
`_build_trust_status`), imported against the **installed** `traigent_schema`
package at the same pin (see `scripts/ci/schema-pin.txt`) from a scratch
clone of TraigentSchema checked out at that commit.

## Test keys (not secrets)

Reused, unmodified, from the Schema's own golden-vector fixtures
(`tests/test_agent_quality_verifier.py::_GV_PRIVATE_KEY` /
`tests/test_process_record_verifier.py`'s own deterministic v0 issuer key):
fixed, publicly-derivable byte sequences used only to produce deterministic,
real ed25519 signatures for test fixtures. Not used anywhere outside this
test fixture set and grant no access to anything.

## Checked-mode feasibility (measured before shipping the checked golden)

`tests/test_process_record_verifier.py::_build_bundle(allow_unchecked_base_status=False)`
was built and its `unsigned_manifest` compared byte-for-byte against
`_GV_PROCESS_RECORD_BUNDLE["unsigned_manifest"]` (the process record the
golden agent-quality bundle is bound to): **identical**, since
`allow_unchecked_base_status` gates the context/trust side only and does not
feed the manifest's own content. The resulting checked bundle/context, with
a trust-status snapshot built via `_build_trust_status`, was then run
through the real `verify_agent_quality_certificate` against the *same*
golden agent-quality bundle and returned `AGENT_QUALITY_VERIFIED` — the
`agent_quality_context_checked.json` / `agent_quality_trust_status_active.json`
/ `agent_quality_trust_anchor.json` fixtures below are exactly that state,
serialized.

## File sha256

```
7f4a4183005222718c46b09d151d34401293f9bae271e9821646381b96853325  agent_quality_abstained_bundle.json
fe30802d9a98da4dac8debcea9e82ea0a0d8954d074fecffa1e87780b2af3ee0  agent_quality_anchor.json
e025e70a81fcca48219beac3f118c31cc9f5736f2ddffda04395c2fc190ecdc0  agent_quality_bundle.json
b572de306fd86a8f2d19692b85fbe10d5da568ecbe20a8dcc7e3abb112b587be  agent_quality_bundle_tamper_interval_minus1.json
3b21b1f4b1bb0dfc2941947ac7d25029db1db15f5fc42a0274cf076e130b2b4a  agent_quality_bundle_tamper_interval_plus1.json
e36c20f483c9fa10f4e0a909846fdbe6f9d1bf3653bc02e66e42e3146d068685  agent_quality_bundle_tamper_signature.json
ac9b74d6259a1984ddd6ca9a288ccc256abb2dc6de08b2d4eccc135d30a02b44  agent_quality_context_checked.json
c92f5cd4a60db3e3b0889b001f77e5865c24d2a5e9315095e52868fcecf9f7f9  agent_quality_context_tamper_commitment_ref.json
9c7469a6eadfd6282b85d9dbdf64c57d7bc9ee65e1a0d500926692b47aba35bc  agent_quality_context_tamper_scope.json
ce424ac8c7e8200abff4a137239206d7932dcf24713132ef349bd504ba007a38  agent_quality_context_unchecked.json
f8da998d4593c1daa730fd953063eb0f913e2fad616b0495dde068b730760d62  agent_quality_process_record.json
56a6e6201061fef87f1b823e1c958847b649ffe76bfe04de640ea1bae6bbded9  agent_quality_process_record_tamper_receipt.json
fe30802d9a98da4dac8debcea9e82ea0a0d8954d074fecffa1e87780b2af3ee0  agent_quality_trust_anchor.json
96babef52699a1662bd3240b84d970ce604c280df984ddd71adc457d47f40284  agent_quality_trust_status_active.json
```

Note: `agent_quality_anchor.json` and `agent_quality_trust_anchor.json`
share a sha256 (both are the same `TrustAnchorKeyV1` payload). This is
expected, not a mistake -- see "Note on `--anchor`" for agent-quality below.

## Note on `--anchor` (agent-quality)

For the unchecked golden (`agent_quality_context_unchecked.json`), the
nested `process_record_context.allow_unchecked_base_status` is `true`, so
`process_record_context.trust_anchor` is absent and the `--anchor` value is
not consulted by the verifier at all -- `agent_quality_anchor.json` is any
well-formed `TrustAnchorKeyV1` payload (in fact the same default anchor
`test_process_record_verifier._default_trust_anchor("ed25519")` returns).
For the checked golden (`agent_quality_context_checked.json`), `--anchor`
MUST be `agent_quality_trust_anchor.json` -- the public key that actually
signs `agent_quality_trust_status_active.json` -- and
`process_record_context.trust_anchor` embeds the same key, so
`_check_context_anchor` (which requires the embedded anchor to equal
`--anchor` when present) passes by construction.

## Regeneration script (pasted in full)

```python
"""Generator for tests/fixtures/certification/agent_quality_*.json.

Run from inside a scratch clone of TraigentSchema checked out at
321d98133645d343297545a155b9549650e3152d, with that clone's venv (which has
traigent-schema installed at the same pin) active, and the SDK worktree's
scripts/ci/schema-pin.txt already updated to that same commit.

    cd /tmp/sa0-schema-scratch
    /path/to/sdk/.venv-sa0/bin/python /tmp/sa0-generate-agent-quality-fixtures.py

Never hand-edited: every JSON file this script writes is produced solely from
the Schema's own test builders (tests/test_agent_quality_verifier.py,
tests/test_process_record_verifier.py) -- see PROVENANCE.md for the paste-in
record.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/sa0-schema-scratch")

import tests.test_agent_quality_verifier as av
import tests.test_process_record_verifier as pv
from traigent_schema.certification import TrustAnchorKeyV1
from traigent_schema.certification.relying_party_verifier import VerificationContext

OUT = Path(
    "/home/nimrod/TraigentProjects/core_project/worktrees/Traigent/"
    "certify-agent-quality/tests/fixtures/certification"
)


def _anchor_dict(anchor: TrustAnchorKeyV1) -> dict:
    return {
        "key_ref": anchor.key_ref,
        "algorithm": anchor.algorithm,
        "public_key_der_b64": anchor.public_key_der_b64,
        "public_key_digest": anchor.public_key_digest,
    }


def _base_context_dict(base: VerificationContext) -> dict:
    return {
        "expected_nonce": base.expected_nonce,
        "expected_build_session_ref": base.expected_build_session_ref,
        "expected_issuer_key_ref": base.expected_issuer_key_ref,
        "expected_issuer_algorithm": base.expected_issuer_algorithm,
        "expected_trust_ring_ref": base.expected_trust_ring_ref,
        "expected_project_ref": base.expected_project_ref,
        "expected_client_key_ref": base.expected_client_key_ref,
        "expected_client_algorithm": base.expected_client_algorithm,
        "client_public_key": base.client_public_key,
    }


def _process_record_context_dict(ctx) -> dict:
    d = {
        "expected_materials_digest": ctx.expected_materials_digest,
        "certificate_ref": ctx.certificate_ref,
        "base_context": _base_context_dict(ctx.base_context),
        "expected_project_ref": ctx.expected_project_ref,
        "expected_build_session_ref": ctx.expected_build_session_ref,
        "expected_agent_commitment_ref": ctx.expected_agent_commitment_ref,
        "expected_dataset_commitment_ref": ctx.expected_dataset_commitment_ref,
        "expected_evaluator_commitment_ref": ctx.expected_evaluator_commitment_ref,
        "expected_build_definition_commitment_ref": ctx.expected_build_definition_commitment_ref,
        "allow_unchecked_base_status": ctx.allow_unchecked_base_status,
        "verification_time": ctx.verification_time,
    }
    if ctx.trust_anchor is not None:
        d["trust_anchor"] = _anchor_dict(ctx.trust_anchor)
    return d


def _agent_quality_context_dict(ctx) -> dict:
    return {
        "process_record_context": _process_record_context_dict(ctx.process_record_context),
        "expected_project_ref": ctx.expected_project_ref,
        "expected_build_session_ref": ctx.expected_build_session_ref,
        "expected_agent_commitment_ref": ctx.expected_agent_commitment_ref,
        "expected_dataset_commitment_ref": ctx.expected_dataset_commitment_ref,
        "expected_evaluator_commitment_ref": ctx.expected_evaluator_commitment_ref,
        "expected_build_definition_commitment_ref": ctx.expected_build_definition_commitment_ref,
        "expected_measurement_contract_ref": ctx.expected_measurement_contract_ref,
        "expected_measurement_contract_record_digest": ctx.expected_measurement_contract_record_digest,
        "accept_abstained_bundle": ctx.accept_abstained_bundle,
        "expected_declared_plan_digest": ctx.expected_declared_plan_digest,
    }


def _write(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    # ---- golden bundle + unchecked context/process-record ----------------
    golden_bundle = av.build_agent_quality_bundle()
    golden_context = av.build_agent_quality_context()
    assert golden_context.process_record_context.allow_unchecked_base_status is True

    _write("agent_quality_bundle.json", golden_bundle)
    _write("agent_quality_process_record.json", av._GV_PROCESS_RECORD_BUNDLE)
    _write(
        "agent_quality_context_unchecked.json",
        _agent_quality_context_dict(golden_context),
    )

    # A dummy anchor -- unused by the unchecked path since the nested
    # process_record_context carries no trust_anchor field, but --anchor is
    # a required CLI flag regardless.
    dummy_anchor = pv._default_trust_anchor("ed25519")
    _write("agent_quality_anchor.json", _anchor_dict(dummy_anchor))

    # ---- checked-mode feasibility: measure it first -----------------------
    checked_pr_bundle, checked_pr_context, _ = pv._build_bundle(
        allow_unchecked_base_status=False
    )
    assert checked_pr_bundle["unsigned_manifest"] == av._GV_PROCESS_RECORD_BUNDLE["unsigned_manifest"], (
        "checked-mode process record's unsigned_manifest diverges from the "
        "golden agent-quality bundle's bound process record -- the checked "
        "golden cannot verify"
    )
    print(
        "CHECKED-MODE FEASIBILITY: checked_pr_bundle['unsigned_manifest'] == "
        "_GV_PROCESS_RECORD_BUNDLE['unsigned_manifest'] -> True"
    )

    trust_status_envelope, status_pr_context = pv._build_trust_status(
        checked_pr_bundle, checked_pr_context
    )
    checked_agent_context = av.build_agent_quality_context(
        process_record_context=status_pr_context
    )
    _write(
        "agent_quality_context_checked.json",
        _agent_quality_context_dict(checked_agent_context),
    )
    _write("agent_quality_trust_status_active.json", trust_status_envelope)
    _write("agent_quality_trust_anchor.json", _anchor_dict(status_pr_context.trust_anchor))

    # Prove the checked golden actually verifies end to end before shipping it.
    from traigent_schema.certification import verify_agent_quality_certificate

    checked_context_with_status = dataclasses.replace(
        checked_agent_context, trust_status=trust_status_envelope
    )
    checked_result = verify_agent_quality_certificate(
        golden_bundle,
        context=checked_context_with_status,
        process_record_bundle=checked_pr_bundle,
    )
    assert checked_result.code == "AGENT_QUALITY_VERIFIED"
    print("CHECKED GOLDEN VERIFIES: AGENT_QUALITY_VERIFIED -> True")

    # ---- abstained bundle ---------------------------------------------
    abstained_bundle = av.build_abstained_agent_quality_bundle()
    _write("agent_quality_abstained_bundle.json", abstained_bundle)

    # ---- tamper: signature byte flip -> ISSUER_SIGNATURE_INVALID -------
    tampered_sig_bundle = copy.deepcopy(golden_bundle)
    sig = bytearray(base64.b64decode(tampered_sig_bundle["signature"]["signature"]))
    sig[0] ^= 0xFF
    tampered_sig_bundle["signature"]["signature"] = base64.b64encode(bytes(sig)).decode("ascii")
    _write("agent_quality_bundle_tamper_signature.json", tampered_sig_bundle)

    # ---- tamper: context commitment ref mismatch (context-side, no resign) --
    other_sha = "sha256:" + "e" * 64
    assert other_sha != golden_context.expected_agent_commitment_ref
    commitment_mismatch_context = av.build_agent_quality_context(
        expected_agent_commitment_ref=other_sha
    )
    _write(
        "agent_quality_context_tamper_commitment_ref.json",
        _agent_quality_context_dict(commitment_mismatch_context),
    )

    # ---- tamper: context scope (build session ref) mismatch ------------
    other_build_session_ref = "bsn:" + "b" * 43
    assert other_build_session_ref != golden_context.expected_build_session_ref
    scope_mismatch_context = av.build_agent_quality_context(
        expected_build_session_ref=other_build_session_ref
    )
    _write(
        "agent_quality_context_tamper_scope.json",
        _agent_quality_context_dict(scope_mismatch_context),
    )

    # ---- tamper: measured-claim interval endpoint +1 ppm, re-signed -----
    plus_one_claims = copy.deepcopy(golden_bundle["measured_claims"])
    plus_one_claims[0]["interval_high"] += 1
    plus_one_bundle = av.build_agent_quality_bundle(measured_claims=plus_one_claims)
    _write("agent_quality_bundle_tamper_interval_plus1.json", plus_one_bundle)

    # ---- tamper: measured-claim interval endpoint -1 ppm, re-signed -----
    minus_one_claims = copy.deepcopy(golden_bundle["measured_claims"])
    minus_one_claims[0]["interval_high"] -= 1
    minus_one_bundle = av.build_agent_quality_bundle(measured_claims=minus_one_claims)
    _write("agent_quality_bundle_tamper_interval_minus1.json", minus_one_bundle)

    # ---- tamper: process record, one receipt digest flipped ------------
    tampered_pr_bundle = copy.deepcopy(av._GV_PROCESS_RECORD_BUNDLE)
    row = tampered_pr_bundle["report"]["rows"][0]
    assert row["status"] == "present"
    original_digest = row["receipt_digest"]
    row["receipt_digest"] = "sha256:" + "0" * 64
    assert row["receipt_digest"] != original_digest
    _write("agent_quality_process_record_tamper_receipt.json", tampered_pr_bundle)

    from traigent_schema.certification import (
        ProcessRecordVerificationError,
        verify_process_record_certificate,
    )

    try:
        verify_process_record_certificate(
            tampered_pr_bundle,
            context=av._GV_PROCESS_RECORD_CONTEXT,
            trust_status=None,
        )
        raise AssertionError("expected ProcessRecordVerificationError")
    except ProcessRecordVerificationError as exc:
        print(f"TAMPERED PROCESS RECORD DIRECT CODE: {exc.code}")

    print("done")


if __name__ == "__main__":
    main()
```
