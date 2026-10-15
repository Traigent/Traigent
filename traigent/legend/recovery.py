"""Conformance check, capture, verified restore and activation.

Everything here is local: bytes go to the customer's :class:`ClientStore`, and
only closed enums, counts and opaque refs go out through the transport.
``capture`` and ``restore`` never write into the live source tree; ``activate``
is a separate, explicit call.
"""

from __future__ import annotations

import hashlib
import io
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from traigent.identity.keys import ContentIdentityKeys
from traigent.legend import _contract as c
from traigent.legend.bundle import (
    pack_bundle,
    unpack_bundle,
    unpack_tree,
)
from traigent.legend.errors import LegendError
from traigent.legend.exporters import export, recompute_fingerprint
from traigent.legend.manifest import (
    build_recovery_manifest,
    identity_reproduced,
    manifest_fingerprint,
    pillar_representation_commitment,
)
from traigent.legend.payloads import (
    build_availability_payload,
    build_capture_payload,
    build_restore_payload,
)
from traigent.legend.store import BUNDLE_NAME, ClientStore
from traigent.legend.transport import CaptureTransport, with_retries

ConformanceStatus = Literal[
    "passed", "conformance_failed", "unreachable", "permission_denied"
]
Grade = Literal["exact", "partial", "substituted", "failed"]


@dataclass(frozen=True)
class ConformanceOutcome:
    status: ConformanceStatus
    kinds_failed: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _synthetic_key() -> str:
    """A conformance key in the reserved ``rmfc0:`` namespace, disjoint from manifest keys."""
    return (
        c.CONFORMANCE_KEY_PREFIX + hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    )


def _read_bundle_file(dest: Path) -> bytes | None:
    try:
        return (dest / BUNDLE_NAME).read_bytes()
    except OSError:
        return None


def conformance_check(store: ClientStore) -> ConformanceOutcome:
    """Per kind: put a synthetic artifact, get it into a fresh temp dir, re-hash, compare.

    Also checks that re-putting different bytes under the same key does not succeed
    or change what ``get`` returns. The store has no delete, so each check leaves
    one small synthetic entry behind under a random key.
    """
    failed: list[str] = []
    worst: ConformanceStatus = "conformance_failed"
    for kind in sorted(c.ENTITY_KINDS):
        key = _synthetic_key()
        payload = pack_bundle({"synthetic": True}, [secrets.token_bytes(2048)])
        other = pack_bundle({"synthetic": True}, [secrets.token_bytes(2048)])
        scratch = Path(tempfile.mkdtemp(prefix="legend-conf-"))
        try:
            status = store.put(kind, key, io.BytesIO(payload))
            if status in ("unreachable", "permission_denied"):
                worst = status
                failed.append(kind)
                continue
            if status != "stored_unverified" or store.has(kind, key) != "available":
                failed.append(kind)
                continue
            # "remove the local copy": nothing local remains; only the store holds the bytes.
            if store.get(kind, key, scratch) != "retrieved":
                failed.append(kind)
                continue
            got = _read_bundle_file(scratch)
            if (
                got is None
                or hashlib.sha256(got).digest() != hashlib.sha256(payload).digest()
                or got != payload
            ):
                failed.append(kind)
                continue
            again: str
            try:
                again = store.put(kind, key, io.BytesIO(other))
            except LegendError:
                again = "refused"
            if again == "stored_unverified":
                failed.append(kind)
                continue
            fresh = Path(tempfile.mkdtemp(prefix="legend-conf-"))
            try:
                if (
                    store.get(kind, key, fresh) != "retrieved"
                    or _read_bundle_file(fresh) != payload
                ):
                    failed.append(kind)
            finally:
                shutil.rmtree(fresh, ignore_errors=True)
        except Exception:  # noqa: BLE001 - any adapter fault is a closed failure, text dropped
            failed.append(kind)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    if not failed:
        return ConformanceOutcome("passed")
    return ConformanceOutcome(worst, tuple(failed))


@dataclass
class CaptureIndex:
    """Local map ``version_ref -> [manifest fingerprints]`` (newest last). Names stay local."""

    entries: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    source_roots: dict[str, Path] = field(default_factory=dict)

    def add(self, version_ref: str, kind: str, fp: str, root: Path | None) -> None:
        self.entries.setdefault(version_ref, []).append((kind, fp))
        if root is not None:
            self.source_roots[fp] = root


DEFAULT_INDEX = CaptureIndex()


@dataclass(frozen=True)
class CaptureRecord:
    version_ref: str
    kind: str
    manifest_fingerprint: str
    recovery_state: str
    coverage: dict[str, Any]
    conformance_passed: bool
    event_ref: str | None
    backend_keyed: bool = False
    representation_commitment: str | None = None
    identity_digests_reproduced: bool = False
    identity_verification_ref: str | None = None


def _coverage(manifest: dict[str, Any]) -> dict[str, Any]:
    cov = manifest["coverage"]
    return {
        "status": cov["status"],
        "covered": list(cov["covered"]),
        "missing": list(cov["missing"]),
    }


def _verify_bundle(
    data: bytes,
    kind: str,
    fp: str,
    keys: ContentIdentityKeys | None,
) -> tuple[dict[str, Any] | None, bool, bool, bool, Any]:
    """Return ``(manifest, bytes_ok, digests_match, pillar_match, bundle)``; manifest None => unusable."""
    try:
        bundle = unpack_bundle(data)
    except LegendError:
        return None, False, False, False, None
    manifest = bundle.header.get("manifest")
    inputs = bundle.header.get("pillar_inputs", {})
    if (
        not isinstance(manifest, dict)
        or not isinstance(inputs, dict)
        or manifest.get("kind") != kind
    ):
        return None, False, False, False, None
    try:
        if manifest_fingerprint(manifest) != fp:
            return None, False, False, False, None
        declared = manifest["representations"]
        match = len(declared) == len(bundle.representations) and all(
            d["ordinal"] == i
            and d["byte_length"] == len(rep)
            and d["digest"] == "sha256:" + hashlib.sha256(rep).hexdigest()
            for i, (d, rep) in enumerate(
                zip(declared, bundle.representations, strict=False)
            )
        )
        _ = manifest["pillar_fingerprint"], manifest["coverage"]["status"]
    except (KeyError, TypeError, ValueError):
        return None, False, False, False, None
    pillar_match = False
    if match:
        kinds = [d["representation_kind"] for d in declared]
        got = recompute_fingerprint(kind, kinds, bundle.representations, inputs, keys)
        pillar_match = got is not None and got == manifest["pillar_fingerprint"]
    return manifest, True, match, pillar_match, bundle


def _post(transport: Any, method: str, sealed: dict[str, Any]) -> str | None:
    """Send one ALREADY SEALED payload (fresh ``idk_`` key minted once; every retry reuses it)."""
    fn: Callable[[dict[str, Any]], str] | None = getattr(transport, method, None)
    if transport is None or fn is None:
        return None
    return with_retries(lambda: fn(sealed))


def _tree_digest(root: Path) -> str:
    """Abstract digest of a staged directory: sorted (relative path, length, bytes) framing."""
    h = hashlib.sha256(b"traigent.legend.staged_artifact.v0\x00")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        body = path.read_bytes()
        h.update(
            len(rel).to_bytes(8, "big") + rel + len(body).to_bytes(8, "big") + body
        )
    return "sha256:" + h.hexdigest()


def capture(
    version_ref: str,
    source: Any,
    *,
    store: ClientStore,
    transport: CaptureTransport | None = None,
    dependencies: Sequence[tuple[str, str]] = (),
    conformance: ConformanceOutcome | None = None,
    keys: ContentIdentityKeys | None = None,
    index: CaptureIndex | None = None,
    source_entity_ref: str | None = None,
    identity_verification_ref: str | None = None,
) -> CaptureRecord:
    """Export real content, store it under its manifest fingerprint and verify the round trip.

    Requires a passing conformance check (run here unless one is supplied). A put
    acknowledgement alone is ``stored_unverified``; a per-capture isolated get plus
    re-hash of representations and the pillar fingerprint makes it ``round_trip_verified``.

    A backend-keyed pillar (search space, evaluator) is only ``complete`` when
    ``identity_verification_ref`` cites the Backend's pre-capture ``identity_verified``
    event for this version; get the commitment that proof must bind from
    :func:`traigent.legend.manifest.pillar_representation_commitment` over
    ``export(source)[0]``. Without it the capture is recorded as ``partial``.
    """
    if not c.VERSION_REF_RE.fullmatch(version_ref or ""):
        raise LegendError(
            "invalid_request", "version_ref must be an opaque version ref"
        )
    pillar, inputs = export(source)
    manifest = build_recovery_manifest(
        pillar,
        dependencies=dependencies,
        source_entity_ref=source_entity_ref,
        identity_verification_ref=identity_verification_ref,
    )
    outcome = conformance if conformance is not None else conformance_check(store)
    if not outcome.passed:
        raise LegendError(
            "conformance_failed", "store did not pass the conformance check"
        )
    fp = manifest.fingerprint
    data = pack_bundle(
        {"manifest": manifest.document, "pillar_inputs": inputs},
        [rep.data for rep in pillar.representations],
    )
    status = store.put(pillar.kind, fp, io.BytesIO(data))
    if status == "store_key_conflict":
        raise LegendError("store_key_conflict", "store key already holds other bytes")
    if status != "stored_unverified":
        raise LegendError("store_unavailable", "store did not accept the capture")
    state = c.RECOVERY_STORED_UNVERIFIED
    scratch = Path(tempfile.mkdtemp(prefix="legend-roundtrip-"))
    try:
        if store.get(pillar.kind, fp, scratch) == "retrieved":
            back = _read_bundle_file(scratch)
            if back is not None:
                m, _ok, digests, pillar_ok, _b = _verify_bundle(
                    back, pillar.kind, fp, keys
                )
                if m is not None and digests and pillar_ok and back == data:
                    state = c.RECOVERY_ROUND_TRIP_VERIFIED
    except Exception:  # noqa: BLE001 - a failing round trip leaves the capture unverified
        state = c.RECOVERY_STORED_UNVERIFIED
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    coverage = _coverage(manifest.document)
    commitment = pillar_representation_commitment(pillar)
    reproduced = identity_reproduced(pillar, identity_verification_ref)
    payload = build_capture_payload(
        version_ref=version_ref,
        kind=pillar.kind,
        manifest_fingerprint=fp,
        coverage=coverage,
        recovery_state=state,
        conformance_passed=True,
        exporter_version=c.EXPORTER_VERSION,
        backend_keyed=pillar.backend_keyed,
        identity_verification_ref=identity_verification_ref,
        representation_commitment=commitment,
        identity_digests_reproduced=reproduced,
    )
    event_ref = _post(transport, "record_capture", payload)
    root = getattr(source, "root", None)
    (index or DEFAULT_INDEX).add(
        version_ref, pillar.kind, fp, Path(root).resolve() if root is not None else None
    )
    return CaptureRecord(
        version_ref,
        pillar.kind,
        fp,
        state,
        coverage,
        True,
        event_ref,
        pillar.backend_keyed,
        commitment,
        reproduced,
        identity_verification_ref,
    )


@dataclass(frozen=True)
class RestoreResult:
    grade: Grade
    kind: str
    manifest_fingerprint: str
    version_ref: str | None
    dest: Path | None
    coverage: dict[str, Any] | None
    representation_digests_match: bool
    pillar_fingerprint_match: bool
    dependencies_satisfied: bool
    store_status: str
    activated: bool = False
    event_ref: str | None = None
    backend_keyed: bool = False
    historical_key_available: bool = False
    verified_artifact_digest: str | None = None


def _historical_key_available(
    manifest: dict[str, Any], backend_keyed: bool, keys: ContentIdentityKeys | None
) -> bool:
    """Is the key the fingerprint was minted under held here? Backend-only keys never are."""
    if backend_keyed:
        return False
    key_id = manifest.get("key_id")
    if key_id is None:
        return True
    from traigent.identity.keys import get_content_identity_keys

    grant = keys if keys is not None else get_content_identity_keys()
    return grant is not None and grant.kid == key_id


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _prepare_dest(dest: Path, guarded: Sequence[Path]) -> Path:
    dest = Path(os.path.abspath(dest))
    real = Path(os.path.realpath(dest))
    for root in guarded:
        if _is_within(real, root) or _is_within(root, real):
            raise LegendError(
                "invalid_request", "restore destination overlaps the source tree"
            )
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise LegendError("invalid_request", "restore destination must be new or empty")
    return dest


def _materialize(
    dest: Path, kind: str, kinds: list[str], reps: tuple[bytes, ...]
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for i, (rkind, data) in enumerate(zip(kinds, reps, strict=False)):
        if rkind == c.REPR_SOURCE_BUNDLE or rkind == c.REPR_EVALUATOR_BUNDLE:
            base = dest / ("tree" if rkind == c.REPR_SOURCE_BUNDLE else "evaluator")
            for path, executable, body in unpack_tree(data) if data else []:
                target = base / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                if executable:
                    target.chmod(0o755)
        else:
            (dest / f"{i:02d}-{rkind}").write_bytes(data)


def _resolve(
    target: Any, kind: str | None, index: CaptureIndex
) -> tuple[str, str, str | None]:
    if isinstance(target, CaptureRecord):
        return target.kind, target.manifest_fingerprint, target.version_ref
    if isinstance(target, str) and c.VERSION_REF_RE.fullmatch(target):
        history = index.entries.get(target)
        if not history:
            raise LegendError("unknown_ref", "no local capture for this version")
        k, fp = history[-1]
        return k, fp, target
    if isinstance(target, str) and target.startswith(c.MANIFEST_FINGERPRINT_PREFIX):
        if kind is None:
            raise LegendError(
                "invalid_request", "a manifest fingerprint needs its kind"
            )
        return kind, target, None
    raise LegendError(
        "invalid_request",
        "restore target must be a capture, version ref or manifest fingerprint",
    )


def restore(
    target: Any,
    dest: str | os.PathLike[str],
    *,
    store: ClientStore,
    kind: str | None = None,
    dependency_present: Callable[[str, str], bool] | None = None,
    keys: ContentIdentityKeys | None = None,
    transport: CaptureTransport | None = None,
    index: CaptureIndex | None = None,
) -> RestoreResult | list[RestoreResult]:
    """Restore into a NEW isolated directory and grade the result; never touches current state.

    ``target`` is a :class:`CaptureRecord` (a step), a version ref (newest local capture),
    a manifest fingerprint with ``kind``, or a list of those (each lands in ``dest/<n>``).
    Grade precedence: failed > partial > substituted > exact.
    """
    idx = index or DEFAULT_INDEX
    if isinstance(target, (list, tuple)):
        base = Path(dest)
        return [
            restore(
                t,
                base / str(n),
                store=store,
                kind=kind,
                dependency_present=dependency_present,
                keys=keys,
                transport=transport,
                index=idx,
            )  # type: ignore[misc]
            for n, t in enumerate(target)
        ]
    rkind, fp, version_ref = _resolve(target, kind, idx)
    if rkind not in c.ENTITY_KINDS:
        raise LegendError("invalid_request", "unknown pillar kind")
    guarded = list(idx.source_roots.values())
    dest_path = _prepare_dest(Path(dest), [Path(os.path.realpath(p)) for p in guarded])

    def failed(status: str, manifest: dict[str, Any] | None = None) -> RestoreResult:
        return RestoreResult(
            "failed",
            rkind,
            fp,
            version_ref,
            None,
            _coverage(manifest) if manifest else None,
            False,
            False,
            False,
            status,
        )

    scratch = Path(tempfile.mkdtemp(prefix="legend-restore-"))
    try:
        try:
            status = store.get(rkind, fp, scratch)
        except Exception:  # noqa: BLE001 - adapter fault => closed failure
            return failed("unreachable")
        if status != "retrieved":
            return failed(status)
        data = _read_bundle_file(scratch)
        if data is None:
            return failed("missing")
        manifest, _ok, digests, pillar_ok, bundle = _verify_bundle(
            data, rkind, fp, keys
        )
        if manifest is None:
            return failed("corrupted")
        deps = manifest.get("dependencies", [])
        deps_ok = all(
            dependency_present is not None
            and dependency_present(d["version_ref"], d["kind"])
            for d in deps
        )
        coverage = _coverage(manifest)
        backend_keyed = rkind in c.BACKEND_KEYED_KINDS
        key_ok = _historical_key_available(manifest, backend_keyed, keys)
        if not digests:
            grade: Grade = "failed"
        elif coverage["status"] != "complete" or not deps_ok:
            grade = "partial"
        elif not pillar_ok:
            grade = "substituted"
        elif backend_keyed:
            # INV-16: a Backend-keyed fingerprint cannot be re-verified client-side;
            # only a server-origin match report may say ``exact``.
            grade = "substituted"
        elif not key_ok:
            grade = "substituted"
        else:
            grade = "exact"
        out_dest: Path | None = None
        if grade != "failed":
            kinds = [d["representation_kind"] for d in manifest["representations"]]
            _materialize(dest_path, rkind, kinds, bundle.representations)
            out_dest = dest_path
        # the abstract digest of the verified staged artifact; recorded only for ``exact``
        artifact = (
            _tree_digest(out_dest)
            if grade == "exact" and out_dest is not None
            else None
        )
        event_ref = None
        if version_ref is not None:
            event_ref = _post(
                transport,
                "record_restore",
                build_restore_payload(
                    version_ref=version_ref,
                    kind=rkind,
                    manifest_fingerprint=fp,
                    outcome=grade,
                    coverage=coverage,
                    isolated_destination=True,
                    representation_digests_match=digests,
                    pillar_fingerprint_match=pillar_ok,
                    historical_key_available=key_ok,
                    dependencies_verified=deps_ok,
                    verified_artifact_digest=artifact,
                    backend_keyed=backend_keyed,
                ),
            )
        return RestoreResult(
            grade,
            rkind,
            fp,
            version_ref,
            out_dest,
            coverage,
            digests,
            pillar_ok,
            deps_ok,
            "retrieved",
            False,
            event_ref,
            backend_keyed,
            key_ok,
            artifact,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@dataclass(frozen=True)
class ActivationResult:
    target: Path
    grade: Grade  # the restore's grade, never upgraded
    accepted_inexact: bool


def activate(
    result: RestoreResult,
    target: str | os.PathLike[str],
    *,
    accept: Literal["partial", "substituted"] | None = None,
    overwrite: bool = False,
) -> ActivationResult:
    """Copy a restored directory into ``target``. Requires ``exact``, or ``accept`` naming the exact grade.

    A ``failed`` restore is never activatable. An accepted partial / substituted
    activation keeps its own grade; it is never reported as ``exact``.
    """
    if result.dest is None or result.grade == "failed":
        raise LegendError("restore_failed", "nothing verified to activate")
    inexact = result.grade != "exact"
    if inexact and accept != result.grade:
        raise LegendError(
            "coverage_insufficient",
            "activation of a non-exact restore needs explicit accept",
        )
    if (
        result.grade == "exact"
        and result.verified_artifact_digest is not None
        and _tree_digest(result.dest) != result.verified_artifact_digest
    ):
        # the staged artifact is not the one that was verified and recorded
        raise LegendError(
            "manifest_mismatch", "staged artifact differs from the verified one"
        )
    out = Path(os.path.abspath(target))
    if out.exists() and not overwrite and (not out.is_dir() or any(out.iterdir())):
        raise LegendError("invalid_request", "activation target must be new or empty")
    shutil.copytree(result.dest, out, dirs_exist_ok=True)
    return ActivationResult(out, result.grade, inexact)


def record_availability(
    version_ref: str,
    manifest_fingerprint: str,
    status: str,
    *,
    transport: CaptureTransport | None = None,
    idempotency_key: str | None = None,
) -> str | None:
    """Report the CURRENT availability of one capture (``available | missing | corrupted | retired``).

    Sends ``RecordAvailabilityRequestV0`` (version ref, manifest fingerprint, closed
    status). The highest ledger position per manifest wins server-side; the report is
    never inferred from a restore. Returns the event ref, or ``None`` without a transport.
    """
    payload = build_availability_payload(
        version_ref, manifest_fingerprint, status, idempotency_key
    )
    return _post(transport, "record_availability", payload)
