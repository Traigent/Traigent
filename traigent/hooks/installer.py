"""Git hooks installer for Traigent.

Installs and manages Git hooks that validate agent configurations
before pushes.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeGuard

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Pre-push hook script content
PRE_PUSH_HOOK_SCRIPT = """#!/bin/bash
# Traigent pre-push hook - validates agent configurations
# Installed by: traigent hooks install

set -e

# Check if traigent is available
if ! command -v traigent &> /dev/null; then
    # Try Python module invocation
    if ! python -m traigent --version &> /dev/null; then
        echo "Warning: traigent CLI not found, skipping validation"
        exit 0
    fi
    TRAIGENT_CMD="python -m traigent"
else
    TRAIGENT_CMD="traigent"
fi

echo "[traigent-validate] Checking agent configurations..."

# Run Traigent validation
if ! $TRAIGENT_CMD hooks validate --exit-code; then
    echo ""
    echo "PUSH REJECTED: Agent configuration violates constraints"
    echo ""
    echo "To bypass (not recommended):"
    echo "  git push --no-verify"
    echo ""
    echo "To fix:"
    echo "  Review the errors above and update your agent configuration"
    exit 1
fi

echo ""
echo "All Traigent hooks passed!"
exit 0
"""

# Pre-commit hook script content (lighter validation)
PRE_COMMIT_HOOK_SCRIPT = """#!/bin/bash
# Traigent pre-commit hook - quick config check
# Installed by: traigent hooks install

set -e

# Check if traigent is available
if ! command -v traigent &> /dev/null; then
    if ! python -m traigent --version &> /dev/null; then
        exit 0  # Skip if traigent not available
    fi
    TRAIGENT_CMD="python -m traigent"
else
    TRAIGENT_CMD="traigent"
fi

# Quick syntax check only (no full validation)
$TRAIGENT_CMD hooks check --quick
"""

# Exported for import compatibility only. This string used to decide whether a
# hook was Traigent's; it no longer does, and nothing in the SDK reads it.
# Ownership is settled by _classify below -- exact template bytes, a known
# historical digest, or the install-time record -- because a marker cannot tell
# a hook Traigent wrote from one a customer copied the header into.
HOOK_MARKER = "# Traigent pre-push hook - validates agent configurations"

# Generated artifacts this SDK has shipped that are NOT the current template,
# extracted with git show + ast.literal_eval.
# 45c067ff, 0320568b, c59cada2, c517a91f, 729eea82 ship what is now the current
# pair; 26902cf5, 1d8ca986, f6be5a66, 7489260a ship the earlier one.
#
# Both the CRLF derivations are listed, including the CRLF form of the CURRENT
# template: historical writers used text mode, where Python translates "\n" to
# os.linesep, so a Windows install really did emit those bytes. They are ours
# to recognise even though their LF original is what we write today. The LF
# digests of the current pair are deliberately absent -- they live in
# _CURRENT_TEMPLATES, which _classify consults first, and duplicating them here
# would make "current" and "superseded" overlap.
_LEGACY_DIGESTS = {
    "pre-push": frozenset(
        {
            "b9d8398357c53b8b128ba0ca6d275a804cd0f674f94037a84333f673605d5b72",
            "bdf3be4f4ae44aa15100ecc6a3409b448ea442b1a048bb5816c0fc0d5a8a069c",
            "5ea26895d95b571ca10ee0625d0e7b44f1d312f378d92cd0c80255cd4dd2b092",
        }
    ),
    "pre-commit": frozenset(
        {
            "9ddbae8ca0016f02820662bb5a18af416c938db7fab74614f5b98c515ab4ef18",
            "105f68f2146a23e5a0cfae9ad5459d9721f55e1cd2127a66c26e7fb5e07c96e9",
            "da32c293794c1a030111120ab5a85950c8268bf61bd5ff65673f123a9db96c82",
        }
    ),
}
_CURRENT_TEMPLATES = {
    "pre-push": PRE_PUSH_HOOK_SCRIPT.encode("utf-8"),
    "pre-commit": PRE_COMMIT_HOOK_SCRIPT.encode("utf-8"),
}

# --- Generated-artifact ownership -------------------------------------------
#
# Bytes cannot answer "did Traigent write this file?". An artifact the SDK
# generated leaves the known set whenever a template changes or a text-mode
# writer rewrites its line endings, and the content classifier then calls it
# foreign. Force-install files it away as the customer's original, uninstall
# hands it back, and the customer is left with a fail-closed hook that blocks
# `git commit` after the SDK is gone.
#
# So install writes down the digest of every artifact it publishes and of any
# original it preserves; status and uninstall read that record. Ownership
# becomes a fact the installer recorded rather than an inference from content:
# the SDK removes exactly what it generated, and a hook it did not generate is
# neither deleted nor resurrected.
#
# The ledger lives beside the hooks directory, inside .git, so it is never
# committed and never mistaken for a hook.
_LEDGER_NAME = "traigent-hooks.json"
_LEDGER_VERSION = 1
# One digest per template the SDK has published into this repository; a cap
# keeps the file bounded if a repository survives many upgrades. Dropping the
# oldest only ever costs recognition of an artifact long replaced, which falls
# back to the content classifier.
_MAX_GENERATED = 32
_BACKUP_ORIGIN_FOREIGN = "foreign"
_DIGEST_CHARS = frozenset("0123456789abcdef")


def _is_digest(value: object) -> TypeGuard[str]:
    """A lowercase hex SHA-256, the only shape the ledger ever stores or trusts."""
    return (
        isinstance(value, str) and len(value) == 64 and _DIGEST_CHARS.issuperset(value)
    )


@dataclass(frozen=True)
class _HookRecord:
    """What the installer knows it did to one hook name."""

    # Digests of artifacts this installer published here, newest last.
    generated: tuple[str, ...] = ()
    # Digest of the customer's own hook that install moved to <hook>.backup.
    backup_digest: str = ""
    backup_origin: str = ""

    def preserved_original(self) -> str:
        """The digest of a customer hook we are holding, or "" if none."""
        if self.backup_origin == _BACKUP_ORIGIN_FOREIGN and _is_digest(
            self.backup_digest
        ):
            return self.backup_digest
        return ""

    def is_empty(self) -> bool:
        return not self.generated and not self.preserved_original()


class _Ledger:
    """Durable ownership record for the artifacts this installer generated.

    Best effort by design. A ledger that is missing, unreadable or malformed
    degrades to exactly the content-only classification the SDK shipped before
    it existed, with a warning, and never blocks an install or an uninstall. It
    is never guessed at or back-filled: an absent record means "unknown", which
    is the conservative answer everywhere it is consulted.

    Trust boundary: the ledger is as trusted as the ``.git`` directory holding
    it. A writer who can forge an entry -- naming a customer hook's digest as
    one of ours, so uninstall removes it -- can already overwrite
    ``.git/hooks/*`` outright, so this grants no new capability and is not a
    privilege boundary. What is enforced is shape, not provenance: only a
    version-1 document is read, only the two known hook names are kept, and
    only lowercase hex SHA-256 digests are accepted, so a malformed or
    partially written file degrades rather than being half-believed.
    """

    def __init__(self, path: Path, records: dict[str, _HookRecord]) -> None:
        self.path = path
        self._records = records

    @classmethod
    def load(cls, path: Path) -> _Ledger:
        try:
            # O_NOFOLLOW for the same reason _read_slot uses it: the record is
            # read with no more trust than a hook, so a symlink planted at this
            # name is refused rather than followed to whatever it points at.
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            )
            fd = os.open(path, flags)
            try:
                chunks = []
                while chunk := os.read(fd, 65536):
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(fd)
        except FileNotFoundError:
            return cls(path, {})
        except OSError as exc:
            logger.warning(
                f"Could not read the Traigent hook ownership record {path} "
                f"({exc.strerror or type(exc).__name__}); falling back to "
                "content-only hook classification for this operation"
            )
            return cls(path, {})
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            logger.warning(
                f"The Traigent hook ownership record {path} is not readable JSON; "
                "falling back to content-only hook classification for this "
                "operation. Remove the file to start a fresh record."
            )
            return cls(path, {})
        return cls(path, cls._parse(path, document))

    @staticmethod
    def _parse(path: Path, document: object) -> dict[str, _HookRecord]:
        if (
            not isinstance(document, dict)
            or document.get("version") != _LEDGER_VERSION
            or not isinstance(document.get("hooks"), dict)
        ):
            logger.warning(
                f"The Traigent hook ownership record {path} is not a version "
                f"{_LEDGER_VERSION} record; falling back to content-only hook "
                "classification for this operation."
            )
            return {}
        records: dict[str, _HookRecord] = {}
        for name, entry in document["hooks"].items():
            if name not in _CURRENT_TEMPLATES or not isinstance(entry, dict):
                continue
            raw_generated = entry.get("generated")
            generated = (
                tuple(d for d in raw_generated if _is_digest(d))
                if isinstance(raw_generated, list)
                else ()
            )
            backup = entry.get("backup")
            digest, origin = "", ""
            if isinstance(backup, dict):
                candidate = backup.get("sha256")
                if _is_digest(candidate) and backup.get("origin") == (
                    _BACKUP_ORIGIN_FOREIGN
                ):
                    digest, origin = candidate, _BACKUP_ORIGIN_FOREIGN
            record = _HookRecord(generated[-_MAX_GENERATED:], digest, origin)
            if not record.is_empty():
                records[name] = record
        return records

    def record(self, hook_name: str) -> _HookRecord:
        return self._records.get(hook_name, _HookRecord())

    def generated(self, hook_name: str) -> tuple[str, ...]:
        return self.record(hook_name).generated

    def preserved_original(self, hook_name: str) -> str:
        return self.record(hook_name).preserved_original()

    def holds_preserved_original(self, hook_name: str, slot: _Slot) -> bool:
        """True when <hook>.backup still holds the customer hook we saved."""
        preserved = self.preserved_original(hook_name)
        return bool(preserved) and slot.kind == "regular" and slot.digest == preserved

    def remember_generated(self, hook_name: str, digest: str) -> None:
        if not _is_digest(digest):
            return
        current = self.record(hook_name)
        if current.generated and current.generated[-1] == digest:
            return
        kept = tuple(d for d in current.generated if d != digest) + (digest,)
        self._set(hook_name, replace(current, generated=kept[-_MAX_GENERATED:]))

    def remember_preserved_original(self, hook_name: str, digest: str) -> None:
        if not _is_digest(digest):
            # Without a digest the record could not be checked later, and an
            # unverifiable claim of ownership is worse than none.
            logger.warning(
                f"Could not record which hook Traigent saved as "
                f"{hook_name}.backup; it will be treated as a customer hook."
            )
            return
        self._set(
            hook_name,
            replace(
                self.record(hook_name),
                backup_digest=digest,
                backup_origin=_BACKUP_ORIGIN_FOREIGN,
            ),
        )

    def forget_preserved_original(self, hook_name: str) -> None:
        self._set(
            hook_name,
            replace(self.record(hook_name), backup_digest="", backup_origin=""),
        )

    def forget(self, hook_name: str) -> None:
        if hook_name in self._records:
            del self._records[hook_name]
            self._save()

    def _set(self, hook_name: str, record: _HookRecord) -> None:
        if record.is_empty():
            self._records.pop(hook_name, None)
        else:
            self._records[hook_name] = record
        self._save()

    def _document(self) -> dict[str, object]:
        hooks: dict[str, object] = {}
        for name, record in sorted(self._records.items()):
            entry: dict[str, object] = {"generated": list(record.generated)}
            preserved = record.preserved_original()
            if preserved:
                entry["backup"] = {
                    "sha256": preserved,
                    "origin": _BACKUP_ORIGIN_FOREIGN,
                }
            hooks[name] = entry
        return {"version": _LEDGER_VERSION, "hooks": hooks}

    def _save(self) -> None:
        try:
            if not self._records:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                return
            # Named apart from _install_hook's `payload`, which is hook bytes:
            # nothing from this record is ever published as a hook.
            serialized = json.dumps(self._document(), indent=2, sort_keys=True) + "\n"
            fd, name = tempfile.mkstemp(
                dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
            )
            staged = Path(name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                    # Through the descriptor, as _stage_hook does: the mode is
                    # set on the object written, never on whatever the name
                    # happens to point at by then. mkstemp already creates 0600;
                    # stating it keeps the record private if that ever changes.
                    if hasattr(os, "fchmod"):
                        os.fchmod(handle.fileno(), 0o600)
                    else:
                        os.chmod(staged, 0o600)
                os.replace(staged, self.path)
            except BaseException:
                try:
                    staged.unlink()
                except OSError:
                    pass
                raise
        except OSError as exc:
            # The hooks themselves are the contract; a ledger write that fails
            # must not fail the install. Say so loudly instead: the next run
            # falls back to content-only classification.
            logger.warning(
                f"Could not update the Traigent hook ownership record "
                f"{self.path} ({exc.strerror or type(exc).__name__}). The hooks "
                "are in place, but a later `traigent hooks install --force` may "
                "not recognise them as Traigent's own."
            )


@dataclass(frozen=True)
class _Slot:
    """One closed-descriptor snapshot; never display customer hook contents."""

    kind: str
    dev: int = 0
    ino: int = 0
    mode: int = 0
    size: int = 0
    data: bytes = field(default=b"", repr=False)
    digest: str = ""
    error: str = ""


def _read_slot(path: Path) -> _Slot:
    try:
        initial = os.lstat(path)
    except FileNotFoundError:
        return _Slot("absent")
    except OSError as exc:
        return _Slot("error", error=exc.strerror or type(exc).__name__)
    if not stat.S_ISREG(initial.st_mode):
        return _Slot("other", initial.st_dev, initial.st_ino, initial.st_mode)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
                initial.st_dev,
                initial.st_ino,
            ):
                return _Slot("error", error="changed while reading")
            chunks = []
            while chunk := os.read(fd, 65536):
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
            size_mismatch = len(data) != after.st_size
            if (
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or size_mismatch:
                return _Slot("error", error="changed while reading")
        finally:
            os.close(fd)
    except OSError as exc:
        return _Slot("error", error=exc.strerror or type(exc).__name__)
    return _Slot(
        "regular",
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        data,
        hashlib.sha256(data).hexdigest(),
    )


def _slot_key(slot: _Slot) -> tuple[str, int, int, int, int, int, str]:
    return (
        slot.kind,
        slot.dev,
        slot.ino,
        stat.S_IFMT(slot.mode),
        stat.S_IMODE(slot.mode),
        slot.size,
        slot.digest,
    )


def _same_inode(first: _Slot, second: _Slot) -> bool:
    return first.kind == second.kind == "regular" and (first.dev, first.ino) == (
        second.dev,
        second.ino,
    )


def _guard_slot(path: Path, expected: _Slot) -> _Slot:
    actual = _read_slot(path)
    if _slot_key(actual) != _slot_key(expected):
        raise RuntimeError(f"{path} changed while Traigent was updating it; stopped")
    return actual


def _classify(hook_name: str, slot: _Slot, generated: tuple[str, ...] = ()) -> str:
    """Classify a slot: absent, unsafe, Traigent's own, or the customer's.

    ``generated`` carries the digests the ownership ledger records this
    installer as having published for ``hook_name``. A match makes the slot
    ours (``Og``) even when its bytes are outside the known template set --
    that is the whole point of the ledger. Everything else is unchanged.
    """
    if slot.kind == "absent":
        return "A"
    if slot.kind != "regular":
        return "X"
    if slot.data == _CURRENT_TEMPLATES.get(hook_name):
        return "Oc"
    if slot.digest in _LEGACY_DIGESTS.get(hook_name, ()):
        return "Ol"
    if slot.digest in generated:
        return "Og"
    return "F"


def _is_owned(kind: str) -> bool:
    """True for a slot holding an artifact this SDK generated."""
    return kind in {"Oc", "Ol", "Og"}


def _remove_staged(path: Path, expected: _Slot) -> None:
    try:
        actual = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISREG(actual.st_mode) and (actual.st_dev, actual.st_ino) == (
        expected.dev,
        expected.ino,
    ):
        os.unlink(path)
    else:
        logger.warning(
            f"Staged path {path} changed; left in place for manual inspection"
        )


def _stage_hook(hook_path: Path, data: bytes) -> tuple[Path, _Slot]:
    fd, name = tempfile.mkstemp(
        dir=hook_path.parent, prefix=f".{hook_path.name}.traigent-", suffix=".tmp"
    )
    path = Path(name)
    expected: _Slot | None = None
    try:
        try:
            identity = os.fstat(fd)
            expected = _Slot(
                "regular", identity.st_dev, identity.st_ino, identity.st_mode
            )
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Could not finish writing the staged Traigent hook")
                view = view[written:]
            os.fsync(fd)
            # Git will not run a hook that is not executable, and the install and
            # status predicates below treat a hook missing any of the 0o111 bits as
            # not installed, so group and other execution are both required: a
            # narrower 0o700 or 0o750 would make a hook we just wrote report as
            # uninstalled.
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o755)  # nosec B103 - see the hook-mode note above
            else:
                os.chmod(path, 0o755)  # nosec B103 - see the hook-mode note above
            final = os.fstat(fd)
            expected = _Slot(
                "regular",
                final.st_dev,
                final.st_ino,
                final.st_mode,
                final.st_size,
                data,
                hashlib.sha256(data).hexdigest(),
            )
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                logger.warning(
                    f"Could not close staged hook {path}; inspect it manually"
                )
            raise
        else:
            os.close(fd)
        _guard_slot(path, expected)
        return path, expected
    except BaseException:
        if expected is None:
            logger.warning(
                f"Could not verify staged hook {path}; left for manual inspection"
            )
        else:
            try:
                _remove_staged(path, expected)
            except OSError:
                logger.warning(
                    f"Could not clean staged hook {path}; inspect it manually"
                )
        raise


def _link_without_overwrite(source: Path, destination: Path) -> None:
    try:
        link = getattr(os, "link", None)
        if link is None:
            raise NotImplementedError("hard links are unavailable")
        link(source, destination)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{destination} was created by another program during install; "
            "left unchanged. Inspect the hook and backup before retrying."
        ) from exc
    except (OSError, NotImplementedError) as exc:
        detail = (
            (exc.strerror or type(exc).__name__)
            if isinstance(exc, OSError)
            else str(exc)
        )
        raise RuntimeError(
            f"Could not create a hard link in {destination.parent} ({detail}). "
            "Traigent needs hard links to add or save a hook without overwriting. "
            "Check filesystem permissions and hard-link support, or move the hook "
            "and backup aside manually before retrying."
        ) from exc


def _unlink_owned(path: Path, expected: _Slot) -> None:
    _guard_slot(path, expected)
    os.unlink(path)
    _guard_slot(path, _Slot("absent"))


def _rollback_original(active: Path, backup: Path, original: _Slot) -> bool:
    """Undo a backup this install created. True only if the backup is gone.

    The caller uses the answer to decide whether it may also drop the ledger's
    record of that backup. Forgetting a record while the file it describes is
    still on disk would leave the customer's own hook sitting at
    ``<hook>.backup`` with nothing recording that it is theirs.
    """
    # Identity, not content: a same-inode edit must survive without a stale backup.
    try:
        active_stat = os.lstat(active)
    except OSError:
        logger.warning(f"Could not roll back {backup}; inspect the current hook state")
        return False
    try:
        backup_stat = os.lstat(backup)
    except FileNotFoundError:
        # Someone else already removed it. There is nothing left to roll back
        # and nothing left for a record to describe, so this is the "gone" answer.
        return True
    except OSError:
        logger.warning(f"Could not roll back {backup}; inspect the current hook state")
        return False
    try:
        if (
            stat.S_ISREG(active_stat.st_mode)
            and stat.S_ISREG(backup_stat.st_mode)
            and (active_stat.st_dev, active_stat.st_ino) == (original.dev, original.ino)
            and (backup_stat.st_dev, backup_stat.st_ino) == (original.dev, original.ino)
        ):
            os.unlink(backup)
            # Confirm rather than assume, as _unlink_owned does: the caller
            # drops the ownership record on True, so True has to mean the file
            # is actually gone and not merely that unlink returned.
            if _read_slot(backup).kind == "absent":
                return True
            logger.warning(
                f"{backup} reappeared immediately after removal; inspect it before retrying"
            )
            return False
        logger.warning(
            f"Backup {backup} left in place; inspect it and the active hook before retrying"
        )
    except OSError:
        logger.warning(f"Could not roll back {backup}; inspect the current hook state")
    return False


class HooksInstaller:
    """Install and remove exact generated artifacts with a protected original slot.

    The transition guarantees require stable active, backup and staged paths.
    Empty-slot publication and backup creation never overwrite an existing name
    (hard-link behavior tested on Linux only). Detected changes stop the operation.
    Overlapping writers of any kind, including editors and other hook managers,
    are unsupported: a write after the final check and before replace/unlink is
    not protected, nor are writes through already-open descriptors or hostile
    same-user processes. Network/FUSE filesystems and crash durability of names
    are not guaranteed. No lock is used. macOS/Windows hook behavior is unverified;
    worktrees and core.hooksPath are outside this installer's scope.
    """

    def __init__(self, repo_path: Path | str | None = None) -> None:
        if repo_path is None:
            repo_path = Path.cwd()
        self.repo_path = Path(repo_path).resolve()
        self.hooks_dir = self.repo_path / ".git" / "hooks"
        self.ledger_path = self.repo_path / ".git" / _LEDGER_NAME

    def is_git_repo(self) -> bool:
        git_dir = self.repo_path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def get_hooks_dir(self) -> Path:
        if not self.is_git_repo():
            raise RuntimeError(f"Not a Git repository: {self.repo_path}")
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        return self.hooks_dir

    def install(self, force: bool = False) -> dict[str, bool]:
        """Install or upgrade both hooks; stop at the first refused/failed mutation."""
        hooks_dir = self.get_hooks_dir()
        ledger = _Ledger.load(self.ledger_path)
        return {
            name: self._install_hook(hooks_dir / name, content, force, ledger)
            for name, content in (
                ("pre-push", PRE_PUSH_HOOK_SCRIPT),
                ("pre-commit", PRE_COMMIT_HOOK_SCRIPT),
            )
        }

    def _install_hook(
        self, hook_path: Path, content: str, force: bool, ledger: _Ledger
    ) -> bool:
        if _CURRENT_TEMPLATES.get(hook_path.name) != content.encode("utf-8"):
            raise ValueError(
                "Only the known generated Traigent hook artifacts may be installed"
            )
        name = hook_path.name
        payload = content.encode("utf-8")
        generated = ledger.generated(name)
        backup_path = hook_path.with_suffix(".backup")
        active, backup = _read_slot(hook_path), _read_slot(backup_path)
        a = _classify(name, active, generated)
        b = _classify(name, backup, generated)
        holds_original = ledger.holds_preserved_original(name, backup)
        if a == "X":
            raise RuntimeError(
                f"{hook_path} is not a regular file or is unreadable; Traigent will not modify it"
            )
        if a == "A" and not (b == "A" or _is_owned(b)):
            if holds_original:
                # We know whose hook this is, so do not call it an orphan and
                # tell the customer to remove it -- that would destroy the file
                # we promised to hand back.
                raise RuntimeError(
                    f"{backup_path} holds the hook Traigent saved for you, and the "
                    "generated hook is gone. Run `traigent hooks uninstall` to put "
                    "it back, then install again."
                )
            raise RuntimeError(
                f"Orphan backup {backup_path}: restore or remove it, then retry"
            )
        if a == "F":
            if not force:
                logger.warning(
                    f"{hook_path} is not an unmodified Traigent hook; left unchanged"
                )
                return False
            if b == "X":
                raise RuntimeError(
                    f"{backup_path} is not a regular file or is unreadable; Traigent will not modify it"
                )
            if b == "F" and not _same_inode(active, backup):
                raise RuntimeError(
                    f"Backup conflict at {backup_path}: move it away first, then retry"
                )
            if holds_original and not _same_inode(active, backup):
                # The ledger says this backup is the customer's own hook, saved
                # by an earlier install. Its bytes happening to match a Traigent
                # template does not make it ours to delete.
                raise RuntimeError(
                    f"Backup conflict at {backup_path}: it holds the hook Traigent "
                    "saved for you. Move it away first, then retry."
                )
        needs_publish = a != "Oc" or active.mode & 0o111 != 0o111
        # Claim the artifact only once every refusal above has been cleared, so
        # a declined install leaves no record of a hook it never wrote -- and
        # still before the first mutation, because recording a digest we then
        # fail to publish is harmless (it only ever makes the installer
        # recognise its own template) while publishing one we failed to record
        # is the defect this ledger exists to prevent.
        ledger.remember_generated(name, hashlib.sha256(payload).hexdigest())
        staged: Path | None = None
        staged_slot: _Slot | None = None
        created_backup = False
        published = False
        try:
            if needs_publish:
                staged, staged_slot = _stage_hook(hook_path, payload)
            if a == "F" and _is_owned(b):
                _guard_slot(hook_path, active)
                _unlink_owned(backup_path, backup)
                ledger.forget_preserved_original(name)
                backup = _read_slot(backup_path)
                b = "A"
            if a == "F" and b == "A":
                _guard_slot(hook_path, active)
                # Written before the link so an interrupted install can never
                # leave a preserved customer hook the installer forgot about.
                ledger.remember_preserved_original(name, active.digest)
                _link_without_overwrite(hook_path, backup_path)
                created_backup = True
                backup = _guard_slot(backup_path, active)
                _guard_slot(hook_path, active)
            if needs_publish:
                assert staged is not None and staged_slot is not None
                _guard_slot(staged, staged_slot)
                if a == "A":
                    _link_without_overwrite(staged, hook_path)
                else:
                    if a == "F":
                        _guard_slot(backup_path, backup)
                    _guard_slot(hook_path, active)
                    os.replace(staged, hook_path)
                published = True
                active = _guard_slot(hook_path, staged_slot)
            else:
                _guard_slot(hook_path, active)
            if a != "F" and _is_owned(b) and not holds_original:
                _unlink_owned(backup_path, backup)
                ledger.forget_preserved_original(name)
            if b == "X":
                logger.warning(
                    f"Backup {backup_path} left in place; inspect it manually"
                )
        except BaseException:
            if created_backup and not published:
                # Only drop the record if the backup it describes is actually
                # gone. If rollback could not prove the inode and left the file
                # in place, that file is the customer's hook and the record is
                # the only thing saying so.
                if _rollback_original(hook_path, backup_path, active):
                    ledger.forget_preserved_original(name)
            if staged is not None and staged_slot is not None:
                try:
                    _remove_staged(staged, staged_slot)
                except OSError:
                    logger.warning(
                        f"Could not clean staged hook {staged}; inspect it manually"
                    )
            raise
        if staged is not None and staged_slot is not None:
            _remove_staged(staged, staged_slot)
        logger.info(f"Installed Traigent hook at {hook_path}")
        return True

    def uninstall(self) -> dict[str, bool]:
        """Remove generated artifacts, restoring only the preserved foreign original."""
        hooks_dir = self.get_hooks_dir()
        ledger = _Ledger.load(self.ledger_path)
        return {
            name: self._uninstall_hook(hooks_dir / name, ledger)
            for name in _CURRENT_TEMPLATES
        }

    def _uninstall_hook(self, hook_path: Path, ledger: _Ledger) -> bool:
        name = hook_path.name
        generated = ledger.generated(name)
        backup_path = hook_path.with_suffix(".backup")
        active, backup = _read_slot(hook_path), _read_slot(backup_path)
        a = _classify(name, active, generated)
        b = _classify(name, backup, generated)
        holds_original = ledger.holds_preserved_original(name, backup)
        if a == "X":
            raise RuntimeError(
                f"{hook_path} is not a regular file or is unreadable; Traigent will not modify it"
            )
        if a == "A":
            if holds_original:
                # Our hook is already gone and we are still holding the
                # customer's. Hand it back rather than delete it, whatever its
                # bytes happen to look like.
                _guard_slot(backup_path, backup)
                os.replace(backup_path, hook_path)
                _guard_slot(hook_path, backup)
                ledger.forget(name)
                logger.info(f"Restored original hook at {hook_path}")
            elif _is_owned(b):
                _unlink_owned(backup_path, backup)
                ledger.forget(name)
            elif b != "A":
                logger.warning(
                    f"Orphan backup {backup_path} left in place; restore or remove it manually"
                )
            else:
                ledger.forget(name)
            return True
        if a == "F":
            if _is_owned(b) and not holds_original:
                _unlink_owned(backup_path, backup)
                ledger.forget_preserved_original(name)
            logger.warning(
                f"{hook_path} is not an unmodified Traigent hook; left unchanged"
            )
            return False
        if b == "X":
            raise RuntimeError(
                f"{backup_path} is not a regular file or is unreadable; Traigent will not modify it"
            )
        if holds_original:
            # Restore only what the record says is the customer's.
            _guard_slot(backup_path, backup)
            _guard_slot(hook_path, active)
            os.replace(backup_path, hook_path)
            _guard_slot(hook_path, backup)
            logger.info(f"Restored original hook at {hook_path}")
        elif b == "F":
            # A backup with no record. Its bytes are not ours, but bytes are
            # exactly what cannot settle this: an artifact generated by some
            # release we do not recognise looks foreign too. Rather than
            # restore the wrong file silently, remove only our own hook and
            # say where theirs is.
            _unlink_owned(hook_path, active)
            logger.warning(
                f"Removed the Traigent hook at {hook_path}. {backup_path} is "
                "not recorded as a hook Traigent saved, so it was left exactly "
                "as it is: check it and move it back yourself if it is yours."
            )
        else:
            _unlink_owned(hook_path, active)
            if _is_owned(b):
                _unlink_owned(backup_path, backup)
        ledger.forget(name)
        return True

    def status(self) -> dict[str, str]:
        """Describe current bytes without following links or decoding foreign scripts."""
        try:
            if not self.is_git_repo():
                return {"error": "Not a Git repository"}
        except OSError:
            return {"error": "Could not inspect Git repository"}
        ledger = _Ledger.load(self.ledger_path)
        result = {}
        for name in _CURRENT_TEMPLATES:
            active = _read_slot(self.hooks_dir / name)
            kind = _classify(name, active, ledger.generated(name))
            if kind == "A":
                backup = _read_slot((self.hooks_dir / name).with_suffix(".backup"))
                result[name] = (
                    "not installed"
                    if backup.kind == "absent"
                    else "not installed (backup present)"
                )
            elif active.kind == "error":
                result[name] = "unreadable"
            elif kind == "X":
                result[name] = "unsafe (not a regular file)"
            elif kind == "Oc" and active.mode & 0o111 == 0o111:
                result[name] = "installed (traigent)"
            elif _is_owned(kind):
                result[name] = "installed (traigent, outdated)"
            else:
                result[name] = "installed (other)"
        return result


def find_git_root(start_path: Path | None = None) -> Path | None:
    """Find the root of the Git repository.

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to Git repository root, or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while True:
        if (current / ".git").exists():
            return current

        parent = current.parent
        if parent == current:
            return None
        current = parent


def install_hooks(repo_path: Path | str | None = None, force: bool = False) -> bool:
    """Convenience function to install Traigent hooks.

    Args:
        repo_path: Path to Git repository (auto-detects if None)
        force: Overwrite existing hooks

    Returns:
        True if all hooks installed successfully
    """
    if repo_path is None:
        repo_path = find_git_root()
        if repo_path is None:
            logger.error("Not in a Git repository")
            return False

    installer = HooksInstaller(repo_path)

    try:
        results = installer.install(force=force)
        return all(results.values())
    except Exception as e:
        logger.error(f"Failed to install hooks: {e}")
        return False


def uninstall_hooks(repo_path: Path | str | None = None) -> bool:
    """Convenience function to uninstall Traigent hooks.

    Args:
        repo_path: Path to Git repository (auto-detects if None)

    Returns:
        True if all hooks uninstalled successfully
    """
    if repo_path is None:
        repo_path = find_git_root()
        if repo_path is None:
            logger.error("Not in a Git repository")
            return False

    installer = HooksInstaller(repo_path)

    try:
        results = installer.uninstall()
        return all(results.values())
    except Exception as e:
        logger.error(f"Failed to uninstall hooks: {e}")
        return False
