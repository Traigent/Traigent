"""First-use creation of the credential store salt is atomic (#3014)."""

from __future__ import annotations

import multiprocessing as mp
import os
import stat
from pathlib import Path

import pytest

from traigent.security import credentials as credentials_module
from traigent.security.credentials import EnhancedCredentialStore, SecurityError

_PASSWORD = "unit-test-master-password-0123456789"  # pragma: allowlist secret


def _open(path: Path) -> EnhancedCredentialStore:
    return EnhancedCredentialStore(
        storage_path=path / "secure_credentials.enc",
        master_password=_PASSWORD,
        use_env_vars=False,
    )


def test_loser_of_create_race_adopts_the_winners_salt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    winner = b"W" * EnhancedCredentialStore.SALT_LENGTH
    real_token_bytes = credentials_module.secrets.token_bytes
    calls = {"n": 0}

    def racing_token_bytes(n: int) -> bytes:
        # Another process creates the salt right after we decided to create it.
        calls["n"] += 1
        if calls["n"] == 1:
            (tmp_path / ".salt").write_bytes(winner)
            return b"L" * n
        return real_token_bytes(n)

    monkeypatch.setattr(credentials_module.secrets, "token_bytes", racing_token_bytes)
    store = _open(tmp_path)
    assert store._salt == winner
    assert (tmp_path / ".salt").read_bytes() == winner


def test_salt_file_is_private_and_leaves_no_temp_files(tmp_path: Path) -> None:
    store = _open(tmp_path)
    salt_path = tmp_path / ".salt"
    assert salt_path.read_bytes() == store._salt
    assert stat.S_IMODE(os.stat(salt_path).st_mode) == 0o600
    assert [p.name for p in tmp_path.glob(".salt.*")] == []


@pytest.mark.parametrize("content", [b"", b"short"])
def test_short_or_empty_salt_is_rejected(tmp_path: Path, content: bytes) -> None:
    (tmp_path / ".salt").write_bytes(content)
    with pytest.raises(SecurityError):
        _open(tmp_path)


def _child(path: str, start, out) -> None:
    start.wait(10)
    out.put(_open(Path(path))._salt)


def test_concurrent_processes_agree_on_one_salt(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    start, out = ctx.Event(), ctx.Queue()
    procs = [
        ctx.Process(target=_child, args=(str(tmp_path), start, out)) for _ in range(4)
    ]
    for p in procs:
        p.start()
    start.set()
    salts = {out.get(timeout=60) for _ in procs}
    for p in procs:
        p.join(30)
    assert len(salts) == 1
    assert salts == {(tmp_path / ".salt").read_bytes()}
