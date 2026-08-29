"""Security floors for *transitive* dependencies must live in package metadata.

Three advisories reach the SDK through packages nobody declares:

    yarl      <- aiohttp                                   (core)
    filelock  <- litellm -> tokenizers -> huggingface-hub   (core)
    h2        <- httpx[http2]                               (hybrid extra)

Bumping them in ``uv.lock`` closes the scanner finding and fixes **nothing** for
users. ``uv.lock`` governs ``uv sync`` inside this repository; it is not shipped
in the sdist or wheel (``[tool.setuptools.packages.find] include`` is
``traigent*`` / ``traigent_validation*``) and it is not consulted by
``pip install traigent``. A customer's resolver reads ``[project.dependencies]``
and the extras -- nothing else -- so a floor that is absent there is a floor that
does not exist for anyone outside this repo.

These tests therefore assert on ``pyproject.toml``, which is what ships. See
GitHub issue #2209.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

# package -> (extra or None for core, vulnerable, first fixed, next major, advisory)
FLOORS: dict[str, tuple[str | None, str, str, str, str]] = {
    "yarl": (None, "1.24.2", "1.24.5", "2.0.0", "AIKIDO-2026-181733"),
    "filelock": (None, "3.29.4", "3.29.5", "4.0.0", "AIKIDO-2026-181731"),
    "h2": ("hybrid", "4.3.0", "4.4.1", "5.0.0", "GHSA-6hr6-w5qg-qmwg"),
}

REQUIREMENTS_DIR = _REPO_ROOT / "requirements"


def _declared_specifier(package: str, extra: str | None) -> str:
    """Return the specifier declared for ``package`` in the shipped metadata."""
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    if extra is None:
        requirements: list[str] = data["project"]["dependencies"]
        where = "[project.dependencies]"
    else:
        requirements = data["project"]["optional-dependencies"][extra]
        where = f"[project.optional-dependencies.{extra}]"

    for raw in requirements:
        parsed = Requirement(raw)
        if parsed.name.lower() == package:
            return str(parsed.specifier)

    raise AssertionError(
        f"{package!r} is not declared in {where} of pyproject.toml. "
        f"A bump in uv.lock does not reach `pip install traigent` -- the floor "
        f"must be declared here to protect users (issue #2209)."
    )


@pytest.mark.parametrize("package", sorted(FLOORS))
def test_floor_is_declared_in_shipped_metadata(package: str) -> None:
    """The floor exists where a customer's resolver will actually read it."""
    extra, _, _, _, _ = FLOORS[package]
    assert _declared_specifier(package, extra), (
        f"{package} is declared with no version specifier; an unbounded "
        f"declaration provides no floor at all."
    )


@pytest.mark.parametrize("package", sorted(FLOORS))
def test_floor_rejects_the_vulnerable_version(package: str) -> None:
    extra, vulnerable, _, _, advisory = FLOORS[package]
    spec = SpecifierSet(_declared_specifier(package, extra))
    assert not spec.contains(vulnerable), (
        f"{package} specifier {spec} still admits {vulnerable}, which is "
        f"affected by {advisory}."
    )


@pytest.mark.parametrize("package", sorted(FLOORS))
def test_floor_admits_the_first_fixed_version(package: str) -> None:
    extra, _, fixed, _, advisory = FLOORS[package]
    spec = SpecifierSet(_declared_specifier(package, extra))
    assert spec.contains(fixed), (
        f"{package} specifier {spec} excludes {fixed}, the first release that "
        f"carries the {advisory} fix."
    )


@pytest.mark.parametrize("package", sorted(FLOORS))
def test_floor_is_bounded_below_the_next_major(package: str) -> None:
    """Repo convention: bounded ``>=X,<Y`` ranges, not bare floors.

    An unbounded floor lets a resolver pull an unverified future major into a
    customer install; every other security floor in this file is bounded.

    Asserting the *next major* is excluded rather than merely that some ``<``
    exists: a nominal upper bound like ``<99`` would satisfy the weaker check
    while providing no protection at all.
    """
    extra, _, _, next_major, _ = FLOORS[package]
    spec = SpecifierSet(_declared_specifier(package, extra))
    assert not spec.contains(next_major), (
        f"{package} specifier {spec} admits {next_major}, an unverified future "
        f"major. Bound it below the next major, e.g. '<{next_major.split('.')[0]}'."
    )


@pytest.mark.parametrize("package", sorted(FLOORS))
def test_installed_distribution_satisfies_the_declared_floor(package: str) -> None:
    """The *installed* distribution must not sit below the metadata floor.

    This reads ``importlib.metadata``, i.e. what uv actually resolved and
    synced into this environment -- not ``uv.lock`` itself. It therefore
    catches a lock that was left behind (or re-resolved downwards) only once
    that lock has been synced, and it skips entirely when the package is
    absent, as ``h2`` is without the hybrid extra. Treat it as a
    reconciliation check between two independent statements about the same
    dependency, not as a lockfile gate.
    """
    from importlib.metadata import PackageNotFoundError, version

    extra, _, _, _, _ = FLOORS[package]
    try:
        installed = version(package)
    except PackageNotFoundError:
        pytest.skip(f"{package} is not installed in this environment")

    spec = SpecifierSet(_declared_specifier(package, extra))
    assert spec.contains(installed), (
        f"installed {package} {installed} violates the declared floor {spec}."
    )


# --- Behavioural coverage for the bumps themselves -------------------------
#
# A lockfile check and an `import traigent` do not exercise any of the
# behaviour these releases changed: h2 4.4 altered protocol-error handling,
# yarl changed host validation, filelock 3.30-3.32 changed soft/strict
# locking. These two tests cover the paths the SDK actually uses.


def test_aiohttp_url_validation_still_accepts_the_urls_the_sdk_builds() -> None:
    """yarl's host parsing must not reject or mangle our own endpoints.

    yarl 1.24.x tightened host normalisation. The SDK composes cloud and hybrid
    URLs by joining a base to a path, so a change in normalisation shows up as
    a wrong Host header rather than an exception.
    """
    yarl = pytest.importorskip("yarl")

    url = yarl.URL("https://api.traigent.ai") / "traigent" / "v1" / "execute"
    assert url.host == "api.traigent.ai"
    assert url.scheme == "https"
    assert url.path == "/traigent/v1/execute"

    with_port = yarl.URL("http://localhost:8080/traigent/v1/health")
    assert with_port.host == "localhost"
    assert with_port.port == 8080

    # Percent-encoding of a query value must survive normalisation intact.
    q = yarl.URL("https://api.traigent.ai/x").with_query({"name": "a b&c"})
    assert q.query["name"] == "a b&c"


def test_hybrid_http2_transport_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hybrid HTTP/2 path constructs and enforces its https precondition.

    ``HTTPTransport._get_client`` opts into HTTP/2 only when ``h2`` imports, so
    a broken or absent h2 silently downgrades the transport to HTTP/1.1 instead
    of failing. Assert the import works and the strict mode is wired.
    """
    pytest.importorskip("h2")
    import asyncio

    import httpx

    from traigent.hybrid.http_transport import HTTPTransport

    strict = HTTPTransport(base_url="https://agent.example.com", require_http2=True)
    assert strict.require_http2 is True
    assert strict.base_url == "https://agent.example.com"

    with pytest.raises(ValueError, match="requires an https:// base_url"):
        HTTPTransport(base_url="http://agent.example.com", require_http2=True)

    lenient = HTTPTransport(base_url="http://agent.example.com")
    assert lenient.require_http2 is False

    # Reach _get_client(), which is where the h2 import decides http2=True.
    # Asserting only on the constructor would still pass if that path silently
    # fell back to HTTP/1.1 -- the exact regression the h2 floor guards.
    #
    # Spy on the public httpx.AsyncClient(http2=...) keyword rather than
    # introspecting client._transport._pool._http2: the private attribute chain
    # is httpcore internals and would start silently passing (or failing) on an
    # httpx upgrade, which is precisely the failure mode this test is meant to
    # detect. The keyword is public, documented API.
    seen: dict[str, object] = {}
    real_client = httpx.AsyncClient

    class _SpyClient(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    async def _build() -> None:
        monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
        client = await strict._get_client()
        try:
            assert seen.get("http2") is True, (
                "HTTPTransport built its client with "
                f"http2={seen.get('http2')!r} even though h2 is installed; the "
                "transport silently downgraded to HTTP/1.1."
            )
        finally:
            await client.aclose()

    asyncio.run(_build())
