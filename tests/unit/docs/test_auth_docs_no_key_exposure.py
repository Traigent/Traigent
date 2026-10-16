"""Guard the authentication docs against recipes that expose the API key.

These check security properties of the published snippets rather than their
wording: a Dockerfile or ``docker build`` must not carry the key or the
credential-store password into the image, the laptop
guide must not write the key to the plaintext credentials file the SDK ignores,
and its smoke check must validate the key the SDK itself resolves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
AUTH_DOC = DOCS_DIR / "features" / "authentication.md"
LAPTOP_DOC = DOCS_DIR / "getting-started" / "portal-dev-laptop-agent.md"

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_SECRET_NAME = r"(?:API_KEY|MASTER_PASSWORD)"
# Dockerfile instructions are case-insensitive; ENV/ARG values and --build-arg
# values persist in image metadata and build history.
_SECRET_IN_IMAGE_RES = (
    re.compile(rf"^\s*(?:ENV|ARG)\s+.*{_SECRET_NAME}", re.IGNORECASE),
    re.compile(rf"--build-arg[\s=]+\S*{_SECRET_NAME}", re.IGNORECASE),
)
_CREDENTIAL_WRITE_RE = re.compile(r"write_text\(|json\.dump|open\(|>>?|\btee\b")
_API_KEY_FIELD_RE = re.compile(r"""["']api_key["']""")
_ENV_KEY_ASSIGNMENT_RE = re.compile(r"""TRAIGENT_API_KEY["']?\]?\s*=(?!=)""")
_WHOAMI = "traigent auth whoami"


def _fenced_blocks(text: str) -> list[str]:
    """Return the content of every fenced code block in a Markdown document."""
    blocks: list[str] = []
    current: list[str] | None = None
    fence = ""
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if current is None:
            if match:
                fence = match.group(1)
                current = []
        elif match and match.group(1).startswith(fence[0] * len(fence)):
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    return blocks


def _logical_lines(source: str) -> list[str]:
    """Join backslash continuations, as the Dockerfile parser and the shell do."""
    lines: list[str] = []
    pending = ""
    for line in source.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append(pending + line)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _secret_baking_lines(source: str) -> list[str]:
    """Logical lines that would bake an API key or the store password into an image."""
    return [
        line.strip()
        for line in _logical_lines(source)
        if any(pattern.search(line) for pattern in _SECRET_IN_IMAGE_RES)
    ]


def _dockerfile_sources() -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    for root in (DOCS_DIR, EXAMPLES_DIR):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".md":
                text = path.read_text(encoding="utf-8")
                sources.extend((path, block) for block in _fenced_blocks(text))
            elif path.name.startswith("Dockerfile"):
                sources.append((path, path.read_text(encoding="utf-8")))
    return sources


@pytest.mark.parametrize(
    "source",
    [
        "ENV TRAIGENT_API_KEY=placeholder",
        "arg TRAIGENT_API_KEY",
        "docker build --build-arg TRAIGENT_API_KEY=$TRAIGENT_API_KEY .",
        "docker build --build-arg=OPENAI_API_KEY .",
        "ENV TRAIGENT_MASTER_PASSWORD=placeholder",
        "ARG TRAIGENT_MASTER_PASSWORD",
        "ENV APP_MODE=prod \\\n    TRAIGENT_API_KEY=placeholder",
        "docker build \\\n  --build-arg TRAIGENT_API_KEY \\\n  .",
    ],
    ids=[
        "env",
        "arg-lowercase",
        "build-arg",
        "build-arg-equals",
        "env-master-password",
        "arg-master-password",
        "env-continuation",
        "build-arg-continuation",
    ],
)
def test_secret_matcher_flags_image_baking_recipes(source: str) -> None:
    assert _secret_baking_lines(source)


@pytest.mark.parametrize(
    "source",
    [
        "ENV APP_MODE=prod",
        "ENV APP_MODE=prod \\\n    LOG_LEVEL=info\nRUN python app.py",
        "docker build --build-arg APP_VERSION=1 .",
        "docker run -e TRAIGENT_API_KEY -e TRAIGENT_MASTER_PASSWORD my-agent",
    ],
    ids=["env", "env-continuation", "build-arg", "runtime-env"],
)
def test_secret_matcher_allows_recipes_without_baked_secrets(source: str) -> None:
    assert not _secret_baking_lines(source)


def test_no_dockerfile_snippet_bakes_a_secret_into_the_image() -> None:
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}: {line}"
        for path, source in _dockerfile_sources()
        for line in _secret_baking_lines(source)
    ]
    assert not offenders, offenders


def test_laptop_guide_does_not_write_the_plaintext_credentials_file() -> None:
    blocks = _fenced_blocks(LAPTOP_DOC.read_text(encoding="utf-8"))
    writers = [
        block
        for block in blocks
        if "credentials.json" in block and _CREDENTIAL_WRITE_RE.search(block)
    ]
    assert not writers, writers
    assert not [block for block in blocks if _API_KEY_FIELD_RE.search(block)]


def test_laptop_smoke_check_validates_the_key_the_sdk_resolves() -> None:
    blocks = _fenced_blocks(LAPTOP_DOC.read_text(encoding="utf-8"))
    commands = [
        line.split("#", 1)[0].strip()
        for block in blocks
        for line in block.splitlines()
        if _WHOAMI in line
    ]
    # A bare invocation, so the key never reaches argv or a hand-built environment.
    assert commands, "laptop guide has no shell `traigent auth whoami` check"
    assert all(command == _WHOAMI for command in commands), commands
    assert not [block for block in blocks if _ENV_KEY_ASSIGNMENT_RE.search(block)]


def test_auth_doc_no_longer_says_whoami_ignores_saved_credentials() -> None:
    words = " ".join(AUTH_DOC.read_text(encoding="utf-8").split())
    assert "does not automatically load a key from saved credentials" not in words
