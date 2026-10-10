"""Guard test: SDK text users can see must only point at the public SDK repo.

Traigent#2415: the 0.27.0 ``safety_constraints`` ``NotImplementedError`` linked
to an issue in a repository that is not public, and the ``optimize()``
docstring kept a short cross-reference to it after the raise was removed.
Users who follow either get a 404, and the text reveals a non-public repo.

Two checks:

* every ``github.com/Traigent/<repo>`` URL anywhere in the ``traigent``
  package (any text file: runtime strings, comments, docstrings) names the
  public SDK repository;
* the docstrings of the public API (``traigent.__all__``), which ``help()``
  shows, carry no ``<repo>#<n>`` cross-reference to any other repository.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import traigent

PACKAGE_ROOT = Path(traigent.__file__).parent
PUBLIC_SDK_REPO = "traigent"  # github.com/Traigent/Traigent, compared lower-case

_GITHUB_URL = re.compile(r"github\.com/Traigent/([A-Za-z0-9_.-]+)", re.IGNORECASE)
# ``Repo#123``; the character before ``#`` must be alphanumeric so prose such as
# "pre-#1234" is not mistaken for a repository name.
_SHORT_REF = re.compile(r"(?<![\w/.-])([A-Za-z][\w.-]*[A-Za-z0-9])#\d+")


def _package_text_files() -> list[Path]:
    files = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files.append(path)
    return files


def test_package_github_links_point_only_at_the_public_sdk_repo() -> None:
    offenders = []
    for path in _package_text_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _GITHUB_URL.finditer(line):
                if match.group(1).lower() != PUBLIC_SDK_REPO:
                    rel = path.relative_to(PACKAGE_ROOT.parent)
                    offenders.append(f"{rel}:{lineno}: {match.group(0)}")
    _failure_detail = (
        "Links to a repository other than the public SDK repo:\n" + "\n".join(offenders)
    )
    assert not offenders, _failure_detail


def _public_docstrings() -> list[tuple[str, str]]:
    docs = []
    for name in getattr(traigent, "__all__", []):
        obj = getattr(traigent, name, None)
        if obj is None:
            continue
        targets = [(name, obj)]
        if inspect.isclass(obj):
            targets += [
                (f"{name}.{member_name}", member)
                for member_name, member in inspect.getmembers(obj)
                if not member_name.startswith("_") and callable(member)
            ]
        for label, target in targets:
            doc = inspect.getdoc(target)
            if doc:
                docs.append((label, doc))
    return docs


def test_public_api_docstrings_cite_no_other_repository() -> None:
    assert _public_docstrings(), "traigent.__all__ exposes no docstrings to check"
    offenders = [
        f"{label}: {match.group(0)}"
        for label, doc in _public_docstrings()
        for match in _SHORT_REF.finditer(doc)
        if match.group(1).lower() != PUBLIC_SDK_REPO
    ]
    _failure_detail = (
        "help() on the public API cites another repository:\n" + "\n".join(offenders)
    )
    assert not offenders, _failure_detail


def test_short_ref_pattern_ignores_prose_and_accepts_sdk_refs() -> None:
    sample = "pre-#1234 fix; see Traigent#2415 and fictional-library#26."
    names = [m.group(1) for m in _SHORT_REF.finditer(sample)]
    assert names == ["Traigent", "fictional-library"]
