"""Canary set and leak detector for captured optimization traffic."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, unquote

from .harness import CapturedRequest, normalize_path

#: kind -> canary value.  Fixed (not random) so the ratchet is deterministic.
#: Mixed case, underscore and hyphen so lowercase/slug/url forms all differ.
CANARIES: dict[str, str] = {
    "dataset_file_name": "cnryFile_Xq7K-2vMb9Rt4",
    "example_id": "cnryExid_Hw3P-8zLc5Nd1",
    "input_text": "cnryInput_Bv6T-4yQe7Gs2",
    "output_text": "cnryExpect_Jm9D-1xRf3Kw8",
    "model_output": "cnryModelOut_Zc5A-7nUp2Lh6",
    "evaluation_set": "cnryEvalSet_Wt2G-6kYd9Vb3",
    "agent_name": "cnryAgent_Rn8F-3hSx1Qj5",
    "run_title": "cnryTitle_Ly4E-9cMa6Ti7",
    "run_description": "cnryDesc_Pk1C-5wOg8Zu2",
    "knob_name": "cnryKnob_Dh7S-2bVn4Xe9",
    "choice_value": "cnryChoice_Fq3M-8jTy6Rk1",
    "metric_name": "cnryMetric_Gx9B_4aWz7Pl3",
    "prompt_template": "cnryPrompt_Ve5H-1uIo3Cd8",
    "metadata_key": "cnryMetaKey_Nb2J-7rKs5Yf4",
    "metadata_value": "cnryMetaVal_Uz6L-3tDq9Ah2",
    "exception_message": "cnryExc_Sa8W-5eGm1Op7",
    "default_config_value": "cnryDefCfg_Kt4R-9mNx2Wq6",
    "constraint_name": "cnryConstr_Yp6V-2dHb8Es3",
    "evaluator_id": "cnryEvalId_Ag1Z-7oLj5Tf9",
    "dataset_identity": "cnryDsId_Mu3Q-4cRy8Xn1",
    "function_name": "cnryFunc_Eb9K-6sPw2Ih4",
    "constraint_knob": "cnryCKnob_Wf4P_8aLz3Rq7",
    "offline_replay": "cnryReplay_Ox7D-1vJm5Gc3",
}


#: Shortest truncated-digest prefix (hex characters) the detector searches for.
MIN_DIGEST_PREFIX = 8

_DIGESTS = (
    ("sha256", lambda b: hashlib.sha256(b).digest()),
    ("md5", lambda b: hashlib.md5(b, usedforsecurity=False).digest()),  # noqa: S324
)


def _b64_needles(raw: bytes) -> set[str]:
    """Base64 substrings that survive any 0/1/2-byte alignment offset."""

    out: set[str] = set()
    for pad in range(3):
        enc = base64.b64encode(b"\x00" * pad + raw).decode().rstrip("=")
        skip = (pad * 4 + 2) // 3 if pad else 0
        core = enc[skip : len(enc) - 1]
        if len(core) >= 8:
            out.add(core)
            out.add(core.replace("+", "-").replace("/", "_"))
    return out


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def forms(value: str) -> dict[str, str]:
    """Every encoded form of ``value`` searched for, as ``form -> needle``."""

    raw_b = value.encode()
    out: dict[str, str] = {
        "raw": value,
        "lowercase": value.lower(),
        "slug": slugify(value),
        "url": quote(value, safe=""),
        "hex": raw_b.hex(),
        "sha256": hashlib.sha256(raw_b).hexdigest(),
        "md5": hashlib.md5(raw_b, usedforsecurity=False).hexdigest(),
        "sha256_lower": hashlib.sha256(value.lower().encode()).hexdigest(),
        "md5_lower": hashlib.md5(  # noqa: S324 - detector, not security use
            value.lower().encode(), usedforsecurity=False
        ).hexdigest(),
    }
    # Truncated digests (and md5/sha256 of the lowercase form) of every length
    # from MIN_DIGEST_PREFIX up to the full digest; base64 of the raw digest.
    for algo_name, digester in _DIGESTS:
        for variant, data in (("", raw_b), ("_lower", value.lower().encode())):
            digest = digester(data)
            hexd = digest.hex()
            for n in range(MIN_DIGEST_PREFIX, len(hexd) + 1):
                out.setdefault(f"{algo_name}{variant}_prefix{n}", hexd[:n])
            b64 = base64.b64encode(digest).decode().rstrip("=")
            out[f"{algo_name}{variant}_b64"] = b64
            out[f"{algo_name}{variant}_b64url"] = b64.replace("+", "-").replace(
                "/", "_"
            )
    for i, needle in enumerate(sorted(_b64_needles(raw_b))):
        out[f"base64#{i}"] = needle
    return out


@dataclass(frozen=True)
class Leak:
    route: str
    location: str  # path | query | header | body.key | body.value | body.raw
    kind: str
    form: str
    field_path: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.route, self.kind)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """``(route, location, field_path, canary_kind)`` -- what the manifest pins."""

        field_path = self.field_path
        if self.location == "path":
            field_path = normalize_path(field_path)
        return (self.route, self.location, field_path, self.kind)


def _walk(obj: object, path: str = "$") -> Iterator[tuple[str, str, str]]:
    """Yield ``(location, field_path, text)`` for every key and string/number value."""

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield "body.key", f"{path}.{k}", str(k)
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield "body.value", path, obj
    elif obj is not None and not isinstance(obj, bool):
        yield "body.value", path, json.dumps(obj)


def _search(text: str, needles: dict[str, dict[str, str]]) -> Iterator[tuple[str, str]]:
    for kind, fm in needles.items():
        for form, needle in fm.items():
            if needle and needle in text:
                yield kind, form
                break  # one form per (text, kind) is enough


def _best_form(text: str, fm: dict[str, str]) -> str | None:
    for form, needle in fm.items():
        if needle and needle in text:
            return form
    return None


def find_leaks(
    requests: Iterable[CapturedRequest],
    canaries: dict[str, str] | None = None,
) -> list[Leak]:
    """Search captured traffic for every canary in every encoded form."""

    canaries = canaries or CANARIES
    needles = {kind: forms(value) for kind, value in canaries.items()}
    leaks: dict[tuple[str, str, str, str], Leak] = {}

    def add(route: str, location: str, kind: str, form: str, field_path: str) -> None:
        key = (route, location, kind, field_path)
        leaks.setdefault(key, Leak(route, location, kind, form, field_path))

    for req in requests:
        route = req.route
        # path: also search the percent-decoded form
        path_text = f"{req.path} {unquote(req.path)}"
        for kind, fm in needles.items():
            form = _best_form(path_text, fm)
            if form:
                add(route, "path", kind, form, req.path)
        query_text = f"{req.query} {unquote(req.query)}"
        for kind, fm in needles.items():
            form = _best_form(query_text, fm)
            if form:
                names = ",".join(
                    k for k, _ in parse_qsl(req.query, keep_blank_values=True)
                )
                add(route, "query", kind, form, names)
        for name, value in req.headers:
            for kind, fm in needles.items():
                form = _best_form(f"{name}: {value}", fm)
                if form:
                    add(route, "header", kind, form, name)
        if not req.body:
            continue
        text = req.body.decode("utf-8", errors="replace")
        found_in_walk: set[tuple[str, str]] = set()
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            for location, field_path, item in _walk(parsed):
                for kind, fm in needles.items():
                    form = _best_form(item, fm)
                    if form:
                        add(route, location, kind, form, field_path)
                        found_in_walk.add((kind, form))
        for kind, fm in needles.items():
            if any(k == kind for k, _ in found_in_walk):
                continue
            form = _best_form(text, fm)
            if form:
                add(route, "body.raw", kind, form, "")
    return sorted(
        leaks.values(), key=lambda x: (x.route, x.kind, x.location, x.field_path)
    )


def distinct_identities(leaks: Iterable[Leak]) -> set[tuple[str, str, str, str]]:
    return {leak.identity for leak in leaks}


def distinct_pairs(leaks: Iterable[Leak]) -> set[tuple[str, str]]:
    return {leak.pair for leak in leaks}
