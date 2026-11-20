"""Who may approve, and whose events count: the authority the controller checks.

The local controller never decides on its own whether a person is who they say they
are, and never trusts a flag in a file.  Two things are configured explicitly:

* an :class:`ApproverAuthority`: the approver references allowed to approve a run
  (**Approve run**), checked again at the instant of the effect (release acceptance
  is planned, not built, and not done here);
* an :class:`EventVerifier`: the issuers whose external events count, each with a
  shared secret.  An event is authenticated only when its ``signature`` is the
  HMAC-SHA256 of the envelope under its issuer's secret; the envelope's own
  ``authenticated`` field is never read.

``local-trusted`` is what a configured local operator list gives: the approver
reference is in the configured list and the operator is trusted to be that person.
No credential is checked, and every record made under it is labelled
``local-trusted``.  Production needs the platform (portal) to authenticate the person:
supply ``mode="platform"`` with an ``authenticator`` that verifies the approver
against the platform.  Without one, nothing authenticates.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.tuning.errors import E_AUTHORITY_CONFIG, TuningError

LOCAL_TRUSTED = "local-trusted"
PLATFORM = "platform"
_MODES = (LOCAL_TRUSTED, PLATFORM)
_NOT_SIGNED = frozenset({"signature", "authenticated"})
Authenticator = Callable[[str, Mapping[str, Any]], bool]


def local_trusted_refused(store_environment: str | None) -> bool:
    """Must a ``local-trusted`` authority be refused here?  Fails closed.

    Refused unless BOTH the process environment is explicitly non-production
    (an unset or unrecognised environment counts as production,
    ``treat_as_production_policy``) AND the store's recorded environment is a
    recognised non-production name (a missing or unknown name is refused): a
    production store never trusts a local operator, whatever the process says.
    """
    from traigent.utils.env_config import (
        _NON_PRODUCTION_ENV_NAMES,
        treat_as_production_policy,
    )

    if treat_as_production_policy():
        return True
    if not isinstance(store_environment, str):
        return True  # a store with no recorded environment is never trusted
    return store_environment.strip().lower() not in _NON_PRODUCTION_ENV_NAMES


@dataclass(frozen=True)
class ApproverAuthority:
    """The approvers allowed to give the two human decisions, and how they are known."""

    approvers: frozenset[str]
    mode: str = LOCAL_TRUSTED
    authenticator: Authenticator | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "approvers", frozenset(self.approvers))
        if self.mode not in _MODES:
            raise TuningError(E_AUTHORITY_CONFIG, f"mode must be one of {list(_MODES)}")
        if not self.approvers:
            raise TuningError(E_AUTHORITY_CONFIG, "at least one approver is required")
        if self.mode == PLATFORM and self.authenticator is None:
            raise TuningError(
                E_AUTHORITY_CONFIG, "platform mode needs an authenticator callable"
            )

    def authorized(self, approver: object) -> bool:
        """True when ``approver`` is on the configured list (checked at effect time)."""
        return isinstance(approver, str) and approver in self.approvers

    def authenticated(
        self,
        approver: object,
        context: Mapping[str, Any],
        store_environment: str | None = None,
    ) -> bool:
        """Whether the approver is authenticated, by this authority's mode.

        ``local-trusted``: a configured local operator is trusted (no credential is
        checked), so only a listed approver is "authenticated".  ``platform``: the
        platform's authenticator decides, whether or not the person is listed
        (authorization is a separate check); an error in it is a refusal.
        """
        if not isinstance(approver, str):
            return False
        if self.mode == LOCAL_TRUSTED:
            # No credential is checked, so in production nobody is authenticated: a
            # saved approval or grant is refused at every use, not only at creation.
            if local_trusted_refused(store_environment):
                return False
            return self.authorized(approver)
        try:
            answer = self.authenticator(str(approver), context)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - an authenticator failure never authenticates
            return False
        if inspect.iscoroutine(answer):
            answer.close()  # an async hook was never awaited: that is not a yes
            return False
        return answer is True  # only a literal True; truthy non-bools are refused


def signing_input(envelope: Mapping[str, Any]) -> bytes:
    """The canonical bytes an issuer signs: the envelope without its proof fields."""
    body = {k: v for k, v in envelope.items() if k not in _NOT_SIGNED}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_event(envelope: Mapping[str, Any], secret: bytes | str) -> str:
    """The ``signature`` an issuer attaches to ``envelope`` (hex HMAC-SHA256)."""
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(key, signing_input(envelope), hashlib.sha256).hexdigest()


class EventVerifier:
    """Authenticates external events against a configured set of trusted issuers."""

    def __init__(self, issuers: Mapping[str, bytes | str]) -> None:
        self._secrets = {
            str(name): (v.encode("utf-8") if isinstance(v, str) else bytes(v))
            for name, v in issuers.items()
            if v
        }

    @property
    def issuers(self) -> frozenset[str]:
        return frozenset(self._secrets)

    def verify(self, envelope: Mapping[str, Any]) -> bool:
        """True only for a trusted issuer's envelope with a matching signature."""
        issuer, signature = envelope.get("issuer"), envelope.get("signature")
        if not isinstance(issuer, str) or not isinstance(signature, str):
            return False
        secret = self._secrets.get(issuer)
        if secret is None:
            return False
        expected = sign_event(envelope, secret)
        return hmac.compare_digest(expected, signature)


def _env_secret(spec: Any, issuer: str) -> bytes:
    name = spec.get("secret_env") if isinstance(spec, Mapping) else None
    if not isinstance(name, str) or not name:
        raise TuningError(
            E_AUTHORITY_CONFIG, f"issuer {issuer!r} needs a secret_env variable name"
        )
    value = os.environ.get(name)
    if not value:
        raise TuningError(
            E_AUTHORITY_CONFIG, f"issuer {issuer!r}: environment variable is not set"
        )
    return value.encode("utf-8")


def authority_from_config(
    config: Mapping[str, Any],
) -> tuple[ApproverAuthority | None, EventVerifier | None]:
    """Build the authority and the event verifier from a parsed config mapping.

    ``mode`` (``local-trusted``), ``approvers`` (a list of references) and
    ``event_issuers`` (each issuer names, under its ``secret_env`` key, the
    environment variable holding its shared key).  Keys are read from those
    environment variables, never from the file.
    """
    approvers = config.get("approvers")
    authority: ApproverAuthority | None = None
    if approvers is not None:
        if not isinstance(approvers, Iterable) or isinstance(approvers, (str, bytes)):
            raise TuningError(E_AUTHORITY_CONFIG, "approvers must be a list")
        mode = str(config.get("mode", LOCAL_TRUSTED))
        if mode == PLATFORM:
            raise TuningError(
                E_AUTHORITY_CONFIG,
                "platform mode needs the platform's authenticator and cannot come "
                "from a file",
            )
        authority = ApproverAuthority(frozenset(str(a) for a in approvers), mode)
    verifier: EventVerifier | None = None
    issuers = config.get("event_issuers")
    if issuers is not None:
        if not isinstance(issuers, Mapping):
            raise TuningError(E_AUTHORITY_CONFIG, "event_issuers must be a mapping")
        verifier = EventVerifier(
            {str(name): _env_secret(spec, str(name)) for name, spec in issuers.items()}
        )
    return authority, verifier


def load_authority(
    path: Path | str,
) -> tuple[ApproverAuthority | None, EventVerifier | None]:
    """Read an authority config file (YAML or JSON)."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise TuningError(E_AUTHORITY_CONFIG, "authority file is unreadable") from None
    if not isinstance(data, Mapping):
        raise TuningError(E_AUTHORITY_CONFIG, "authority file must hold a mapping")
    return authority_from_config(data)


__all__ = [
    "LOCAL_TRUSTED",
    "PLATFORM",
    "ApproverAuthority",
    "EventVerifier",
    "local_trusted_refused",
    "authority_from_config",
    "load_authority",
    "sign_event",
    "signing_input",
]
