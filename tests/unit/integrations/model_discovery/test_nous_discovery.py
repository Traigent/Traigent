"""Offline unit tests for Nous Portal (Hermes) model discovery.

All offline/mock: the ``openai`` client is a fake, the JWT-refresh helper is
monkeypatched, and the models.yaml fallback is served from the shipped catalog.
No network, no credentials, no spend.

The load-bearing regression these guard (both plans flagged it): a verbatim copy
of the OpenAI discovery would construct ``OpenAI()`` with NO ``base_url`` and
silently hit ``api.openai.com``. NousDiscovery MUST pass
``base_url=NOUS_BASE_URL`` and an ``api_key`` sourced from
:func:`traigent.integrations.llms.nous_auth.get_nous_api_key`, not
``OPENAI_API_KEY``.
"""

from __future__ import annotations

import logging
import json
import types

import pytest

from traigent.config.provider_support import load_models_yaml
from traigent.integrations.llms import nous_auth
from traigent.integrations.llms.nous_auth import NOUS_BASE_URL
from traigent.integrations.model_discovery import nous_discovery
from traigent.integrations.model_discovery.cache import ModelCache, reset_global_cache
from traigent.integrations.model_discovery.nous_discovery import NousDiscovery
from traigent.integrations.model_discovery.registry import get_model_discovery
from traigent.integrations.utils import Framework

openai = pytest.importorskip("openai")

pytestmark = pytest.mark.unit


# The models.yaml catalog served as the discovery fallback (single source; keep
# in lockstep with the file rather than hardcoding it here).
_YAML_NOUS_MODELS = load_models_yaml()["nous"]["known_models"]


def _fresh_discovery() -> NousDiscovery:
    """A NousDiscovery bound to a private, file-less cache for isolation."""
    return NousDiscovery(cache=ModelCache(enable_file_cache=False))


def _fake_openai_factory(captured: dict, model_ids: list[str]):
    """Return a fake ``OpenAI`` class capturing its ctor kwargs + serving ids."""

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id=mid) for mid in model_ids]
                )
            )

    return _FakeOpenAI


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No ambient Nous creds; a private global cache per test."""
    for var in ("NOUS_API_KEY", "NOUS_REFRESH_TOKEN", "NOUS_PORTAL_REFRESH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(tmp_path / "absent-auth.json"))
    reset_global_cache()
    yield
    reset_global_cache()


# --------------------------------------------------------------------------- #
# Provider / framework identity                                               #
# --------------------------------------------------------------------------- #
def test_provider_and_framework_attributes():
    assert NousDiscovery.PROVIDER == "nous"
    assert NousDiscovery.FRAMEWORK == Framework.NOUS


def _set_static_credential(monkeypatch, _tmp_path):
    monkeypatch.setenv("NOUS_API_KEY", "static-identity-canary")


def _set_refresh_credential(monkeypatch, _tmp_path):
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "refresh-identity-canary")


def _set_file_credential(monkeypatch, tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"refresh_token": "file-identity-canary"}', encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))


@pytest.mark.parametrize(
    ("identity_setup", "identity_material", "alternate_identity_material"),
    [
        (_set_static_credential, "static-identity-canary", "alternate-static-identity"),
        (
            _set_refresh_credential,
            "refresh-identity-canary",
            "alternate-refresh-identity",
        ),
        (_set_file_credential, "file-identity-canary", "alternate-file-identity"),
    ],
)
def test_cache_partitions_persistent_models_by_real_nous_identity(
    monkeypatch,
    tmp_path,
    identity_setup,
    identity_material,
    alternate_identity_material,
):
    cache_dir = tmp_path / "model-cache"
    identity_setup(monkeypatch, tmp_path)
    first = NousDiscovery(cache=ModelCache(cache_dir=cache_dir))
    monkeypatch.setattr(
        first, "_fetch_models_from_identity", lambda _identity: ["account-one"]
    )
    assert first.list_models() == ["account-one"]

    if identity_material == "file-identity-canary":
        (tmp_path / "auth.json").write_text(
            f'{{"refresh_token": "{alternate_identity_material}"}}', encoding="utf-8"
        )
    else:
        monkeypatch.setenv(
            "NOUS_API_KEY"
            if identity_material == "static-identity-canary"
            else "NOUS_REFRESH_TOKEN",
            alternate_identity_material,
        )
    second = NousDiscovery(cache=ModelCache(cache_dir=cache_dir))
    monkeypatch.setattr(
        second, "_fetch_models_from_identity", lambda _identity: ["account-two"]
    )
    assert second.list_models() == ["account-two"]

    if identity_material == "file-identity-canary":
        (tmp_path / "auth.json").write_text(
            f'{{"refresh_token": "{identity_material}"}}', encoding="utf-8"
        )
    else:
        monkeypatch.setenv(
            "NOUS_API_KEY"
            if identity_material == "static-identity-canary"
            else "NOUS_REFRESH_TOKEN",
            identity_material,
        )
    same = NousDiscovery(cache=ModelCache(cache_dir=cache_dir))
    monkeypatch.setattr(
        same, "_fetch_models_from_sdk", lambda: pytest.fail("cache miss")
    )
    assert same.list_models() == ["account-one"]
    cache_text = "".join(
        path.read_text(encoding="utf-8") for path in cache_dir.iterdir()
    )
    for value in (identity_material, alternate_identity_material):
        assert value not in cache_text
        assert all(value not in path.name for path in cache_dir.iterdir())


def test_cache_identity_helper_uses_precedence_and_file_content(monkeypatch, tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"refresh_token": "file-refresh"}', encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))
    monkeypatch.setenv("NOUS_PORTAL_REFRESH_TOKEN", "alias-refresh")
    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "primary-refresh")
    monkeypatch.setenv("NOUS_API_KEY", "static-identity")
    assert nous_auth._get_nous_cache_identity() == ("static", "static-identity")

    monkeypatch.delenv("NOUS_API_KEY")
    assert nous_auth._get_nous_cache_identity() == (
        "$NOUS_REFRESH_TOKEN",
        "primary-refresh",
    )

    monkeypatch.delenv("NOUS_REFRESH_TOKEN")
    monkeypatch.delenv("NOUS_PORTAL_REFRESH_TOKEN")
    assert nous_auth._get_nous_cache_identity() == (
        str(auth_file.resolve()),
        "file-refresh",
    )


def test_malformed_auth_file_uses_private_nonsecret_cache_partition(
    monkeypatch, tmp_path
):
    auth_file = tmp_path / "raw-auth-path.json"
    malformed = "malformed-refresh-material"
    auth_file.write_text(malformed, encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))

    identity = nous_auth._get_nous_cache_identity()

    assert identity == ("invalid-auth-file-v1", str(auth_file.resolve()))
    assert malformed not in identity[1]


def test_malformed_auth_file_list_models_falls_back_without_reusing_valid_cache(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "model-cache"
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"refresh_token": "valid-refresh"}', encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))

    valid = NousDiscovery(cache=ModelCache(cache_dir=cache_dir))
    monkeypatch.setattr(
        valid, "_fetch_models_from_identity", lambda _identity: ["valid-account"]
    )
    assert valid.list_models() == ["valid-account"]

    malformed = "malformed-refresh-material"
    auth_file.write_text(malformed, encoding="utf-8")
    broken = NousDiscovery(cache=ModelCache(cache_dir=cache_dir))
    monkeypatch.setattr(
        broken,
        "_fetch_models_from_identity",
        lambda _identity: (_ for _ in ()).throw(
            nous_auth.NousAuthError("malformed auth")
        ),
    )
    assert broken.list_models() == _YAML_NOUS_MODELS

    cache_text = "".join(
        path.read_text(encoding="utf-8") for path in cache_dir.iterdir()
    )
    assert str(auth_file) not in cache_text
    assert malformed not in cache_text
    assert all(str(auth_file) not in path.name for path in cache_dir.iterdir())
    assert all(malformed not in path.name for path in cache_dir.iterdir())


# --------------------------------------------------------------------------- #
# _fetch_models_from_sdk — base_url + minted key + unfiltered sort            #
# --------------------------------------------------------------------------- #
def test_fetch_uses_nous_base_url_and_minted_key(monkeypatch):
    payload = [
        "openai/gpt-4o",
        "NousResearch/Hermes-4-405B",
        "Hermes-3-Llama-3.1-8B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "DeepHermes-3-Mistral-24B-Preview",
    ]
    captured: dict = {}
    monkeypatch.setattr(openai, "OpenAI", _fake_openai_factory(captured, payload))
    monkeypatch.setattr(nous_discovery, "has_nous_credentials", lambda: True)
    monkeypatch.setattr(nous_discovery, "get_nous_api_key", lambda: "minted-token-xyz")

    result = _fresh_discovery()._fetch_models_from_sdk()

    # The api.openai.com regression guard: explicit Nous base_url + minted key.
    assert captured["base_url"] == NOUS_BASE_URL
    assert (
        captured["api_key"] == "minted-token-xyz"
    )  # from the helper, not OPENAI_API_KEY
    # All advertised ids returned, sorted, and UNFILTERED (third-party kept).
    assert result == sorted(payload)
    assert "openai/gpt-4o" in result
    assert "meta-llama/Llama-3.3-70B-Instruct" in result


def test_fetch_returns_empty_without_credentials(monkeypatch):
    monkeypatch.setattr(nous_discovery, "has_nous_credentials", lambda: False)

    def _explode(**_k):  # pragma: no cover - must not be constructed
        raise AssertionError("OpenAI client must not be built without credentials")

    monkeypatch.setattr(openai, "OpenAI", _explode)

    assert _fresh_discovery()._fetch_models_from_sdk() == []


# --------------------------------------------------------------------------- #
# list_models fallback behaviour                                              #
# --------------------------------------------------------------------------- #
def test_list_models_falls_back_to_yaml_when_no_credentials(monkeypatch):
    monkeypatch.setattr(nous_discovery, "has_nous_credentials", lambda: False)
    models = _fresh_discovery().list_models()
    assert models == _YAML_NOUS_MODELS
    assert models, "models.yaml nous fallback must be non-empty"


def test_list_models_falls_back_when_mint_fails(monkeypatch, caplog):
    # Credentials present, but the token mint blows up mid-discovery: the error
    # must propagate out of _fetch_models_from_sdk, be caught by list_models,
    # and fall back to the models.yaml catalog (never a mock). Because creds ARE
    # present, the degraded state must be VISIBLE — a WARNING, not a silent debug
    # line that reads like a clean discovery (#1978 #2).
    from traigent.integrations.llms.nous_auth import NousAuthError

    monkeypatch.setenv("NOUS_REFRESH_TOKEN", "refresh-mint-failure")

    def _boom(*_args, **_kwargs):
        raise NousAuthError("token mint failed (offline test)")

    monkeypatch.setattr(nous_auth, "_get_nous_api_key_for_identity", _boom)

    with caplog.at_level(logging.WARNING):
        models = _fresh_discovery().list_models()

    assert models == _YAML_NOUS_MODELS
    # The broken credential is surfaced at WARNING (not swallowed at debug).
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "credentials present" in r.getMessage().lower()
        and "fail" in r.getMessage().lower()
        for r in warnings
    ), "a broken Nous credential must be logged at WARNING, not silently swallowed"


def test_prepared_models_list_failure_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("NOUS_API_KEY", "static-discovery-token")

    class _FailingOpenAI:
        def __init__(self, **_kwargs):
            self.models = types.SimpleNamespace(
                list=lambda: (_ for _ in ()).throw(
                    RuntimeError("models endpoint failed")
                )
            )

    monkeypatch.setattr(openai, "OpenAI", _FailingOpenAI)

    with caplog.at_level(logging.WARNING):
        models = _fresh_discovery().list_models(force_refresh=True)

    assert models == _YAML_NOUS_MODELS
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "credentials present but token mint/discovery failed" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()


def test_prepared_fetch_keeps_rotated_auth_file_on_original_cache_partition(
    monkeypatch, tmp_path
):
    """A file rotation after preparation cannot cross-contaminate discovery."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"refresh_token": "refresh-A"}', encoding="utf-8")
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))

    minted: list[str] = []

    def _mint(refresh_material, *, token_url, **_kwargs):
        minted.append(refresh_material)
        return {"access_token": f"jwt-{refresh_material}", "expires_in": 3600}

    class _FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            assert base_url == NOUS_BASE_URL
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id=f"models-for-{api_key}")]
                )
            )

    monkeypatch.setattr(nous_auth, "_request_new_token", _mint)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    cache = ModelCache(cache_dir=tmp_path / "model-cache")
    discovery = NousDiscovery(cache=cache)
    original_prepare = discovery._prepare_model_fetch

    def _prepare_then_rotate():
        prepared = original_prepare()
        auth_file.write_text('{"refresh_token": "refresh-B"}', encoding="utf-8")
        return prepared

    monkeypatch.setattr(discovery, "_prepare_model_fetch", _prepare_then_rotate)
    assert discovery.list_models(force_refresh=True) == ["models-for-jwt-refresh-A"]
    assert minted == ["refresh-A"]

    first_key = next(key for key in cache._cache if key.startswith("nous-"))
    second = NousDiscovery(cache=cache)
    assert second.list_models(force_refresh=True) == ["models-for-jwt-refresh-B"]
    second_key = next(
        key for key in cache._cache if key.startswith("nous-") and key != first_key
    )
    assert first_key != second_key
    assert cache.get(first_key) == ["models-for-jwt-refresh-A"]
    assert cache.get(second_key) == ["models-for-jwt-refresh-B"]
    assert minted == ["refresh-A", "refresh-B"]


@pytest.mark.parametrize("invalid_material", ["bad\x00material", "bad\ud800material"])
def test_invalid_auth_file_identity_falls_back_before_cache_hmac(
    monkeypatch, tmp_path, invalid_material
):
    auth_file = tmp_path / "invalid-auth.json"
    auth_file.write_text(
        json.dumps({"refresh_token": invalid_material}, ensure_ascii=True),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAIGENT_NOUS_AUTH_FILE", str(auth_file))
    discovery = NousDiscovery(cache=ModelCache(cache_dir=tmp_path / "model-cache"))

    models = discovery.list_models(force_refresh=True)

    assert models == _YAML_NOUS_MODELS
    assert all(
        invalid_material not in path.name
        for path in discovery._cache._cache_dir.iterdir()
    )
    assert all(
        invalid_material not in path.read_text(encoding="utf-8")
        for path in discovery._cache._cache_dir.iterdir()
    )


def test_list_models_no_credentials_stays_silent(monkeypatch, caplog):
    # The no-credential path is the expected offline case: it returns the yaml
    # catalog WITHOUT emitting a WARNING (that would be noise on every offline
    # run). Only a broken-but-present credential warrants a WARNING (#1978 #2).
    monkeypatch.setattr(nous_discovery, "has_nous_credentials", lambda: False)

    with caplog.at_level(logging.WARNING):
        models = _fresh_discovery().list_models()

    assert models == _YAML_NOUS_MODELS
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
        "the no-credential offline path must not emit a WARNING"
    )


def test_every_yaml_nous_model_is_tier_classified():
    # #1978 #4: every models.yaml-served Nous model must be classified in
    # providers._MODEL_TIERS["nous"], so the yaml catalog and the tier table
    # never drift (DeepHermes-3-Mistral-24B-Preview was previously yaml-only).
    from traigent.integrations.providers import _MODEL_TIERS

    tier_ids = {mid for tier in _MODEL_TIERS["nous"].values() for mid in tier}
    missing = set(_YAML_NOUS_MODELS) - tier_ids
    assert not missing, f"yaml Nous models absent from _MODEL_TIERS['nous']: {missing}"


# --------------------------------------------------------------------------- #
# get_pattern — config wins, else the Hermes-family default                   #
# --------------------------------------------------------------------------- #
def test_get_pattern_defaults_to_hermes_family(monkeypatch, tmp_path):
    # A config without a nous.pattern falls back to the module default.
    cfg = tmp_path / "models.yaml"
    cfg.write_text("nous:\n  known_models: []\n", encoding="utf-8")
    discovery = NousDiscovery(
        cache=ModelCache(enable_file_cache=False), config_path=cfg
    )
    assert discovery.get_pattern() == nous_discovery.NOUS_MODEL_PATTERN


def test_get_pattern_prefers_config_override(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text('nous:\n  pattern: "^CUSTOM-"\n', encoding="utf-8")
    discovery = NousDiscovery(
        cache=ModelCache(enable_file_cache=False), config_path=cfg
    )
    assert discovery.get_pattern() == "^CUSTOM-"


def test_shipped_config_pattern_matches_hermes_family():
    # The shipped models.yaml pattern is what discovery actually uses.
    assert NousDiscovery().get_pattern() == nous_discovery.NOUS_MODEL_PATTERN


# --------------------------------------------------------------------------- #
# is_valid_model — known hit / pattern hit / miss (no creds -> yaml catalog)   #
# --------------------------------------------------------------------------- #
def test_is_valid_model_behaviour(monkeypatch):
    monkeypatch.setattr(nous_discovery, "has_nous_credentials", lambda: False)
    discovery = _fresh_discovery()

    # Exact catalog hit.
    assert discovery.is_valid_model("NousResearch/Hermes-3-Llama-3.1-8B") is True
    # Not catalogued but matches the Hermes-family pattern fallback.
    assert discovery.is_valid_model("Hermes-4-Future-999B") is True
    assert discovery.is_valid_model("NousResearch/SomethingNew-7B") is True
    # Neither catalogued nor pattern-matching.
    assert discovery.is_valid_model("gpt-4o") is False
    assert discovery.is_valid_model("") is False


# --------------------------------------------------------------------------- #
# Registry resolution by string and by Framework enum                         #
# --------------------------------------------------------------------------- #
def test_get_model_discovery_by_string():
    discovery = get_model_discovery("nous")
    assert isinstance(discovery, NousDiscovery)


def test_get_model_discovery_by_framework():
    discovery = get_model_discovery(Framework.NOUS)
    assert isinstance(discovery, NousDiscovery)
