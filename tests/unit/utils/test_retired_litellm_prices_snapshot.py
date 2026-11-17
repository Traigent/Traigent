"""litellm >= 1.103.10 dropped 736 cost-map ids that 1.103.4 priced.

The SDK restores them from its snapshots (OpenRouter + retired non-OpenRouter).
The fixture was generated with the SDK's own ``cost_from_tokens(100, 10, id)``
and ``litellm.get_llm_provider(id)`` under litellm 1.103.4; these tests pin
the restored pricing and routing to those values.
"""

import json
import tomllib
from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")

from traigent.utils import cost_calculator as cc  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "litellm_1_103_4_removed_ids_pricing.json"
    ).read_text("utf-8")
)


@pytest.fixture(autouse=True)
def _no_copilot_oauth(monkeypatch):
    """litellm routes github_copilot/* through an interactive device login."""
    try:
        from litellm.llms.github_copilot import authenticator
    except ImportError:  # pragma: no cover - layout differs in other litellm lines
        return

    def _refuse(self, *args, **kwargs):
        raise RuntimeError("oauth login blocked in tests")

    monkeypatch.setattr(authenticator.Authenticator, "_login", _refuse)


def test_fixture_covers_all_736_removed_ids():
    assert len(_FIXTURE["prices"]) == 736
    assert all(isinstance(v, list) for v in _FIXTURE["prices"].values())


def test_every_restored_id_prices_as_under_litellm_1_103_4():
    cc.ensure_bundled_provider_prices()
    mismatches = {}
    for model, expected in _FIXTURE["prices"].items():
        got = list(cc.cost_from_tokens(100, 10, model, strict=True))
        if got != expected:
            mismatches[model] = (got, expected)
    assert not mismatches, list(mismatches.items())[:5]


def test_every_restored_id_routes_as_under_litellm_1_103_4():
    cc.ensure_bundled_provider_prices()
    mismatches = {}
    for model, expected in _FIXTURE["routes"].items():
        try:
            got = list(litellm.get_llm_provider(model)[:2])
        except Exception as exc:  # fixture records the exception type name
            got = type(exc).__name__
        if got != expected:
            mismatches[model] = (got, expected)
    assert not mismatches, list(mismatches.items())[:5]


def test_retired_snapshot_never_overwrites_existing_entries(monkeypatch):
    snapshot = cc._load_sdk_retired_snapshot()
    kept_key = "mistral/codestral-2405"
    assert kept_key in snapshot
    sentinel = {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    stripped = {k: v for k, v in litellm.model_cost.items() if k not in snapshot}
    stripped[kept_key] = sentinel
    monkeypatch.setattr(litellm, "model_cost", stripped)
    added = cc.ensure_bundled_provider_prices()
    assert added >= len(snapshot) - 1
    assert litellm.model_cost[kept_key] is sentinel
    assert litellm.model_cost[kept_key] == {
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
    }
    assert cc.ensure_bundled_provider_prices() == 0


def test_retired_snapshot_is_valid_and_declared_in_package_data():
    snapshot = cc._load_sdk_retired_snapshot()
    assert len(snapshot) == 274
    assert not any(k.startswith("openrouter/") for k in snapshot)
    package_data = tomllib.loads((_ROOT / "pyproject.toml").read_text("utf-8"))["tool"][
        "setuptools"
    ]["package-data"]["traigent"]
    assert "utils/litellm_retired_prices_1_103_4.json" in package_data
    # sdist: MANIFEST.in globs every traigent/**/*.json
    assert "recursive-include traigent *.json" in (_ROOT / "MANIFEST.in").read_text(
        "utf-8"
    )


def test_unknown_model_still_unpriced():
    assert not cc.model_pricing_resolves("mistral/invented-zz-none")
