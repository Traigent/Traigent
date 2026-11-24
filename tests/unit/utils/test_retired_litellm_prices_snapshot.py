"""litellm >= 1.103.10 dropped 736 cost-map ids that 1.103.4 priced.

The SDK restores them from its snapshots (OpenRouter + retired non-OpenRouter).
The fixture was generated with the SDK's own ``cost_from_tokens(100, 10, id)``
and ``litellm.get_llm_provider(id)`` under litellm 1.103.4; these tests pin
the restored pricing and routing to those values.

The pins check the SDK's own snapshot data, not whatever the installed litellm
ships: ``_snapshot_only_cost_map`` removes every snapshot id from litellm's
table first, so a later litellm that re-adds a retired id under new pricing
(for example moonshot/kimi-latest, an alias of kimi-k3 since 1.110) cannot
change the outcome. ``ensure_bundled_provider_prices`` never overrides live
entries, so the restored value is what the SDK would price if litellm dropped
the id again.
"""

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")

from litellm._internal_context import pinned_billing_time  # noqa: E402

from traigent.utils import cost_calculator as cc  # noqa: E402

# litellm prices from its billing clock (litellm._internal_context.
# current_billing_time, which reads datetime.now(UTC)): a snapshot entry's
# off_peak_pricing applies inside its window (hy3: 16:00-00:00 UTC every day;
# deepseek-v4.1-flash: weekday windows), and moonshot/kimi-latest stops pricing
# after its deprecation_date (2026-01-28). The fixture was generated at a
# fixed moment, so the autouse _pin_billing_time fixture pins the clock to
# Monday 2026-01-05 12:00Z, a fixed moment matching how the fixture was
# generated (hy3 peak; deepseek-v4.1-flash off-peak; kimi pre-deprecation).
_PEAK_TIME = datetime(2026, 1, 14, 12, 0, tzinfo=UTC)  # Wed: hy3 peak
_OFF_PEAK_TIME = datetime(2026, 1, 14, 20, 0, tzinfo=UTC)  # Wed: hy3 off-peak
_DEEPSEEK = "openrouter/deepseek/deepseek-v4.1-flash"
_DEEPSEEK_PEAK_TIME = datetime(2026, 1, 14, 2, 0, tzinfo=UTC)  # weekday 01:00-04:00

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "litellm_1_103_4_removed_ids_pricing.json"
    ).read_text("utf-8")
)


@pytest.fixture(autouse=True)
def _pin_billing_time():
    """Price at a fixed moment matching how the fixture was generated.

    Some restored OpenRouter entries (openrouter/tencent/hy3) carry an
    ``off_peak_pricing`` block that litellm applies by wall-clock UTC
    (16:00-00:00). The fixture holds the rates as of the pinned moment, so
    without a pinned clock these tests drift with the wall clock.
    """
    with pinned_billing_time(datetime(2026, 1, 5, 12, 0, tzinfo=UTC)):
        yield


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


@pytest.fixture
def _snapshot_only_cost_map(monkeypatch):
    """litellm's table without any id the SDK snapshots restore."""
    snapshot_ids = set(cc._load_sdk_retired_snapshot()) | set(
        cc._load_sdk_openrouter_snapshot()
    )
    stripped = {k: v for k, v in litellm.model_cost.items() if k not in snapshot_ids}
    monkeypatch.setattr(litellm, "model_cost", stripped)
    return stripped


def test_fixture_covers_all_736_removed_ids():
    assert len(_FIXTURE["prices"]) == 736
    assert all(isinstance(v, list) for v in _FIXTURE["prices"].values())


def test_every_restored_id_prices_as_under_litellm_1_103_4(_snapshot_only_cost_map):
    cc.ensure_bundled_provider_prices()
    mismatches = {}
    for model, expected in _FIXTURE["prices"].items():
        got = list(cc.cost_from_tokens(100, 10, model, strict=True))
        if got != expected:
            mismatches[model] = (got, expected)
    assert not mismatches, list(mismatches.items())[:5]


def test_off_peak_pricing_is_kept_for_restored_openrouter_entry():
    """The freeze above hides nothing: at an off-peak moment hy3 prices at the
    snapshot's ``off_peak_pricing`` rates."""
    cc.ensure_bundled_provider_prices()
    entry = cc._load_sdk_openrouter_snapshot()["openrouter/tencent/hy3"]
    off_peak = entry["off_peak_pricing"]
    assert off_peak["hours_utc"] == "16:00-00:00"
    with pinned_billing_time(_OFF_PEAK_TIME):
        got = cc.cost_from_tokens(100, 10, "openrouter/tencent/hy3", strict=True)
    assert got == pytest.approx(
        (100 * off_peak["input_cost_per_token"], 10 * off_peak["output_cost_per_token"])
    )
    with pinned_billing_time(_PEAK_TIME):
        peak = cc.cost_from_tokens(100, 10, "openrouter/tencent/hy3", strict=True)
    assert peak == pytest.approx(
        (100 * entry["input_cost_per_token"], 10 * entry["output_cost_per_token"])
    )
    assert got != peak


def test_off_peak_weekday_windows_are_kept_for_restored_deepseek_entry():
    cc.ensure_bundled_provider_prices()
    entry = cc._load_sdk_openrouter_snapshot()[_DEEPSEEK]
    off_peak = entry["off_peak_pricing"]
    with pinned_billing_time(_PEAK_TIME):  # Wed 12:00Z is inside 10:00-00:00
        got = cc.cost_from_tokens(100, 10, _DEEPSEEK, strict=True)
    assert got == pytest.approx(
        (100 * off_peak["input_cost_per_token"], 10 * off_peak["output_cost_per_token"])
    )
    with pinned_billing_time(_DEEPSEEK_PEAK_TIME):  # Wed 02:00Z is in no window
        base = cc.cost_from_tokens(100, 10, _DEEPSEEK, strict=True)
    assert base == pytest.approx(
        (100 * entry["input_cost_per_token"], 10 * entry["output_cost_per_token"])
    )
    assert got != base


def test_every_restored_id_routes_as_under_litellm_1_103_4(_snapshot_only_cost_map):
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
