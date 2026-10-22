"""#2652: bundled OpenRouter prices survive live-price mode and import order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

litellm = pytest.importorskip("litellm")

_PROBE = """
import json, sys
if sys.argv[1] == "litellm-first":
    import litellm
    litellm.model_cost = {k: v for k, v in litellm.model_cost.items() if not k.startswith("openrouter/")}
import traigent
import litellm
from traigent.utils import cost_calculator as cc
if sys.argv[1] == "live":
    litellm.model_cost = {k: v for k, v in litellm.model_cost.items() if not k.startswith("openrouter/")}
    cc.ensure_bundled_provider_prices()
bundled_or = [k for k in litellm.model_cost if k.startswith("openrouter/")]
print(json.dumps({"n": len(bundled_or), "unknown": cc.model_pricing_resolves("openrouter/invented/zz-none")}))
"""


@pytest.mark.parametrize("mode", ["default", "litellm-first", "live"])
def test_openrouter_entries_present_in_every_mode(mode):
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    if mode == "live":
        env["TRAIGENT_LITELLM_LIVE_PRICES"] = "1"
        env["LITELLM_MODEL_COST_MAP_URL"] = "http://127.0.0.1:9/x.json"
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, mode],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
        check=True,
    )
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["n"] > 0
    assert data["unknown"] is False  # unknown stays unpriced, never $0


def test_merge_adds_missing_without_overriding(monkeypatch):
    from traigent.utils import cost_calculator as cc

    stripped = {
        k: v for k, v in litellm.model_cost.items() if not k.startswith("openrouter/")
    }
    sentinel = {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    stripped["openrouter/live/keep-me"] = sentinel
    monkeypatch.setattr(litellm, "model_cost", stripped)
    added = cc.ensure_bundled_provider_prices()
    assert added > 0
    assert litellm.model_cost["openrouter/live/keep-me"] is sentinel
    assert cc.ensure_bundled_provider_prices() == 0
    assert not cc.model_pricing_resolves("openrouter/invented/zz-none")


def test_sdk_openrouter_snapshot_is_valid_and_declared_in_package_data():
    """The SDK-owned snapshot exists, parses, and ships in the wheel."""
    import tomllib

    from traigent.utils import cost_calculator as cc

    snapshot = cc._load_sdk_openrouter_snapshot()
    assert len(snapshot) > 100
    assert all(k.startswith("openrouter/") for k in snapshot)

    root = Path(__file__).resolve().parents[3]
    package_data = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["tool"][
        "setuptools"
    ]["package-data"]["traigent"]
    assert "utils/openrouter_prices_litellm_1_103_4.json" in package_data
    rel = cc._SDK_OPENROUTER_SNAPSHOT.relative_to(root / "traigent").as_posix()
    assert rel in package_data


def test_litellm_price_wins_over_sdk_snapshot(monkeypatch):
    from traigent.utils import cost_calculator as cc

    key = next(iter(cc._load_sdk_openrouter_snapshot()))
    sentinel = {"input_cost_per_token": 9.0, "output_cost_per_token": 9.0}
    monkeypatch.setattr(litellm, "model_cost", {key: sentinel})
    cc.ensure_bundled_provider_prices()
    assert litellm.model_cost[key] is sentinel


def test_provenance_names_snapshot_for_supplied_models_only(monkeypatch):
    from traigent.utils import cost_calculator as cc

    snap_key = next(iter(cc._load_sdk_openrouter_snapshot()))
    held = {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
    table = {
        k: v for k, v in litellm.model_cost.items() if not k.startswith("openrouter/")
    }
    table["openrouter/live/held-by-litellm"] = held
    monkeypatch.setattr(litellm, "model_cost", table)
    monkeypatch.setattr(cc, "_sdk_snapshot_supplied", {})
    # Older litellm bundles openrouter/* itself; isolate the SDK snapshot path.
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    monkeypatch.setattr(
        GetModelCostMap, "load_local_model_cost_map", staticmethod(lambda: {})
    )
    cc.ensure_bundled_provider_prices()

    assert cc.get_model_token_pricing(snap_key)[2] == "sdk_snapshot"
    assert cc.get_model_token_pricing("openrouter/live/held-by-litellm")[2] == "litellm"
    assert cc.get_model_token_pricing("gpt-4.1-mini")[2] == "litellm"
    prov = cc.get_pricing_provenance()
    assert prov["sdk_snapshot_version"].startswith("litellm-")
    assert prov["sdk_snapshot_models"] > 0
