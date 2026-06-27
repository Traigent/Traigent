"""TVL 1.1 loader support (RFC 0001): cvars, policies, require_calibration.

The SDK loader accepts the new sections, exposes them as RAW declarations on
the artifact (typed knob bindings are the traigent.knobs surface's job), and
keeps CVARs OUT of the configuration space (P2/P5: governed, not searched).
"""

from __future__ import annotations

import textwrap

import pytest

from traigent.tvl.models import PromotionPolicy, RequireCalibration
from traigent.tvl.spec_loader import TVLValidationError, load_tvl_spec

SPEC_11 = textwrap.dedent(
    """
    tvl:
      module: corp.test.tvl11

    environment:
      snapshot_id: "2026-06-05T00:00:00Z"

    evaluation_set:
      dataset: s3://datasets/test/dev.parquet

    tvars:
      - name: model
        type: enum[str]
        domain: ["gpt-4o-mini", "gpt-4o"]
      - name: retriever.k
        type: int
        domain:
          range: [0, 20]

    cvars:
      - name: router.margin_threshold
        type: float
        domain:
          range: [0.0, 1.0]
        calibration:
          source: margin_eval_pool
          signal: vote_margin_v1
          depends_on: [model, retriever.k]
        governance:
          require_calibration: true

    policies:
      - name: cheap_strong_cascade
        kind: policy
        strategy: cascade
        stages: [cheap, strong]
        gates:
          - kind: margin_below
            threshold: router.margin_threshold

    constraints:
      structural: []

    objectives:
      - name: quality
        direction: maximize

    promotion_policy:
      dominance: epsilon_pareto
      alpha: 0.05
      min_effect:
        quality: 0.02
      require_calibration:
        enabled: true
        hash_covered_context: [model_versions]
    """
)


def _write(tmp_path, content: str):
    spec_path = tmp_path / "spec.tvl.yml"
    spec_path.write_text(content, encoding="utf-8")
    return spec_path


def test_tvl11_spec_loads(tmp_path):
    artifact = load_tvl_spec(spec_path=_write(tmp_path, SPEC_11))
    assert artifact.cvars is not None and len(artifact.cvars) == 1
    assert artifact.cvars[0]["name"] == "router.margin_threshold"
    assert artifact.policies is not None
    assert artifact.policies[0]["strategy"] == "cascade"


def test_cvars_never_enter_configuration_space(tmp_path):
    """P2/P5: governed, not searched."""
    artifact = load_tvl_spec(spec_path=_write(tmp_path, SPEC_11))
    assert set(artifact.configuration_space) == {"model", "retriever.k"}
    assert "router.margin_threshold" not in artifact.configuration_space


def test_require_calibration_parsed(tmp_path):
    artifact = load_tvl_spec(spec_path=_write(tmp_path, SPEC_11))
    policy = artifact.promotion_policy
    assert isinstance(policy, PromotionPolicy)
    assert isinstance(policy.require_calibration, RequireCalibration)
    assert policy.require_calibration.enabled is True
    assert policy.require_calibration.hash_covered_context == ["model_versions"]


def test_require_calibration_rejects_core_keys(tmp_path):
    bad = SPEC_11.replace(
        "hash_covered_context: [model_versions]",
        "hash_covered_context: [tuned_parent_values]",
    )
    with pytest.raises(
        TVLValidationError, match="not extension keys|Invalid promotion_policy"
    ):
        load_tvl_spec(spec_path=_write(tmp_path, bad))


def test_cvar_shadowing_tvar_rejected(tmp_path):
    bad = SPEC_11.replace("name: router.margin_threshold", "name: model")
    with pytest.raises(TVLValidationError, match="shadows a TVAR"):
        load_tvl_spec(spec_path=_write(tmp_path, bad))


def test_cvar_missing_source_rejected(tmp_path):
    bad = SPEC_11.replace("source: margin_eval_pool", "signal_only: x")
    with pytest.raises(TVLValidationError, match="calibration.source"):
        load_tvl_spec(spec_path=_write(tmp_path, bad))


def test_policy_kind_literal_enforced(tmp_path):
    bad = SPEC_11.replace("kind: policy", "kind: router")
    with pytest.raises(TVLValidationError, match="literal 'policy'"):
        load_tvl_spec(spec_path=_write(tmp_path, bad))


def test_legacy_spec_without_11_sections_unchanged(tmp_path):
    """P1: a 1.0 spec loads exactly as before — cvars/policies stay None."""
    legacy_spec = textwrap.dedent(
        """
        tvl:
          module: corp.test.legacy

        environment:
          snapshot_id: "2026-06-05T00:00:00Z"

        evaluation_set:
          dataset: s3://datasets/test/dev.parquet

        tvars:
          - name: model
            type: enum[str]
            domain: ["a", "b"]

        constraints:
          structural: []

        objectives:
          - name: quality
            direction: maximize

        promotion_policy:
          dominance: epsilon_pareto
          alpha: 0.05
          min_effect:
            quality: 0.02
        """
    )
    artifact = load_tvl_spec(spec_path=_write(tmp_path, legacy_spec))
    assert artifact.cvars is None
    assert artifact.policies is None
    assert artifact.promotion_policy.require_calibration is None
