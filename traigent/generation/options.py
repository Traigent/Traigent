"""Options for client-side guided generation (Pydantic, extra='forbid')."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptRewriteOptions(BaseModel):
    """Controls for LLM prompt rewrite that folds candidates into the config space."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(1, ge=1, le=20)
    candidates_per_round: int = Field(3, ge=1, le=50)
    max_total_candidates: int = Field(20, ge=1, le=500)
    prompt_param: str | None = Field(
        None,
        description="Config-space parameter holding the prompt Choices; required if ambiguous.",
    )
    rewrite_model: str | None = Field(
        None, description="Hint passed to the user's LLM provider."
    )
    privacy_mode: bool = Field(
        True,
        description="When True, generation runs only on the user's LLM; content never leaves the client.",
    )


class DatasetGrowthOptions(BaseModel):
    """Controls for benchmark example synthesis that grows the eval dataset."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(1, ge=1, le=20)
    examples_per_round: int = Field(5, ge=1, le=200)
    max_total_examples_added: int = Field(50, ge=1, le=5000)
    gen_model: str | None = Field(
        None, description="Hint passed to the user's LLM provider."
    )
    privacy_mode: bool = Field(
        True,
        description="When True, synthesis runs only on the user's LLM; content never leaves the client.",
    )


class SkillTrainOptions(BaseModel):
    """Controls for local trajectory-driven skill-document training."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(4, ge=1, le=20)
    rollout_batch: int = Field(40, ge=4, le=500)
    steps_per_epoch: int = Field(1, ge=1, le=20)
    reflection_minibatch: int = Field(8, ge=2, le=32)
    edit_budget: int = Field(4, ge=1, le=16)
    edit_budget_floor: int = Field(2, ge=1, le=16)
    edit_budget_schedule: Literal["constant", "cosine"] = "cosine"
    rejected_buffer: bool = True
    rejected_buffer_max: int = Field(50, ge=0, le=500)
    slow_update: bool = True
    slow_update_probe_size: int = Field(20, ge=4, le=100)
    meta_skill: bool = True
    selection_split: float = Field(0.2, gt=0.0, lt=0.9)
    test_split: float = Field(0.0, ge=0.0, lt=0.5)
    split_seed: int = 42
    score_metric: str | None = Field(
        None,
        description=(
            "Metric used for aggregate scoring and per-example failure checks. "
            "When omitted, train_skill resolves the trial score from result.best_score, "
            "then trial metrics by primary objective/accuracy/score/first numeric metric."
        ),
    )
    failure_threshold: float = Field(
        1.0,
        description=(
            "A rollout is treated as a failure when its success flag is false or "
            "metrics[score_metric] is below this threshold for maximize metrics, "
            "or above this threshold for minimize metrics."
        ),
    )
    higher_is_better: bool = Field(
        True,
        description=(
            "Controls strict gate direction: True accepts strictly higher selection "
            "scores; False accepts strictly lower selection scores."
        ),
    )
    optimizer_model: str | None = Field(
        None,
        description=(
            "Advisory model hint for the caller's optimizer LLM provider; used only "
            "when the supplied RewriteLLM complete() method accepts a model hint."
        ),
    )
    privacy_mode: bool = Field(
        True,
        description="When True, generation runs only on the user's LLM; content never leaves the client.",
    )
    max_optimizer_calls: int | None = Field(None, ge=0)
    max_gate_evaluations: int | None = Field(None, ge=0)
    doc_param: str | None = None
    artifacts_dir: str | None = None
    write_artifacts: bool = Field(
        True,
        description=(
            "When True, write local skill-training artifacts. When False, no "
            "best_skill.md, reports, logs, or artifact directories are written."
        ),
    )


__all__ = ["PromptRewriteOptions", "DatasetGrowthOptions", "SkillTrainOptions"]
