"""Options for client-side guided generation (Pydantic, extra='forbid')."""

from __future__ import annotations

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


__all__ = ["PromptRewriteOptions", "DatasetGrowthOptions"]
