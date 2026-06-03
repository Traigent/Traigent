"""Built-in effectuation strategies."""

from traigent.effectuation.strategies.framework_param import (
    FrameworkParamEffect,
    FrameworkParamStrategy,
)
from traigent.effectuation.strategies.self_consistency import (
    SELF_CONSISTENCY_NAMES,
    SelfConsistencyEffect,
    SelfConsistencyStrategy,
    majority_vote,
)

__all__ = [
    "FrameworkParamEffect",
    "FrameworkParamStrategy",
    "SELF_CONSISTENCY_NAMES",
    "SelfConsistencyEffect",
    "SelfConsistencyStrategy",
    "majority_vote",
]
