"""Prompt templates for the LangGraph document routing demo."""

from .financial_prompts import FINANCIAL_PROMPTS
from .legal_prompts import LEGAL_PROMPTS
from .router_prompts import ROUTER_PROMPTS

__all__ = ["ROUTER_PROMPTS", "LEGAL_PROMPTS", "FINANCIAL_PROMPTS"]
