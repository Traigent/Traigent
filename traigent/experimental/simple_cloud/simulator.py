"""⚠️  EXPERIMENTAL: Simple Cloud Simulator - NOT FOR PRODUCTION.

🚨 WARNING: This is a NAIVE simulation of cloud execution for local testing only.
This is NOT the real Traigent cloud implementation and does NOT represent
Traigent's proprietary IP.

This module provides a simple orchestrator for testing platform integrations
locally while the OptiGen backend is under development.

Real Traigent cloud execution happens in the OptiGen backend (proprietary).
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import warnings
from typing import Any

from traigent.utils.logging import get_logger

# Import experimental platform executors
try:
    from .platforms.anthropic_executor import AnthropicAgentExecutor

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from .platforms.cohere_executor import CohereAgentExecutor

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from .platforms.huggingface_executor import HuggingFaceAgentExecutor

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

logger = get_logger(__name__)


class SimpleCloudSimulator:
    """⚠️  EXPERIMENTAL: Naive cloud simulation for local testing.

    🚨 WARNING: This is NOT the real Traigent cloud!

    This is a simplified simulator that helps test platform integrations
    locally while the OptiGen backend is being developed.

    Features that are MISSING (present in real Traigent cloud):
    - Advanced optimization algorithms
    - Smart subset selection
    - Cost optimization
    - Massive parallelization
    - Enterprise security
    - Real-time analytics
    - Multi-objective optimization
    - And much more...
    """

    def __init__(self) -> None:
        """Initialize the simple cloud simulator."""
        # Issue warning
        warnings.warn(
            "SimpleCloudSimulator is experimental and NOT for production use. "
            "This is NOT the real Traigent cloud implementation.",
            UserWarning,
            stacklevel=2,
        )

        self.available_platforms: dict[str, Any] = {}
        self._register_available_platforms()

        logger.warning(
            "🚨 Using EXPERIMENTAL SimpleCloudSimulator - NOT production cloud!"
        )

    def _register_available_platforms(self) -> None:
        """Register available experimental platform executors."""
        if ANTHROPIC_AVAILABLE:
            self.available_platforms["anthropic"] = AnthropicAgentExecutor

        if COHERE_AVAILABLE:
            self.available_platforms["cohere"] = CohereAgentExecutor

        if HUGGINGFACE_AVAILABLE:
            self.available_platforms["huggingface"] = HuggingFaceAgentExecutor

    def get_available_platforms(self) -> list[str]:
        """Get list of available platforms for testing.

        Returns:
            List of platform names
        """
        return list(self.available_platforms.keys())

    async def test_platform_completion(
        self, platform: str, prompt: str, **kwargs
    ) -> dict[str, Any]:
        """Test a simple completion on a platform.

        🚨 WARNING: This is for testing only!

        Args:
            platform: Platform name (anthropic, cohere, huggingface)
            prompt: Test prompt
            **kwargs: Platform-specific parameters

        Returns:
            Test result dictionary
        """
        if platform not in self.available_platforms:
            raise ValueError(f"Platform {platform} not available for testing") from None

        executor_class = self.available_platforms[platform]
        executor = executor_class(**kwargs)

        try:
            result = await executor.complete(prompt, **kwargs)
            return {
                "platform": platform,
                "prompt": prompt,
                "result": result,
                "success": True,
                "warning": "🚨 This is experimental testing, not real Traigent cloud!",
            }
        except Exception as e:
            return {
                "platform": platform,
                "prompt": prompt,
                "result": None,
                "success": False,
                "error": str(e),
                "warning": "🚨 This is experimental testing, not real Traigent cloud!",
            }

    async def test_parameter_mapping(self, platform: str) -> dict[str, Any]:
        """Test parameter mapping for a platform.

        Args:
            platform: Platform name

        Returns:
            Parameter mapping test results
        """
        if platform not in self.available_platforms:
            raise ValueError(f"Platform {platform} not available for testing") from None

        executor_class = self.available_platforms[platform]
        executor = executor_class()

        # Test with various unified parameters
        test_params = {
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }

        try:
            mapped_params = executor.translate_parameters(test_params)
            return {
                "platform": platform,
                "unified_params": test_params,
                "mapped_params": mapped_params,
                "success": True,
                "warning": "🚨 This is experimental testing, not real Traigent cloud!",
            }
        except Exception as e:
            return {
                "platform": platform,
                "unified_params": test_params,
                "mapped_params": None,
                "success": False,
                "error": str(e),
                "warning": "🚨 This is experimental testing, not real Traigent cloud!",
            }

    def get_disclaimer(self) -> str:
        """Get important disclaimer about this experimental module.

        Returns:
            Disclaimer text
        """
        return """
⚠️  EXPERIMENTAL SIMPLE CLOUD SIMULATOR - NOT FOR PRODUCTION

🚨 IMPORTANT DISCLAIMERS:
- This is NOT the real Traigent cloud implementation
- This is NOT suitable for production use
- This does NOT represent Traigent's proprietary IP
- This is a naive, simplified testing utility only

REAL TRAIGENT FEATURES (not in this simulator):
- Advanced optimization algorithms
- Smart cost optimization
- Massive parallelization
- Enterprise security
- Real-time analytics
- Multi-objective optimization
- Intelligent subset selection
- Advanced caching and performance optimization

For production use:
1. Use @traigent.optimize(auto_override_frameworks=True) for seamless integration
2. Use Traigent cloud services (via OptiGen backend) when available
3. Do NOT rely on this experimental simulator

This module exists only to help with local development and testing
while the OptiGen backend is being built.
        """


# Global instance for easy access during testing
_simple_cloud_simulator = None


def get_simple_cloud_simulator() -> SimpleCloudSimulator:
    """Get global simple cloud simulator instance.

    🚨 WARNING: This is experimental and NOT for production!

    Returns:
        SimpleCloudSimulator instance
    """
    global _simple_cloud_simulator
    if _simple_cloud_simulator is None:
        _simple_cloud_simulator = SimpleCloudSimulator()
    return _simple_cloud_simulator
