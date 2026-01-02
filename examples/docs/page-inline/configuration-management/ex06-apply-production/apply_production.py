#!/usr/bin/env python3
"""Example: Production Configuration Management with Traigent."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")


class OptimizedChatService:
    """Production-ready chat service with Traigent configuration management."""

    def __init__(self, config_path: str | None = None):
        """Initialize service with configuration management.

        Args:
            config_path: Path to saved configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        self._setup_optimized_functions()

    def _load_configuration(self) -> dict[str, Any]:
        """Load configuration based on environment."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path) as f:
                return json.load(f)

        # Default configuration if no saved config exists
        return {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 200,
            "response_style": "professional",
        }

    def _setup_optimized_functions(self):
        """Set up Traigent-optimized functions."""

        # Define the optimized greeting function
        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
                "temperature": [0.1, 0.3, 0.5],
                "max_tokens": [50, 100, 150],
                "response_style": ["friendly", "professional", "casual"],
            },
            objectives=["cost", "quality"],
            execution_mode="edge_analytics",
            max_trials=10,
        )
        def generate_greeting(name: str) -> str:
            """Generate personalized greeting."""
            config = self.config

            style_prompt = {
                "friendly": "in a warm and friendly manner",
                "professional": "in a professional business tone",
                "casual": "in a casual, relaxed way",
            }.get(config.get("response_style", "professional"))

            llm = ChatOpenAI(
                model=config.get("model", "gpt-3.5-turbo"),
                temperature=config.get("temperature", 0.3),
                model_kwargs={"max_tokens": config.get("max_tokens", 100)},
            )

            prompt = f"Generate a greeting for {name} {style_prompt}."
            response = llm.invoke([HumanMessage(content=prompt)])
            return getattr(response, "content", str(response))

        # Define the optimized response function
        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.3, 0.5, 0.7, 0.9],
                "max_tokens": [200, 300, 500],
                "creativity_mode": ["low", "medium", "high"],
            },
            eval_dataset=os.path.join(
                os.path.dirname(__file__), "production_queries.jsonl"
            ),
            objectives=["cost", "quality", "response_time"],
            execution_mode="edge_analytics",
            max_trials=10,
        )
        def generate_response(query: str, context: str = "") -> str:
            """Generate response using optimized parameters."""
            config = self.config

            creativity_map = {"low": 0.1, "medium": 0.5, "high": 0.9}

            llm = ChatOpenAI(
                model=config.get("model", "gpt-3.5-turbo"),
                temperature=creativity_map.get(
                    config.get("creativity_mode", "medium"),
                    config.get("temperature", 0.5),
                ),
                model_kwargs={"max_tokens": config.get("max_tokens", 300)},
            )

            prompt = f"""Context: {context}
            User Query: {query}

            Please provide a helpful and accurate response."""

            response = llm.invoke([HumanMessage(content=prompt)])
            return getattr(response, "content", str(response))

        # Apply saved configuration if available
        if self.config_path:
            generate_greeting.set_config(self.config)
            generate_response.set_config(self.config)

        self.generate_greeting = generate_greeting
        self.generate_response = generate_response

    def update_config(self, new_config_path: str):
        """Hot-swap configuration without restarting.

        Args:
            new_config_path: Path to new configuration file
        """
        with open(new_config_path) as f:
            self.config = json.load(f)

        # Update all optimized functions
        self.generate_greeting.set_config(self.config)
        self.generate_response.set_config(self.config)

        print(f"Configuration updated from {new_config_path}")

    def save_current_config(self, save_path: str):
        """Save current configuration to file.

        Args:
            save_path: Path where to save configuration
        """
        with open(save_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {save_path}")


# Production usage examples
class ProductionDeployment:
    """Example production deployment patterns."""

    @staticmethod
    def environment_based_config():
        """Load configuration based on environment."""
        environment = os.getenv("ENVIRONMENT", "development")

        config_map = {
            "development": "configs/dev_config.json",
            "staging": "configs/staging_config.json",
            "production": "configs/prod_config.json",
        }

        config_file = config_map.get(environment, "configs/dev_config.json")
        return OptimizedChatService(config_path=config_file)

    @staticmethod
    def ab_testing_setup():
        """Set up A/B testing with different configurations."""
        # Create services with different configs
        service_a = OptimizedChatService("configs/variant_a.json")
        service_b = OptimizedChatService("configs/variant_b.json")

        # Use based on user segment or random assignment
        import random

        selected_service = service_a if random.random() < 0.5 else service_b
        return selected_service

    @staticmethod
    def gradual_rollout():
        """Gradual rollout of new configuration."""
        import random

        # 10% of traffic gets new config
        rollout_percentage = 0.1

        if random.random() < rollout_percentage:
            return OptimizedChatService("configs/new_config.json")
        else:
            return OptimizedChatService("configs/stable_config.json")


def demo_production_usage():
    """Demonstrate production usage patterns."""
    print("Traigent Production Configuration Management")
    print("=" * 50)

    # Example 1: Basic production service
    print("\n1. Basic Production Service:")
    service = OptimizedChatService()
    greeting = service.generate_greeting("Alice")
    print(f"Greeting: {greeting[:100]}...")

    # Example 2: Load saved configuration
    print("\n2. Using Saved Configuration:")
    saved_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 250,
        "response_style": "professional",
        "creativity_mode": "medium",
    }

    # Save config
    with open("production_config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    # Create service with saved config
    prod_service = OptimizedChatService("production_config.json")
    response = prod_service.generate_response(
        "What are the benefits of cloud computing?", context="Enterprise IT discussion"
    )
    print(f"Response: {response[:150]}...")

    # Example 3: Hot-swap configuration
    print("\n3. Hot-Swap Configuration:")
    new_config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
        "max_tokens": 200,
        "response_style": "friendly",
        "creativity_mode": "high",
    }

    with open("updated_config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    prod_service.update_config("updated_config.json")
    print("Configuration updated successfully!")

    return prod_service


if __name__ == "__main__":
    # Run demonstration
    service = demo_production_usage()

    print("\n" + "=" * 50)
    print("Production deployment ready with optimal configuration!")
