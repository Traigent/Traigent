"""
Progressive feature hints and incentive system for Traigent Edge Analytics mode.
Inspired by DeepEval's approach to encourage cloud adoption.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, cast

from ..api.types import OptimizationStatus
from ..config.types import TraigentConfig
from ..storage.local_storage import LocalStorageManager
from ..utils.logging import get_logger
from .local_analytics import LocalAnalytics, collect_and_submit_analytics

logger = get_logger(__name__)


class IncentiveManager:
    """
    Manages progressive feature hints and incentives for Edge Analytics mode users.

    Tracks usage patterns and shows contextual upgrade hints to encourage
    cloud adoption without being intrusive.
    """

    def __init__(self, config: TraigentConfig) -> None:
        """Initialize incentive manager."""
        self.config = config
        storage_path_option = config.get_local_storage_path()
        if storage_path_option is None:
            storage_path_str = str(Path.home() / ".traigent")
        else:
            storage_path_str = storage_path_option
        self.storage = LocalStorageManager(storage_path_str)
        self._state_file = Path(storage_path_str) / "incentive_state.json"
        self._state: dict[str, Any] = self._load_state()

        # Initialize analytics integration
        self.analytics: LocalAnalytics | None = (
            LocalAnalytics(config) if config.enable_usage_analytics else None
        )

        # Save initial state if file doesn't exist
        if not self._state_file.exists():
            self._save_state()

    def _load_state(self) -> dict[str, Any]:
        """Load incentive state from disk."""
        default_state = {
            "first_use": datetime.now(timezone.utc).isoformat(),
            "total_sessions": 0,
            "total_trials": 0,
            "hints_shown": [],
            "last_hint": None,
            "upgrade_dismissed": False,
            "achievement_unlocked": [],
        }

        if not self._state_file.exists():
            return default_state

        try:
            with open(self._state_file) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return cast(Dict[str, Any], data)
                logger.warning(
                    "Incentive state file had unexpected structure; resetting"
                )
                return default_state
        except Exception as e:
            logger.warning(f"Failed to load incentive state: {e}")
            return default_state  # Return default instead of recursing

    def _save_state(self) -> None:
        """Save incentive state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save incentive state: {e}")

    def update_usage_stats(self) -> None:
        """Update usage statistics."""
        sessions = self.storage.list_sessions()
        completed_sessions = [
            s for s in sessions if s.status == OptimizationStatus.COMPLETED.value
        ]

        self._state["total_sessions"] = len(sessions)
        self._state["total_trials"] = sum(s.completed_trials for s in sessions)
        self._state["completed_sessions"] = len(completed_sessions)

        # Check for achievements
        self._check_achievements(completed_sessions)

        # Submit analytics in background if enabled
        if self.analytics:
            try:
                collect_and_submit_analytics(self.config)
            except Exception as e:
                logger.debug(f"Analytics submission failed: {e}")

        self._save_state()

    def should_show_hint(self, context: str = "general") -> bool:
        """Determine if a hint should be shown based on context and timing."""
        if self._state.get("upgrade_dismissed", False):
            return False

        # Don't show hints too frequently
        last_hint = self._state.get("last_hint")
        if isinstance(last_hint, str):
            last_hint_time = datetime.fromisoformat(last_hint)
            if last_hint_time.tzinfo is None:
                last_hint_time = last_hint_time.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_hint_time < timedelta(hours=24):
                return False

        # Context-specific rules
        completed_raw = self._state.get("completed_sessions", 0)
        completed_count = int(completed_raw or 0)

        hint_triggers: dict[str, bool] = {
            "session_complete": completed_count in [3, 7, 15, 30],
            "cli_usage": completed_count >= 5 and completed_count % 10 == 0,
            "storage_info": completed_count >= 10,
            "general": completed_count in [3, 7, 15, 30]
            or (completed_count >= 5 and completed_count % 5 == 0),
        }

        return bool(hint_triggers.get(context, False))

    def get_contextual_hint(self, context: str = "general") -> str | None:
        """Get a contextual hint message."""
        if not self.should_show_hint(context):
            return None

        completed_count = int(self._state.get("completed_sessions", 0) or 0)
        total_trials = int(self._state.get("total_trials", 0) or 0)

        hints = {
            "session_complete": [
                f"🎉 {completed_count} optimizations completed! Upgrade to Traigent Cloud for:",
                "   • Advanced Bayesian optimization (vs. random search)",
                "   • Web dashboard with beautiful visualizations",
                "   • Team collaboration and sharing features",
                "   • Unlimited optimization trials",
                f"   \n   Your {completed_count} sessions can be synced instantly!",
                "   Run 'traigent login' to get started",
            ],
            "storage_growing": [
                f"📈 Your optimization library is growing! ({completed_count} sessions, {total_trials} trials)",
                "   Cloud benefits at your scale:",
                "   • Cross-project optimization insights",
                "   • Performance regression detection",
                "   • Automated A/B testing workflows",
                "   • Integration with CI/CD pipelines",
            ],
            "power_user": [
                f"🚀 Power user detected! ({completed_count} optimizations completed)",
                "   You're ready for Traigent Cloud's advanced features:",
                "   • Multi-objective Pareto optimization",
                "   • Distributed optimization across multiple GPUs",
                "   • Custom metric plugins and integrations",
                "   • White-label deployment options",
            ],
        }

        # Select appropriate hint category
        if completed_count <= 5:
            hint_key = "session_complete"
        elif completed_count <= 20:
            hint_key = "storage_growing"
        else:
            hint_key = "power_user"

        self._state["last_hint"] = datetime.now(timezone.utc).isoformat()
        self._state["hints_shown"].append(
            {
                "context": context,
                "hint_key": hint_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "completed_sessions": completed_count,
            }
        )

        self._save_state()

        return "\n".join(hints[hint_key])

    def show_achievement_unlock(self, achievement: str) -> str | None:
        """Show achievement unlock message."""
        if achievement in self._state.get("achievement_unlocked", []):
            return None

        achievements = {
            "first_optimization": {
                "title": "🌟 First Optimization Complete!",
                "description": "You've completed your first LLM optimization with Traigent.",
                "hint": "Try running more optimizations to unlock cloud features!",
            },
            "optimization_explorer": {
                "title": "🔍 Optimization Explorer",
                "description": "5 optimizations completed! You're getting the hang of this.",
                "hint": "Traigent Cloud offers 10x faster optimization with advanced algorithms.",
            },
            "efficiency_expert": {
                "title": "⚡ Efficiency Expert",
                "description": "10 optimizations completed! You're optimizing like a pro.",
                "hint": "Cloud users get access to collaborative optimization workspaces.",
            },
            "optimization_master": {
                "title": "🏆 Optimization Master",
                "description": "25 optimizations completed! You've mastered local optimization.",
                "hint": "Ready for enterprise-grade features? Traigent Cloud awaits!",
            },
        }

        if achievement not in achievements:
            return None

        ach = achievements[achievement]
        self._state["achievement_unlocked"].append(achievement)
        self._save_state()

        return f"\n{ach['title']}\n{ach['description']}\n💡 {ach['hint']}\n"

    def _check_achievements(self, completed_sessions: list[Any]) -> None:
        """Check and unlock achievements based on usage."""
        completed_count = len(completed_sessions)

        achievement_thresholds = {
            "first_optimization": 1,
            "optimization_explorer": 5,
            "efficiency_expert": 10,
            "optimization_master": 25,
        }

        for achievement, threshold in achievement_thresholds.items():
            if completed_count >= threshold and achievement not in self._state.get(
                "achievement_unlocked", []
            ):
                message = self.show_achievement_unlock(achievement)
                if message:
                    print(message)

    def get_upgrade_value_proposition(self) -> dict[str, Any]:
        """Get personalized upgrade value proposition based on usage."""
        completed_count = int(self._state.get("completed_sessions", 0) or 0)
        total_trials = int(self._state.get("total_trials", 0) or 0)

        # Get enhanced analytics data if available
        if self.analytics:
            try:
                analytics_data = self.analytics.get_cloud_incentive_data()
                if analytics_data:
                    # Return analytics-enhanced value proposition
                    return analytics_data
            except Exception as e:
                logger.debug(f"Failed to get analytics-enhanced value proposition: {e}")

        # Fallback to basic value proposition calculation
        avg_trials_per_session = total_trials / max(completed_count, 1)
        estimated_time_saved_hours = (
            completed_count * 0.5
        )  # Assume 30min saved per optimization

        # Estimate cloud benefits
        cloud_trial_speedup = 3  # Cloud is 3x faster on average
        cloud_accuracy_improvement = 0.15  # 15% better accuracy

        value_props = {
            "time_savings": {
                "current_time_investment": f"{estimated_time_saved_hours:.1f} hours",
                "potential_cloud_savings": f"{estimated_time_saved_hours * cloud_trial_speedup:.1f} hours saved",
                "roi_message": f"Cloud could save you {estimated_time_saved_hours * 2:.1f} additional hours",
            },
            "performance_gains": {
                "current_avg_trials": f"{avg_trials_per_session:.1f} trials per optimization",
                "cloud_improvement": f"Up to {cloud_accuracy_improvement * 100:.0f}% better results",
                "advanced_algorithms": "Bayesian optimization vs. random search",
            },
            "scale_benefits": {
                "current_sessions": completed_count,
                "collaboration_value": "Share optimizations across your team",
                "automation_value": "Automated optimization workflows",
                "integration_value": "CI/CD pipeline integration",
            },
        }

        return value_props

    def show_context_sensitive_hint(self, context: str, **kwargs) -> None:
        """Show context-sensitive hints during specific actions."""
        hints = {
            "large_config_space": [
                "💡 Large configuration space detected!",
                "   Edge Analytics mode uses random search, but Traigent Cloud offers:",
                "   • Intelligent Bayesian optimization",
                "   • 3-5x faster convergence",
                "   • Better handling of high-dimensional spaces",
            ],
            "multiple_objectives": [
                "🎯 Multi-objective optimization detected!",
                "   Traigent Cloud provides advanced features:",
                "   • True Pareto front optimization",
                "   • Interactive objective trade-off analysis",
                "   • Automated constraint handling",
            ],
            "slow_evaluation": [
                "⏱️ Slow evaluations detected!",
                "   Cloud features that can help:",
                "   • Parallel evaluation across multiple instances",
                "   • Intelligent early stopping",
                "   • Cached evaluation results",
            ],
            "high_trial_count": [
                f"🔢 High trial count optimization ({kwargs.get('trial_count', 'many')} trials)!",
                "   Perfect use case for Traigent Cloud:",
                "   • Distributed optimization",
                "   • Progress tracking and visualization",
                "   • Automatic result analysis",
            ],
        }

        if context in hints and self.should_show_hint(context):
            hint_message = "\n".join(hints[context])
            print(f"Upgrade hint: {hint_message}")
            self._state["last_hint"] = datetime.now(timezone.utc).isoformat()
            self._save_state()

    def dismiss_upgrade_hints(self) -> None:
        """Allow user to dismiss upgrade hints."""
        self._state["upgrade_dismissed"] = True
        self._state["dismiss_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        print(
            "✅ Upgrade hints dismissed. You can re-enable them by deleting the incentive_state.json file."
        )

    def get_local_vs_cloud_comparison(self) -> str:
        """Get a comparison table of local vs cloud features."""
        comparison = """
┌─────────────────────────────────┬──────────────┬─────────────────┐
│ Feature                         │ Local Mode   │ Traigent Cloud  │
├─────────────────────────────────┼──────────────┼─────────────────┤
│ Optimization Algorithm          │ Random/Grid  │ Bayesian/Multi  │
│ Max Trials per Session          │ 20           │ Unlimited       │
│ Web Dashboard                   │ ❌           │ ✅              │
│ Team Collaboration              │ ❌           │ ✅              │
│ Results Persistence             │ Local Files  │ Cloud Storage   │
│ Performance Analytics           │ Basic        │ Advanced        │
│ Multi-objective Optimization    │ Basic        │ Pareto-optimal  │
│ Parallel Execution              │ ❌           │ ✅              │
│ Integration APIs                │ Limited      │ Full REST API   │
│ Support                         │ Community    │ Priority        │
└─────────────────────────────────┴──────────────┴─────────────────┘
"""
        return comparison

    def show_onboarding_tips(self) -> list[str]:
        """Show helpful tips for new users."""
        tips = [
            "💡 Pro tip: Use environment variables to configure local storage:",
            "   export TRAIGENT_RESULTS_FOLDER='./my_optimizations'",
            "",
            "🔧 Quick commands:",
            "   traigent local list          # List your optimizations",
            "   traigent local show <id>     # View detailed results",
            "   traigent local info          # Storage information",
            "",
            "🚀 Ready for more? Run 'traigent login' to unlock cloud features!",
        ]
        return tips


def show_upgrade_hint(context: str = "general", **kwargs) -> None:
    """Global function to show upgrade hints."""
    config = TraigentConfig.from_environment()
    if config.is_edge_analytics_mode():
        incentive_manager = IncentiveManager(config)
        incentive_manager.update_usage_stats()

        hint = incentive_manager.get_contextual_hint(context)
        if hint:
            logger.info(f"Incentive hint: {hint}")

        # Show context-sensitive hints
        incentive_manager.show_context_sensitive_hint(context, **kwargs)


def show_achievement(achievement: str) -> None:
    """Global function to show achievement unlock."""
    config = TraigentConfig.from_environment()
    if config.is_edge_analytics_mode():
        incentive_manager = IncentiveManager(config)
        message = incentive_manager.show_achievement_unlock(achievement)
        if message:
            logger.info(message)
