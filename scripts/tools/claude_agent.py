#!/usr/bin/env python3
"""
Claude Agent Helper for TraiGent/OptiGen Multi-Project Communication

This module provides a helper class for Claude Code to communicate between
projects using Redis as a message broker. It enables Claude to report issues,
notify about dependencies, suggest improvements, and coordinate work across
the entire TraiGent/OptiGen ecosystem.

Usage:
    from tools.claude_agent import ClaudeAgent

    agent = ClaudeAgent("project-name")
    agent.report_issue(
        target="other-project",
        issue_type="api_mismatch",
        title="Issue title",
        description="Issue description"
    )
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import redis


class ClaudeAgent:
    """Helper class for Claude Code to communicate via Redis.

    This class provides methods for Claude to:
    - Report issues found in one project that affect another
    - Notify services about breaking changes
    - Suggest improvements and optimizations
    - Request code/architecture reviews
    - Check for messages from other services
    """

    def __init__(
        self,
        project_name: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 3,
    ):
        """Initialize the Claude agent for a specific project.

        Args:
            project_name: Name of the current project (e.g., 'optigen-frontend')
            redis_host: Redis server hostname (default: localhost)
            redis_port: Redis server port (default: 6379)
            redis_db: Redis database number for agent communication (default: 3)
        """
        self.project_name = project_name
        self.agent_id = f"claude-{project_name}"

        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, db=redis_db, decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
        except redis.ConnectionError as e:
            print(f"⚠️ Warning: Could not connect to Redis at {redis_host}:{redis_port}")
            print(f"   Error: {e}")
            print("   Messages will not be sent. Please ensure Redis is running.")
            self.redis_client = None

    def report_issue(
        self,
        target: str,
        issue_type: str,
        title: str,
        description: str,
        file_path: str = None,
        suggested_fix: str = None,
        priority: str = "medium",
    ) -> Optional[str]:
        """Report an issue found in the current project that affects another service.

        Args:
            target: Target service name (e.g., 'optigen-backend')
            issue_type: Type of issue (e.g., 'api_mismatch', 'type_error', 'performance')
            title: Brief title of the issue
            description: Detailed description of the issue
            file_path: Path to the file where issue was found (optional)
            suggested_fix: Suggested solution (optional)
            priority: Issue priority - 'critical', 'high', 'medium', 'low' (default: 'medium')

        Returns:
            Message ID if sent successfully, None otherwise
        """
        if not self.redis_client:
            return None

        message = self._create_message(
            to=target,
            content_type="issue_report",
            content={
                "issue_type": issue_type,
                "title": title,
                "description": description,
                "file_path": file_path,
                "suggested_fix": suggested_fix,
                "priority": priority,
                "detected_in": self.project_name,
            },
        )
        return self._send_message(target, message)

    def notify_dependency(
        self,
        affected_services: List[str],
        change_type: str,
        description: str,
        migration_guide: str = None,
        deadline: str = None,
    ) -> List[str]:
        """Notify dependent services about breaking changes.

        Args:
            affected_services: List of service names that will be affected
            change_type: Type of change (e.g., 'breaking', 'deprecation', 'enhancement')
            description: Description of the change
            migration_guide: Instructions for migrating to the new version (optional)
            deadline: Deadline for migration in ISO format (optional)

        Returns:
            List of message IDs for sent messages
        """
        if not self.redis_client:
            return []

        message_ids = []
        for service in affected_services:
            message = self._create_message(
                to=service,
                content_type="dependency_notification",
                content={
                    "change_type": change_type,
                    "description": description,
                    "migration_guide": migration_guide,
                    "deadline": deadline,
                    "source_service": self.project_name,
                },
            )
            msg_id = self._send_message(service, message)
            if msg_id:
                message_ids.append(msg_id)
        return message_ids

    def suggest_improvement(
        self,
        target: str,
        improvement_type: str,
        description: str,
        expected_benefit: str,
        implementation_notes: str = None,
    ) -> Optional[str]:
        """Suggest an improvement or optimization to another service.

        Args:
            target: Target service name
            improvement_type: Type of improvement (e.g., 'performance', 'security', 'refactoring')
            description: Description of the improvement
            expected_benefit: Expected benefits of implementing the improvement
            implementation_notes: Notes on how to implement (optional)

        Returns:
            Message ID if sent successfully, None otherwise
        """
        if not self.redis_client:
            return None

        message = self._create_message(
            to=target,
            content_type="improvement_suggestion",
            content={
                "improvement_type": improvement_type,
                "description": description,
                "expected_benefit": expected_benefit,
                "implementation_notes": implementation_notes,
                "suggested_by": self.project_name,
            },
        )
        return self._send_message(target, message)

    def request_review(
        self,
        reviewer: str,
        review_type: str,
        description: str,
        urgency: str = "normal",
        context: Dict = None,
    ) -> Optional[str]:
        """Request code review or architectural review from another team.

        Args:
            reviewer: Service/team to review
            review_type: Type of review (e.g., 'code', 'architecture', 'security')
            description: What needs to be reviewed
            urgency: Review urgency - 'urgent', 'high', 'normal', 'low' (default: 'normal')
            context: Additional context information (optional)

        Returns:
            Message ID if sent successfully, None otherwise
        """
        if not self.redis_client:
            return None

        message = self._create_message(
            to=reviewer,
            content_type="review_request",
            content={
                "review_type": review_type,
                "description": description,
                "urgency": urgency,
                "context": context or {},
                "requested_by": self.project_name,
            },
        )
        return self._send_message(reviewer, message)

    def report_performance_issue(
        self,
        component: str,
        issue: str,
        metrics: Dict = None,
        suggested_fix: str = None,
        affected_workflows: List[str] = None,
    ) -> Optional[str]:
        """Report performance issues detected in the system.

        Args:
            component: Component with performance issue
            issue: Description of the performance issue
            metrics: Performance metrics (optional)
            suggested_fix: Suggested solution (optional)
            affected_workflows: List of affected workflows (optional)

        Returns:
            Message ID if broadcast successfully, None otherwise
        """
        if not self.redis_client:
            return None

        # Broadcast to all services
        message = self._create_message(
            to="broadcast",
            content_type="performance_alert",
            content={
                "component": component,
                "issue": issue,
                "metrics": metrics or {},
                "suggested_fix": suggested_fix,
                "affected_workflows": affected_workflows or [],
                "detected_by": self.project_name,
            },
        )
        return self._broadcast_message(message)

    def check_messages(self, limit: int = 10) -> List[Dict]:
        """Check for messages addressed to this project.

        Args:
            limit: Maximum number of messages to retrieve (default: 10)

        Returns:
            List of message dictionaries
        """
        if not self.redis_client:
            return []

        messages = []
        raw_messages = self.redis_client.lrange(
            f"agent:messages:{self.project_name}:inbox", 0, limit - 1
        )

        for raw_msg in raw_messages:
            try:
                messages.append(json.loads(raw_msg))
            except json.JSONDecodeError:
                continue

        return messages

    def clear_inbox(self) -> int:
        """Clear all messages from the inbox.

        Returns:
            Number of messages cleared
        """
        if not self.redis_client:
            return 0

        inbox_key = f"agent:messages:{self.project_name}:inbox"
        count = self.redis_client.llen(inbox_key)
        self.redis_client.delete(inbox_key)
        print(f"🗑️ Cleared {count} messages from inbox")
        return count

    def get_outbox_messages(self, limit: int = 10) -> List[Dict]:
        """Get messages sent from this project.

        Args:
            limit: Maximum number of messages to retrieve (default: 10)

        Returns:
            List of sent message dictionaries
        """
        if not self.redis_client:
            return []

        messages = []
        raw_messages = self.redis_client.lrange(
            f"agent:messages:{self.agent_id}:outbox", 0, limit - 1
        )

        for raw_msg in raw_messages:
            try:
                messages.append(json.loads(raw_msg))
            except json.JSONDecodeError:
                continue

        return messages

    def _create_message(self, to: str, content_type: str, content: Dict) -> Dict:
        """Create a standardized message.

        Args:
            to: Target service or 'broadcast'
            content_type: Type of message content
            content: Message payload

        Returns:
            Formatted message dictionary
        """
        return {
            "header": {
                "message_id": f"msg_{datetime.utcnow().isoformat()}_{self.agent_id}_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.utcnow().isoformat(),
                "priority": content.get("priority", "medium"),
                "ttl": 86400,  # 24 hours
            },
            "routing": {
                "from": self.agent_id,
                "to": to,
                "reply_to": f"agent:messages:{self.agent_id}:inbox",
            },
            "content": {
                "type": content_type,
                "payload": content,
                "schema_version": "1.0.0",
            },
            "metadata": {
                "source": "claude_code",
                "project": self.project_name,
                "environment": "development",
            },
        }

    def _send_message(self, target: str, message: Dict) -> Optional[str]:
        """Send a message to a specific service.

        Args:
            target: Target service name
            message: Message dictionary to send

        Returns:
            Message ID if sent successfully, None otherwise
        """
        if not self.redis_client:
            return None

        try:
            message_json = json.dumps(message)

            # Add to target's inbox
            self.redis_client.lpush(f"agent:messages:{target}:inbox", message_json)

            # Add to our outbox for tracking
            self.redis_client.lpush(
                f"agent:messages:{self.agent_id}:outbox", message_json
            )

            # Publish notification
            self.redis_client.publish(
                "agent:channels:messages", f"message_sent:{self.agent_id}:{target}"
            )

            print(f"✅ Message sent to {target}: {message['header']['message_id']}")
            return message["header"]["message_id"]

        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            return None

    def _broadcast_message(self, message: Dict) -> Optional[str]:
        """Broadcast a message to all services.

        Args:
            message: Message dictionary to broadcast

        Returns:
            Message ID if broadcast successfully, None otherwise
        """
        if not self.redis_client:
            return None

        try:
            message_json = json.dumps(message)

            # Publish to broadcast channel
            self.redis_client.publish("agent:channels:broadcast", message_json)

            # Add to our outbox
            self.redis_client.lpush(
                f"agent:messages:{self.agent_id}:outbox", message_json
            )

            print(f"📢 Broadcast message: {message['header']['message_id']}")
            return message["header"]["message_id"]

        except Exception as e:
            print(f"❌ Failed to broadcast message: {e}")
            return None


# Quick test function
def test_connection(project_name: str = "test-project"):
    """Test Redis connection and basic messaging.

    Args:
        project_name: Name to use for test project
    """
    print(f"🧪 Testing Claude Agent for project: {project_name}")

    agent = ClaudeAgent(project_name)

    if agent.redis_client:
        print("✅ Redis connection successful")

        # Test sending a message
        msg_id = agent.report_issue(
            target="test-target",
            issue_type="test",
            title="Connection test",
            description="This is a test message",
        )

        if msg_id:
            print(f"✅ Test message sent: {msg_id}")
        else:
            print("❌ Failed to send test message")
    else:
        print("❌ Redis connection failed")


if __name__ == "__main__":
    # Run test when executed directly
    test_connection()
