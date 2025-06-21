"""
Jules Coding Agent - High-level wrapper for autonomous coding capabilities

This module provides a clean, high-level interface to Jules' autonomous
coding capabilities with built-in logging, monitoring, and error handling.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...integrations.google.jules_client import (
    JulesClient,
    JulesTask,
    JulesTaskStatus,
    JulesTaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class JulesAgentConfig:
    """Configuration for Jules Agent"""

    project_id: str
    auto_approve_prs: bool = False
    max_retries: int = 3
    timeout_minutes: int = 30
    enable_monitoring: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    github_org: Optional[str] = None
    default_branch: str = "main"
    test_coverage_threshold: float = 0.8


@dataclass
class CodingTaskResult:
    """Result of a coding task"""

    task_id: str
    status: str
    repository_url: Optional[str] = None
    pr_url: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    test_coverage: Optional[float] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


class JulesAgent:
    """
    High-level Jules coding agent with enterprise features.

    This agent provides:
    - Simplified API for common coding tasks
    - Built-in logging and monitoring
    - Error handling and retries
    - Progress tracking
    - Performance metrics
    """

    def __init__(self, config: JulesAgentConfig, skip_auth: bool = False):
        self.config = config
        self.client = JulesClient(config.project_id, skip_auth=skip_auth)
        self.active_tasks: Dict[str, CodingTaskResult] = {}
        self._setup_logging()
        self._metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_test_coverage": 0,
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def create_feature(
        self,
        description: str,
        repository: str,
        requirements: Optional[Dict[str, Any]] = None,
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Create a new feature with Jules.

        Args:
            description: Natural language description of the feature
            repository: Target repository (org/repo format)
            requirements: Additional requirements and constraints
            branch: Target branch (auto-generated if not provided)

        Returns:
            CodingTaskResult with task outcome
        """
        return await self._execute_task(
            task_type=JulesTaskType.ADD_FEATURE,
            description=description,
            repository=repository,
            requirements=requirements,
            branch=branch,
        )

    async def fix_bug(
        self,
        bug_description: str,
        repository: str,
        issue_number: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Fix a bug with Jules.

        Args:
            bug_description: Description of the bug to fix
            repository: Target repository
            issue_number: GitHub issue number (if exists)
            branch: Target branch

        Returns:
            CodingTaskResult with fix details
        """
        requirements = {}
        if issue_number:
            requirements["issue_number"] = issue_number

        return await self._execute_task(
            task_type=JulesTaskType.FIX_BUG,
            description=bug_description,
            repository=repository,
            requirements=requirements,
            branch=branch,
        )

    async def create_api_endpoint(
        self,
        endpoint_spec: Dict[str, Any],
        repository: str,
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Create a new API endpoint.

        Args:
            endpoint_spec: Endpoint specification including path, method, etc.
            repository: Target repository
            branch: Target branch

        Returns:
            CodingTaskResult with endpoint implementation
        """
        description = f"Create {endpoint_spec.get('method', 'GET')} endpoint at {endpoint_spec.get('path', '/api/endpoint')}"

        return await self._execute_task(
            task_type=JulesTaskType.CREATE_ENDPOINT,
            description=description,
            repository=repository,
            requirements={"endpoint_spec": endpoint_spec},
            branch=branch,
        )

    async def refactor_code(
        self,
        refactoring_description: str,
        repository: str,
        files: List[str],
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Refactor existing code.

        Args:
            refactoring_description: What to refactor and why
            repository: Target repository
            files: List of files to refactor
            branch: Target branch

        Returns:
            CodingTaskResult with refactoring details
        """
        return await self._execute_task(
            task_type=JulesTaskType.REFACTOR_CODE,
            description=refactoring_description,
            repository=repository,
            requirements={"target_files": files},
            branch=branch,
        )

    async def write_tests(
        self,
        test_description: str,
        repository: str,
        target_files: List[str],
        coverage_target: Optional[float] = None,
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Write tests for existing code.

        Args:
            test_description: What to test
            repository: Target repository
            target_files: Files to write tests for
            coverage_target: Target test coverage (default from config)
            branch: Target branch

        Returns:
            CodingTaskResult with test details
        """
        coverage_target = coverage_target or self.config.test_coverage_threshold

        return await self._execute_task(
            task_type=JulesTaskType.WRITE_TESTS,
            description=test_description,
            repository=repository,
            requirements={
                "target_files": target_files,
                "coverage_target": coverage_target,
            },
            branch=branch,
        )

    async def build_app(
        self,
        app_spec: Dict[str, Any],
        organization: Optional[str] = None,
        create_repo: bool = True,
    ) -> CodingTaskResult:
        """
        Build a complete application from specification.

        Args:
            app_spec: Complete application specification
            organization: GitHub organization (uses config default if not provided)
            create_repo: Whether to create new repository

        Returns:
            CodingTaskResult with app details
        """
        org = organization or self.config.github_org or "nexus-forge"
        repo_name = app_spec["name"].lower().replace(" ", "-")
        repository = f"{org}/{repo_name}"

        start_time = datetime.utcnow()
        result = CodingTaskResult(
            task_id=self.client._generate_task_id(app_spec["description"]),
            status="started",
        )

        try:
            self._log(result, f"Starting app build for {app_spec['name']}")
            self._metrics["tasks_created"] += 1

            async with self.client as client:
                # Build the app
                build_result = await client.autonomous_app_builder(
                    app_spec=app_spec, repository=repository, create_repo=create_repo
                )

                # Update result
                result.status = "completed"
                result.repository_url = build_result["repository_url"]
                result.pr_url = build_result.get("pr_url")
                result.execution_time = (datetime.utcnow() - start_time).total_seconds()

                self._log(result, f"App build completed: {result.repository_url}")
                self._metrics["tasks_completed"] += 1

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self._log(result, f"App build failed: {str(e)}", level="ERROR")
            self._metrics["tasks_failed"] += 1

        finally:
            self._update_metrics(result)

        return result

    async def _execute_task(
        self,
        task_type: JulesTaskType,
        description: str,
        repository: str,
        requirements: Optional[Dict[str, Any]] = None,
        branch: Optional[str] = None,
    ) -> CodingTaskResult:
        """
        Execute a Jules task with monitoring and error handling.
        """
        start_time = datetime.utcnow()
        result = CodingTaskResult(
            task_id="", status="started"  # Will be set after task creation
        )

        try:
            self._log(result, f"Starting {task_type.value}: {description}")
            self._metrics["tasks_created"] += 1

            async with self.client as client:
                # Create task
                task = await client.create_task(
                    task_type=task_type,
                    description=description,
                    repository=repository,
                    branch=branch,
                    requirements=requirements,
                )

                result.task_id = task.task_id
                self.active_tasks[task.task_id] = result

                # Monitor task execution
                execution_result = await self._execute_with_monitoring(
                    client, task, result
                )

                # Update result
                result.status = "completed"
                result.pr_url = execution_result.get("pr_url")
                result.files_created = execution_result.get("files_created", [])
                result.files_modified = execution_result.get("files_modified", [])
                result.test_coverage = execution_result.get("test_coverage")
                result.execution_time = (datetime.utcnow() - start_time).total_seconds()

                self._log(result, f"Task completed: {result.pr_url}")
                self._metrics["tasks_completed"] += 1

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._log(result, f"Task failed: {str(e)}", level="ERROR")
            self._metrics["tasks_failed"] += 1

            if self.config.max_retries > 0:
                self._log(result, "Retrying task...")
                # Implement retry logic here

        finally:
            self._update_metrics(result)
            if result.task_id in self.active_tasks:
                del self.active_tasks[result.task_id]

        return result

    async def _execute_with_monitoring(
        self, client: JulesClient, task: JulesTask, result: CodingTaskResult
    ) -> Dict[str, Any]:
        """
        Execute task with progress monitoring.
        """
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_task_progress(task, result))

        try:
            # Execute the task
            execution_result = await client.execute_task(
                task, auto_commit=not self.config.auto_approve_prs
            )

            return execution_result

        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_task_progress(self, task: JulesTask, result: CodingTaskResult):
        """
        Monitor task progress and update logs.
        """
        previous_status = None

        while True:
            if task.status != previous_status:
                self._log(result, f"Task status: {task.status.value}")
                previous_status = task.status

            if task.status in [JulesTaskStatus.COMPLETED, JulesTaskStatus.FAILED]:
                break

            await asyncio.sleep(5)  # Check every 5 seconds

    def _log(self, result: CodingTaskResult, message: str, level: str = "INFO"):
        """Add log message to result and logger"""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        result.logs.append(log_entry)

        if self.config.enable_logging:
            getattr(logger, level.lower())(f"[{result.task_id}] {message}")

    def _update_metrics(self, result: CodingTaskResult):
        """Update agent metrics"""
        if result.execution_time:
            self._metrics["total_execution_time"] += result.execution_time

        if result.test_coverage:
            # Update average test coverage
            completed = self._metrics["tasks_completed"]
            if completed > 0:
                current_avg = self._metrics["average_test_coverage"]
                self._metrics["average_test_coverage"] = (
                    current_avg * (completed - 1) + result.test_coverage
                ) / completed

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        metrics = self._metrics.copy()

        # Calculate additional metrics
        total_tasks = metrics["tasks_created"]
        if total_tasks > 0:
            metrics["success_rate"] = metrics["tasks_completed"] / total_tasks
            metrics["failure_rate"] = metrics["tasks_failed"] / total_tasks

        if metrics["tasks_completed"] > 0:
            metrics["average_execution_time"] = (
                metrics["total_execution_time"] / metrics["tasks_completed"]
            )

        return metrics

    def get_active_tasks(self) -> Dict[str, CodingTaskResult]:
        """Get currently active tasks"""
        return self.active_tasks.copy()

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an active task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if cancelled, False if not found
        """
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = "cancelled"
            self._log(
                self.active_tasks[task_id], "Task cancelled by user", level="WARNING"
            )
            del self.active_tasks[task_id]
            return True
        return False

    async def generate_code_snippet(
        self,
        prompt: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a code snippet without full task workflow.

        Args:
            prompt: Description of code to generate
            language: Programming language
            context: Additional context

        Returns:
            Generated code as string
        """
        async with self.client as client:
            code_files = await client.generate_code(
                prompt=prompt, context=context, language=language
            )

            # Return the first generated file content
            if code_files:
                return list(code_files.values())[0]
            return "# No code generated"

    async def review_pr(
        self, repository: str, pr_number: int, feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Have Jules review and potentially improve a pull request.

        Args:
            repository: Repository (org/repo format)
            pr_number: Pull request number
            feedback: Optional feedback to address

        Returns:
            Review results
        """
        description = f"Review PR #{pr_number}"
        if feedback:
            description += f" and address feedback: {feedback}"

        # This would integrate with Jules' PR review capabilities
        # For now, return a placeholder
        return {"status": "reviewed", "suggestions": [], "improvements_made": False}


# Convenience functions for common operations
async def quick_feature(description: str, repository: str) -> CodingTaskResult:
    """Quick function to add a feature with default settings"""
    config = JulesAgentConfig(project_id="default-project", auto_approve_prs=False)
    agent = JulesAgent(config)
    return await agent.create_feature(description, repository)


async def quick_fix(bug_description: str, repository: str) -> CodingTaskResult:
    """Quick function to fix a bug with default settings"""
    config = JulesAgentConfig(project_id="default-project", auto_approve_prs=False)
    agent = JulesAgent(config)
    return await agent.fix_bug(bug_description, repository)


# Example usage
async def main():
    """Example of using Jules Agent"""

    # Configure Jules agent
    config = JulesAgentConfig(
        project_id="my-project",
        github_org="my-org",
        auto_approve_prs=False,
        test_coverage_threshold=0.9,
        enable_monitoring=True,
    )

    # Create agent
    agent = JulesAgent(config)

    # Create a new feature
    result = await agent.create_feature(
        description="Add user authentication with JWT tokens",
        repository="my-org/my-app",
        requirements={
            "auth_methods": ["email", "google_oauth"],
            "include_refresh_tokens": True,
        },
    )

    print(f"Feature created: {result.pr_url}")
    print(f"Execution time: {result.execution_time}s")
    print(f"Test coverage: {result.test_coverage}")

    # Get metrics
    metrics = agent.get_metrics()
    print(f"Agent metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
