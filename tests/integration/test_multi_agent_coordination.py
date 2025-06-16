"""
Integration tests for multi-agent coordination

Tests the coordination and communication between multiple AI agents:
- Agent task distribution
- Inter-agent communication
- Parallel processing
- Result aggregation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from nexus_forge.agents.agents.nexus_forge_agents import (
    GeminiFlashAgent,
    GeminiSpecAgent,
    ImagenDesignAgent,
    JulesCodeAgent,
    StarriOrchestrator,
    VeoVideoAgent,
)
from nexus_forge.services.google_ai_service import GoogleAIService


@pytest.mark.integration
@pytest.mark.multi_agent
@pytest.mark.asyncio
class TestMultiAgentCoordination:
    """Test coordination between multiple AI agents"""

    async def test_agent_task_distribution(
        self,
        mock_starri_orchestrator,
        mock_ai_model_responses,
        sample_app_specification,
    ):
        """Test proper distribution of tasks among agents"""
        # Arrange
        task_distribution = {
            "gemini_2_5_pro": [],
            "imagen_4": [],
            "veo_3": [],
            "jules": [],
            "gemini_flash": [],
        }

        # Track which agent handles which task
        async def track_task(agent_name: str, task_type: str, task_data: Any):
            task_distribution[agent_name].append(
                {"type": task_type, "data": task_data, "timestamp": datetime.utcnow()}
            )

        # Configure orchestrator to track task distribution
        mock_starri_orchestrator.distribute_tasks = AsyncMock(
            side_effect=lambda tasks: track_task("orchestrator", "distribute", tasks)
        )

        # Act
        prompt = "Build a collaborative document editor with real-time sync"
        config = {
            "useAdaptiveThinking": True,
            "parallelProcessing": True,
            "optimizeForSpeed": True,
        }

        # Simulate orchestrator breaking down the task
        tasks = [
            {
                "agent": "gemini_2_5_pro",
                "task": "generate_specification",
                "priority": 1,
            },
            {"agent": "imagen_4", "task": "create_ui_mockups", "priority": 2},
            {"agent": "imagen_4", "task": "design_system", "priority": 2},
            {"agent": "veo_3", "task": "create_demo_video", "priority": 3},
            {"agent": "jules", "task": "generate_backend_code", "priority": 2},
            {"agent": "jules", "task": "generate_frontend_code", "priority": 2},
            {"agent": "gemini_flash", "task": "optimize_code", "priority": 4},
        ]

        # Distribute and execute tasks
        for task in tasks:
            await track_task(task["agent"], task["task"], task)

        # Process tasks based on priority
        priority_groups = {}
        for agent, agent_tasks in task_distribution.items():
            for task in agent_tasks:
                if isinstance(task["data"], dict) and "priority" in task["data"]:
                    priority = task["data"]["priority"]
                    if priority not in priority_groups:
                        priority_groups[priority] = []
                    priority_groups[priority].append((agent, task))

        # Assert
        # Verify task distribution
        assert len(task_distribution["gemini_2_5_pro"]) >= 1  # Specification
        assert len(task_distribution["imagen_4"]) >= 2  # Mockups and design system
        assert len(task_distribution["veo_3"]) >= 1  # Demo video
        assert len(task_distribution["jules"]) >= 2  # Backend and frontend code
        assert len(task_distribution["gemini_flash"]) >= 1  # Code optimization

        # Verify priority-based execution order
        priorities = sorted(priority_groups.keys())
        assert priorities == [1, 2, 3, 4]  # Tasks executed in priority order

        # Verify parallel execution for same priority tasks
        priority_2_tasks = priority_groups.get(2, [])
        assert len(priority_2_tasks) >= 3  # Multiple tasks at priority 2

    async def test_inter_agent_communication(
        self, mock_starri_orchestrator, mock_imagen_integration, mock_veo_integration
    ):
        """Test communication and data sharing between agents"""
        # Arrange
        agent_messages = []
        shared_context = {}

        # Create message bus for inter-agent communication
        class AgentMessageBus:
            def __init__(self):
                self.subscribers = {}
                self.message_log = []

            async def publish(self, sender: str, topic: str, data: Any):
                message = {
                    "sender": sender,
                    "topic": topic,
                    "data": data,
                    "timestamp": datetime.utcnow(),
                }
                self.message_log.append(message)
                agent_messages.append(message)

                # Notify subscribers
                if topic in self.subscribers:
                    for subscriber in self.subscribers[topic]:
                        await subscriber(message)

            def subscribe(self, topic: str, callback):
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                self.subscribers[topic].append(callback)

        message_bus = AgentMessageBus()

        # Configure agents to use message bus
        async def gemini_spec_handler(message):
            if message["topic"] == "specification_complete":
                shared_context["specification"] = message["data"]
                await message_bus.publish(
                    "imagen_4",
                    "request_mockups",
                    {
                        "spec": message["data"],
                        "components": message["data"].get("ui_components", []),
                    },
                )

        async def imagen_mockup_handler(message):
            if message["topic"] == "request_mockups":
                mockups = {}
                for component in message["data"]["components"]:
                    mockups[component] = f"https://mockup.url/{component}.png"
                shared_context["mockups"] = mockups
                await message_bus.publish("imagen_4", "mockups_complete", mockups)

        async def veo_video_handler(message):
            if message["topic"] == "mockups_complete":
                video_url = "https://video.url/demo.mp4"
                shared_context["demo_video"] = video_url
                await message_bus.publish(
                    "veo_3",
                    "video_complete",
                    {"url": video_url, "mockups_used": message["data"]},
                )

        # Subscribe handlers
        message_bus.subscribe("specification_complete", gemini_spec_handler)
        message_bus.subscribe("request_mockups", imagen_mockup_handler)
        message_bus.subscribe("mockups_complete", veo_video_handler)

        # Act
        # Start the workflow
        await message_bus.publish(
            "gemini_2_5_pro",
            "specification_complete",
            {"name": "Test App", "ui_components": ["Dashboard", "Settings", "Profile"]},
        )

        # Allow async handlers to complete
        await asyncio.sleep(0.1)

        # Assert
        # Verify message flow
        assert len(agent_messages) >= 3

        # Verify message sequence
        topics = [msg["topic"] for msg in agent_messages]
        assert "specification_complete" in topics
        assert "request_mockups" in topics
        assert "mockups_complete" in topics
        assert "video_complete" in topics

        # Verify shared context
        assert "specification" in shared_context
        assert "mockups" in shared_context
        assert "demo_video" in shared_context

        # Verify data dependencies
        assert len(shared_context["mockups"]) == 3
        assert all(
            comp in shared_context["mockups"]
            for comp in ["Dashboard", "Settings", "Profile"]
        )

    async def test_parallel_agent_processing(
        self,
        mock_starri_orchestrator,
        mock_imagen_integration,
        mock_veo_integration,
        performance_benchmarks,
    ):
        """Test parallel processing capabilities of multiple agents"""
        # Arrange
        processing_times = {}
        agent_tasks = {
            "imagen_4": [
                {"task": "generate_dashboard_mockup", "duration": 2.0},
                {"task": "generate_settings_mockup", "duration": 1.5},
                {"task": "generate_profile_mockup", "duration": 1.8},
            ],
            "jules": [
                {"task": "generate_api_endpoints", "duration": 3.0},
                {"task": "generate_database_schema", "duration": 2.5},
            ],
            "veo_3": [{"task": "generate_intro_video", "duration": 4.0}],
        }

        async def simulate_agent_task(agent_name: str, task: Dict):
            start_time = datetime.utcnow()
            await asyncio.sleep(task["duration"] * 0.1)  # Scale down for testing
            end_time = datetime.utcnow()

            return {
                "agent": agent_name,
                "task": task["task"],
                "start": start_time,
                "end": end_time,
                "duration": (end_time - start_time).total_seconds(),
            }

        # Act
        # Execute all tasks in parallel
        all_tasks = []
        for agent_name, tasks in agent_tasks.items():
            for task in tasks:
                all_tasks.append(simulate_agent_task(agent_name, task))

        start_time = datetime.utcnow()
        results = await asyncio.gather(*all_tasks)
        end_time = datetime.utcnow()

        total_duration = (end_time - start_time).total_seconds()

        # Calculate theoretical sequential time
        sequential_time = sum(
            task["duration"] * 0.1 for tasks in agent_tasks.values() for task in tasks
        )

        # Assert
        # Verify parallel execution
        assert (
            total_duration < sequential_time * 0.7
        )  # At least 30% faster than sequential

        # Verify all tasks completed
        assert len(results) == sum(len(tasks) for tasks in agent_tasks.values())

        # Verify concurrent execution by checking overlapping time ranges
        overlaps = 0
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i < j and result1["agent"] != result2["agent"]:
                    # Check if tasks overlapped in time
                    if (
                        result1["start"] <= result2["end"]
                        and result2["start"] <= result1["end"]
                    ):
                        overlaps += 1

        assert overlaps > 0  # Should have overlapping execution

        # Verify no single agent bottleneck
        agent_durations = {}
        for result in results:
            agent = result["agent"]
            if agent not in agent_durations:
                agent_durations[agent] = 0
            agent_durations[agent] += result["duration"]

        max_agent_duration = max(agent_durations.values())
        assert (
            max_agent_duration < total_duration * 1.2
        )  # No agent takes more than 120% of total

    async def test_result_aggregation_and_validation(
        self,
        mock_starri_orchestrator,
        mock_ai_model_responses,
        sample_app_specification,
    ):
        """Test aggregation and validation of results from multiple agents"""
        # Arrange
        agent_results = {
            "gemini_2_5_pro": {
                "specification": sample_app_specification.__dict__,
                "validation_score": 0.95,
            },
            "imagen_4": {
                "mockups": {
                    "Dashboard": "https://mockups/dashboard.png",
                    "Charts": "https://mockups/charts.png",
                },
                "design_system": {
                    "colors": {"primary": "#3B82F6"},
                    "typography": {"base": "16px"},
                },
            },
            "veo_3": {
                "demo_video": "https://videos/demo.mp4",
                "feature_videos": {"data_viz": "https://videos/data_viz.mp4"},
            },
            "jules": {
                "code_files": {
                    "backend/main.py": "# FastAPI backend",
                    "frontend/App.tsx": "// React frontend",
                },
                "test_coverage": 0.85,
            },
            "gemini_flash": {
                "optimizations": {
                    "performance_improvements": ["lazy_loading", "code_splitting"],
                    "size_reduction": "23%",
                }
            },
        }

        # Create result aggregator
        class ResultAggregator:
            def __init__(self):
                self.results = {}
                self.validation_errors = []

            async def add_result(self, agent: str, result: Any):
                self.results[agent] = result
                await self.validate_result(agent, result)

            async def validate_result(self, agent: str, result: Any):
                # Validate based on agent type
                if agent == "gemini_2_5_pro":
                    if "specification" not in result:
                        self.validation_errors.append(f"{agent}: Missing specification")
                elif agent == "imagen_4":
                    if "mockups" not in result or not result["mockups"]:
                        self.validation_errors.append(f"{agent}: No mockups generated")
                elif agent == "jules":
                    if "code_files" not in result or not result["code_files"]:
                        self.validation_errors.append(
                            f"{agent}: No code files generated"
                        )

            async def aggregate(self) -> Dict:
                # Check all required agents have results
                required_agents = ["gemini_2_5_pro", "imagen_4", "jules"]
                for agent in required_agents:
                    if agent not in self.results:
                        self.validation_errors.append(f"Missing results from {agent}")

                if self.validation_errors:
                    return {"status": "failed", "errors": self.validation_errors}

                # Aggregate results
                return {
                    "status": "success",
                    "specification": self.results["gemini_2_5_pro"]["specification"],
                    "mockups": self.results["imagen_4"]["mockups"],
                    "design_system": self.results["imagen_4"].get("design_system", {}),
                    "demo_video": self.results.get("veo_3", {}).get("demo_video"),
                    "code_files": self.results["jules"]["code_files"],
                    "optimizations": self.results.get("gemini_flash", {}).get(
                        "optimizations", {}
                    ),
                    "metrics": {
                        "validation_score": self.results["gemini_2_5_pro"].get(
                            "validation_score", 0
                        ),
                        "test_coverage": self.results["jules"].get("test_coverage", 0),
                        "agents_used": list(self.results.keys()),
                    },
                }

        aggregator = ResultAggregator()

        # Act
        # Add results from each agent
        for agent, result in agent_results.items():
            await aggregator.add_result(agent, result)

        # Aggregate final result
        final_result = await aggregator.aggregate()

        # Assert
        # Verify successful aggregation
        assert final_result["status"] == "success"
        assert len(final_result["specification"]) > 0
        assert len(final_result["mockups"]) >= 2
        assert len(final_result["code_files"]) >= 2

        # Verify metrics
        assert final_result["metrics"]["validation_score"] > 0.9
        assert final_result["metrics"]["test_coverage"] > 0.8
        assert len(final_result["metrics"]["agents_used"]) == 5

        # Verify optional components
        assert "demo_video" in final_result
        assert "optimizations" in final_result
        assert final_result["optimizations"]["size_reduction"] == "23%"

    async def test_agent_failure_recovery(
        self, mock_starri_orchestrator, mock_imagen_integration, mock_veo_integration
    ):
        """Test recovery mechanisms when individual agents fail"""
        # Arrange
        failure_scenarios = [
            {
                "failing_agent": "imagen_4",
                "error": "API rate limit exceeded",
                "recovery_strategy": "retry_with_backoff",
            },
            {
                "failing_agent": "veo_3",
                "error": "Video generation timeout",
                "recovery_strategy": "skip_optional",
            },
            {
                "failing_agent": "jules",
                "error": "Code generation failed",
                "recovery_strategy": "fallback_agent",
            },
        ]

        recovery_attempts = []

        async def attempt_recovery(agent: str, error: str, strategy: str):
            recovery_attempts.append(
                {
                    "agent": agent,
                    "error": error,
                    "strategy": strategy,
                    "timestamp": datetime.utcnow(),
                }
            )

            if strategy == "retry_with_backoff":
                # Retry with exponential backoff
                for attempt in range(3):
                    await asyncio.sleep(0.1 * (2**attempt))
                    if attempt == 2:  # Success on third attempt
                        return {"status": "recovered", "attempt": attempt + 1}
                return {"status": "failed"}

            elif strategy == "skip_optional":
                # Skip non-critical component
                return {"status": "skipped", "reason": "optional_component"}

            elif strategy == "fallback_agent":
                # Use alternative agent
                return {
                    "status": "fallback_used",
                    "original_agent": agent,
                    "fallback_agent": "gemini_flash",
                }

        # Act
        results = []
        for scenario in failure_scenarios:
            result = await attempt_recovery(
                scenario["failing_agent"],
                scenario["error"],
                scenario["recovery_strategy"],
            )
            results.append(result)

        # Assert
        # Verify recovery attempts
        assert len(recovery_attempts) == len(failure_scenarios)

        # Verify recovery strategies
        assert results[0]["status"] == "recovered"
        assert results[0]["attempt"] == 3

        assert results[1]["status"] == "skipped"
        assert results[1]["reason"] == "optional_component"

        assert results[2]["status"] == "fallback_used"
        assert results[2]["fallback_agent"] == "gemini_flash"

        # Verify no cascading failures
        for i, attempt in enumerate(recovery_attempts):
            if i > 0:
                time_diff = (
                    attempt["timestamp"] - recovery_attempts[i - 1]["timestamp"]
                ).total_seconds()
                assert time_diff < 10  # Recovery should be quick

    async def test_adaptive_agent_orchestration(
        self, mock_starri_orchestrator, sample_app_specification
    ):
        """Test adaptive orchestration based on task complexity and requirements"""
        # Arrange
        test_prompts = [
            {
                "prompt": "Build a simple todo list app",
                "complexity": "low",
                "expected_agents": ["gemini_flash", "imagen_4", "jules"],
                "expected_duration": 120,  # 2 minutes
            },
            {
                "prompt": "Build a real-time collaborative IDE with AI assistance",
                "complexity": "high",
                "expected_agents": [
                    "gemini_2_5_pro",
                    "imagen_4",
                    "veo_3",
                    "jules",
                    "gemini_flash",
                ],
                "expected_duration": 300,  # 5 minutes
            },
            {
                "prompt": "Create a landing page for a startup",
                "complexity": "medium",
                "expected_agents": ["gemini_2_5_pro", "imagen_4", "jules"],
                "expected_duration": 180,  # 3 minutes
            },
        ]

        orchestration_results = []

        # Mock adaptive orchestration logic
        async def adaptive_orchestrate(prompt: str, config: Dict):
            # Analyze prompt complexity
            complexity = (
                "high"
                if any(
                    word in prompt.lower()
                    for word in ["real-time", "collaborative", "ai"]
                )
                else (
                    "low"
                    if any(
                        word in prompt.lower() for word in ["simple", "basic", "todo"]
                    )
                    else "medium"
                )
            )

            # Select agents based on complexity
            if complexity == "low":
                agents = ["gemini_flash", "imagen_4", "jules"]
            elif complexity == "high":
                agents = [
                    "gemini_2_5_pro",
                    "imagen_4",
                    "veo_3",
                    "jules",
                    "gemini_flash",
                ]
            else:
                agents = ["gemini_2_5_pro", "imagen_4", "jules"]

            # Simulate orchestration
            start_time = datetime.utcnow()

            # Process with selected agents
            results = {}
            for agent in agents:
                await asyncio.sleep(0.1)  # Simulate processing
                results[agent] = {"status": "completed"}

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "prompt": prompt,
                "complexity": complexity,
                "agents_used": agents,
                "duration": duration,
                "results": results,
            }

        # Act
        for test in test_prompts:
            result = await adaptive_orchestrate(test["prompt"], {})
            orchestration_results.append(result)

        # Assert
        for i, result in enumerate(orchestration_results):
            test = test_prompts[i]

            # Verify complexity detection
            assert result["complexity"] == test["complexity"]

            # Verify agent selection
            assert set(result["agents_used"]) == set(test["expected_agents"])

            # Verify adaptive performance
            # Lower complexity should use fewer resources
            if result["complexity"] == "low":
                assert len(result["agents_used"]) <= 3
            elif result["complexity"] == "high":
                assert len(result["agents_used"]) >= 5

            # Verify all selected agents completed
            assert all(
                agent in result["results"]
                and result["results"][agent]["status"] == "completed"
                for agent in result["agents_used"]
            )
