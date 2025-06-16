"""
Enhanced test suite for Starri Orchestrator with Gemini-2.5-Flash-Thinking integration

Tests the new deep thinking capabilities, task decomposition, and MCP tool integrations.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_forge.agents.starri.orchestrator import (
    AgentCapability,
    StarriOrchestrator,
    ThinkingMode,
)
from nexus_forge.core.exceptions import OrchestrationError, TaskDecompositionError

pytestmark = pytest.mark.asyncio


class TestStarriOrchestratorEnhanced:
    """Test suite for enhanced Starri Orchestrator with deep thinking"""

    @pytest.fixture
    async def orchestrator(self):
        """Create enhanced orchestrator with mocked dependencies"""
        with patch("nexus_forge.agents.starri.orchestrator.GeminiClient"):
            with patch(
                "nexus_forge.agents.starri.orchestrator.SupabaseCoordinationClient"
            ):
                with patch(
                    "nexus_forge.agents.starri.orchestrator.Mem0KnowledgeClient"
                ):
                    orchestrator = StarriOrchestrator(
                        project_id="test-project",
                        supabase_url="https://test.supabase.co",
                        supabase_key="test-key",
                    )

                    # Mock the clients
                    orchestrator.gemini_client = AsyncMock()
                    orchestrator.coordination_client = AsyncMock()
                    orchestrator.knowledge_client = AsyncMock()
                    orchestrator.cache = AsyncMock()

                    yield orchestrator

    async def test_deep_thinking_analysis_mode(self, orchestrator):
        """Test deep thinking in analysis mode"""
        # Mock Gemini Flash-Thinking response
        orchestrator.gemini_client.generate_content = AsyncMock(
            return_value={
                "content": """
                Deep analysis step 1: Breaking down the problem into components...
                Confidence: 0.7
                I need to think more about the implications...
                
                Step 2: Considering multiple perspectives...
                Confidence: 0.9
                Based on my analysis, the conclusion is clear.
                """,
                "usage_metadata": {"total_token_count": 150},
            }
        )

        result = await orchestrator.think_deeply(
            prompt="How should we optimize the task allocation algorithm?",
            mode=ThinkingMode.DEEP_ANALYSIS,
            max_thinking_steps=3,
        )

        assert result["mode"] == ThinkingMode.DEEP_ANALYSIS.value
        assert len(result["thinking_chain"]) > 0
        assert result["confidence"] > 0.0
        assert "conclusion" in result
        assert result["thinking_time"] > 0

        # Verify Gemini Flash-Thinking model was used
        orchestrator.gemini_client.generate_content.assert_called()
        call_args = orchestrator.gemini_client.generate_content.call_args
        assert (
            call_args[1]["model_type"] == "flash"
        )  # Should use flash model for thinking

    async def test_task_decomposition_with_thinking(self, orchestrator):
        """Test task decomposition using planning mode"""
        # Mock thinking result for task decomposition
        orchestrator.think_deeply = AsyncMock(
            return_value={
                "thinking_chain": [
                    {
                        "step": 1,
                        "thought": "Analyzing requirements...",
                        "confidence": 0.8,
                    },
                    {
                        "step": 2,
                        "thought": "Breaking into subtasks...",
                        "confidence": 0.9,
                    },
                ],
                "conclusion": {
                    "conclusion": """
                    Subtask 1: Design user interface
                    - Requires: UI design capabilities
                    - Estimated duration: 30m
                    
                    Subtask 2: Implement backend API
                    - Requires: Code generation capabilities
                    - Estimated duration: 45m
                    
                    Subtask 3: Set up database
                    - Requires: Database configuration capabilities
                    - Estimated duration: 20m
                    """,
                    "confidence": 0.9,
                },
                "confidence": 0.9,
            }
        )

        # Mock coordination client
        orchestrator.coordination_client.create_workflow = AsyncMock(
            return_value="workflow_123"
        )

        result = await orchestrator.decompose_task(
            task_description="Build a user management system",
            requirements=["User authentication", "Profile management", "Admin panel"],
            constraints={"priority": 8},
        )

        assert result["workflow_id"] == "workflow_123"
        assert "decomposition" in result
        assert result["confidence"] > 0.8

        # Verify thinking was used with planning mode
        orchestrator.think_deeply.assert_called_once()
        call_args = orchestrator.think_deeply.call_args
        assert call_args[1]["mode"] == ThinkingMode.PLANNING

    async def test_agent_coordination_with_reflection(self, orchestrator):
        """Test agent coordination with reflection capabilities"""
        # Setup workflow
        workflow_id = "test_workflow_123"
        orchestrator.active_workflows[workflow_id] = {
            "decomposition": {
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Design UI components",
                        "required_capabilities": ["ui_design"],
                        "estimated_duration": "30m",
                    },
                    {
                        "id": "task_2",
                        "description": "Implement API endpoints",
                        "required_capabilities": ["code_generation"],
                        "estimated_duration": "45m",
                    },
                ]
            },
            "status": "initialized",
        }

        # Mock registered agents
        orchestrator.registered_agents = {
            "agent_1": {
                "type": "ui_designer",
                "capabilities": ["ui_design"],
                "status": "online",
            },
            "agent_2": {
                "type": "code_generator",
                "capabilities": ["code_generation"],
                "status": "online",
            },
        }

        # Mock thinking for coordination
        orchestrator.think_deeply = AsyncMock(
            return_value={
                "conclusion": {
                    "coordination_plan": "parallel_execution",
                    "agent_assignments": {"task_1": "agent_1", "task_2": "agent_2"},
                }
            }
        )

        # Mock execution methods
        orchestrator._execute_coordination_plan = AsyncMock(
            return_value={
                "status": "completed",
                "results": {"task_1": "success", "task_2": "success"},
                "metrics": {"tasks_completed": 2, "tasks_failed": 0},
            }
        )

        orchestrator._monitor_workflow_execution = AsyncMock()
        orchestrator._reflect_on_execution = AsyncMock(
            return_value={
                "reflection": "Execution was efficient and successful",
                "quality_score": 0.95,
            }
        )

        result = await orchestrator.coordinate_agents(workflow_id, "parallel")

        assert result["status"] == "completed"
        assert result["workflow_id"] == workflow_id
        assert "reflection" in result

        # Verify coordination thinking was used
        orchestrator.think_deeply.assert_called()
        call_args = orchestrator.think_deeply.call_args
        assert call_args[1]["mode"] == ThinkingMode.COORDINATION

    async def test_agent_registration_with_knowledge_graph(self, orchestrator):
        """Test agent registration with knowledge graph integration"""
        agent_id = "test_agent_001"
        capabilities = [AgentCapability.CODE_GENERATION, AgentCapability.UI_DESIGN]

        # Mock knowledge client
        orchestrator.knowledge_client.add_agent_entity = AsyncMock(
            return_value="entity_123"
        )

        await orchestrator.register_agent(
            agent_id=agent_id,
            agent_type="full_stack_developer",
            capabilities=capabilities,
            configuration={"model": "gemini-pro", "specialization": "web_apps"},
        )

        # Verify agent was registered
        assert agent_id in orchestrator.registered_agents
        agent_info = orchestrator.registered_agents[agent_id]
        assert agent_info["type"] == "full_stack_developer"
        assert "code_generation" in agent_info["capabilities"]
        assert agent_info["status"] == "online"

        # Verify capability mapping
        assert (
            agent_id in orchestrator.agent_capabilities[AgentCapability.CODE_GENERATION]
        )
        assert agent_id in orchestrator.agent_capabilities[AgentCapability.UI_DESIGN]

        # Verify knowledge graph integration
        orchestrator.knowledge_client.add_agent_entity.assert_called_once_with(
            agent_id=agent_id,
            agent_type="full_stack_developer",
            capabilities=["code_generation", "ui_design"],
        )

    async def test_confidence_extraction_from_thinking(self, orchestrator):
        """Test confidence extraction from thinking content"""
        # Test explicit confidence
        confidence = orchestrator._extract_confidence(
            "I am 85% confident in this solution"
        )
        assert confidence == 0.85

        # Test percentage confidence
        confidence = orchestrator._extract_confidence("Confidence: 92%")
        assert confidence == 0.92

        # Test decimal confidence
        confidence = orchestrator._extract_confidence("confidence: 0.75")
        assert confidence == 0.75

        # Test high certainty words
        confidence = orchestrator._extract_confidence(
            "I am certain this is the correct approach"
        )
        assert confidence == 0.9

        # Test uncertain words
        confidence = orchestrator._extract_confidence(
            "This might possibly work, but I'm uncertain"
        )
        assert confidence == 0.3

        # Test default confidence
        confidence = orchestrator._extract_confidence("Here is a solution")
        assert confidence == 0.5

    async def test_thinking_chain_continuation(self, orchestrator):
        """Test continuation of thinking chains"""
        orchestrator.gemini_client.generate_content = AsyncMock(
            side_effect=[
                {
                    "content": "Initial thought: Let me analyze this problem. Confidence: 0.6. I need to think more...",
                    "usage_metadata": {"total_token_count": 50},
                },
                {
                    "content": "Deeper analysis: Now I see the patterns. Confidence: 0.8. Almost there...",
                    "usage_metadata": {"total_token_count": 60},
                },
                {
                    "content": "Final conclusion: The solution is clear. Confidence: 0.95. I'm satisfied with this analysis.",
                    "usage_metadata": {"total_token_count": 70},
                },
            ]
        )

        result = await orchestrator.think_deeply(
            prompt="Complex optimization problem",
            mode=ThinkingMode.DEEP_ANALYSIS,
            max_thinking_steps=5,
        )

        # Should have multiple thinking steps
        assert len(result["thinking_chain"]) == 3

        # Should show increasing confidence
        confidences = [step["confidence"] for step in result["thinking_chain"]]
        assert confidences[-1] > confidences[0]  # Final confidence higher than initial

        # Should stop when confidence is high enough
        assert result["thinking_chain"][-1]["confidence"] > 0.9

    async def test_orchestrator_status_monitoring(self, orchestrator):
        """Test orchestrator status monitoring"""
        # Add some registered agents
        orchestrator.registered_agents = {
            "agent_1": {"status": "online"},
            "agent_2": {"status": "offline"},
            "agent_3": {"status": "online"},
        }

        # Add some active workflows
        orchestrator.active_workflows = {
            "workflow_1": {"status": "running"},
            "workflow_2": {"status": "completed"},
        }

        # Update metrics
        orchestrator.metrics = {
            "tasks_completed": 15,
            "tasks_failed": 2,
            "average_thinking_time": 2.5,
            "reflection_count": 8,
        }

        # Mock cache stats
        orchestrator.cache.get_cache_stats = MagicMock(
            return_value={"hit_rate": 75.5, "total_hits": 150}
        )

        status = await orchestrator.get_orchestrator_status()

        assert status["orchestrator_id"] == orchestrator.orchestrator_id
        assert status["status"] == "operational"
        assert status["registered_agents"] == 3
        assert status["active_agents"] == 2  # Only online agents
        assert status["active_workflows"] == 2
        assert status["metrics"]["tasks_completed"] == 15
        assert "cache_stats" in status
        assert "timestamp" in status

    async def test_error_handling_in_thinking(self, orchestrator):
        """Test error handling during thinking process"""
        # Mock Gemini client to raise exception
        orchestrator.gemini_client.generate_content = AsyncMock(
            side_effect=Exception("Gemini API error")
        )

        with pytest.raises(OrchestrationError) as exc_info:
            await orchestrator.think_deeply(
                prompt="Test prompt", mode=ThinkingMode.QUICK_DECISION
            )

        assert "Thinking process failed" in str(exc_info.value)

    async def test_task_decomposition_validation(self, orchestrator):
        """Test validation of task decomposition"""
        # Mock invalid decomposition (no subtasks)
        orchestrator.think_deeply = AsyncMock(
            return_value={
                "conclusion": {
                    "conclusion": "Invalid decomposition with no clear subtasks",
                    "confidence": 0.5,
                }
            }
        )

        orchestrator._parse_task_decomposition = MagicMock(
            return_value={"subtasks": []}  # Empty subtasks
        )

        with pytest.raises(TaskDecompositionError) as exc_info:
            await orchestrator.decompose_task(
                task_description="Build something", requirements=["requirement 1"]
            )

        assert "Invalid decomposition" in str(exc_info.value)

    async def test_parallel_vs_sequential_execution(self, orchestrator):
        """Test parallel vs sequential execution modes"""
        workflow_id = "test_workflow"

        # Setup test workflow
        orchestrator.active_workflows[workflow_id] = {
            "decomposition": {
                "subtasks": [
                    {"id": "task_1", "description": "Task 1"},
                    {"id": "task_2", "description": "Task 2"},
                ]
            }
        }

        # Mock single task execution
        orchestrator._execute_single_task = AsyncMock(
            return_value={"status": "completed", "result": "success"}
        )

        orchestrator.coordination_client.update_workflow_status = AsyncMock()

        # Test parallel execution
        parallel_result = await orchestrator._execute_coordination_plan(
            workflow_id, {"execution_mode": "parallel"}, "parallel"
        )

        # Test sequential execution
        sequential_result = await orchestrator._execute_coordination_plan(
            workflow_id, {"execution_mode": "sequential"}, "sequential"
        )

        # Both should succeed
        assert parallel_result["status"] == "completed"
        assert sequential_result["status"] == "completed"

        # Both should have executed all tasks
        assert parallel_result["metrics"]["tasks_completed"] == 2
        assert sequential_result["metrics"]["tasks_completed"] == 2


class TestThinkingModes:
    """Test different thinking modes"""

    @pytest.fixture
    def orchestrator(self):
        """Simple orchestrator for thinking mode tests"""
        with patch("nexus_forge.agents.starri.orchestrator.GeminiClient"):
            orchestrator = StarriOrchestrator("test", "url", "key")
            orchestrator.gemini_client = AsyncMock()
            return orchestrator

    def test_thinking_mode_prompt_preparation(self, orchestrator):
        """Test prompt preparation for different thinking modes"""
        base_prompt = "Solve this optimization problem"

        # Test deep analysis mode
        deep_prompt = orchestrator._prepare_thinking_prompt(
            base_prompt, ThinkingMode.DEEP_ANALYSIS, {}
        )
        assert "deep analysis" in deep_prompt.lower()
        assert "fundamental components" in deep_prompt

        # Test quick decision mode
        quick_prompt = orchestrator._prepare_thinking_prompt(
            base_prompt, ThinkingMode.QUICK_DECISION, {}
        )
        assert "quick" in quick_prompt.lower()
        assert "critical factors" in quick_prompt

        # Test planning mode
        planning_prompt = orchestrator._prepare_thinking_prompt(
            base_prompt, ThinkingMode.PLANNING, {}
        )
        assert "plan" in planning_prompt.lower()
        assert "objectives" in planning_prompt

        # Test coordination mode
        coordination_prompt = orchestrator._prepare_thinking_prompt(
            base_prompt, ThinkingMode.COORDINATION, {}
        )
        assert "coordination" in coordination_prompt.lower()
        assert "agent capabilities" in coordination_prompt

        # Test reflection mode
        reflection_prompt = orchestrator._prepare_thinking_prompt(
            base_prompt, ThinkingMode.REFLECTION, {}
        )
        assert "reflect" in reflection_prompt.lower()
        assert "lessons learned" in reflection_prompt
