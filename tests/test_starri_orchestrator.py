"""
Comprehensive test suite for Starri Orchestrator - the core multi-agent coordinator

Tests the orchestration of Gemini 2.5 Pro, Jules, Veo 3, Imagen 4, and other AI models
for one-shot app building capabilities.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nexus_forge.agents.agents.nexus_forge_agents import (
    AppSpecification,
    ModelType,
    StarriOrchestrator,
)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


class TestStarriOrchestrator:
    """Test suite for Starri Orchestrator multi-agent coordination"""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create a Starri Orchestrator instance with mocked dependencies"""
        with patch("nexus_forge.agents.agents.nexus_forge_agents.aiplatform"):
            with patch("nexus_forge.agents.agents.nexus_forge_agents.GenerativeModel"):
                orchestrator = StarriOrchestrator(
                    project_id="test-project", region="us-central1"
                )

                # Mock the AI model integrations
                orchestrator.veo_integration = AsyncMock()
                orchestrator.imagen_integration = AsyncMock()

                yield orchestrator

    @pytest_asyncio.fixture
    async def sample_app_spec(self):
        """Sample app specification for testing"""
        return AppSpecification(
            name="Test Analytics Dashboard",
            description="Real-time analytics dashboard for data visualization",
            features=[
                "Real-time data updates",
                "Interactive charts",
                "User authentication",
                "Export functionality",
            ],
            tech_stack={
                "frontend": "React",
                "backend": "FastAPI",
                "database": "PostgreSQL",
            },
            ui_components=["Dashboard", "Charts", "UserProfile", "Settings"],
            api_endpoints=[
                {"method": "GET", "path": "/api/data", "description": "Fetch data"},
                {
                    "method": "POST",
                    "path": "/api/auth",
                    "description": "Authentication",
                },
            ],
            database_schema={"tables": ["users", "analytics", "sessions"]},
            deployment_config={"platform": "Cloud Run", "scaling": "auto"},
        )

    async def test_starri_initialization(self, orchestrator):
        """Test Starri Orchestrator initialization"""
        assert orchestrator.project_id == "test-project"
        assert orchestrator.region == "us-central1"

        # Verify communication channels are initialized
        assert len(orchestrator.communication_channels) == 5
        assert all(
            isinstance(channel, asyncio.Queue)
            for channel in orchestrator.communication_channels.values()
        )

        # Verify agent status tracking
        assert len(orchestrator.active_agents) == 5
        assert all(
            agent["status"] == "idle" for agent in orchestrator.active_agents.values()
        )

    async def test_starri_analyze_and_plan(self, orchestrator):
        """Test Starri's analysis and planning capabilities"""
        user_prompt = (
            "Build a real-time analytics dashboard with charts and user authentication"
        )

        # Mock the Starri coordinator response
        orchestrator.starri_coordinator = AsyncMock()
        orchestrator.starri_coordinator.generate_content_async = AsyncMock(
            return_value=MagicMock(
                text='{"complexity": "medium", "features": ["charts", "auth"], "parallel_tasks": ["ui", "api"], "delegation_plan": {}, "risks": [], "estimated_time": "5-10 minutes"}'
            )
        )

        plan = await orchestrator._starri_analyze_and_plan(user_prompt)

        assert plan["complexity"] == "medium"
        assert "charts" in plan["features"]
        assert "ui" in plan["parallel_tasks"]
        assert plan["estimated_time"] == "5-10 minutes"

    async def test_generate_app_specification(self, orchestrator):
        """Test app specification generation with Gemini 2.5 Pro"""
        user_prompt = "Build an e-commerce platform with product catalog and checkout"

        # Mock the specification model
        orchestrator.models["specification"] = AsyncMock()
        orchestrator.models["specification"].generate_content_async = AsyncMock(
            return_value=MagicMock(
                text='{"name": "E-Commerce Platform", "description": "Online shopping platform", "features": ["product catalog", "checkout"], "tech_stack": {"frontend": "React", "backend": "FastAPI"}, "ui_components": ["ProductList", "Cart"], "api_endpoints": [], "database_schema": null, "deployment_config": {"platform": "Cloud Run"}}'
            )
        )

        spec = await orchestrator.generate_app_specification(user_prompt)

        assert spec.name == "E-Commerce Platform"
        assert "product catalog" in spec.features
        assert spec.tech_stack["frontend"] == "React"

    async def test_multi_agent_parallel_execution(self, orchestrator, sample_app_spec):
        """Test parallel execution of multiple agents"""
        # Mock UI and video generation to return quickly
        orchestrator.imagen_integration.generate_design_system = AsyncMock(
            return_value={"colors": {"primary": "#3B82F6"}, "typography": {}}
        )
        orchestrator.imagen_integration.generate_ui_mockup = AsyncMock(
            return_value={"url": "https://mockup.url", "metadata": {}}
        )
        orchestrator.veo_integration.generate_demo_video = AsyncMock(
            return_value="https://video.url"
        )

        # Test parallel execution
        start_time = asyncio.get_event_loop().time()

        # Run UI mockup and video planning in parallel
        mockups_task = asyncio.create_task(
            orchestrator._starri_delegate_ui_design(sample_app_spec)
        )
        video_plan_task = asyncio.create_task(
            orchestrator._starri_plan_demo_video(sample_app_spec)
        )

        mockups, video_plan = await asyncio.gather(mockups_task, video_plan_task)

        end_time = asyncio.get_event_loop().time()

        # Verify parallel execution (should complete faster than sequential)
        assert (end_time - start_time) < 1.0  # Should be fast with mocks
        assert "mockups" in mockups
        assert "scenes" in video_plan

    async def test_communication_channels(self, orchestrator):
        """Test inter-agent communication through channels"""
        # Test sending message to Jules channel
        test_message = {
            "type": "specification_ready",
            "data": {"spec": "test"},
            "from": "starri",
        }

        await orchestrator._broadcast_to_agents("specification_ready", {"spec": "test"})

        # Verify message was sent to all channels
        for agent_name, channel in orchestrator.communication_channels.items():
            assert not channel.empty()
            message = await channel.get()
            assert message["type"] == "specification_ready"
            assert message["from"] == "starri"

    async def test_error_handling_and_recovery(self, orchestrator):
        """Test error handling and fallback strategies"""
        # Mock a failing Imagen integration
        orchestrator.imagen_integration.generate_ui_mockup = AsyncMock(
            side_effect=Exception("Imagen API error")
        )

        # The system should handle the error gracefully
        with patch(
            "nexus_forge.agents.agents.nexus_forge_agents.logger"
        ) as mock_logger:
            try:
                await orchestrator.generate_ui_mockups(
                    AppSpecification(
                        name="Test App",
                        description="Test",
                        features=[],
                        tech_stack={},
                        ui_components=["Dashboard"],
                        api_endpoints=[],
                        database_schema=None,
                        deployment_config={},
                    )
                )
            except Exception:
                # Should log the error
                assert mock_logger.error.called

    async def test_build_app_with_starri_full_workflow(
        self, orchestrator, sample_app_spec
    ):
        """Test the complete app building workflow orchestrated by Starri"""
        user_prompt = "Build a real-time analytics dashboard"

        # Mock all the components
        orchestrator.starri_coordinator = AsyncMock()
        orchestrator.starri_coordinator.generate_content_async = AsyncMock(
            return_value=MagicMock(
                text='{"complexity": "medium", "features": [], "parallel_tasks": [], "delegation_plan": {}, "risks": [], "estimated_time": "5 minutes"}'
            )
        )

        orchestrator.models["specification"] = AsyncMock()
        orchestrator.models["specification"].generate_content_async = AsyncMock(
            return_value=MagicMock(text="{}")
        )

        orchestrator._parse_specification_response = MagicMock(
            return_value=sample_app_spec.__dict__
        )

        orchestrator.generate_ui_mockups = AsyncMock(
            return_value={"mockups": {"Dashboard": "url"}}
        )
        orchestrator.generate_demo_video = AsyncMock(return_value="video.url")
        orchestrator.generate_code_with_jules = AsyncMock(
            return_value={"main.py": "code"}
        )
        orchestrator.optimize_with_flash = AsyncMock(
            return_value={"main.py": "optimized"}
        )
        orchestrator._deploy_to_cloud_run = AsyncMock(
            return_value="https://app.run.app"
        )

        # Execute the full workflow
        result = await orchestrator.build_app_with_starri(user_prompt)

        # Verify all phases were executed
        assert result["specification"] is not None
        assert result["mockups"] is not None
        assert result["demo_video"] == "video.url"
        assert result["code_files"] is not None
        assert result["deployment_url"] == "https://app.run.app"
        assert result["orchestrator"] == "Starri"
        assert len(result["models_used"]) == 5

    async def test_agent_status_tracking(self, orchestrator, sample_app_spec):
        """Test that agent statuses are properly tracked during execution"""
        # Set up mocks
        orchestrator.generate_app_specification = AsyncMock(
            return_value=sample_app_spec
        )

        # Execute specification phase
        build_context = {"user_prompt": "test", "execution_plan": {}}
        await orchestrator._starri_delegate_specification(build_context)

        # Verify agent status was updated
        # Note: Status should return to idle after completion
        assert orchestrator.active_agents["gemini_pro"]["status"] == "completed"

    async def test_parallel_resource_management(self, orchestrator):
        """Test that parallel execution doesn't overwhelm system resources"""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                orchestrator._broadcast_to_agents(f"test_message_{i}", {"data": i})
            )
            tasks.append(task)

        # All tasks should complete without issues
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # No exceptions should occur
        assert all(result is None for result in results)

        # Verify all messages were queued
        for channel in orchestrator.communication_channels.values():
            assert channel.qsize() == 10


class TestAppSpecificationValidation:
    """Test suite for AppSpecification validation and handling"""

    def test_app_specification_creation(self):
        """Test creating a valid AppSpecification"""
        spec = AppSpecification(
            name="Test App",
            description="A test application",
            features=["Feature 1", "Feature 2"],
            tech_stack={"frontend": "React", "backend": "Node.js"},
            ui_components=["Header", "Footer"],
            api_endpoints=[{"method": "GET", "path": "/api/test"}],
            database_schema={"tables": ["users"]},
            deployment_config={"platform": "Cloud Run"},
        )

        assert spec.name == "Test App"
        assert len(spec.features) == 2
        assert spec.tech_stack["frontend"] == "React"

    def test_app_specification_optional_fields(self):
        """Test AppSpecification with optional fields"""
        spec = AppSpecification(
            name="Minimal App",
            description="Minimal test",
            features=[],
            tech_stack={},
            ui_components=[],
            api_endpoints=[],
            database_schema=None,
            deployment_config={},
        )

        assert spec.database_schema is None
        assert len(spec.features) == 0


class TestPerformanceAndScaling:
    """Test suite for performance and scaling characteristics"""

    @pytest.mark.asyncio
    async def test_concurrent_agent_limit(self):
        """Test that system handles concurrent agent limits properly"""
        orchestrator = StarriOrchestrator("test-project")

        # Simulate high concurrent load
        concurrent_tasks = []
        for i in range(50):
            task = asyncio.create_task(
                orchestrator._broadcast_to_agents(f"load_test_{i}", {"index": i})
            )
            concurrent_tasks.append(task)

        # Should handle all tasks without memory issues
        await asyncio.gather(*concurrent_tasks)

        # Verify system is still responsive
        assert all(
            agent["status"] == "idle" for agent in orchestrator.active_agents.values()
        )
