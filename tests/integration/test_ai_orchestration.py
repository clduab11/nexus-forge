"""
Integration tests for AI orchestration end-to-end workflow

Tests the complete flow from user request to final output, including:
- Multi-agent coordination
- WebSocket real-time updates
- Error handling and recovery
- Performance benchmarks
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from src.backend.agents.agents.nexus_forge_agents import StarriOrchestrator
from src.backend.api.v1.nexus_forge_router import src.backend_router
from src.backend.services.google_ai_service import GoogleAIService


@pytest.mark.integration
@pytest.mark.nexus_forge
@pytest.mark.asyncio
class TestAIOrchestrationFlow:
    """Test complete AI orchestration workflow"""

    async def test_end_to_end_app_build_workflow(
        self,
        authenticated_client,
        mock_starri_orchestrator,
        mock_imagen_integration,
        mock_veo_integration,
        mock_nexus_forge_websocket,
        sample_app_specification,
        integration_test_environment,
    ):
        """Test complete app build from request to deployment"""
        # Arrange
        build_request = {
            "prompt": "Build a real-time analytics dashboard with AI insights",
            "config": {
                "useAdaptiveThinking": True,
                "enableVideoDemo": True,
                "deployToCloudRun": True,
                "targetFramework": "react",
            },
        }

        # Mock orchestrator phases
        mock_starri_orchestrator.build_app_with_starri.return_value = {
            "specification": sample_app_specification.__dict__,
            "mockups": {
                "Dashboard": "https://storage.googleapis.com/mockups/dashboard.png",
                "Analytics": "https://storage.googleapis.com/mockups/analytics.png",
            },
            "demo_video": "https://storage.googleapis.com/videos/demo.mp4",
            "code_files": {
                "main.py": "# FastAPI backend\nfrom fastapi import FastAPI\napp = FastAPI()",
                "frontend/App.tsx": "// React frontend\nimport React from 'react';",
                "tests/test_main.py": "# Unit tests\nimport pytest",
            },
            "deployment_url": "https://analytics-app-abc123.a.run.app",
            "build_time": "5 minutes 23 seconds",
            "orchestrator": "Starri",
            "models_used": [
                "gemini_2_5_pro",
                "imagen_4",
                "veo_3",
                "jules",
                "gemini_flash",
            ],
        }

        # Configure WebSocket to simulate real-time updates
        update_sequence = [
            {
                "type": "phase_start",
                "phase": "specification",
                "message": "Analyzing requirements",
            },
            {"type": "progress", "phase": "specification", "progress": 25},
            {"type": "phase_complete", "phase": "specification", "progress": 25},
            {
                "type": "phase_start",
                "phase": "design",
                "message": "Creating UI mockups",
            },
            {"type": "progress", "phase": "design", "progress": 50},
            {"type": "phase_complete", "phase": "design", "progress": 50},
            {
                "type": "phase_start",
                "phase": "video",
                "message": "Generating demo video",
            },
            {"type": "progress", "phase": "video", "progress": 75},
            {"type": "phase_complete", "phase": "video", "progress": 75},
            {
                "type": "phase_start",
                "phase": "code",
                "message": "Writing application code",
            },
            {"type": "progress", "phase": "code", "progress": 90},
            {"type": "phase_complete", "phase": "code", "progress": 90},
            {
                "type": "phase_start",
                "phase": "deployment",
                "message": "Deploying to Cloud Run",
            },
            {"type": "progress", "phase": "deployment", "progress": 100},
            {
                "type": "build_complete",
                "result": mock_starri_orchestrator.build_app_with_starri.return_value,
            },
        ]

        websocket_updates = []

        async def capture_websocket_update(data):
            websocket_updates.append(data)

        mock_nexus_forge_websocket.send_json.side_effect = capture_websocket_update

        # Act
        # Simulate WebSocket connection
        await mock_nexus_forge_websocket.accept()

        # Send build request
        await mock_nexus_forge_websocket.receive_json.return_value

        # Process build through orchestrator
        async for update in mock_starri_orchestrator.build_app_with_starri(
            build_request["prompt"], build_request["config"]
        ):
            await mock_nexus_forge_websocket.send_json(update)

        # Assert
        # Verify all phases were executed
        assert len(websocket_updates) >= len(update_sequence)

        # Verify orchestrator was called correctly
        mock_starri_orchestrator.build_app_with_starri.assert_called_once_with(
            build_request["prompt"], build_request["config"]
        )

        # Verify final result contains all required components
        final_result = websocket_updates[-1]["result"]
        assert "specification" in final_result
        assert "mockups" in final_result
        assert "demo_video" in final_result
        assert "code_files" in final_result
        assert "deployment_url" in final_result

        # Verify performance metrics
        assert "build_time" in final_result
        build_time_seconds = self._parse_build_time(final_result["build_time"])
        assert build_time_seconds < 600  # Should complete within 10 minutes

    async def test_multi_agent_coordination(
        self,
        mock_starri_orchestrator,
        mock_imagen_integration,
        mock_veo_integration,
        mock_ai_model_responses,
        performance_benchmarks,
    ):
        """Test coordination between multiple AI agents"""
        # Arrange
        prompt = "Create an e-commerce platform with AI recommendations"

        # Track agent calls
        agent_calls = []

        async def track_agent_call(agent_name, method, *args, **kwargs):
            agent_calls.append(
                {
                    "agent": agent_name,
                    "method": method,
                    "timestamp": datetime.utcnow(),
                    "args": args,
                    "kwargs": kwargs,
                }
            )

        # Wrap agent methods to track calls
        original_spec = mock_starri_orchestrator.generate_app_specification
        mock_starri_orchestrator.generate_app_specification.side_effect = (
            lambda *args, **kwargs: track_agent_call(
                "gemini_2_5_pro", "generate_specification", *args, **kwargs
            )
        )

        original_mockups = mock_imagen_integration.generate_ui_mockup
        mock_imagen_integration.generate_ui_mockup.side_effect = (
            lambda *args, **kwargs: track_agent_call(
                "imagen_4", "generate_mockup", *args, **kwargs
            )
        )

        original_video = mock_veo_integration.generate_demo_video
        mock_veo_integration.generate_demo_video.side_effect = (
            lambda *args, **kwargs: track_agent_call(
                "veo_3", "generate_video", *args, **kwargs
            )
        )

        # Act
        start_time = datetime.utcnow()
        result = await mock_starri_orchestrator.build_app_with_starri(prompt)
        end_time = datetime.utcnow()

        # Assert
        # Verify agents were called in correct order
        assert len(agent_calls) >= 3
        assert agent_calls[0]["agent"] == "gemini_2_5_pro"
        assert agent_calls[1]["agent"] == "imagen_4"
        assert agent_calls[2]["agent"] == "veo_3"

        # Verify timing constraints
        total_time = (end_time - start_time).total_seconds()
        assert total_time < performance_benchmarks["build_completion_time"]

        # Verify parallel execution where applicable
        # Mockups and video generation should overlap
        mockup_start = next(
            c["timestamp"] for c in agent_calls if c["agent"] == "imagen_4"
        )
        video_start = next(c["timestamp"] for c in agent_calls if c["agent"] == "veo_3")
        time_diff = abs((mockup_start - video_start).total_seconds())
        assert time_diff < 1.0  # Should start within 1 second of each other

    async def test_error_handling_and_recovery(
        self,
        mock_starri_orchestrator,
        mock_imagen_integration,
        mock_veo_integration,
        mock_nexus_forge_websocket,
    ):
        """Test error handling and recovery mechanisms"""
        # Arrange
        error_scenarios = [
            {
                "phase": "specification",
                "error": Exception("Gemini API rate limit exceeded"),
                "expected_recovery": "retry_with_backoff",
            },
            {
                "phase": "mockups",
                "error": Exception("Imagen service temporarily unavailable"),
                "expected_recovery": "fallback_to_alternative",
            },
            {
                "phase": "video",
                "error": Exception("Video generation timeout"),
                "expected_recovery": "skip_optional_component",
            },
        ]

        for scenario in error_scenarios:
            # Configure mock to raise error
            if scenario["phase"] == "specification":
                mock_starri_orchestrator.generate_app_specification.side_effect = (
                    scenario["error"]
                )
            elif scenario["phase"] == "mockups":
                mock_imagen_integration.generate_ui_mockup.side_effect = scenario[
                    "error"
                ]
            elif scenario["phase"] == "video":
                mock_veo_integration.generate_demo_video.side_effect = scenario["error"]

            # Act
            try:
                result = await mock_starri_orchestrator.build_app_with_starri(
                    "Build a test app", {"handleErrors": True}
                )

                # Assert recovery behavior
                if scenario["expected_recovery"] == "retry_with_backoff":
                    # Should have retried the operation
                    assert (
                        mock_starri_orchestrator.generate_app_specification.call_count
                        >= 2
                    )
                elif scenario["expected_recovery"] == "fallback_to_alternative":
                    # Should use fallback mockup generation
                    assert "mockups" in result
                    assert result["mockups"] is not None
                elif scenario["expected_recovery"] == "skip_optional_component":
                    # Should complete without video
                    assert "demo_video" not in result or result["demo_video"] is None
                    assert "code_files" in result  # Core functionality preserved

            except Exception as e:
                # Only critical errors should propagate
                assert (
                    scenario["phase"] == "specification"
                )  # Only spec generation is critical

            # Reset mocks for next scenario
            mock_starri_orchestrator.generate_app_specification.side_effect = None
            mock_imagen_integration.generate_ui_mockup.side_effect = None
            mock_veo_integration.generate_demo_video.side_effect = None

    async def test_concurrent_build_requests(
        self, mock_starri_orchestrator, performance_benchmarks
    ):
        """Test handling of concurrent build requests"""
        # Arrange
        num_concurrent_requests = 5
        build_requests = [
            {
                "prompt": f"Build app {i}: {['CRM', 'Dashboard', 'E-commerce', 'Blog', 'Portfolio'][i]}",
                "config": {"useAdaptiveThinking": True},
            }
            for i in range(num_concurrent_requests)
        ]

        # Track request processing
        request_times = []

        async def process_request(request):
            start = datetime.utcnow()
            result = await mock_starri_orchestrator.build_app_with_starri(
                request["prompt"], request["config"]
            )
            end = datetime.utcnow()
            return {
                "request": request,
                "result": result,
                "duration": (end - start).total_seconds(),
            }

        # Act
        # Process requests concurrently
        results = await asyncio.gather(
            *[process_request(req) for req in build_requests]
        )

        # Assert
        # All requests should complete
        assert len(results) == num_concurrent_requests

        # Each request should have a valid result
        for result in results:
            assert "result" in result
            assert result["result"] is not None
            assert result["duration"] < performance_benchmarks["build_completion_time"]

        # Verify concurrent execution
        # Total time should be less than sequential execution would take
        total_time = max(r["duration"] for r in results)
        average_time = sum(r["duration"] for r in results) / len(results)
        assert (
            total_time < average_time * num_concurrent_requests * 0.5
        )  # At least 50% concurrency gain

    async def test_resource_cleanup(
        self,
        mock_starri_orchestrator,
        mock_imagen_integration,
        mock_veo_integration,
        mock_nexus_forge_websocket,
    ):
        """Test proper resource cleanup after build completion or failure"""
        # Arrange
        resources_to_track = {
            "websocket_connections": [],
            "temporary_files": [],
            "ai_model_sessions": [],
            "memory_usage": [],
        }

        # Track resource allocation
        original_accept = mock_nexus_forge_websocket.accept

        async def track_websocket(*args, **kwargs):
            resources_to_track["websocket_connections"].append(datetime.utcnow())
            return await original_accept(*args, **kwargs)

        mock_nexus_forge_websocket.accept = track_websocket

        # Act
        # Successful build
        await mock_nexus_forge_websocket.accept()
        result = await mock_starri_orchestrator.build_app_with_starri(
            "Build a test app", {"cleanup": True}
        )
        await mock_nexus_forge_websocket.close()

        # Failed build
        mock_starri_orchestrator.build_app_with_starri.side_effect = Exception(
            "Build failed"
        )
        try:
            await mock_nexus_forge_websocket.accept()
            await mock_starri_orchestrator.build_app_with_starri(
                "Build another app", {"cleanup": True}
            )
        except:
            pass
        finally:
            await mock_nexus_forge_websocket.close()

        # Assert
        # All WebSocket connections should be closed
        assert mock_nexus_forge_websocket.close.call_count == 2

        # Temporary resources should be cleaned up
        # (In real implementation, check file system, memory, etc.)
        assert len(resources_to_track["websocket_connections"]) == 2

    def _parse_build_time(self, build_time_str: str) -> float:
        """Parse build time string to seconds"""
        # Example: "5 minutes 23 seconds" -> 323.0
        parts = build_time_str.split()
        total_seconds = 0

        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                value = int(parts[i])
                unit = parts[i + 1]
                if "minute" in unit:
                    total_seconds += value * 60
                elif "second" in unit:
                    total_seconds += value

        return total_seconds


@pytest.mark.integration
@pytest.mark.nexus_forge
@pytest.mark.asyncio
class TestGoogleAIServiceIntegration:
    """Test Google AI Service integration with all models"""

    async def test_google_ai_service_initialization(self, integration_test_environment):
        """Test Google AI Service initialization and model loading"""
        # Act
        service = GoogleAIService()

        # Assert
        assert service is not None
        assert hasattr(service, "gemini_client")
        assert hasattr(service, "imagen_integration")
        assert hasattr(service, "veo_integration")
        assert hasattr(service, "jules_integration")

        # Verify all models are accessible
        models = await service.list_available_models()
        expected_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "imagen-4",
            "veo-3",
            "jules",
        ]
        for model in expected_models:
            assert model in models

    async def test_model_orchestration_flow(
        self, mock_ai_model_responses, integration_test_environment
    ):
        """Test orchestrated flow between different AI models"""
        # Arrange
        service = GoogleAIService()
        prompt = "Create a machine learning dashboard"

        # Act
        # Step 1: Generate specification with Gemini 2.5 Pro
        spec = await service.generate_specification(prompt)

        # Step 2: Generate mockups with Imagen 4
        mockups = await service.generate_mockups(spec)

        # Step 3: Generate demo video with Veo 3
        video = await service.generate_demo_video(spec, mockups)

        # Step 4: Generate code with Jules
        code = await service.generate_code(spec)

        # Step 5: Optimize with Gemini Flash
        optimized_code = await service.optimize_code(code)

        # Assert
        assert spec is not None
        assert "name" in spec
        assert "features" in spec

        assert mockups is not None
        assert len(mockups) > 0

        assert video is not None
        assert video.startswith("https://")

        assert code is not None
        assert len(code) > 0

        assert optimized_code is not None
        assert optimized_code != code  # Should be different after optimization
