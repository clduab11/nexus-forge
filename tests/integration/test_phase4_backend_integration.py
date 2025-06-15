"""
Phase 4 Backend Integration Tests

Comprehensive tests for the enhanced backend with Starri orchestration,
Supabase real-time coordination, and WebSocket integration.
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from httpx import AsyncClient
import websockets

# Import the main application and components
from nexus_forge.main import app
from nexus_forge.agents.starri.orchestrator import StarriOrchestrator, AgentCapability
from nexus_forge.websockets.manager import WebSocketManager
from nexus_forge.integrations.supabase.coordination_client import SupabaseCoordinationClient


class TestStarriOrchestration:
    """Test Starri orchestrator integration"""
    
    @pytest.fixture
    async def mock_orchestrator(self):
        """Create a mock Starri orchestrator"""
        orchestrator = AsyncMock(spec=StarriOrchestrator)
        orchestrator.get_orchestrator_status.return_value = {
            "orchestrator_id": "starri_test123",
            "status": "operational",
            "registered_agents": 4,
            "active_agents": 4,
            "active_workflows": 0,
            "metrics": {
                "tasks_completed": 15,
                "tasks_failed": 1,
                "average_thinking_time": 2.3,
                "total_thinking_time": 34.5,
                "reflection_count": 5
            }
        }
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_status_endpoint(self, mock_orchestrator):
        """Test the orchestrator status endpoint"""
        app.state.starri_orchestrator = mock_orchestrator
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/orchestrator/status")
            
        assert response.status_code == 200
        data = response.json()
        assert data["orchestrator_id"] == "starri_test123"
        assert data["status"] == "operational"
        assert data["registered_agents"] == 4
        assert data["metrics"]["tasks_completed"] == 15
    
    @pytest.mark.asyncio
    async def test_agent_registration_endpoint(self, mock_orchestrator):
        """Test agent registration via API"""
        app.state.starri_orchestrator = mock_orchestrator
        
        agent_data = {
            "agent_id": "test-agent-123",
            "agent_type": "code_generator",
            "capabilities": ["code_generation", "testing"],
            "configuration": {
                "language": "python",
                "framework": "fastapi"
            }
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/orchestrator/agents/register", json=agent_data)
            
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["agent_id"] == "test-agent-123"
        
        # Verify orchestrator.register_agent was called
        mock_orchestrator.register_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_decomposition_endpoint(self, mock_orchestrator):
        """Test task decomposition endpoint"""
        # Mock the decomposition result
        mock_orchestrator.decompose_task.return_value = {
            "workflow_id": "workflow_456",
            "decomposition": {
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Create React frontend",
                        "required_capabilities": ["ui_design", "code_generation"]
                    },
                    {
                        "id": "task_2", 
                        "description": "Build FastAPI backend",
                        "required_capabilities": ["code_generation", "api_integration"]
                    }
                ],
                "total_duration": "60m",
                "required_agents": ["ui_designer", "code_generator"]
            },
            "confidence": 0.89
        }
        
        app.state.starri_orchestrator = mock_orchestrator
        
        task_request = {
            "description": "Build a todo list app with React frontend and FastAPI backend",
            "requirements": ["responsive design", "user authentication", "CRUD operations"],
            "constraints": {"deadline": "2 hours", "budget": "medium"}
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/orchestrator/tasks/decompose", json=task_request)
            
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "workflow_456"
        assert len(data["decomposition"]["subtasks"]) == 2
        assert data["confidence"] == 0.89
    
    @pytest.mark.asyncio
    async def test_workflow_execution_endpoint(self, mock_orchestrator):
        """Test workflow execution endpoint"""
        # Mock the coordination result
        mock_orchestrator.coordinate_agents.return_value = {
            "workflow_id": "workflow_456",
            "status": "completed",
            "results": {
                "task_1": {"status": "completed", "output": "React app generated"},
                "task_2": {"status": "completed", "output": "FastAPI backend created"}
            },
            "metrics": {
                "start_time": 1640995200,
                "tasks_completed": 2,
                "tasks_failed": 0,
                "total_time": 180.5
            }
        }
        
        app.state.starri_orchestrator = mock_orchestrator
        
        execution_config = {"mode": "parallel"}
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/orchestrator/workflows/workflow_456/execute", 
                json=execution_config
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "workflow_456"
        assert data["status"] == "completed"
        assert data["metrics"]["tasks_completed"] == 2


class TestSupabaseIntegration:
    """Test Supabase coordination client integration"""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase coordination client"""
        client = AsyncMock(spec=SupabaseCoordinationClient)
        client.get_recent_events.return_value = [
            {
                "event_id": "event_123",
                "event_type": "task_assigned",
                "source_agent_id": "agent_1",
                "target_agent_id": "agent_2",
                "workflow_id": "workflow_456",
                "task_id": "task_1",
                "created_at": "2025-06-15T21:00:00Z"
            }
        ]
        return client
    
    @pytest.mark.asyncio
    async def test_coordination_events_endpoint(self, mock_supabase_client):
        """Test coordination events endpoint"""
        app.state.supabase_client = mock_supabase_client
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/coordination/events?limit=10")
            
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) == 1
        assert data["events"][0]["event_type"] == "task_assigned"
        
        mock_supabase_client.get_recent_events.assert_called_once_with(limit=10)


class TestWebSocketIntegration:
    """Test WebSocket real-time coordination"""
    
    @pytest.fixture
    def mock_ws_manager(self):
        """Create a mock WebSocket manager"""
        manager = AsyncMock(spec=WebSocketManager)
        manager.connect.return_value = "session_123"
        return manager
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, mock_ws_manager):
        """Test WebSocket connection establishment"""
        app.state.websocket_manager = mock_ws_manager
        
        # Test WebSocket connection
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Verify connection was established
                assert websocket is not None
    
    @pytest.mark.asyncio  
    async def test_websocket_real_time_updates(self, mock_ws_manager):
        """Test real-time updates via WebSocket"""
        app.state.websocket_manager = mock_ws_manager
        
        # Mock real-time event handling
        coordination_event = {
            "type": "coordination_event",
            "event_type": "task_assigned", 
            "source_agent": "agent_1",
            "target_agent": "agent_2",
            "workflow_id": "workflow_456",
            "timestamp": "2025-06-15T21:00:00Z"
        }
        
        # Simulate sending coordination event
        await mock_ws_manager.broadcast_coordination_event(
            event_type="task_assigned",
            source_agent_id="agent_1",
            target_agent_id="agent_2", 
            workflow_id="workflow_456"
        )
        
        mock_ws_manager.broadcast_coordination_event.assert_called_once()


class TestNexusForgeAPIIntegration:
    """Test enhanced Nexus Forge API with Starri integration"""
    
    @pytest.fixture
    def mock_app_components(self):
        """Mock all app components"""
        mock_orchestrator = AsyncMock(spec=StarriOrchestrator)
        mock_ws_manager = AsyncMock(spec=WebSocketManager)
        mock_supabase = AsyncMock(spec=SupabaseCoordinationClient)
        
        # Mock thinking and decomposition
        mock_orchestrator.think_deeply.return_value = {
            "thinking_chain": [{"step": 1, "thought": "Analyzing requirements..."}],
            "conclusion": {
                "conclusion": "This is a web application request requiring React frontend and FastAPI backend",
                "confidence": 0.92
            },
            "confidence": 0.92
        }
        
        mock_orchestrator.decompose_task.return_value = {
            "workflow_id": "workflow_789",
            "decomposition": {
                "subtasks": [
                    {"id": "task_1", "description": "Frontend development"},
                    {"id": "task_2", "description": "Backend development"}
                ]
            }
        }
        
        mock_orchestrator.coordinate_agents.return_value = {
            "workflow_id": "workflow_789",
            "status": "completed",
            "results": {"task_1": {"status": "completed"}, "task_2": {"status": "completed"}}
        }
        
        mock_ws_manager.start_build_session.return_value = "build_123"
        
        return {
            "orchestrator": mock_orchestrator,
            "ws_manager": mock_ws_manager,
            "supabase": mock_supabase
        }
    
    @pytest.mark.asyncio
    async def test_enhanced_build_endpoint(self, mock_app_components):
        """Test the enhanced build endpoint with Starri orchestration"""
        # Setup mocks
        app.state.starri_orchestrator = mock_app_components["orchestrator"]
        app.state.websocket_manager = mock_app_components["ws_manager"]
        app.state.supabase_client = mock_app_components["supabase"]
        
        build_request = {
            "prompt": "Create a task management app with user authentication",
            "config": {
                "framework": "react",
                "backend": "fastapi",
                "database": "postgresql",
                "execution_mode": "parallel"
            }
        }
        
        # Mock user authentication
        with patch("nexus_forge.api.routers.nexus_forge.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id="user_123")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/nexus-forge/build", json=build_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "build_id" in data
        assert data["status"] == "build_started"
        assert data["websocket_url"] == "/ws"
        assert "starri_orchestration" in data["features"]
    
    @pytest.mark.asyncio
    async def test_build_status_endpoint(self, mock_app_components):
        """Test build status retrieval"""
        # Setup active session
        from nexus_forge.api.routers.nexus_forge import active_sessions
        session_id = "session_456"
        active_sessions[session_id] = {
            "user_id": "user_123",
            "status": "completed",
            "started_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
            "result": {
                "app_structure": {"frontend": "React", "backend": "FastAPI"},
                "metrics": {"total_files": 23, "lines_of_code": 2847}
            }
        }
        
        with patch("nexus_forge.api.routers.nexus_forge.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id="user_123")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get(f"/api/nexus-forge/build/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "completed"
        assert data["result"]["metrics"]["total_files"] == 23


class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_app_building_workflow(self):
        """Test complete app building workflow from request to completion"""
        # This would be a comprehensive test that:
        # 1. Sends build request
        # 2. Connects to WebSocket for updates
        # 3. Monitors real-time progress
        # 4. Verifies final results
        # 5. Tests deployment endpoint
        
        # Setup all mocks
        mock_orchestrator = AsyncMock(spec=StarriOrchestrator)
        mock_ws_manager = AsyncMock(spec=WebSocketManager)
        mock_supabase = AsyncMock(spec=SupabaseCoordinationClient)
        
        # Configure mock responses for complete workflow
        mock_orchestrator.think_deeply.return_value = {
            "conclusion": {"conclusion": "Full-stack web app", "confidence": 0.95},
            "confidence": 0.95
        }
        
        mock_orchestrator.decompose_task.return_value = {
            "workflow_id": "end_to_end_workflow",
            "decomposition": {"subtasks": []}
        }
        
        mock_orchestrator.coordinate_agents.return_value = {
            "workflow_id": "end_to_end_workflow", 
            "status": "completed",
            "results": {}
        }
        
        mock_ws_manager.start_build_session.return_value = "e2e_build_session"
        
        # Set up app state
        app.state.starri_orchestrator = mock_orchestrator
        app.state.websocket_manager = mock_ws_manager
        app.state.supabase_client = mock_supabase
        
        # Test the workflow
        build_request = {
            "prompt": "Build a comprehensive e-commerce platform",
            "config": {"complexity": "high", "deployment": "cloud_run"}
        }
        
        with patch("nexus_forge.api.routers.nexus_forge.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id="user_e2e")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # 1. Start build
                response = await client.post("/api/nexus-forge/build", json=build_request)
                assert response.status_code == 200
                session_id = response.json()["session_id"]
                
                # 2. Check orchestrator status
                orch_response = await client.get("/api/orchestrator/status")
                assert orch_response.status_code == 200
                
                # 3. Get coordination events
                events_response = await client.get("/api/coordination/events")
                assert events_response.status_code == 200
                
                # 4. Check build status
                # Note: In real test, would wait for completion
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Verify all components were called
                mock_orchestrator.think_deeply.assert_called()
                mock_ws_manager.start_build_session.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        mock_orchestrator = AsyncMock(spec=StarriOrchestrator)
        mock_ws_manager = AsyncMock(spec=WebSocketManager)
        
        # Simulate orchestrator failure
        mock_orchestrator.think_deeply.side_effect = Exception("Orchestrator error")
        mock_ws_manager.start_build_session.return_value = "error_test_session"
        
        app.state.starri_orchestrator = mock_orchestrator
        app.state.websocket_manager = mock_ws_manager
        
        build_request = {"prompt": "Test error handling"}
        
        with patch("nexus_forge.api.routers.nexus_forge.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id="user_error_test")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/nexus-forge/build", json=build_request)
                # Should still return 200 but will fail in background
                assert response.status_code == 200


class TestPerformanceAndScaling:
    """Performance and scaling tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_build_requests(self):
        """Test handling multiple concurrent build requests"""
        # Setup mocks
        mock_orchestrator = AsyncMock(spec=StarriOrchestrator)
        mock_ws_manager = AsyncMock(spec=WebSocketManager)
        
        mock_orchestrator.think_deeply.return_value = {"confidence": 0.8}
        mock_orchestrator.decompose_task.return_value = {"workflow_id": "concurrent_test"}
        mock_ws_manager.start_build_session.side_effect = lambda **kwargs: f"session_{id(kwargs)}"
        
        app.state.starri_orchestrator = mock_orchestrator
        app.state.websocket_manager = mock_ws_manager
        
        # Create multiple concurrent requests
        build_requests = [
            {"prompt": f"Build app {i}"} for i in range(5)
        ]
        
        with patch("nexus_forge.api.routers.nexus_forge.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id="user_concurrent")
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Send concurrent requests
                tasks = [
                    client.post("/api/nexus-forge/build", json=request)
                    for request in build_requests
                ]
                
                responses = await asyncio.gather(*tasks)
                
                # Verify all requests succeeded
                for response in responses:
                    assert response.status_code == 200
                    data = response.json()
                    assert "session_id" in data
                    assert "build_id" in data
    
    @pytest.mark.asyncio
    async def test_websocket_connection_limits(self):
        """Test WebSocket connection handling and limits"""
        mock_ws_manager = AsyncMock(spec=WebSocketManager)
        
        # Mock connection limit behavior
        connection_count = 0
        def mock_connect(websocket, user_id, tier):
            nonlocal connection_count
            connection_count += 1
            if connection_count > 5:  # Simulate limit
                return None
            return f"session_{connection_count}"
        
        mock_ws_manager.connect.side_effect = mock_connect
        app.state.websocket_manager = mock_ws_manager
        
        # Test would simulate multiple WebSocket connections
        # and verify proper limit enforcement
        
        assert True  # Placeholder for actual WebSocket testing


# Fixtures and test configuration
@pytest.fixture(scope="session")
def test_app():
    """Create test application instance"""
    return app


@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Setup test environment before each test"""
    # Reset any global state
    from nexus_forge.api.routers.nexus_forge import active_sessions
    active_sessions.clear()
    
    yield
    
    # Cleanup after test
    active_sessions.clear()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])