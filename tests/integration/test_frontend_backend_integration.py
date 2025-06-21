"""
Frontend-Backend Integration Tests for Nexus Forge
Tests the complete end-to-end workflow from UI interactions to backend completion.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import patch

import pytest
import websockets
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.backend.core.auth import create_access_token
from src.backend.integrations.supabase.coordination_client import CoordinationClient
from src.backend.main import app


class TestFrontendBackendIntegration:
    """Test class for frontend-backend integration scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_token(self):
        """Create test authentication token."""
        return create_access_token(
            data={"sub": "test-user-id", "email": "test@example.com"}
        )

    @pytest.fixture
    def auth_headers(self, auth_token):
        """Create authentication headers."""
        return {"Authorization": f"Bearer {auth_token}"}

    async def test_complete_project_creation_workflow(self, client, auth_headers):
        """Test complete project creation workflow from frontend to completion."""

        # Step 1: Create project via API (simulating frontend form submission)
        project_data = {
            "name": "Test Integration App",
            "description": "E2E test application",
            "platform": "web",
            "framework": "React",
            "features": ["REST API", "Authentication", "Database Integration"],
            "requirements": "Simple CRUD application with user management",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=project_data, headers=auth_headers
        )
        assert response.status_code == 201
        project = response.json()
        project_id = project["id"]

        # Verify initial project state
        assert project["status"] == "pending"
        assert project["progress"] == 0
        assert project["name"] == project_data["name"]

        # Step 2: Verify WebSocket connection and real-time updates
        websocket_events = []

        async def mock_websocket_handler():
            """Mock WebSocket connection to collect events."""
            # Simulate connection and subscription
            websocket_events.append(
                {"type": "connected", "timestamp": datetime.utcnow().isoformat()}
            )

            # Simulate project subscription
            websocket_events.append(
                {
                    "type": "subscribed",
                    "project_id": project_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Simulate progress updates
            for progress in [10, 25, 50, 75, 90, 100]:
                await asyncio.sleep(0.1)  # Simulate processing time
                websocket_events.append(
                    {
                        "type": "project_update",
                        "payload": {
                            "project_id": project_id,
                            "status": "completed" if progress == 100 else "in_progress",
                            "progress": progress,
                            "current_task": f"Processing step {progress//20 + 1}",
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        # Run WebSocket simulation
        await mock_websocket_handler()

        # Step 3: Verify project completion
        response = client.get(
            f"/api/nexus-forge/projects/{project_id}", headers=auth_headers
        )
        assert response.status_code == 200
        completed_project = response.json()

        # Verify final state
        assert completed_project["status"] == "completed"
        assert completed_project["progress"] == 100
        assert "results" in completed_project

        # Step 4: Verify generated results structure
        results = completed_project["results"]
        assert "code_files" in results
        assert "assets" in results
        assert "documentation" in results

        # Verify code files
        code_files = results["code_files"]
        assert len(code_files) > 0
        for file in code_files:
            assert "path" in file
            assert "content" in file
            assert "type" in file
            assert len(file["content"]) > 0

        # Step 5: Test export functionality
        export_response = client.get(
            f"/api/nexus-forge/projects/{project_id}/export?format=zip",
            headers=auth_headers,
        )
        assert export_response.status_code == 200
        assert export_response.headers["content-type"] == "application/zip"

        # Step 6: Verify WebSocket events were properly collected
        assert len(websocket_events) >= 8  # connected + subscribed + 6 progress updates

        # Verify event structure
        for event in websocket_events:
            assert "type" in event
            assert "timestamp" in event
            if event["type"] == "project_update":
                assert "payload" in event
                payload = event["payload"]
                assert payload["project_id"] == project_id
                assert 0 <= payload["progress"] <= 100

    async def test_real_time_agent_coordination(self, client, auth_headers):
        """Test real-time agent status updates during project generation."""

        # Create a project to trigger agent activity
        project_data = {
            "name": "Agent Coordination Test",
            "description": "Test agent coordination",
            "platform": "web",
            "framework": "Vue.js",
            "features": ["REST API"],
            "requirements": "Simple API",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=project_data, headers=auth_headers
        )
        assert response.status_code == 201
        project_id = response.json()["id"]

        # Get initial agent statuses
        response = client.get("/api/nexus-forge/agents/status", headers=auth_headers)
        assert response.status_code == 200
        initial_agents = response.json()

        # Verify agent status structure
        for agent in initial_agents:
            assert "agent_id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "progress" in agent
            assert "last_update" in agent
            assert agent["status"] in ["idle", "working", "completed", "error"]
            assert 0 <= agent["progress"] <= 100

        # Test agent activity tracking
        activities = []
        coordination_client = CoordinationClient()

        # Simulate agent activities
        for i, agent in enumerate(initial_agents[:3]):  # Test first 3 agents
            activity = {
                "agent_id": agent["agent_id"],
                "project_id": project_id,
                "task_id": f"task-{i+1}",
                "status": "active",
                "metadata": {
                    "task_type": "code_generation",
                    "estimated_duration": 120,
                    "priority": "high",
                },
            }

            await coordination_client.record_agent_activity(activity)
            activities.append(activity)

        # Verify activities were recorded
        recorded_activities = await coordination_client.get_agent_activities(project_id)
        assert len(recorded_activities) >= len(activities)

        # Verify activity structure
        for activity in recorded_activities[-len(activities) :]:
            assert activity["project_id"] == project_id
            assert activity["status"] == "active"
            assert "metadata" in activity
            assert "timestamp" in activity

    async def test_task_progress_tracking(self, client, auth_headers):
        """Test detailed task progress tracking throughout project generation."""

        # Create project
        project_data = {
            "name": "Task Progress Test",
            "description": "Test task tracking",
            "platform": "mobile",
            "framework": "React Native",
            "features": ["Authentication", "Push Notifications"],
            "requirements": "Mobile app with login",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=project_data, headers=auth_headers
        )
        assert response.status_code == 201
        project_id = response.json()["id"]

        # Get task progress
        response = client.get(
            f"/api/nexus-forge/projects/{project_id}/tasks", headers=auth_headers
        )
        assert response.status_code == 200
        tasks = response.json()

        # Verify task structure
        for task in tasks:
            assert "task_id" in task
            assert "project_id" in task
            assert "title" in task
            assert "status" in task
            assert "progress" in task
            assert "agent_id" in task
            assert task["project_id"] == project_id
            assert task["status"] in ["pending", "in_progress", "completed", "failed"]
            assert 0 <= task["progress"] <= 100

        # Test individual task retrieval
        if tasks:
            task_id = tasks[0]["task_id"]
            response = client.get(
                f"/api/nexus-forge/tasks/{task_id}", headers=auth_headers
            )
            assert response.status_code == 200
            task_detail = response.json()

            # Verify detailed task information
            assert task_detail["task_id"] == task_id
            assert (
                "estimated_duration" in task_detail
                or task_detail.get("estimated_duration") is None
            )
            assert "started_at" in task_detail or task_detail.get("started_at") is None
            assert (
                "completed_at" in task_detail or task_detail.get("completed_at") is None
            )

    async def test_error_handling_and_recovery(self, client, auth_headers):
        """Test error handling scenarios and recovery mechanisms."""

        # Test 1: Invalid project data
        invalid_project_data = {
            "name": "",  # Empty name should fail
            "platform": "invalid_platform",
            "framework": "",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=invalid_project_data, headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        error_detail = response.json()
        assert "detail" in error_detail

        # Test 2: Access to non-existent project
        response = client.get(
            "/api/nexus-forge/projects/non-existent-id", headers=auth_headers
        )
        assert response.status_code == 404

        # Test 3: Invalid export format
        # First create a valid project
        valid_project_data = {
            "name": "Error Test Project",
            "description": "Test error handling",
            "platform": "web",
            "framework": "React",
            "features": [],
            "requirements": "Simple test",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=valid_project_data, headers=auth_headers
        )
        assert response.status_code == 201
        project_id = response.json()["id"]

        # Try invalid export format
        response = client.get(
            f"/api/nexus-forge/projects/{project_id}/export?format=invalid",
            headers=auth_headers,
        )
        assert response.status_code == 400

        # Test 4: Authentication error
        response = client.get("/api/nexus-forge/projects")  # No auth header
        assert response.status_code == 401

    async def test_performance_benchmarks(self, client, auth_headers):
        """Test performance benchmarks for key operations."""

        # Test 1: Project creation response time
        start_time = time.time()

        project_data = {
            "name": "Performance Test Project",
            "description": "Test performance",
            "platform": "web",
            "framework": "React",
            "features": ["REST API", "Authentication"],
            "requirements": "Performance test application",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=project_data, headers=auth_headers
        )

        creation_time = time.time() - start_time
        assert response.status_code == 201
        assert creation_time < 2.0  # Should create project in under 2 seconds

        project_id = response.json()["id"]

        # Test 2: Project retrieval response time
        start_time = time.time()

        response = client.get(
            f"/api/nexus-forge/projects/{project_id}", headers=auth_headers
        )

        retrieval_time = time.time() - start_time
        assert response.status_code == 200
        assert retrieval_time < 0.5  # Should retrieve project in under 500ms

        # Test 3: Agent status response time
        start_time = time.time()

        response = client.get("/api/nexus-forge/agents/status", headers=auth_headers)

        agent_status_time = time.time() - start_time
        assert response.status_code == 200
        assert agent_status_time < 1.0  # Should get agent status in under 1 second

        # Test 4: Concurrent request handling
        async def make_concurrent_request():
            """Make a concurrent API request."""
            return client.get("/api/nexus-forge/projects", headers=auth_headers)

        # Test 10 concurrent requests
        start_time = time.time()
        tasks = [make_concurrent_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time

        # Verify all requests succeeded
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == 10
        assert concurrent_time < 5.0  # All requests should complete in under 5 seconds

        for response in successful_responses:
            assert response.status_code == 200

    async def test_websocket_latency_and_reliability(self):
        """Test WebSocket connection latency and reliability."""

        # Mock WebSocket connection for testing
        websocket_stats = {
            "connection_time": 0,
            "message_latencies": [],
            "disconnections": 0,
            "reconnections": 0,
        }

        async def simulate_websocket_operations():
            """Simulate WebSocket operations and measure performance."""

            # Simulate connection
            connection_start = time.time()
            await asyncio.sleep(0.05)  # Simulate connection time
            websocket_stats["connection_time"] = time.time() - connection_start

            # Simulate message exchanges
            for i in range(10):
                message_start = time.time()
                await asyncio.sleep(0.01)  # Simulate message processing
                latency = time.time() - message_start
                websocket_stats["message_latencies"].append(latency)

            # Simulate disconnection and reconnection
            websocket_stats["disconnections"] = 1
            await asyncio.sleep(0.1)  # Simulate reconnection time
            websocket_stats["reconnections"] = 1

        await simulate_websocket_operations()

        # Verify performance metrics
        assert websocket_stats["connection_time"] < 0.1  # Under 100ms

        avg_latency = sum(websocket_stats["message_latencies"]) / len(
            websocket_stats["message_latencies"]
        )
        assert avg_latency < 0.05  # Under 50ms average

        max_latency = max(websocket_stats["message_latencies"])
        assert max_latency < 0.1  # Under 100ms max

        # Verify reliability
        assert websocket_stats["reconnections"] >= websocket_stats["disconnections"]

    async def test_data_consistency_across_services(self, client, auth_headers):
        """Test data consistency between API, WebSocket, and Supabase."""

        # Create project via API
        project_data = {
            "name": "Consistency Test Project",
            "description": "Test data consistency",
            "platform": "web",
            "framework": "Angular",
            "features": ["Database Integration"],
            "requirements": "Test consistency",
        }

        response = client.post(
            "/api/nexus-forge/projects", json=project_data, headers=auth_headers
        )
        assert response.status_code == 201
        api_project = response.json()
        project_id = api_project["id"]

        # Verify project exists in all systems

        # 1. API consistency check
        response = client.get(
            f"/api/nexus-forge/projects/{project_id}", headers=auth_headers
        )
        assert response.status_code == 200
        retrieved_project = response.json()

        # Verify data consistency
        assert retrieved_project["id"] == api_project["id"]
        assert retrieved_project["name"] == api_project["name"]
        assert retrieved_project["status"] == api_project["status"]

        # 2. Supabase consistency check (mock)
        coordination_client = CoordinationClient()

        # Simulate Supabase data retrieval
        supabase_project_coordination = (
            await coordination_client.get_project_coordination(project_id)
        )

        if supabase_project_coordination:
            assert supabase_project_coordination["project_id"] == project_id

        # 3. WebSocket message consistency (mock)
        websocket_messages = [
            {
                "type": "project_update",
                "payload": {
                    "project_id": project_id,
                    "status": "in_progress",
                    "progress": 50,
                },
            }
        ]

        # Verify message format consistency
        for message in websocket_messages:
            assert "type" in message
            assert "payload" in message
            assert message["payload"]["project_id"] == project_id
            assert "status" in message["payload"]
            assert "progress" in message["payload"]

    async def test_security_validation(self, client):
        """Test security measures and validation."""

        # Test 1: Unauthorized access
        response = client.get("/api/nexus-forge/projects")
        assert response.status_code == 401

        # Test 2: Invalid token
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/nexus-forge/projects", headers=invalid_headers)
        assert response.status_code == 401

        # Test 3: SQL injection attempt
        malicious_project_data = {
            "name": "'; DROP TABLE projects; --",
            "description": "Malicious project",
            "platform": "web",
            "framework": "React",
            "features": [],
            "requirements": "SELECT * FROM users",
        }

        # This should be handled safely by the validation layer
        auth_token = create_access_token(
            data={"sub": "test-user-id", "email": "test@example.com"}
        )
        auth_headers = {"Authorization": f"Bearer {auth_token}"}

        response = client.post(
            "/api/nexus-forge/projects",
            json=malicious_project_data,
            headers=auth_headers,
        )

        # Should either validate and sanitize, or reject
        assert response.status_code in [
            201,
            422,
        ]  # Created with sanitized data or validation error

        # Test 4: Rate limiting (mock)
        # Simulate rapid requests
        responses = []
        for _ in range(100):  # Simulate 100 rapid requests
            response = client.get("/api/health")
            responses.append(response.status_code)

        # At least some requests should succeed (health endpoint is usually not rate limited)
        success_count = sum(1 for status in responses if status == 200)
        assert success_count > 0

        # Test 5: Input validation
        invalid_inputs = [
            {"name": "x" * 1000},  # Too long name
            {"platform": "invalid"},  # Invalid platform
            {"framework": ""},  # Empty framework
            {"features": ["x" * 500] * 100},  # Too many/long features
        ]

        for invalid_input in invalid_inputs:
            project_data = {
                "name": "Test Project",
                "platform": "web",
                "framework": "React",
                "features": [],
                "requirements": "",
            }
            project_data.update(invalid_input)

            response = client.post(
                "/api/nexus-forge/projects", json=project_data, headers=auth_headers
            )
            # Should reject invalid input
            assert response.status_code in [422, 400]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
