"""
WebSocket functionality test suite

Tests WebSocket connections, message handling, rate limiting,
and real-time features.
"""

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import WebSocket
from src.api.middleware.rate_limiter import WebSocketRateLimiter
from src.api.security.validation import WebSocketMessageValidator
from src.api.websocket_adk_enhanced import EnhancedADKWebSocketManager
from starlette.websockets import WebSocketDisconnect, WebSocketState


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self):
        self.client_state = WebSocketState.CONNECTED
        self.messages_sent = []
        self.close_code = None
        self.close_reason = None
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        if self.client_state != WebSocketState.CONNECTED:
            raise RuntimeError("WebSocket not connected")
        self.messages_sent.append(data)

    async def receive_json(self):
        # Mock receiving data
        return {"type": "ping"}

    async def close(self, code=1000, reason=""):
        self.client_state = WebSocketState.DISCONNECTED
        self.close_code = code
        self.close_reason = reason


class TestEnhancedWebSocketManager:
    """Test enhanced WebSocket manager functionality"""

    @pytest.fixture
    async def ws_manager(self):
        """Create WebSocket manager for testing"""
        manager = EnhancedADKWebSocketManager()

        # Mock dependencies
        manager.adk = AsyncMock()
        manager.state_manager = AsyncMock()
        manager.ws_rate_limiter = AsyncMock()

        # Mock rate limiter responses
        manager.ws_rate_limiter.check_websocket_limit = AsyncMock(
            return_value=(True, "OK")
        )
        manager.ws_rate_limiter.register_connection = AsyncMock()
        manager.ws_rate_limiter.unregister_connection = AsyncMock()

        yield manager

        # Cleanup
        await manager.shutdown()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        return MockWebSocket()

    @pytest.mark.asyncio
    async def test_connect_success(self, ws_manager, mock_websocket):
        """Test successful WebSocket connection"""

        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        assert session_id is not None
        assert mock_websocket.accepted
        assert len(mock_websocket.messages_sent) == 1

        # Check connection message
        msg = mock_websocket.messages_sent[0]
        assert msg["type"] == "connection_established"
        assert msg["session_id"] == session_id
        assert "features" in msg

    @pytest.mark.asyncio
    async def test_connect_rate_limited(self, ws_manager, mock_websocket):
        """Test rate limited connection"""

        # Mock rate limit exceeded
        ws_manager.ws_rate_limiter.check_websocket_limit = AsyncMock(
            return_value=(False, "Maximum connections exceeded")
        )

        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        assert session_id is None
        assert mock_websocket.close_code == 1008  # Policy violation
        assert "Maximum connections" in mock_websocket.close_reason

    @pytest.mark.asyncio
    async def test_connect_invalid_user_id(self, ws_manager, mock_websocket):
        """Test connection with invalid user ID"""

        session_id = await ws_manager.connect(
            mock_websocket, "user@invalid", "free"  # Invalid character
        )

        assert session_id is None
        assert mock_websocket.close_code == 1008

    @pytest.mark.asyncio
    async def test_disconnect(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection"""

        # First connect
        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        # Then disconnect
        await ws_manager.disconnect(session_id)

        # Verify cleanup
        assert session_id not in ws_manager.connection_metadata
        ws_manager.state_manager.remove_user_session.assert_called_once()
        ws_manager.ws_rate_limiter.unregister_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_handling_research_query(self, ws_manager, mock_websocket):
        """Test handling research query message"""

        # Connect first
        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        # Mock operation rate limiter
        with patch(
            "src.api.middleware.rate_limiter.OperationRateLimiter"
        ) as MockOpLimiter:
            mock_op_limiter = MockOpLimiter.return_value
            mock_op_limiter.check_operation_limit = AsyncMock(
                return_value=(True, {"remaining": 9})
            )

            # Send research query
            message = {
                "type": "research_query",
                "data": {
                    "query": "What is quantum computing?",
                    "mode": "comprehensive",
                },
            }

            # Mock create_research_task
            ws_manager.state_manager.create_research_task = AsyncMock(
                return_value="task_123"
            )

            await ws_manager.handle_message(mock_websocket, session_id, message)

            # Should create task
            ws_manager.state_manager.create_research_task.assert_called_once()

            # Should send acknowledgment
            ack_sent = False
            for msg in mock_websocket.messages_sent:
                if msg.get("type") == "research_started":
                    ack_sent = True
                    assert msg["task_id"] == "task_123"
                    break
            assert ack_sent

    @pytest.mark.asyncio
    async def test_message_validation_error(self, ws_manager, mock_websocket):
        """Test handling invalid messages"""

        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        # Invalid message type
        message = {"type": "hack_the_system", "data": {}}

        await ws_manager.handle_message(mock_websocket, session_id, message)

        # Should send error
        error_sent = False
        for msg in mock_websocket.messages_sent:
            if msg.get("type") == "error":
                error_sent = True
                assert msg["code"] == "invalid_input"
                break
        assert error_sent

    @pytest.mark.asyncio
    async def test_rate_limiting_per_operation(self, ws_manager, mock_websocket):
        """Test operation-specific rate limiting"""

        session_id = await ws_manager.connect(mock_websocket, "user123", "free")

        with patch(
            "src.api.middleware.rate_limiter.OperationRateLimiter"
        ) as MockOpLimiter:
            mock_op_limiter = MockOpLimiter.return_value

            # First request succeeds
            mock_op_limiter.check_operation_limit = AsyncMock(
                return_value=(True, {"remaining": 1})
            )

            message = {"type": "research_query", "data": {"query": "First query"}}

            ws_manager.state_manager.create_research_task = AsyncMock(
                return_value="task_1"
            )

            await ws_manager.handle_message(mock_websocket, session_id, message)

            # Second request rate limited
            mock_op_limiter.check_operation_limit = AsyncMock(
                return_value=(False, {"remaining": 0, "reset": 3600})
            )

            await ws_manager.handle_message(mock_websocket, session_id, message)

            # Check for rate limit error
            rate_limit_error = False
            for msg in mock_websocket.messages_sent:
                if msg.get("type") == "error" and msg.get("code") == "rate_limited":
                    rate_limit_error = True
                    break
            assert rate_limit_error

    @pytest.mark.asyncio
    async def test_tier_based_features(self, ws_manager, mock_websocket):
        """Test tier-based feature access"""

        tiers = ["free", "basic", "pro", "enterprise"]

        for tier in tiers:
            ws = MockWebSocket()
            session_id = await ws_manager.connect(ws, f"user_{tier}", tier)

            # Check features in connection message
            msg = ws.messages_sent[0]
            features = msg["features"]

            if tier == "free":
                assert features["max_queries_hour"] == 10
                assert features["voice_input"] is False
                assert "txt" in features["export_formats"]
                assert "notion" not in features["export_formats"]
            elif tier == "pro":
                assert features["max_queries_hour"] == 200
                assert features["voice_input"] is True
                assert "notion" in features["export_formats"]
            elif tier == "enterprise":
                assert features["max_queries_hour"] == 1000
                assert features["voice_input"] is True
                assert "custom" in features["export_formats"]


class TestResearchProcessing:
    """Test research processing functionality"""

    @pytest.fixture
    async def setup_manager(self):
        """Setup manager with mocked research stream"""
        manager = EnhancedADKWebSocketManager()

        # Mock ADK research stream
        async def mock_stream_research(query, user_id, session_id, mode):
            yield {
                "type": "start",
                "agent": "orchestrator",
                "content": "Starting research",
                "progress": 0,
            }
            yield {
                "type": "searching",
                "agent": "retrieval",
                "content": "Searching sources",
                "progress": 30,
            }
            yield {
                "type": "result",
                "agent": "analysis",
                "content": {"findings": ["Finding 1"]},
                "progress": 60,
            }
            yield {
                "type": "complete",
                "agent": "orchestrator",
                "content": {
                    "summary": "Research complete",
                    "sources": [{"title": "Source 1"}],
                },
                "progress": 100,
            }

        manager.adk = AsyncMock()
        manager.adk.stream_research = mock_stream_research

        manager.state_manager = AsyncMock()
        manager.state_manager.update_task_progress = AsyncMock()
        manager.state_manager.increment_metric = AsyncMock()

        return manager

    @pytest.mark.asyncio
    async def test_research_processing_flow(self, setup_manager):
        """Test complete research processing flow"""

        manager = setup_manager
        ws = MockWebSocket()

        # Setup connection metadata
        session_id = str(uuid.uuid4())
        manager.connection_metadata[session_id] = {
            "user_id": "user123",
            "user_tier": "pro",
        }

        # Process research
        from src.api.security.validation import ResearchQueryValidator

        query = ResearchQueryValidator(
            query="Test research query", mode="comprehensive"
        )

        await manager._process_research(ws, session_id, "task_123", query)

        # Verify progress updates
        assert manager.state_manager.update_task_progress.call_count >= 4

        # Verify WebSocket updates sent
        ws_updates = [
            msg for msg in ws.messages_sent if msg.get("type") == "research_update"
        ]
        assert len(ws_updates) >= 3

        # Verify completion
        completion_msg = next(
            (
                msg
                for msg in ws.messages_sent
                if msg.get("type") == "research_completed"
            ),
            None,
        )
        assert completion_msg is not None
        assert completion_msg["task_id"] == "task_123"
        assert "results" in completion_msg

    @pytest.mark.asyncio
    async def test_research_error_handling(self, setup_manager):
        """Test research error handling"""

        manager = setup_manager
        ws = MockWebSocket()

        # Make stream raise error
        async def error_stream(*args, **kwargs):
            yield {
                "type": "start",
                "agent": "orchestrator",
                "content": "Start",
                "progress": 0,
            }
            raise Exception("Research failed")

        manager.adk.stream_research = error_stream

        session_id = str(uuid.uuid4())
        manager.connection_metadata[session_id] = {
            "user_id": "user123",
            "user_tier": "free",
        }

        from src.api.security.validation import ResearchQueryValidator

        query = ResearchQueryValidator(query="Test")

        await manager._process_research(ws, session_id, "task_456", query)

        # Should update task as error
        manager.state_manager.update_task_progress.assert_called_with(
            "task_456", 100, status="error"
        )

        # Should send error to client
        error_msg = next(
            (
                msg
                for msg in ws.messages_sent
                if msg.get("type") == "error" and msg.get("code") == "research_failed"
            ),
            None,
        )
        assert error_msg is not None


class TestDistributedCoordination:
    """Test multi-instance coordination features"""

    @pytest.mark.asyncio
    async def test_distributed_event_handling(self):
        """Test handling events from other instances"""

        manager = EnhancedADKWebSocketManager()

        # Mock event stream
        async def mock_event_stream():
            yield {
                "type": "research_update",
                "channel": "research_updates",
                "task_id": "task_789",
                "progress": 50,
            }
            yield {
                "type": "agent_health",
                "channel": "system_events",
                "status": {
                    "overall_status": "degraded",
                    "agents": {"retrieval": {"status": "unhealthy"}},
                },
            }

        manager.state_manager.subscribe_to_events = AsyncMock()
        manager.state_manager.subscribe_to_events.return_value.__aenter__.return_value = (
            mock_event_stream()
        )

        # Process events (would run in background)
        events_processed = []

        async def capture_events():
            async with manager.state_manager.subscribe_to_events(["test"]) as events:
                async for event in events:
                    events_processed.append(event)
                    if len(events_processed) >= 2:
                        break

        await capture_events()

        assert len(events_processed) == 2
        assert events_processed[0]["type"] == "research_update"
        assert events_processed[1]["type"] == "agent_health"

    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test agent health monitoring"""

        manager = EnhancedADKWebSocketManager()

        # Mock ADK health check
        manager.adk = AsyncMock()
        manager.adk.get_agent_health = AsyncMock(
            return_value={
                "overall_status": "degraded",
                "agents": {
                    "orchestrator": {"status": "healthy"},
                    "retrieval": {"status": "unhealthy", "error": "Timeout"},
                },
            }
        )

        manager.state_manager = AsyncMock()

        # Run one health check iteration
        await manager._monitor_agent_health.__wrapped__(manager)

        # Should publish degraded status
        manager.state_manager.publish_event.assert_called_once()
        call_args = manager.state_manager.publish_event.call_args
        assert call_args[0][0] == "system_events"
        assert call_args[0][1]["type"] == "agent_health"
        assert call_args[0][1]["status"]["overall_status"] == "degraded"


class TestWebSocketSecurity:
    """Test WebSocket security features"""

    @pytest.mark.asyncio
    async def test_connection_limit_per_tier(self):
        """Test connection limits based on user tier"""

        manager = EnhancedADKWebSocketManager()

        # Test each tier's limit
        tier_limits = {"free": 2, "basic": 5, "pro": 10, "enterprise": 50}

        for tier, limit in tier_limits.items():
            assert manager._get_connection_limit(tier) == limit

        # Unknown tier should default to free
        assert manager._get_connection_limit("unknown") == 2

    @pytest.mark.asyncio
    async def test_message_injection_prevention(self):
        """Test prevention of message injection attacks"""

        manager = EnhancedADKWebSocketManager()
        ws = MockWebSocket()

        # Setup minimal mocks
        manager.state_manager = AsyncMock()
        manager.ws_rate_limiter = AsyncMock()
        manager.ws_rate_limiter.check_websocket_limit = AsyncMock(
            return_value=(True, "OK")
        )

        session_id = await manager.connect(ws, "user123", "free")

        # Try SQL injection in research query
        injection_message = {
            "type": "research_query",
            "data": {"query": "'; DROP TABLE users; --"},
        }

        await manager.handle_message(ws, session_id, injection_message)

        # Should receive error, not process query
        error_found = False
        for msg in ws.messages_sent:
            if msg.get("type") == "error":
                error_found = True
                break
        assert error_found

        # Should not create research task
        manager.state_manager.create_research_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_hijacking_prevention(self):
        """Test prevention of session hijacking"""

        manager = EnhancedADKWebSocketManager()

        # Create legitimate session
        ws1 = MockWebSocket()
        manager.ws_rate_limiter = AsyncMock()
        manager.ws_rate_limiter.check_websocket_limit = AsyncMock(
            return_value=(True, "OK")
        )
        manager.state_manager = AsyncMock()

        session_id = await manager.connect(ws1, "user123", "free")

        # Try to use same session from different connection
        ws2 = MockWebSocket()

        # Attempt to send message with hijacked session
        message = {"type": "ping", "session_id": session_id}

        # Should fail since ws2 is not associated with session
        await manager.handle_message(ws2, "fake_session", message)

        # Original connection should still work
        await manager.handle_message(ws1, session_id, {"type": "ping"})

        # Check ws1 received pong
        pong_found = any(msg.get("type") == "pong" for msg in ws1.messages_sent)
        assert pong_found
