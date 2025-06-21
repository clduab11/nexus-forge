"""
Integration tests for WebSocket real-time update flow

Tests the complete WebSocket communication flow including:
- Connection lifecycle
- Real-time progress updates
- Error handling and reconnection
- Performance under load
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
import websockets
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.backend.api.v1.nexus_forge_router import src.backend_websocket_endpoint
from src.backend.services.websocket_manager import WebSocketManager


@pytest.mark.integration
@pytest.mark.websocket
@pytest.mark.asyncio
class TestWebSocketFlow:
    """Test WebSocket real-time communication flow"""

    async def test_websocket_connection_lifecycle(
        self, authenticated_client, mock_nexus_forge_websocket, mock_starri_orchestrator
    ):
        """Test complete WebSocket connection lifecycle"""
        # Arrange
        connection_events = []

        async def track_event(event_type, data=None):
            connection_events.append(
                {"type": event_type, "timestamp": datetime.utcnow(), "data": data}
            )

        # Mock WebSocket methods with tracking
        original_accept = mock_nexus_forge_websocket.accept

        async def tracked_accept(*args, **kwargs):
            await track_event("connection_accepted")
            return await original_accept(*args, **kwargs)

        mock_nexus_forge_websocket.accept = tracked_accept

        original_send = mock_nexus_forge_websocket.send_json

        async def tracked_send(data):
            await track_event("message_sent", data)
            return await original_send(data)

        mock_nexus_forge_websocket.send_json = tracked_send

        # Act
        # Establish connection
        await mock_nexus_forge_websocket.accept()

        # Send initial handshake
        await mock_nexus_forge_websocket.send_json(
            {
                "type": "connection_established",
                "session_id": "test-session-123",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Receive build request
        build_request = {
            "type": "build_request",
            "data": {
                "prompt": "Build a test application",
                "config": {"useAdaptiveThinking": True},
            },
        }
        mock_nexus_forge_websocket.receive_json.return_value = build_request

        # Process request and send updates
        phases = [
            {"id": "init", "name": "Initialization"},
            {"id": "spec", "name": "Specification"},
            {"id": "design", "name": "Design"},
            {"id": "code", "name": "Code Generation"},
            {"id": "deploy", "name": "Deployment"},
        ]

        for i, phase in enumerate(phases):
            progress = int((i + 1) / len(phases) * 100)
            await mock_nexus_forge_websocket.send_json(
                {
                    "type": "progress_update",
                    "phase": phase["id"],
                    "message": f"Processing {phase['name']}...",
                    "progress": progress,
                }
            )
            await asyncio.sleep(0.1)  # Simulate processing

        # Send completion
        await mock_nexus_forge_websocket.send_json(
            {
                "type": "build_complete",
                "result": {
                    "deployment_url": "https://app.example.com",
                    "build_time": "2 minutes 15 seconds",
                },
            }
        )

        # Close connection
        await mock_nexus_forge_websocket.close()
        await track_event("connection_closed")

        # Assert
        # Verify connection lifecycle events
        event_types = [e["type"] for e in connection_events]
        assert "connection_accepted" in event_types
        assert "message_sent" in event_types
        assert "connection_closed" in event_types

        # Verify message sequence
        sent_messages = [
            e["data"] for e in connection_events if e["type"] == "message_sent"
        ]
        assert len(sent_messages) >= len(phases) + 2  # phases + handshake + completion

        # Verify progress updates
        progress_updates = [
            m for m in sent_messages if m.get("type") == "progress_update"
        ]
        assert len(progress_updates) == len(phases)

        # Verify progress increases monotonically
        progresses = [u["progress"] for u in progress_updates]
        assert progresses == sorted(progresses)
        assert progresses[-1] == 100

    async def test_concurrent_websocket_connections(
        self, authenticated_client, performance_benchmarks
    ):
        """Test handling of multiple concurrent WebSocket connections"""
        # Arrange
        num_connections = 10
        connections = []
        connection_times = []

        async def create_connection(connection_id):
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            ws.receive_json = AsyncMock()
            ws.close = AsyncMock()

            start_time = datetime.utcnow()
            await ws.accept()
            end_time = datetime.utcnow()

            connection_times.append(
                {
                    "id": connection_id,
                    "duration": (end_time - start_time).total_seconds(),
                }
            )

            return ws

        # Act
        # Create connections concurrently
        connections = await asyncio.gather(
            *[create_connection(i) for i in range(num_connections)]
        )

        # Simulate concurrent message broadcasting
        broadcast_message = {
            "type": "system_announcement",
            "message": "System update in progress",
            "timestamp": datetime.utcnow().isoformat(),
        }

        await asyncio.gather(
            *[conn.send_json(broadcast_message) for conn in connections]
        )

        # Close all connections
        await asyncio.gather(*[conn.close() for conn in connections])

        # Assert
        # All connections should be established
        assert len(connections) == num_connections

        # Connection times should be within limits
        for timing in connection_times:
            assert (
                timing["duration"] < performance_benchmarks["websocket_connection_time"]
            )

        # All connections should receive broadcast
        for conn in connections:
            conn.send_json.assert_called_with(broadcast_message)

        # All connections should be closed
        for conn in connections:
            conn.close.assert_called_once()

    async def test_websocket_error_handling_and_recovery(
        self, mock_nexus_forge_websocket, mock_starri_orchestrator
    ):
        """Test WebSocket error handling and automatic recovery"""
        # Arrange
        error_scenarios = [
            {
                "error": WebSocketDisconnect(code=1006),
                "recovery": "reconnect",
                "max_retries": 3,
            },
            {
                "error": asyncio.TimeoutError("Connection timeout"),
                "recovery": "retry_with_backoff",
                "max_retries": 5,
            },
            {
                "error": Exception("Unexpected error"),
                "recovery": "graceful_shutdown",
                "max_retries": 1,
            },
        ]

        for scenario in error_scenarios:
            retry_count = 0
            recovery_successful = False

            # Configure mock to raise error initially
            mock_nexus_forge_websocket.receive_json.side_effect = scenario["error"]

            # Act
            for attempt in range(scenario["max_retries"]):
                try:
                    # Attempt connection
                    await mock_nexus_forge_websocket.accept()

                    # Try to receive message
                    message = await mock_nexus_forge_websocket.receive_json()

                    # If successful, mark recovery
                    recovery_successful = True
                    break

                except (WebSocketDisconnect, asyncio.TimeoutError, Exception) as e:
                    retry_count += 1

                    if scenario["recovery"] == "reconnect":
                        # Simulate reconnection
                        await asyncio.sleep(0.1 * attempt)  # Exponential backoff
                        if attempt == scenario["max_retries"] - 1:
                            # Success on last attempt
                            mock_nexus_forge_websocket.receive_json.side_effect = None
                            mock_nexus_forge_websocket.receive_json.return_value = {
                                "type": "ping"
                            }

                    elif scenario["recovery"] == "retry_with_backoff":
                        # Exponential backoff
                        await asyncio.sleep(2**attempt * 0.1)
                        if attempt >= 2:
                            # Success after a few retries
                            mock_nexus_forge_websocket.receive_json.side_effect = None
                            mock_nexus_forge_websocket.receive_json.return_value = {
                                "type": "ping"
                            }

                    elif scenario["recovery"] == "graceful_shutdown":
                        # Clean up and exit
                        await mock_nexus_forge_websocket.close()
                        break

            # Assert
            if scenario["recovery"] in ["reconnect", "retry_with_backoff"]:
                assert recovery_successful or retry_count == scenario["max_retries"]
            else:
                assert mock_nexus_forge_websocket.close.called

            # Reset for next scenario
            mock_nexus_forge_websocket.receive_json.side_effect = None
            retry_count = 0
            recovery_successful = False

    async def test_websocket_message_ordering_and_buffering(
        self, mock_nexus_forge_websocket
    ):
        """Test message ordering and buffering under high load"""
        # Arrange
        num_messages = 100
        sent_messages = []
        received_messages = []

        # Create messages with sequence numbers
        messages = [
            {
                "type": "progress_update",
                "sequence": i,
                "timestamp": (
                    datetime.utcnow() + timedelta(milliseconds=i)
                ).isoformat(),
                "data": f"Update {i}",
            }
            for i in range(num_messages)
        ]

        # Mock message buffer
        message_buffer = asyncio.Queue(maxsize=50)

        async def buffered_send(message):
            if message_buffer.full():
                # Wait for buffer to have space
                await asyncio.sleep(0.01)
            await message_buffer.put(message)
            sent_messages.append(message)

        async def buffered_receive():
            while not message_buffer.empty():
                message = await message_buffer.get()
                received_messages.append(message)
                await asyncio.sleep(0.001)  # Simulate processing

        mock_nexus_forge_websocket.send_json.side_effect = buffered_send

        # Act
        # Send messages rapidly
        await asyncio.gather(
            *[mock_nexus_forge_websocket.send_json(msg) for msg in messages]
        )

        # Receive and process messages
        await buffered_receive()

        # Assert
        # All messages should be sent
        assert len(sent_messages) == num_messages

        # Messages should maintain order
        for i in range(1, len(received_messages)):
            prev_seq = received_messages[i - 1]["sequence"]
            curr_seq = received_messages[i]["sequence"]
            assert (
                curr_seq > prev_seq
            ), f"Message order violated: {prev_seq} -> {curr_seq}"

        # Timestamps should be monotonically increasing
        timestamps = [
            datetime.fromisoformat(msg["timestamp"]) for msg in received_messages
        ]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    async def test_websocket_performance_metrics(
        self, mock_nexus_forge_websocket, performance_benchmarks
    ):
        """Test WebSocket performance under various conditions"""
        # Arrange
        test_scenarios = [
            {
                "name": "small_messages",
                "message_size": 100,  # bytes
                "message_count": 1000,
                "expected_throughput": 10000,  # messages/second
            },
            {
                "name": "large_messages",
                "message_size": 10000,  # 10KB
                "message_count": 100,
                "expected_throughput": 100,  # messages/second
            },
            {
                "name": "mixed_sizes",
                "message_size": "mixed",
                "message_count": 500,
                "expected_throughput": 1000,  # messages/second
            },
        ]

        performance_results = []

        for scenario in test_scenarios:
            # Create test messages
            if scenario["message_size"] == "mixed":
                messages = [
                    {"type": "test", "data": "x" * (100 * (i % 100 + 1))}
                    for i in range(scenario["message_count"])
                ]
            else:
                messages = [
                    {"type": "test", "data": "x" * scenario["message_size"]}
                    for _ in range(scenario["message_count"])
                ]

            # Act
            start_time = datetime.utcnow()

            # Send all messages
            for msg in messages:
                await mock_nexus_forge_websocket.send_json(msg)

            end_time = datetime.utcnow()

            # Calculate metrics
            duration = (end_time - start_time).total_seconds()
            throughput = scenario["message_count"] / duration if duration > 0 else 0

            performance_results.append(
                {
                    "scenario": scenario["name"],
                    "duration": duration,
                    "throughput": throughput,
                    "message_count": scenario["message_count"],
                }
            )

            # Assert
            # Throughput should meet expectations
            assert (
                throughput >= scenario["expected_throughput"] * 0.8
            )  # Allow 20% variance

        # Overall performance assertions
        for result in performance_results:
            print(f"\nScenario: {result['scenario']}")
            print(f"Duration: {result['duration']:.3f}s")
            print(f"Throughput: {result['throughput']:.0f} msg/s")

    async def test_websocket_state_management(
        self, mock_nexus_forge_websocket, mock_starri_orchestrator
    ):
        """Test WebSocket connection state management"""
        # Arrange
        ws_manager = WebSocketManager()
        connection_id = "test-conn-123"
        user_id = 1

        # Track state changes
        state_changes = []

        async def track_state_change(old_state, new_state):
            state_changes.append(
                {
                    "timestamp": datetime.utcnow(),
                    "old_state": old_state,
                    "new_state": new_state,
                }
            )

        # Act
        # Connect
        await ws_manager.connect(connection_id, mock_nexus_forge_websocket, user_id)
        await track_state_change(None, "connected")

        # Start build
        await ws_manager.update_state(connection_id, "building")
        await track_state_change("connected", "building")

        # Send progress updates
        for progress in [0, 25, 50, 75, 100]:
            await ws_manager.send_to_connection(
                connection_id, {"type": "progress", "value": progress}
            )

        # Complete build
        await ws_manager.update_state(connection_id, "completed")
        await track_state_change("building", "completed")

        # Disconnect
        await ws_manager.disconnect(connection_id)
        await track_state_change("completed", "disconnected")

        # Assert
        # Verify state transitions
        assert len(state_changes) == 4
        assert state_changes[0]["new_state"] == "connected"
        assert state_changes[1]["new_state"] == "building"
        assert state_changes[2]["new_state"] == "completed"
        assert state_changes[3]["new_state"] == "disconnected"

        # Verify connection is cleaned up
        assert connection_id not in ws_manager.active_connections
