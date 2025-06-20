"""
Comprehensive test suite for ADK Agent2Agent Protocol

Tests the core protocol, discovery, security, negotiation, and integration
components.
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from src.backend.protocols.agent2agent import (
    Agent2AgentMessage,
    MessageType,
    Agent2AgentProtocol,
    AgentDiscoveryService,
    AgentInfo,
    DiscoveryQuery,
    SecureAgentChannel,
    AgentCertificateManager,
    MessageEncryption,
    CapabilityNegotiationEngine,
    Task,
    TaskContract,
    Agent2AgentIntegration
)


class TestAgent2AgentCore:
    """Test core protocol functionality"""
    
    @pytest.fixture
    def protocol(self):
        """Create protocol instance for testing"""
        return Agent2AgentProtocol("test_agent_1")
        
    @pytest.fixture  
    def sample_message(self):
        """Create sample message for testing"""
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_PROPOSE,
            sender="test_agent_1",
            recipient="test_agent_2",
            payload={"task": "test_task", "priority": 1},
            timestamp=time.time()
        )
        
    @pytest.mark.asyncio
    async def test_message_serialization(self, sample_message):
        """Test message serialization and deserialization"""
        # To dict
        msg_dict = sample_message.to_dict()
        assert msg_dict["id"] == sample_message.id
        assert msg_dict["type"] == sample_message.type.value
        assert msg_dict["sender"] == sample_message.sender
        
        # From dict
        restored = Agent2AgentMessage.from_dict(msg_dict)
        assert restored.id == sample_message.id
        assert restored.type == sample_message.type
        assert restored.payload == sample_message.payload
        
        # To bytes
        msg_bytes = sample_message.to_bytes()
        assert isinstance(msg_bytes, bytes)
        
        # From bytes
        restored_from_bytes = Agent2AgentMessage.from_bytes(msg_bytes)
        assert restored_from_bytes.id == sample_message.id
        
    @pytest.mark.asyncio
    async def test_message_expiration(self):
        """Test message TTL and expiration"""
        # Create message with short TTL
        message = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.HEALTH_CHECK,
            sender="agent1",
            recipient="agent2",
            payload={},
            timestamp=time.time() - 10,  # 10 seconds ago
            ttl=5  # 5 second TTL
        )
        
        assert message.is_expired() is True
        
        # Fresh message should not be expired
        fresh_message = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.HEALTH_CHECK,
            sender="agent1",
            recipient="agent2",
            payload={},
            timestamp=time.time(),
            ttl=60
        )
        
        assert fresh_message.is_expired() is False
        
    @pytest.mark.asyncio
    async def test_protocol_handshake(self, protocol):
        """Test protocol handshake mechanism"""
        await protocol.start()
        
        # Create handshake message
        handshake = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.PROTOCOL_HANDSHAKE,
            sender="remote_agent",
            recipient=protocol.agent_id,
            payload={"version": "2.0.0"},
            timestamp=time.time()
        )
        
        # Process handshake
        response = await protocol.process_incoming(handshake)
        
        assert response is not None
        assert response.type == MessageType.PROTOCOL_HANDSHAKE
        assert response.payload["version"] == "2.0.0"
        assert response.payload["status"] == "ready"
        
        await protocol.stop()
        
    @pytest.mark.asyncio
    async def test_health_check_handling(self, protocol):
        """Test health check message handling"""
        await protocol.start()
        
        # Send health check
        health_check = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.HEALTH_CHECK,
            sender="monitor_agent",
            recipient=protocol.agent_id,
            payload={},
            timestamp=time.time()
        )
        
        response = await protocol.process_incoming(health_check)
        
        assert response is not None
        assert response.type == MessageType.HEALTH_REPORT
        assert response.payload["status"] == "healthy"
        assert "protocol_version" in response.payload
        
        await protocol.stop()


class TestAgentDiscovery:
    """Test agent discovery service"""
    
    @pytest.fixture
    async def discovery_service(self):
        """Create discovery service for testing"""
        protocol = Agent2AgentProtocol("discovery_test")
        service = AgentDiscoveryService(protocol)
        await service.initialize()
        yield service
        await service.shutdown()
        
    @pytest.fixture
    def sample_agent(self):
        """Create sample agent info"""
        return AgentInfo(
            id="test_agent",
            name="Test Agent",
            type="development",
            description="Test agent for unit tests",
            capabilities=["code_generation", "testing", "debugging"],
            resources={"memory": "2Gi", "cpu": 1.0},
            performance_metrics={
                "avg_response_time": 150,
                "success_rate": 0.95,
                "availability": 0.99
            },
            adk_version="2.0",
            protocols_supported=["agent2agent/2.0"]
        )
        
    @pytest.mark.asyncio
    async def test_agent_registration(self, discovery_service, sample_agent):
        """Test agent registration in discovery"""
        # Register agent
        await discovery_service.register_agent(sample_agent)
        
        # Verify registration
        registered = await discovery_service.registry.get(sample_agent.id)
        assert registered is not None
        assert registered.id == sample_agent.id
        assert registered.name == sample_agent.name
        
        # Check capability index
        agents_with_capability = discovery_service.capability_index.find_agents_with_capability("code_generation")
        assert sample_agent.id in agents_with_capability
        
    @pytest.mark.asyncio
    async def test_agent_discovery_by_capability(self, discovery_service, sample_agent):
        """Test discovering agents by capability"""
        # Register multiple agents
        await discovery_service.register_agent(sample_agent)
        
        another_agent = AgentInfo(
            id="another_agent",
            name="Another Agent",
            type="testing",
            description="Another test agent",
            capabilities=["testing", "monitoring"],
            resources={"memory": "1Gi", "cpu": 0.5},
            performance_metrics={
                "avg_response_time": 100,
                "success_rate": 0.98
            },
            adk_version="2.0",
            protocols_supported=["agent2agent/2.0"]
        )
        await discovery_service.register_agent(another_agent)
        
        # Find agents with testing capability
        testing_agents = await discovery_service.find_agents_by_capability("testing")
        
        assert len(testing_agents) == 2
        agent_ids = [agent.id for agent in testing_agents]
        assert sample_agent.id in agent_ids
        assert another_agent.id in agent_ids
        
    @pytest.mark.asyncio
    async def test_discovery_query(self, discovery_service, sample_agent):
        """Test complex discovery queries"""
        await discovery_service.register_agent(sample_agent)
        
        # Query by type
        query = DiscoveryQuery(
            agent_type="development",
            status="active"
        )
        
        results = await discovery_service.discover_agents(query)
        assert len(results) == 1
        assert results[0].id == sample_agent.id
        
        # Query by capabilities
        query = DiscoveryQuery(
            capabilities=["code_generation", "testing"],
            min_performance={"success_rate": 0.9}
        )
        
        results = await discovery_service.discover_agents(query)
        assert len(results) == 1
        assert results[0].id == sample_agent.id
        
        # Query with no matches
        query = DiscoveryQuery(
            capabilities=["nonexistent_capability"]
        )
        
        results = await discovery_service.discover_agents(query)
        assert len(results) == 0


class TestSecureCommunication:
    """Test secure communication layer"""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for testing"""
        return MessageEncryption()
        
    @pytest.fixture
    def cert_manager(self):
        """Create certificate manager for testing"""
        return AgentCertificateManager()
        
    @pytest.mark.asyncio
    async def test_session_key_generation(self, encryption_service):
        """Test session key generation"""
        key_id, key = encryption_service.generate_session_key()
        
        assert key_id is not None
        assert key is not None
        assert len(key) > 0
        assert key_id in encryption_service.symmetric_keys
        
    @pytest.mark.asyncio
    async def test_payload_encryption_decryption(self, encryption_service):
        """Test message payload encryption and decryption"""
        # Generate session key
        key_id, key = encryption_service.generate_session_key()
        
        # Original payload
        original_payload = {
            "task": "test_task",
            "data": {"value": 42, "message": "secret"},
            "timestamp": time.time()
        }
        
        # Encrypt
        encrypted = encryption_service.encrypt_payload(original_payload, key_id)
        
        assert encrypted["encrypted"] is True
        assert "data" in encrypted
        assert encrypted["algorithm"] == "fernet"
        
        # Decrypt
        decrypted = encryption_service.decrypt_payload(encrypted, key_id)
        
        assert decrypted == original_payload
        
    @pytest.mark.asyncio
    async def test_secure_channel_establishment(self, cert_manager, encryption_service):
        """Test secure channel establishment between agents"""
        channel = SecureAgentChannel(
            "agent_a",
            "agent_b",
            cert_manager,
            encryption_service
        )
        
        # Mock certificates
        with patch.object(cert_manager, 'get_certificate', return_value=Mock(is_valid=lambda: True)):
            with patch.object(cert_manager, 'verify_certificate', return_value=True):
                success = await channel.establish()
                
        assert success is True
        assert channel.established is True
        assert channel.session_key_id is not None
        
        await channel.close()
        assert channel.established is False


class TestCapabilityNegotiation:
    """Test capability negotiation engine"""
    
    @pytest.fixture
    async def negotiation_engine(self):
        """Create negotiation engine for testing"""
        protocol = Agent2AgentProtocol("negotiator")
        engine = CapabilityNegotiationEngine(protocol)
        return engine
        
    @pytest.fixture
    def sample_task(self):
        """Create sample task for negotiation"""
        return Task(
            id=str(uuid.uuid4()),
            name="Generate API endpoints",
            description="Generate REST API endpoints for user management",
            required_capabilities=["code_generation", "api_design"],
            optional_capabilities=["testing", "documentation"],
            requirements={"language": "python", "framework": "fastapi"},
            constraints={"max_time": 300},
            priority=1,
            max_duration=300,
            success_criteria={"endpoints": ["GET /users", "POST /users", "PUT /users/{id}"]},
            reward=10.0
        )
        
    @pytest.mark.asyncio
    async def test_capability_matching(self, negotiation_engine, sample_task):
        """Test semantic capability matching"""
        matcher = negotiation_engine.capability_matcher
        
        # Test exact match
        score = matcher.match_capabilities(
            ["code_generation", "api_design"],
            ["code_generation", "api_design", "testing"]
        )
        assert score == 1.0
        
        # Test partial match
        score = matcher.match_capabilities(
            ["code_generation", "api_design", "testing"],
            ["code_generation", "api_design"]
        )
        assert score < 1.0
        
        # Test no match
        score = matcher.match_capabilities(
            ["video_generation"],
            ["code_generation", "api_design"]
        )
        assert score < 0.5
        
    @pytest.mark.asyncio
    async def test_agent_ranking(self, negotiation_engine, sample_task):
        """Test agent ranking for task"""
        agents = [
            AgentInfo(
                id="agent1",
                name="Agent 1",
                type="development",
                description="Development agent",
                capabilities=["code_generation", "api_design", "testing"],
                resources={},
                performance_metrics={"response_time": 100, "success_rate": 0.95},
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0"]
            ),
            AgentInfo(
                id="agent2",
                name="Agent 2",
                type="development",
                description="Another development agent",
                capabilities=["code_generation"],
                resources={},
                performance_metrics={"response_time": 200, "success_rate": 0.85},
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0"]
            )
        ]
        
        ranked = await negotiation_engine._rank_agents_for_task(sample_task, agents)
        
        assert len(ranked) >= 1
        assert ranked[0][0].id == "agent1"  # agent1 should rank higher
        assert ranked[0][1] > ranked[1][1] if len(ranked) > 1 else True


class TestIntegration:
    """Test full Agent2Agent integration"""
    
    @pytest.fixture
    async def integration(self):
        """Create integration instance for testing"""
        ws_manager = Mock()
        adk_service = Mock()
        
        integration = Agent2AgentIntegration(
            "integration_test_agent",
            ws_manager,
            adk_service
        )
        
        # Mock Supabase client
        integration.supabase_client = AsyncMock()
        
        await integration.initialize()
        yield integration
        await integration.shutdown()
        
    @pytest.mark.asyncio
    async def test_message_sending(self, integration):
        """Test sending messages through integration"""
        # Mock WebSocket bridge
        integration.ws_bridge.send_to_agent = AsyncMock()
        
        # Send message
        message = await integration.send_message(
            recipient="target_agent",
            message_type=MessageType.TASK_PROPOSE,
            payload={"task": "test"},
            secure=False
        )
        
        assert message is not None
        assert message.sender == integration.agent_id
        assert message.recipient == "target_agent"
        assert message.type == MessageType.TASK_PROPOSE
        
        # Verify WebSocket bridge was called
        integration.ws_bridge.send_to_agent.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_task_creation_and_negotiation(self, integration):
        """Test task creation and negotiation flow"""
        # Create task
        task = await integration.create_task(
            name="Test Task",
            description="A test task for unit testing",
            required_capabilities=["testing"],
            priority=1,
            max_duration=60
        )
        
        assert task.id in integration.active_tasks
        assert task.name == "Test Task"
        
        # Mock discovery to return capable agents
        mock_agents = [
            AgentInfo(
                id="capable_agent",
                name="Capable Agent",
                type="testing",
                description="Agent capable of testing",
                capabilities=["testing", "validation"],
                resources={},
                performance_metrics={"success_rate": 0.95},
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0"]
            )
        ]
        
        with patch.object(integration, 'discover_agents', return_value=mock_agents):
            # Mock negotiation engine
            if integration.negotiation_engine:
                mock_contract = TaskContract(
                    contract_id=str(uuid.uuid4()),
                    task_id=task.id,
                    requester_id=integration.agent_id,
                    provider_id="capable_agent",
                    terms=Mock(),
                    created_at=time.time()
                )
                
                with patch.object(
                    integration.negotiation_engine,
                    'negotiate_task',
                    return_value=mock_contract
                ):
                    # Negotiate task execution
                    contract = await integration.negotiate_task_execution(task)
                    
                    assert contract is not None
                    assert contract.task_id == task.id
                    assert contract.contract_id in integration.active_contracts
                    
    @pytest.mark.asyncio
    async def test_stats_collection(self, integration):
        """Test statistics collection"""
        # Send some messages
        for i in range(5):
            await integration.send_message(
                recipient=f"agent_{i}",
                message_type=MessageType.HEALTH_CHECK,
                payload={},
                secure=False
            )
            
        stats = integration.stats.get_stats()
        
        assert stats["messages_sent"] == 5
        assert stats["messages_received"] == 0
        assert stats["bytes_sent"] > 0


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_agent_discovery_and_task_execution(self):
        """Test complete flow from discovery to task execution"""
        # Create two agents
        agent1 = Agent2AgentIntegration(
            "coordinator_agent",
            Mock(),
            Mock()
        )
        
        agent2 = Agent2AgentIntegration(
            "worker_agent",
            Mock(),
            Mock()
        )
        
        await agent1.initialize()
        await agent2.initialize()
        
        try:
            # Register worker agent
            worker_info = AgentInfo(
                id="worker_agent",
                name="Worker Agent",
                type="development",
                description="Agent that performs development tasks",
                capabilities=["code_generation", "testing"],
                resources={"memory": "2Gi", "cpu": 1.0},
                performance_metrics={"success_rate": 0.95},
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0"]
            )
            
            if agent2.discovery_service:
                await agent2.discovery_service.register_agent(worker_info)
                
            # Coordinator discovers worker
            if agent1.discovery_service:
                # Mock cross-agent discovery
                with patch.object(
                    agent1.discovery_service,
                    'discover_agents',
                    return_value=[worker_info]
                ):
                    agents = await agent1.discover_agents(
                        capabilities=["code_generation"]
                    )
                    
                    assert len(agents) == 1
                    assert agents[0].id == "worker_agent"
                    
        finally:
            await agent1.shutdown()
            await agent2.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])