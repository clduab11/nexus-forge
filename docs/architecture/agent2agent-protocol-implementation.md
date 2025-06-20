# ADK Agent2Agent Protocol Implementation Strategy

## Executive Summary

This document provides a comprehensive implementation strategy for completing the ADK Agent2Agent protocol integration in Nexus Forge. The implementation builds upon existing WebSocket infrastructure and integrates with Google ADK's native capabilities to enable secure, bidirectional agent communication.

## Current State Analysis

### Existing Infrastructure
1. **WebSocket Manager** (`/src/backend/websockets/manager.py`)
   - Real-time communication infrastructure
   - Session management and tracking
   - Rate limiting and security
   - Supabase integration for persistence

2. **ADK Integration** (`/src/backend/integrations/google/adk.py`)
   - Basic ADK framework integration
   - Agent initialization patterns
   - Streaming capabilities

3. **Agent Orchestration** (`/src/backend/agents/agents/nexus_forge_agents.py`)
   - Starri orchestrator for coordination
   - Multi-model integration (Gemini, Jules, Veo, Imagen)
   - Communication channels (asyncio.Queue)

### Gaps to Address
1. No Agent2Agent protocol implementation
2. Missing agent discovery mechanism
3. No secure agent-to-agent channels
4. Limited inter-agent messaging capabilities
5. No capability negotiation system

## Implementation Architecture

### 1. Core Protocol Components

```python
# /src/backend/protocols/agent2agent/core.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import json
import time
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

class MessageType(Enum):
    """Agent2Agent message types"""
    # Discovery Messages
    AGENT_ANNOUNCE = "agent.announce"
    AGENT_DISCOVER = "agent.discover"
    AGENT_QUERY = "agent.query"
    
    # Capability Messages  
    CAPABILITY_REGISTER = "capability.register"
    CAPABILITY_REQUEST = "capability.request"
    CAPABILITY_MATCH = "capability.match"
    
    # Task Coordination
    TASK_PROPOSE = "task.propose"
    TASK_ACCEPT = "task.accept"
    TASK_REJECT = "task.reject"
    TASK_DELEGATE = "task.delegate"
    TASK_COMPLETE = "task.complete"
    
    # Resource Sharing
    RESOURCE_OFFER = "resource.offer"
    RESOURCE_REQUEST = "resource.request"
    RESOURCE_TRANSFER = "resource.transfer"
    
    # Health & Monitoring
    HEALTH_CHECK = "health.check"
    HEALTH_REPORT = "health.report"
    METRICS_REPORT = "metrics.report"

@dataclass
class Agent2AgentMessage:
    """Core message structure for agent communication"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str]  # None for broadcast
    payload: Dict[str, Any]
    timestamp: float
    signature: str  # Cryptographic signature
    priority: int = 0
    ttl: Optional[int] = None  # Time to live in seconds
    encryption_key_id: Optional[str] = None  # For encrypted payloads
```

### 2. Agent Discovery Service

```python
# /src/backend/protocols/agent2agent/discovery.py

class AgentDiscoveryService:
    """Service for agent discovery and registration"""
    
    def __init__(self):
        self.registry = {}  # agent_id -> agent_info
        self.capability_index = {}  # capability -> [agent_ids]
        self.discovery_cache = TTLCache(maxsize=1000, ttl=300)
        self.broadcast_channel = None
        
    async def register_agent(self, agent: Agent) -> None:
        """Register agent in discovery service"""
        announcement = self._create_announcement(agent)
        
        # Store in registry
        self.registry[agent.id] = {
            "info": agent.to_dict(),
            "capabilities": agent.get_capabilities(),
            "last_seen": time.time(),
            "status": "active"
        }
        
        # Index capabilities
        for capability in agent.get_capabilities():
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(agent.id)
        
        # Broadcast announcement
        await self.broadcast_announcement(announcement)
    
    async def discover_agents(self, query: Dict[str, Any]) -> List[Agent]:
        """Discover agents matching query criteria"""
        # Check cache first
        cache_key = json.dumps(query, sort_keys=True)
        if cache_key in self.discovery_cache:
            return self.discovery_cache[cache_key]
        
        # Search registry
        matching_agents = []
        for agent_id, agent_info in self.registry.items():
            if self._matches_query(agent_info, query):
                matching_agents.append(agent_info)
        
        # Cache results
        self.discovery_cache[cache_key] = matching_agents
        return matching_agents
```

### 3. Secure Communication Layer

```python
# /src/backend/protocols/agent2agent/security.py

class SecureAgentChannel:
    """Secure communication channel between agents"""
    
    def __init__(self, agent_a: Agent, agent_b: Agent):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.channel_id = str(uuid.uuid4())
        self.session_key = None
        self.established = False
        
    async def establish_channel(self) -> bool:
        """Establish secure channel with mTLS and session keys"""
        # Exchange certificates
        cert_a = await self.agent_a.get_certificate()
        cert_b = await self.agent_b.get_certificate()
        
        # Verify certificates with ADK CA
        if not await self._verify_certificates(cert_a, cert_b):
            return False
        
        # Generate session key
        self.session_key = Fernet.generate_key()
        
        # Exchange session key using public key encryption
        await self._exchange_session_key()
        
        self.established = True
        return True
    
    async def send_message(self, message: Agent2AgentMessage) -> None:
        """Send encrypted message through channel"""
        if not self.established:
            raise SecurityError("Channel not established")
        
        # Encrypt payload
        encrypted_payload = self._encrypt_payload(message.payload)
        message.payload = encrypted_payload
        message.encryption_key_id = self.channel_id
        
        # Sign message
        message.signature = await self._sign_message(message)
        
        # Send through transport
        await self._send_through_transport(message)
```

### 4. Capability Negotiation Engine

```python
# /src/backend/protocols/agent2agent/negotiation.py

class CapabilityNegotiationEngine:
    """Engine for capability matching and task negotiation"""
    
    def __init__(self):
        self.negotiation_sessions = {}
        self.capability_matcher = SemanticCapabilityMatcher()
        
    async def negotiate_task(
        self,
        requester: Agent,
        task: Task,
        available_agents: List[Agent]
    ) -> Optional[TaskContract]:
        """Negotiate task execution with available agents"""
        
        # Find capable agents
        capable_agents = await self._find_capable_agents(task, available_agents)
        
        if not capable_agents:
            return None
        
        # Score and rank agents
        ranked_agents = await self._rank_agents(task, capable_agents)
        
        # Negotiate with top agents
        for agent in ranked_agents[:3]:  # Try top 3
            contract = await self._negotiate_contract(requester, agent, task)
            if contract:
                return contract
        
        return None
    
    async def _negotiate_contract(
        self,
        requester: Agent,
        provider: Agent,
        task: Task
    ) -> Optional[TaskContract]:
        """Negotiate contract between agents"""
        
        # Create negotiation session
        session_id = str(uuid.uuid4())
        self.negotiation_sessions[session_id] = {
            "requester": requester.id,
            "provider": provider.id,
            "task": task,
            "status": "negotiating"
        }
        
        # Send task proposal
        proposal = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_PROPOSE,
            sender=requester.id,
            recipient=provider.id,
            payload={
                "task": task.to_dict(),
                "requirements": task.requirements,
                "constraints": task.constraints,
                "proposed_sla": {
                    "response_time": task.max_duration,
                    "success_criteria": task.success_criteria
                }
            },
            timestamp=time.time(),
            signature=""  # Will be signed
        )
        
        # Wait for response
        response = await self._send_and_wait_response(provider, proposal)
        
        if response.type == MessageType.TASK_ACCEPT:
            # Create contract
            contract = TaskContract(
                contract_id=session_id,
                task_id=task.id,
                requester_id=requester.id,
                provider_id=provider.id,
                terms=response.payload["terms"],
                sla=response.payload["sla"],
                created_at=time.time()
            )
            
            self.negotiation_sessions[session_id]["status"] = "contracted"
            return contract
        
        return None
```

### 5. Integration Points

```python
# /src/backend/protocols/agent2agent/integration.py

class Agent2AgentIntegration:
    """Integration with existing Nexus Forge architecture"""
    
    def __init__(self, ws_manager: WebSocketManager, adk_service: ADKService):
        self.ws_manager = ws_manager
        self.adk_service = adk_service
        self.discovery_service = AgentDiscoveryService()
        self.negotiation_engine = CapabilityNegotiationEngine()
        self.secure_channels = {}
        
    async def initialize(self):
        """Initialize Agent2Agent protocol"""
        # Register existing agents
        await self._register_existing_agents()
        
        # Setup WebSocket handlers
        self._setup_websocket_handlers()
        
        # Start discovery heartbeat
        asyncio.create_task(self._discovery_heartbeat())
        
    async def send_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> None:
        """Send message between agents"""
        
        # Get or create secure channel
        channel = await self._get_secure_channel(from_agent, to_agent)
        
        # Create message
        message = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            sender=from_agent,
            recipient=to_agent,
            payload=payload,
            timestamp=time.time(),
            signature=""  # Will be signed in channel
        )
        
        # Send through secure channel
        await channel.send_message(message)
        
        # Broadcast to WebSocket clients for monitoring
        await self.ws_manager.broadcast_coordination_event(
            event_type="agent_message",
            source_agent_id=from_agent,
            target_agent_id=to_agent,
            event_data={
                "message_type": message_type.value,
                "timestamp": message.timestamp
            }
        )
```

## Implementation Phases

### Phase 1: Core Protocol (Week 1)
1. Implement message types and structures
2. Create basic serialization/deserialization
3. Setup protocol versioning
4. Unit tests for core components

### Phase 2: Discovery Service (Week 2)
1. Implement agent registry
2. Create capability indexing
3. Build discovery API
4. Add caching layer
5. Integration tests

### Phase 3: Security Layer (Week 3)
1. Implement certificate management
2. Create secure channels
3. Add encryption/decryption
4. Setup signature verification
5. Security audit

### Phase 4: Negotiation Engine (Week 4)
1. Build capability matcher
2. Implement negotiation protocol
3. Create contract system
4. Add SLA monitoring
5. Performance testing

### Phase 5: Integration (Week 5)
1. WebSocket integration
2. ADK service integration
3. Monitoring and metrics
4. Documentation
5. End-to-end testing

## Testing Strategy

### Unit Tests
```python
# /tests/test_agent2agent_protocol.py

class TestAgent2AgentProtocol:
    async def test_message_serialization(self):
        """Test message serialization/deserialization"""
        
    async def test_discovery_registration(self):
        """Test agent registration in discovery"""
        
    async def test_secure_channel_establishment(self):
        """Test secure channel creation"""
        
    async def test_capability_negotiation(self):
        """Test capability matching and negotiation"""
```

### Integration Tests
```python
# /tests/integration/test_agent2agent_integration.py

class TestAgent2AgentIntegration:
    async def test_end_to_end_communication(self):
        """Test complete agent communication flow"""
        
    async def test_multi_agent_coordination(self):
        """Test coordination between multiple agents"""
        
    async def test_failure_recovery(self):
        """Test protocol resilience"""
```

## Performance Targets

- Message latency: <10ms (p99)
- Discovery query: <50ms
- Channel establishment: <100ms
- Negotiation completion: <200ms
- Throughput: 10,000 messages/second

## Security Considerations

1. **Authentication**: mTLS with ADK-issued certificates
2. **Encryption**: AES-256-GCM for message payloads
3. **Integrity**: HMAC-SHA256 signatures
4. **Authorization**: Capability-based access control
5. **Audit**: Complete message logging with privacy controls

## Monitoring and Observability

```python
# Metrics to track
metrics = {
    "agent2agent_messages_total": Counter,
    "agent2agent_message_latency": Histogram,
    "discovery_queries_total": Counter,
    "secure_channels_active": Gauge,
    "negotiation_success_rate": Gauge,
    "protocol_errors_total": Counter
}
```

## Migration Strategy

1. Deploy protocol alongside existing system
2. Gradually migrate agents to use Agent2Agent
3. Maintain backward compatibility
4. Monitor performance and errors
5. Full cutover after stability proven

## Future Enhancements

1. **Protocol v3**: Support for multi-party negotiations
2. **Federation**: Cross-organization agent communication
3. **ML Optimization**: Learn optimal agent pairings
4. **Blockchain**: Immutable contract storage
5. **5G Integration**: Ultra-low latency communication

## Conclusion

This implementation strategy provides a comprehensive roadmap for completing the ADK Agent2Agent protocol integration. The phased approach ensures systematic development while maintaining system stability. The architecture leverages existing infrastructure while adding powerful new capabilities for autonomous agent coordination.