"""
Integration module for Agent2Agent Protocol with Nexus Forge

Connects the Agent2Agent protocol with existing WebSocket infrastructure,
ADK services, and monitoring systems.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass

from ...websockets.manager import WebSocketManager
from ...services.adk_service import ADKService
from ...integrations.supabase.coordination_client import SupabaseCoordinationClient
from ...monitoring.cloud_monitoring import CloudMonitoringService

from .core import (
    Agent2AgentProtocol,
    Agent2AgentMessage,
    MessageType,
    ProtocolStats
)
from .discovery import (
    AgentDiscoveryService,
    AgentInfo,
    DiscoveryQuery
)
from .security import (
    SecureChannelManager,
    AgentCertificateManager,
    MessageEncryption
)
from .negotiation import (
    CapabilityNegotiationEngine,
    Task,
    TaskContract
)

logger = logging.getLogger(__name__)


@dataclass
class Agent2AgentConfig:
    """Configuration for Agent2Agent integration"""
    enable_discovery: bool = True
    enable_security: bool = True
    enable_negotiation: bool = True
    enable_monitoring: bool = True
    redis_url: Optional[str] = None
    ca_cert_path: Optional[str] = None
    heartbeat_interval: int = 30
    channel_timeout: int = 3600
    max_message_size: int = 10 * 1024 * 1024  # 10MB


class WebSocketBridge:
    """Bridge between Agent2Agent protocol and WebSocket infrastructure"""
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.agent_to_session: Dict[str, str] = {}
        self.session_to_agent: Dict[str, str] = {}
        
    async def register_agent_session(self, agent_id: str, session_id: str):
        """Register WebSocket session for an agent"""
        self.agent_to_session[agent_id] = session_id
        self.session_to_agent[session_id] = agent_id
        
    async def unregister_agent_session(self, agent_id: str):
        """Unregister WebSocket session for an agent"""
        session_id = self.agent_to_session.get(agent_id)
        if session_id:
            del self.agent_to_session[agent_id]
            del self.session_to_agent[session_id]
            
    async def send_to_agent(self, agent_id: str, message: Agent2AgentMessage):
        """Send message to agent via WebSocket"""
        session_id = self.agent_to_session.get(agent_id)
        if not session_id:
            logger.warning(f"No WebSocket session found for agent {agent_id}")
            return
            
        # Convert to WebSocket message format
        ws_message = {
            "type": "agent2agent_message",
            "message": message.to_dict(),
            "timestamp": time.time()
        }
        
        await self.ws_manager.send_to_session(session_id, ws_message)
        
    async def broadcast_to_agents(self, message: Agent2AgentMessage):
        """Broadcast message to all connected agents"""
        ws_message = {
            "type": "agent2agent_broadcast",
            "message": message.to_dict(),
            "timestamp": time.time()
        }
        
        # Send to all agent sessions
        for agent_id, session_id in self.agent_to_session.items():
            if agent_id != message.sender:  # Don't send back to sender
                await self.ws_manager.send_to_session(session_id, ws_message)


class ADKProtocolAdapter:
    """Adapter for ADK services to use Agent2Agent protocol"""
    
    def __init__(self, adk_service: ADKService, protocol: Agent2AgentProtocol):
        self.adk_service = adk_service
        self.protocol = protocol
        
    async def register_adk_agents(self) -> List[AgentInfo]:
        """Register ADK agents with discovery service"""
        agents = []
        
        # Register orchestrator
        orchestrator_info = AgentInfo(
            id="starri_orchestrator",
            name="Starri Orchestrator",
            type="orchestrator",
            description="Master AI coordinator for Nexus Forge",
            capabilities=[
                "task_decomposition",
                "agent_coordination",
                "workflow_management",
                "result_synthesis"
            ],
            resources={"memory": "4Gi", "cpu": 2.0},
            performance_metrics={
                "avg_response_time": 100,
                "success_rate": 0.98,
                "availability": 0.999
            },
            adk_version="2.0",
            protocols_supported=["agent2agent/2.0", "adk/1.0"]
        )
        agents.append(orchestrator_info)
        
        # Register specialized agents
        specialized_agents = [
            {
                "id": "jules_coder",
                "name": "Jules Autonomous Coder",
                "type": "development",
                "capabilities": ["code_generation", "testing", "optimization", "debugging"]
            },
            {
                "id": "gemini_architect",
                "name": "Gemini Architecture Agent",
                "type": "architecture",
                "capabilities": ["system_design", "api_design", "database_design", "security_analysis"]
            },
            {
                "id": "veo_media",
                "name": "Veo Media Generator",
                "type": "media",
                "capabilities": ["video_generation", "animation", "demo_creation"]
            },
            {
                "id": "imagen_designer",
                "name": "Imagen UI Designer",
                "type": "design",
                "capabilities": ["ui_design", "mockup_generation", "design_system", "prototyping"]
            }
        ]
        
        for agent_data in specialized_agents:
            agent_info = AgentInfo(
                id=agent_data["id"],
                name=agent_data["name"],
                type=agent_data["type"],
                description=f"Specialized {agent_data['type']} agent",
                capabilities=agent_data["capabilities"],
                resources={"memory": "2Gi", "cpu": 1.0},
                performance_metrics={
                    "avg_response_time": 200,
                    "success_rate": 0.95,
                    "availability": 0.99
                },
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0", "adk/1.0"]
            )
            agents.append(agent_info)
            
        return agents


class Agent2AgentIntegration:
    """Main integration class for Agent2Agent protocol"""
    
    def __init__(
        self,
        agent_id: str,
        ws_manager: WebSocketManager,
        adk_service: ADKService,
        config: Optional[Agent2AgentConfig] = None
    ):
        self.agent_id = agent_id
        self.ws_manager = ws_manager
        self.adk_service = adk_service
        self.config = config or Agent2AgentConfig()
        
        # Core protocol
        self.protocol = Agent2AgentProtocol(agent_id)
        
        # Protocol components
        self.discovery_service = None
        self.security_manager = None
        self.negotiation_engine = None
        
        # Integration components
        self.ws_bridge = WebSocketBridge(ws_manager)
        self.adk_adapter = ADKProtocolAdapter(adk_service, self.protocol)
        
        # Monitoring
        self.monitoring = CloudMonitoringService() if self.config.enable_monitoring else None
        self.stats = ProtocolStats()
        
        # Supabase integration
        self.supabase_client = None
        
        # Active tasks and contracts
        self.active_tasks: Dict[str, Task] = {}
        self.active_contracts: Dict[str, TaskContract] = {}
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self, supabase_client: Optional[SupabaseCoordinationClient] = None):
        """Initialize Agent2Agent integration"""
        logger.info(f"Initializing Agent2Agent protocol for agent {self.agent_id}")
        
        # Setup Supabase
        self.supabase_client = supabase_client
        
        # Initialize protocol
        await self.protocol.start()
        
        # Initialize discovery service
        if self.config.enable_discovery:
            self.discovery_service = AgentDiscoveryService(
                self.protocol,
                self.config.redis_url
            )
            await self.discovery_service.initialize()
            
            # Register ADK agents
            adk_agents = await self.adk_adapter.register_adk_agents()
            for agent in adk_agents:
                await self.discovery_service.register_agent(agent)
                
        # Initialize security
        if self.config.enable_security:
            cert_manager = AgentCertificateManager(self.config.ca_cert_path)
            encryption_service = MessageEncryption()
            self.security_manager = SecureChannelManager(cert_manager, encryption_service)
            
        # Initialize negotiation
        if self.config.enable_negotiation:
            self.negotiation_engine = CapabilityNegotiationEngine(self.protocol)
            
        # Setup message routing
        self._setup_message_routing()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Agent2Agent protocol initialized successfully")
        
    async def shutdown(self):
        """Shutdown Agent2Agent integration"""
        logger.info("Shutting down Agent2Agent protocol")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown components
        if self.discovery_service:
            await self.discovery_service.shutdown()
            
        await self.protocol.stop()
        
        logger.info("Agent2Agent protocol shutdown complete")
        
    def _setup_message_routing(self):
        """Setup message routing between protocol and WebSocket"""
        # Route incoming WebSocket messages to protocol
        async def route_ws_to_protocol(session_id: str, message: Dict[str, Any]):
            if message.get("type") == "agent2agent_message":
                agent_message = Agent2AgentMessage.from_dict(message["message"])
                await self.handle_incoming_message(agent_message)
                
        # This would be registered with WebSocket manager in production
        
    def _start_background_tasks(self):
        """Start background tasks"""
        # Heartbeat task
        if self.discovery_service:
            task = asyncio.create_task(self._heartbeat_loop())
            self.background_tasks.add(task)
            
        # Stats reporting task
        if self.monitoring:
            task = asyncio.create_task(self._stats_reporting_loop())
            self.background_tasks.add(task)
            
        # Channel cleanup task
        if self.security_manager:
            task = asyncio.create_task(self._channel_cleanup_loop())
            self.background_tasks.add(task)
            
    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        secure: bool = True,
        priority: int = 0
    ) -> Optional[Agent2AgentMessage]:
        """Send message to another agent"""
        # Create message
        message = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        # Use secure channel if enabled and requested
        if secure and self.security_manager:
            try:
                channel = await self.security_manager.get_or_create_channel(
                    self.agent_id,
                    recipient
                )
                await channel.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send secure message: {e}")
                return None
        else:
            # Send via protocol
            await self.protocol.send_message(message)
            
        # Route to WebSocket if recipient is connected
        await self.ws_bridge.send_to_agent(recipient, message)
        
        # Update stats
        self.stats.record_sent(message, len(message.to_bytes()))
        
        # Monitor if enabled
        if self.monitoring:
            self.monitoring.increment_counter(
                "agent2agent_messages_sent",
                labels={
                    "agent": self.agent_id,
                    "message_type": message_type.value,
                    "recipient": recipient
                }
            )
            
        return message
        
    async def handle_incoming_message(self, message: Agent2AgentMessage) -> None:
        """Handle incoming Agent2Agent message"""
        # Update stats
        self.stats.record_received(message, len(message.to_bytes()))
        
        # Process through security layer if needed
        if message.encryption_key_id and self.security_manager:
            try:
                # Find appropriate channel
                channel = None  # Would look up by key_id
                if channel:
                    message = await channel.receive_message(message)
            except Exception as e:
                logger.error(f"Failed to decrypt message: {e}")
                return
                
        # Process through protocol
        response = await self.protocol.process_incoming(message)
        
        # Send response if generated
        if response:
            await self.send_message(
                response.recipient,
                response.type,
                response.payload,
                priority=response.priority
            )
            
        # Monitor if enabled
        if self.monitoring:
            self.monitoring.increment_counter(
                "agent2agent_messages_received",
                labels={
                    "agent": self.agent_id,
                    "message_type": message.type.value,
                    "sender": message.sender
                }
            )
            
    async def discover_agents(
        self,
        capabilities: Optional[List[str]] = None,
        agent_type: Optional[str] = None
    ) -> List[AgentInfo]:
        """Discover available agents"""
        if not self.discovery_service:
            return []
            
        query = DiscoveryQuery(
            capabilities=capabilities,
            agent_type=agent_type,
            status="active"
        )
        
        return await self.discovery_service.discover_agents(query)
        
    async def create_task(
        self,
        name: str,
        description: str,
        required_capabilities: List[str],
        **kwargs
    ) -> Task:
        """Create a new task for negotiation"""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            required_capabilities=required_capabilities,
            optional_capabilities=kwargs.get("optional_capabilities", []),
            requirements=kwargs.get("requirements", {}),
            constraints=kwargs.get("constraints", {}),
            priority=kwargs.get("priority", 0),
            max_duration=kwargs.get("max_duration"),
            success_criteria=kwargs.get("success_criteria", {}),
            reward=kwargs.get("reward")
        )
        
        self.active_tasks[task.id] = task
        return task
        
    async def negotiate_task_execution(self, task: Task) -> Optional[TaskContract]:
        """Negotiate task execution with available agents"""
        if not self.negotiation_engine:
            logger.warning("Negotiation engine not enabled")
            return None
            
        # Discover capable agents
        agents = await self.discover_agents(capabilities=task.required_capabilities)
        
        if not agents:
            logger.warning(f"No agents found with capabilities: {task.required_capabilities}")
            return None
            
        # Negotiate with agents
        contract = await self.negotiation_engine.negotiate_task(task, agents)
        
        if contract:
            self.active_contracts[contract.contract_id] = contract
            
            # Store in Supabase if available
            if self.supabase_client:
                await self._store_contract_in_supabase(contract)
                
            # Emit WebSocket event
            await self.ws_manager.broadcast_coordination_event(
                event_type="contract_created",
                source_agent_id=contract.requester_id,
                target_agent_id=contract.provider_id,
                task_id=task.id,
                event_data=contract.to_dict()
            )
            
        return contract
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send heartbeat
                await self.discovery_service.heartbeat(self.agent_id)
                
                # Broadcast heartbeat message
                heartbeat = Agent2AgentMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.STATUS_UPDATE,
                    sender=self.agent_id,
                    recipient=None,  # Broadcast
                    payload={
                        "status": "active",
                        "timestamp": time.time(),
                        "stats": self.stats.get_stats()
                    },
                    timestamp=time.time()
                )
                
                await self.ws_bridge.broadcast_to_agents(heartbeat)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                
    async def _stats_reporting_loop(self):
        """Report statistics to monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                stats = self.stats.get_stats()
                
                # Report to monitoring
                self.monitoring.observe_histogram(
                    "agent2agent_message_rate",
                    stats["avg_messages_per_second"],
                    labels={"agent": self.agent_id}
                )
                
                self.monitoring.set_gauge(
                    "agent2agent_active_connections",
                    stats["active_connections"],
                    labels={"agent": self.agent_id}
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats reporting: {e}")
                
    async def _channel_cleanup_loop(self):
        """Clean up expired secure channels"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.security_manager.cleanup_expired_channels()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in channel cleanup: {e}")
                
    async def _store_contract_in_supabase(self, contract: TaskContract):
        """Store contract in Supabase for persistence"""
        try:
            await self.supabase_client.create_coordination_event(
                event_type="contract_created",
                source_agent_id=contract.requester_id,
                target_agent_id=contract.provider_id,
                task_id=contract.task_id,
                event_data=contract.to_dict()
            )
        except Exception as e:
            logger.error(f"Failed to store contract in Supabase: {e}")
