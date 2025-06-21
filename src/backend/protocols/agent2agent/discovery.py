"""
Agent Discovery Service Implementation

Provides discovery, registration, and capability indexing for agents
in the ADK ecosystem.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

from cachetools import TTLCache
import redis.asyncio as redis

from .core import Agent2AgentMessage, MessageType, Agent2AgentProtocol

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    id: str
    name: str
    type: str
    description: str
    capabilities: List[str]
    resources: Dict[str, Any]
    performance_metrics: Dict[str, float]
    adk_version: str
    protocols_supported: List[str]
    status: str = "active"
    last_seen: float = 0
    registered_at: float = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        return cls(**data)


@dataclass
class DiscoveryQuery:
    """Query for discovering agents"""
    capabilities: Optional[List[str]] = None
    agent_type: Optional[str] = None
    status: Optional[str] = None
    min_performance: Optional[Dict[str, float]] = None
    max_results: int = 10
    include_inactive: bool = False
    
    def matches(self, agent: AgentInfo) -> bool:
        """Check if agent matches query criteria"""
        # Check status
        if not self.include_inactive and agent.status != "active":
            return False
            
        if self.status and agent.status != self.status:
            return False
            
        # Check type
        if self.agent_type and agent.type != self.agent_type:
            return False
            
        # Check capabilities
        if self.capabilities:
            agent_caps = set(agent.capabilities)
            required_caps = set(self.capabilities)
            if not required_caps.issubset(agent_caps):
                return False
                
        # Check performance
        if self.min_performance:
            for metric, min_value in self.min_performance.items():
                if agent.performance_metrics.get(metric, 0) < min_value:
                    return False
                    
        return True


class CapabilityIndex:
    """Index for fast capability-based agent lookup"""
    
    def __init__(self):
        self.capability_to_agents: Dict[str, Set[str]] = defaultdict(set)
        self.agent_to_capabilities: Dict[str, Set[str]] = defaultdict(set)
        
    def add_agent(self, agent_id: str, capabilities: List[str]):
        """Add agent to capability index"""
        for capability in capabilities:
            self.capability_to_agents[capability].add(agent_id)
            self.agent_to_capabilities[agent_id].add(capability)
            
    def remove_agent(self, agent_id: str):
        """Remove agent from capability index"""
        capabilities = self.agent_to_capabilities.get(agent_id, set())
        for capability in capabilities:
            self.capability_to_agents[capability].discard(agent_id)
        del self.agent_to_capabilities[agent_id]
        
    def find_agents_with_capability(self, capability: str) -> Set[str]:
        """Find all agents with a specific capability"""
        return self.capability_to_agents.get(capability, set())
        
    def find_agents_with_all_capabilities(self, capabilities: List[str]) -> Set[str]:
        """Find agents with all specified capabilities"""
        if not capabilities:
            return set()
            
        # Start with agents having the first capability
        result = self.capability_to_agents.get(capabilities[0], set()).copy()
        
        # Intersect with agents having remaining capabilities
        for capability in capabilities[1:]:
            result &= self.capability_to_agents.get(capability, set())
            
        return result


class AgentRegistry:
    """Registry for storing agent information"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.agents: Dict[str, AgentInfo] = {}
        self.redis_client = redis_client
        self._lock = asyncio.Lock()
        
    async def register(self, agent: AgentInfo):
        """Register an agent"""
        async with self._lock:
            agent.registered_at = time.time()
            agent.last_seen = time.time()
            self.agents[agent.id] = agent
            
            # Store in Redis if available
            if self.redis_client:
                await self.redis_client.hset(
                    "nexus_forge:agents",
                    agent.id,
                    json.dumps(agent.to_dict())
                )
                
    async def update(self, agent_id: str, updates: Dict[str, Any]):
        """Update agent information"""
        async with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                for key, value in updates.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                agent.last_seen = time.time()
                
                # Update in Redis
                if self.redis_client:
                    await self.redis_client.hset(
                        "nexus_forge:agents",
                        agent_id,
                        json.dumps(agent.to_dict())
                    )
                    
    async def unregister(self, agent_id: str):
        """Unregister an agent"""
        async with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.hdel("nexus_forge:agents", agent_id)
                    
    async def get(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information"""
        # Check local cache first
        if agent_id in self.agents:
            return self.agents[agent_id]
            
        # Check Redis
        if self.redis_client:
            data = await self.redis_client.hget("nexus_forge:agents", agent_id)
            if data:
                agent = AgentInfo.from_dict(json.loads(data))
                self.agents[agent_id] = agent
                return agent
                
        return None
        
    async def get_all(self) -> List[AgentInfo]:
        """Get all registered agents"""
        # Load from Redis if available
        if self.redis_client and not self.agents:
            agents_data = await self.redis_client.hgetall("nexus_forge:agents")
            for agent_id, data in agents_data.items():
                if isinstance(agent_id, bytes):
                    agent_id = agent_id.decode()
                if isinstance(data, bytes):
                    data = data.decode()
                self.agents[agent_id] = AgentInfo.from_dict(json.loads(data))
                
        return list(self.agents.values())


class AgentDiscoveryService:
    """Service for agent discovery and registration"""
    
    def __init__(self, protocol: Agent2AgentProtocol, redis_url: Optional[str] = None):
        self.protocol = protocol
        self.registry = AgentRegistry()
        self.capability_index = CapabilityIndex()
        self.discovery_cache = TTLCache(maxsize=1000, ttl=300)
        self.redis_client = None
        self.redis_url = redis_url
        
        # Heartbeat tracking
        self.heartbeat_interval = 30  # seconds
        self.inactive_threshold = 90  # seconds
        self.heartbeat_task = None
        
        # Discovery callbacks
        self.discovery_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Initialize discovery service"""
        # Connect to Redis if URL provided
        if self.redis_url:
            self.redis_client = await redis.from_url(self.redis_url)
            self.registry.redis_client = self.redis_client
            
        # Setup protocol handlers
        self.protocol.message_handler.register_handler(
            MessageType.AGENT_ANNOUNCE,
            self._handle_announce
        )
        self.protocol.message_handler.register_handler(
            MessageType.AGENT_DISCOVER,
            self._handle_discover
        )
        self.protocol.message_handler.register_handler(
            MessageType.AGENT_QUERY,
            self._handle_query
        )
        self.protocol.message_handler.register_handler(
            MessageType.AGENT_GOODBYE,
            self._handle_goodbye
        )
        
        # Start heartbeat monitoring
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        logger.info("Agent discovery service initialized")
        
    async def shutdown(self):
        """Shutdown discovery service"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        if self.redis_client:
            await self.redis_client.close()
            
    async def register_agent(self, agent_info: AgentInfo) -> None:
        """Register agent in discovery service"""
        # Add to registry
        await self.registry.register(agent_info)
        
        # Update capability index
        self.capability_index.add_agent(agent_info.id, agent_info.capabilities)
        
        # Create announcement message
        announcement = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.AGENT_ANNOUNCE,
            sender=agent_info.id,
            recipient=None,  # Broadcast
            payload=agent_info.to_dict(),
            timestamp=time.time(),
            priority=1  # High priority for announcements
        )
        
        # Broadcast announcement
        await self.broadcast_announcement(announcement)
        
        # Notify callbacks
        for callback in self.discovery_callbacks:
            try:
                await callback("registered", agent_info)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
        logger.info(f"Agent {agent_info.id} registered")
        
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent from discovery service"""
        agent = await self.registry.get(agent_id)
        if not agent:
            return
            
        # Remove from registry
        await self.registry.unregister(agent_id)
        
        # Remove from capability index
        self.capability_index.remove_agent(agent_id)
        
        # Send goodbye message
        goodbye = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.AGENT_GOODBYE,
            sender=agent_id,
            recipient=None,  # Broadcast
            payload={"agent_id": agent_id, "reason": "unregistering"},
            timestamp=time.time()
        )
        
        await self.broadcast_announcement(goodbye)
        
        # Notify callbacks
        for callback in self.discovery_callbacks:
            try:
                await callback("unregistered", agent)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")
                
        logger.info(f"Agent {agent_id} unregistered")
        
    async def discover_agents(self, query: DiscoveryQuery) -> List[AgentInfo]:
        """Discover agents matching query criteria"""
        # Check cache
        cache_key = json.dumps(asdict(query), sort_keys=True)
        if cache_key in self.discovery_cache:
            return self.discovery_cache[cache_key]
            
        # Search registry
        all_agents = await self.registry.get_all()
        matching_agents = []
        
        for agent in all_agents:
            if query.matches(agent):
                matching_agents.append(agent)
                
            if len(matching_agents) >= query.max_results:
                break
                
        # Cache results
        self.discovery_cache[cache_key] = matching_agents
        
        return matching_agents
        
    async def find_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Find all agents with a specific capability"""
        agent_ids = self.capability_index.find_agents_with_capability(capability)
        agents = []
        
        for agent_id in agent_ids:
            agent = await self.registry.get(agent_id)
            if agent and agent.status == "active":
                agents.append(agent)
                
        return agents
        
    async def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        await self.registry.update(agent_id, {"status": status, "last_seen": time.time()})
        
    async def heartbeat(self, agent_id: str):
        """Record agent heartbeat"""
        await self.registry.update(agent_id, {"last_seen": time.time()})
        
    async def broadcast_announcement(self, message: Agent2AgentMessage):
        """Broadcast announcement to all agents"""
        # In a real implementation, this would use a pub/sub system
        # For now, we'll use Redis pub/sub if available
        if self.redis_client:
            await self.redis_client.publish(
                "nexus_forge:agent_announcements",
                message.to_bytes()
            )
            
    async def subscribe_to_discovery(self, callback: Callable):
        """Subscribe to discovery events"""
        self.discovery_callbacks.append(callback)
        
    async def _handle_announce(self, message: Agent2AgentMessage) -> None:
        """Handle agent announcement"""
        agent_info = AgentInfo.from_dict(message.payload)
        await self.register_agent(agent_info)
        
    async def _handle_discover(self, message: Agent2AgentMessage) -> Agent2AgentMessage:
        """Handle discovery request"""
        query = DiscoveryQuery(**message.payload)
        agents = await self.discover_agents(query)
        
        return Agent2AgentMessage(
            id=str(uuid.uuid4()),
            correlation_id=message.id,
            type=MessageType.AGENT_QUERY,
            sender=self.protocol.agent_id,
            recipient=message.sender,
            payload={
                "agents": [agent.to_dict() for agent in agents],
                "total": len(agents)
            },
            timestamp=time.time()
        )
        
    async def _handle_query(self, message: Agent2AgentMessage) -> Agent2AgentMessage:
        """Handle agent query"""
        # Similar to discover but with different query format
        return await self._handle_discover(message)
        
    async def _handle_goodbye(self, message: Agent2AgentMessage) -> None:
        """Handle agent goodbye"""
        agent_id = message.payload.get("agent_id", message.sender)
        await self.unregister_agent(agent_id)
        
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and mark inactive agents"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check all agents
                all_agents = await self.registry.get_all()
                current_time = time.time()
                
                for agent in all_agents:
                    if agent.status == "active":
                        time_since_seen = current_time - agent.last_seen
                        
                        if time_since_seen > self.inactive_threshold:
                            # Mark as inactive
                            await self.update_agent_status(agent.id, "inactive")
                            logger.warning(f"Agent {agent.id} marked inactive (no heartbeat)")
                            
                            # Notify callbacks
                            for callback in self.discovery_callbacks:
                                try:
                                    await callback("inactive", agent)
                                except Exception as e:
                                    logger.error(f"Error in discovery callback: {e}")
                                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                
import uuid  # Add this import at the top
