"""
Google Agent Development Kit (ADK) Integration for Nexus Forge
Enables integration with Google's open-source agent framework
Supports Agent2Agent protocol for cross-framework communication
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from nexus_forge.core.exceptions import AgentError, IntegrationError
from nexus_forge.core.monitoring import get_logger
from nexus_forge.integrations.supabase.coordination_client import (
    SupabaseCoordinationClient,
)

logger = get_logger(__name__)


@dataclass
class AgentCapability:
    """Agent capability definition for ADK integration"""

    name: str
    type: str  # 'action', 'query', 'transform', 'generate'
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any] = None


@dataclass
class Agent2AgentMessage:
    """Agent2Agent protocol message structure"""

    id: str
    source_agent_id: str
    target_agent_id: str
    message_type: str  # 'request', 'response', 'notification', 'error'
    action: str
    payload: Dict[str, Any]
    timestamp: datetime
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None


class ADKAgent(BaseModel):
    """ADK Agent registration model"""

    agent_id: str
    name: str
    type: str
    version: str
    capabilities: List[AgentCapability]
    endpoint_url: str
    authentication: Dict[str, Any]
    metadata: Dict[str, Any] = {}


class GoogleADKIntegration:
    """
    Google Agent Development Kit Integration
    Enables Nexus Forge agents to work with ADK framework and Agent2Agent protocol
    """

    def __init__(
        self,
        coordination_client: SupabaseCoordinationClient,
        adk_registry_url: str = "https://agent-registry.googleapis.com",
        a2a_protocol_version: str = "1.0",
        enable_bidirectional_av: bool = True,
    ):
        """Initialize ADK integration"""
        self.coordination_client = coordination_client
        self.adk_registry_url = adk_registry_url
        self.a2a_protocol_version = a2a_protocol_version
        self.enable_bidirectional_av = enable_bidirectional_av

        # Agent registry
        self.registered_agents: Dict[str, ADKAgent] = {}
        self.external_agents: Dict[str, ADKAgent] = {}

        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}

        # HTTP client for external communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # ADK capabilities
        self.supported_frameworks = [
            "nexus-forge",
            "langraph",
            "crewai",
            "autogen",
            "custom",
        ]

    async def initialize(self):
        """Initialize ADK integration and register with Agent2Agent protocol"""
        try:
            # Register Nexus Forge agents with ADK framework
            await self._register_nexus_forge_agents()

            # Discover external agents supporting Agent2Agent protocol
            await self._discover_external_agents()

            # Setup message routing
            await self._setup_message_routing()

            logger.info("Google ADK integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ADK integration: {e}")
            raise IntegrationError(f"ADK initialization failed: {e}")

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        endpoint_url: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Register agent with ADK framework"""
        try:
            adk_agent = ADKAgent(
                agent_id=agent_id,
                name=name,
                type=agent_type,
                version="1.0.0",
                capabilities=capabilities,
                endpoint_url=endpoint_url,
                authentication={
                    "type": "bearer_token",
                    "config": {"header": "Authorization", "prefix": "Bearer"},
                },
                metadata=metadata or {},
            )

            # Register with local registry
            self.registered_agents[agent_id] = adk_agent

            # Register with Google ADK registry
            registration_data = {
                "agent": asdict(adk_agent),
                "protocol_version": self.a2a_protocol_version,
                "frameworks_supported": self.supported_frameworks,
                "bidirectional_av_enabled": self.enable_bidirectional_av,
            }

            response = await self.http_client.post(
                f"{self.adk_registry_url}/v1/agents/register", json=registration_data
            )

            if response.status_code == 201:
                logger.info(f"Successfully registered agent {name} with ADK")
                return True
            else:
                logger.error(f"ADK registration failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to register agent {name}: {e}")
            return False

    async def discover_agents(
        self,
        agent_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        frameworks: Optional[List[str]] = None,
    ) -> List[ADKAgent]:
        """Discover agents in the ADK ecosystem"""
        try:
            query_params = {"protocol_version": self.a2a_protocol_version}

            if agent_type:
                query_params["type"] = agent_type
            if capabilities:
                query_params["capabilities"] = ",".join(capabilities)
            if frameworks:
                query_params["frameworks"] = ",".join(frameworks)

            response = await self.http_client.get(
                f"{self.adk_registry_url}/v1/agents/discover", params=query_params
            )

            if response.status_code == 200:
                agents_data = response.json()
                discovered_agents = []

                for agent_data in agents_data.get("agents", []):
                    try:
                        agent = ADKAgent(**agent_data)
                        discovered_agents.append(agent)
                        self.external_agents[agent.agent_id] = agent
                    except ValidationError as e:
                        logger.warning(f"Invalid agent data: {e}")

                logger.info(f"Discovered {len(discovered_agents)} ADK agents")
                return discovered_agents
            else:
                logger.error(f"Agent discovery failed: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return []

    async def send_agent_message(
        self,
        source_agent_id: str,
        target_agent_id: str,
        action: str,
        payload: Dict[str, Any],
        conversation_id: Optional[str] = None,
        message_type: str = "request",
    ) -> Optional[Agent2AgentMessage]:
        """Send message to agent using Agent2Agent protocol"""
        try:
            # Get target agent info
            target_agent = self.external_agents.get(target_agent_id)
            if not target_agent:
                logger.error(f"Target agent {target_agent_id} not found")
                return None

            # Create A2A message
            message = Agent2AgentMessage(
                id=str(uuid4()),
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                message_type=message_type,
                action=action,
                payload=payload,
                timestamp=datetime.now(timezone.utc),
                conversation_id=conversation_id or str(uuid4()),
            )

            # Send via Agent2Agent protocol
            message_data = {
                "protocol": "agent2agent",
                "version": self.a2a_protocol_version,
                "message": asdict(message),
            }

            response = await self.http_client.post(
                f"{target_agent.endpoint_url}/a2a/message",
                json=message_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"nexus-forge-adk/1.0.0",
                },
            )

            if response.status_code == 200:
                response_data = response.json()

                # Parse response message
                if "message" in response_data:
                    response_message = Agent2AgentMessage(**response_data["message"])

                    # Store conversation context
                    if message.conversation_id:
                        self.conversation_contexts[message.conversation_id] = {
                            "source_agent": source_agent_id,
                            "target_agent": target_agent_id,
                            "last_message": message.id,
                            "created_at": datetime.now(timezone.utc),
                            "message_count": self.conversation_contexts.get(
                                message.conversation_id, {}
                            ).get("message_count", 0)
                            + 1,
                        }

                    logger.debug(f"Sent A2A message {message.id} to {target_agent_id}")
                    return response_message
                else:
                    logger.warning(
                        f"Invalid A2A response format from {target_agent_id}"
                    )
                    return None
            else:
                logger.error(
                    f"A2A message failed: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return None

    async def handle_incoming_message(
        self, message_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming Agent2Agent message"""
        try:
            # Validate message format
            if message_data.get("protocol") != "agent2agent":
                raise ValueError("Invalid protocol")

            message = Agent2AgentMessage(**message_data["message"])

            # Check if target agent is registered locally
            target_agent = self.registered_agents.get(message.target_agent_id)
            if not target_agent:
                return {"error": "Target agent not found", "message_id": message.id}

            # Route message to appropriate handler
            handler = self.message_handlers.get(message.action)
            if handler:
                response_payload = await handler(message)
            else:
                response_payload = await self._default_message_handler(message)

            # Create response message
            response = Agent2AgentMessage(
                id=str(uuid4()),
                source_agent_id=message.target_agent_id,
                target_agent_id=message.source_agent_id,
                message_type="response",
                action=f"{message.action}_response",
                payload=response_payload,
                timestamp=datetime.now(timezone.utc),
                conversation_id=message.conversation_id,
                reply_to=message.id,
            )

            return {
                "protocol": "agent2agent",
                "version": self.a2a_protocol_version,
                "message": asdict(response),
            }

        except Exception as e:
            logger.error(f"Failed to handle incoming A2A message: {e}")
            return {
                "error": str(e),
                "protocol": "agent2agent",
                "version": self.a2a_protocol_version,
            }

    def register_message_handler(self, action: str, handler: Callable):
        """Register handler for specific Agent2Agent action"""
        self.message_handlers[action] = handler
        logger.info(f"Registered A2A message handler for action: {action}")

    async def _register_nexus_forge_agents(self):
        """Register Nexus Forge agents with ADK"""
        # Starri Orchestrator
        starri_capabilities = [
            AgentCapability(
                name="orchestrate_workflow",
                type="action",
                description="Orchestrate multi-agent workflows",
                input_schema={
                    "type": "object",
                    "properties": {"workflow_definition": {"type": "object"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"workflow_id": {"type": "string"}},
                },
            ),
            AgentCapability(
                name="coordinate_agents",
                type="action",
                description="Coordinate multiple AI agents",
                input_schema={
                    "type": "object",
                    "properties": {"agents": {"type": "array"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"coordination_result": {"type": "object"}},
                },
            ),
        ]

        await self.register_agent(
            agent_id="starri-orchestrator",
            name="Starri Orchestrator",
            agent_type="orchestrator",
            capabilities=starri_capabilities,
            endpoint_url="http://localhost:8000/agents/starri",
            metadata={"framework": "nexus-forge", "version": "1.0.0"},
        )

        # Gemini Analysis Agent
        gemini_capabilities = [
            AgentCapability(
                name="analyze_requirements",
                type="query",
                description="Analyze and understand requirements using Gemini 2.5 Pro",
                input_schema={
                    "type": "object",
                    "properties": {"requirements": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"analysis": {"type": "object"}},
                },
            ),
            AgentCapability(
                name="adaptive_thinking",
                type="transform",
                description="Apply adaptive thinking to complex problems",
                input_schema={
                    "type": "object",
                    "properties": {"problem": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"solution": {"type": "object"}},
                },
            ),
        ]

        await self.register_agent(
            agent_id="gemini-analysis",
            name="Gemini Analysis Agent",
            agent_type="llm",
            capabilities=gemini_capabilities,
            endpoint_url="http://localhost:8000/agents/gemini",
            metadata={"framework": "nexus-forge", "model": "gemini-2.5-pro"},
        )

        # Jules Coding Agent
        jules_capabilities = [
            AgentCapability(
                name="generate_code",
                type="generate",
                description="Generate production-ready code",
                input_schema={
                    "type": "object",
                    "properties": {"specification": {"type": "object"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            ),
            AgentCapability(
                name="create_tests",
                type="generate",
                description="Create comprehensive test suites",
                input_schema={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"tests": {"type": "string"}},
                },
            ),
        ]

        await self.register_agent(
            agent_id="jules-coding",
            name="Jules Coding Agent",
            agent_type="code",
            capabilities=jules_capabilities,
            endpoint_url="http://localhost:8000/agents/jules",
            metadata={"framework": "nexus-forge", "model": "jules"},
        )

        # Content Generation Agents
        imagen_capabilities = [
            AgentCapability(
                name="generate_image",
                type="generate",
                description="Generate high-quality images using Imagen 4",
                input_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"image_url": {"type": "string"}},
                },
            )
        ]

        await self.register_agent(
            agent_id="imagen-generator",
            name="Imagen Generator Agent",
            agent_type="image",
            capabilities=imagen_capabilities,
            endpoint_url="http://localhost:8000/agents/imagen",
            metadata={"framework": "nexus-forge", "model": "imagen-4"},
        )

        veo_capabilities = [
            AgentCapability(
                name="generate_video",
                type="generate",
                description="Generate videos using Veo 3",
                input_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                },
                output_schema={
                    "type": "object",
                    "properties": {"video_url": {"type": "string"}},
                },
            )
        ]

        await self.register_agent(
            agent_id="veo-generator",
            name="Veo Generator Agent",
            agent_type="video",
            capabilities=veo_capabilities,
            endpoint_url="http://localhost:8000/agents/veo",
            metadata={"framework": "nexus-forge", "model": "veo-3"},
        )

    async def _discover_external_agents(self):
        """Discover external agents supporting Agent2Agent protocol"""
        # Discover agents by category
        categories = [
            {"type": "orchestrator", "capabilities": ["workflow_management"]},
            {"type": "llm", "capabilities": ["text_generation", "analysis"]},
            {"type": "code", "capabilities": ["code_generation", "testing"]},
            {"type": "image", "capabilities": ["image_generation", "image_analysis"]},
            {"type": "video", "capabilities": ["video_generation", "video_editing"]},
            {"type": "data", "capabilities": ["data_processing", "analytics"]},
            {
                "type": "integration",
                "capabilities": ["api_integration", "workflow_automation"],
            },
        ]

        for category in categories:
            agents = await self.discover_agents(
                agent_type=category["type"],
                capabilities=category["capabilities"],
                frameworks=["langraph", "crewai", "autogen"],
            )
            logger.info(f"Discovered {len(agents)} {category['type']} agents")

    async def _setup_message_routing(self):
        """Setup message routing for common actions"""
        # Register default handlers
        self.register_message_handler("health_check", self._handle_health_check)
        self.register_message_handler(
            "capabilities_query", self._handle_capabilities_query
        )
        self.register_message_handler(
            "workflow_collaboration", self._handle_workflow_collaboration
        )
        self.register_message_handler("resource_sharing", self._handle_resource_sharing)

    async def _default_message_handler(
        self, message: Agent2AgentMessage
    ) -> Dict[str, Any]:
        """Default message handler for unregistered actions"""
        return {
            "status": "error",
            "message": f"No handler registered for action: {message.action}",
            "supported_actions": list(self.message_handlers.keys()),
        }

    async def _handle_health_check(self, message: Agent2AgentMessage) -> Dict[str, Any]:
        """Handle health check messages"""
        return {
            "status": "healthy",
            "agent_id": message.target_agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capabilities": len(
                self.registered_agents.get(
                    message.target_agent_id,
                    ADKAgent(
                        agent_id="",
                        name="",
                        type="",
                        version="",
                        capabilities=[],
                        endpoint_url="",
                        authentication={},
                    ),
                ).capabilities
            ),
        }

    async def _handle_capabilities_query(
        self, message: Agent2AgentMessage
    ) -> Dict[str, Any]:
        """Handle capabilities query messages"""
        agent = self.registered_agents.get(message.target_agent_id)
        if agent:
            return {
                "capabilities": [asdict(cap) for cap in agent.capabilities],
                "metadata": agent.metadata,
            }
        else:
            return {"error": "Agent not found"}

    async def _handle_workflow_collaboration(
        self, message: Agent2AgentMessage
    ) -> Dict[str, Any]:
        """Handle workflow collaboration requests"""
        # Extract workflow details from payload
        workflow_data = message.payload.get("workflow")
        collaboration_type = message.payload.get("type", "execute")

        if collaboration_type == "execute":
            # Execute workflow step
            try:
                # Route to appropriate Nexus Forge agent
                target_agent_id = message.target_agent_id

                # Create task in coordination system
                task_id = await self.coordination_client.create_task(
                    workflow_id=workflow_data.get("workflow_id", str(uuid4())),
                    name=f"A2A_Collaboration_{message.id}",
                    task_type="agent_collaboration",
                    input_data=message.payload,
                    agent_id=target_agent_id,
                )

                return {
                    "status": "accepted",
                    "task_id": task_id,
                    "estimated_completion": "5m",
                }

            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "error",
                "message": f"Unsupported collaboration type: {collaboration_type}",
            }

    async def _handle_resource_sharing(
        self, message: Agent2AgentMessage
    ) -> Dict[str, Any]:
        """Handle resource sharing requests"""
        resource_type = message.payload.get("resource_type")
        resource_request = message.payload.get("request")

        if resource_type == "model_access":
            # Share access to Nexus Forge AI models
            return {
                "status": "available",
                "models": ["gemini-2.5-pro", "jules", "imagen-4", "veo-3"],
                "access_endpoint": "http://localhost:8000/api/v1/models",
                "authentication_required": True,
            }
        elif resource_type == "computation":
            # Share computational resources
            return {
                "status": "available",
                "resources": {"cpu_cores": 8, "memory_gb": 32, "gpu_available": True},
                "reservation_endpoint": "http://localhost:8000/api/v1/resources",
            }
        else:
            return {
                "status": "error",
                "message": f"Unsupported resource type: {resource_type}",
            }

    async def cleanup(self):
        """Cleanup ADK integration resources"""
        try:
            # Unregister agents from ADK registry
            for agent_id in self.registered_agents:
                await self.http_client.delete(
                    f"{self.adk_registry_url}/v1/agents/{agent_id}"
                )

            # Close HTTP client
            await self.http_client.aclose()

            logger.info("ADK integration cleanup completed")

        except Exception as e:
            logger.error(f"Error during ADK cleanup: {e}")

    # Monitoring and Analytics
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get ADK integration performance metrics"""
        return {
            "registered_agents": len(self.registered_agents),
            "discovered_agents": len(self.external_agents),
            "active_conversations": len(self.conversation_contexts),
            "message_handlers": len(self.message_handlers),
            "supported_frameworks": self.supported_frameworks,
            "protocol_version": self.a2a_protocol_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
