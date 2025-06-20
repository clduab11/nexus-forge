# Agent2Agent Protocol Integration Example

## Quick Start Guide

This document shows how to integrate the Agent2Agent protocol into the existing Nexus Forge application.

### 1. Update Main Application

```python
# In src/backend/main.py, add the following imports:

from src.backend.protocols.agent2agent import (
    Agent2AgentIntegration,
    Agent2AgentConfig
)

# In the startup event handler, add:

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    
    # ... existing initialization code ...
    
    # Initialize Agent2Agent Protocol
    agent2agent_config = Agent2AgentConfig(
        enable_discovery=True,
        enable_security=True,
        enable_negotiation=True,
        enable_monitoring=True,
        redis_url=os.getenv("REDIS_URL"),
        ca_cert_path=os.getenv("ADK_CA_CERT_PATH")
    )
    
    app.state.agent2agent = Agent2AgentIntegration(
        agent_id="nexus_forge_main",
        ws_manager=app.state.ws_manager,
        adk_service=app.state.adk_service,
        config=agent2agent_config
    )
    
    await app.state.agent2agent.initialize(
        supabase_client=app.state.supabase_client
    )
    
    logger.info("Agent2Agent protocol initialized")
```

### 2. Add WebSocket Handler

```python
# In src/backend/api/routers/nexus_forge.py, add:

@router.websocket("/agent2agent/{agent_id}")
async def agent2agent_websocket(
    websocket: WebSocket,
    agent_id: str,
    current_user: User = Depends(get_current_user_ws)
):
    """WebSocket endpoint for Agent2Agent communication"""
    
    # Get integration instance
    agent2agent = websocket.app.state.agent2agent
    
    # Accept connection
    await websocket.accept()
    
    try:
        # Register agent session
        session_id = str(uuid.uuid4())
        await agent2agent.ws_bridge.register_agent_session(agent_id, session_id)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "agent_id": agent_id,
            "session_id": session_id,
            "protocol_version": "2.0.0"
        })
        
        # Message loop
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "agent2agent_message":
                # Convert to Agent2Agent message
                message = Agent2AgentMessage.from_dict(data["message"])
                
                # Handle through integration
                await agent2agent.handle_incoming_message(message)
                
    except WebSocketDisconnect:
        # Cleanup on disconnect
        await agent2agent.ws_bridge.unregister_agent_session(agent_id)
    except Exception as e:
        logger.error(f"Error in Agent2Agent WebSocket: {e}")
        await websocket.close()
```

### 3. Add REST API Endpoints

```python
# In src/backend/api/routers/adk.py, add:

@router.post("/agent2agent/discover")
async def discover_agents(
    query: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    agent2agent: Agent2AgentIntegration = Depends(get_agent2agent)
):
    """Discover available agents"""
    
    discovery_query = DiscoveryQuery(
        capabilities=query.get("capabilities"),
        agent_type=query.get("agent_type"),
        status=query.get("status", "active"),
        max_results=query.get("max_results", 10)
    )
    
    agents = await agent2agent.discover_agents(
        capabilities=discovery_query.capabilities,
        agent_type=discovery_query.agent_type
    )
    
    return {
        "agents": [agent.to_dict() for agent in agents],
        "total": len(agents)
    }


@router.post("/agent2agent/negotiate")
async def negotiate_task(
    task_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    agent2agent: Agent2AgentIntegration = Depends(get_agent2agent)
):
    """Negotiate task execution with agents"""
    
    # Create task
    task = await agent2agent.create_task(
        name=task_data["name"],
        description=task_data["description"],
        required_capabilities=task_data["required_capabilities"],
        optional_capabilities=task_data.get("optional_capabilities", []),
        requirements=task_data.get("requirements", {}),
        constraints=task_data.get("constraints", {}),
        priority=task_data.get("priority", 0),
        max_duration=task_data.get("max_duration", 300)
    )
    
    # Negotiate execution
    contract = await agent2agent.negotiate_task_execution(task)
    
    if contract:
        return {
            "success": True,
            "contract": contract.to_dict()
        }
    else:
        return {
            "success": False,
            "error": "No suitable agents found or negotiation failed"
        }
```

### 4. Integrate with Existing Agents

```python
# In src/backend/agents/agents/nexus_forge_agents.py, update StarriOrchestrator:

class StarriOrchestrator:
    # ... existing code ...
    
    async def register_with_agent2agent(self, agent2agent: Agent2AgentIntegration):
        """Register Starri and sub-agents with Agent2Agent protocol"""
        
        # Register Starri
        starri_info = AgentInfo(
            id="starri_orchestrator",
            name="Starri Orchestrator",
            type="orchestrator",
            description="Master AI coordinator for Nexus Forge",
            capabilities=[
                "task_decomposition",
                "agent_coordination", 
                "workflow_management",
                "app_building",
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
        
        await agent2agent.discovery_service.register_agent(starri_info)
        
        # Register sub-agents
        sub_agents = [
            ("jules_coder", "Jules Autonomous Coder", "development", 
             ["code_generation", "testing", "optimization"]),
            ("gemini_architect", "Gemini Architect", "architecture",
             ["system_design", "api_design", "database_design"]),
            ("veo_media", "Veo Media Generator", "media",
             ["video_generation", "animation", "demo_creation"]),
            ("imagen_designer", "Imagen Designer", "design",
             ["ui_design", "mockup_generation", "prototyping"])
        ]
        
        for agent_id, name, agent_type, capabilities in sub_agents:
            agent_info = AgentInfo(
                id=agent_id,
                name=name,
                type=agent_type,
                description=f"Specialized {agent_type} agent",
                capabilities=capabilities,
                resources={"memory": "2Gi", "cpu": 1.0},
                performance_metrics={
                    "avg_response_time": 200,
                    "success_rate": 0.95,
                    "availability": 0.99
                },
                adk_version="2.0",
                protocols_supported=["agent2agent/2.0"]
            )
            
            await agent2agent.discovery_service.register_agent(agent_info)
```

### 5. Usage Example

```python
# Example: Building an app with Agent2Agent coordination

async def build_app_with_agent2agent(user_prompt: str):
    """Build an app using Agent2Agent protocol for coordination"""
    
    # Get Agent2Agent integration
    agent2agent = app.state.agent2agent
    
    # Create main task
    main_task = await agent2agent.create_task(
        name="Build Complete Application",
        description=user_prompt,
        required_capabilities=["app_building", "orchestration"],
        priority=1,
        max_duration=600  # 10 minutes
    )
    
    # Find orchestrator agent
    orchestrators = await agent2agent.discover_agents(
        capabilities=["orchestration", "app_building"]
    )
    
    if not orchestrators:
        raise Exception("No orchestrator agents available")
    
    # Negotiate with Starri
    contract = await agent2agent.negotiate_task_execution(main_task)
    
    if not contract:
        raise Exception("Failed to negotiate with orchestrator")
    
    # Starri will now coordinate sub-agents using Agent2Agent protocol
    # Each sub-agent can discover and negotiate with other agents
    
    # Monitor progress via WebSocket events
    # Results will be delivered through Agent2Agent messages
    
    return {
        "task_id": main_task.id,
        "contract_id": contract.contract_id,
        "orchestrator": contract.provider_id
    }
```

### 6. Client-Side Integration

```typescript
// In frontend, connect to Agent2Agent WebSocket

class Agent2AgentClient {
    private ws: WebSocket;
    private agentId: string;
    
    constructor(agentId: string) {
        this.agentId = agentId;
    }
    
    connect() {
        this.ws = new WebSocket(
            `ws://localhost:8000/agent2agent/${this.agentId}`
        );
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'agent2agent_message') {
                this.handleAgentMessage(data.message);
            } else if (data.type === 'agent2agent_broadcast') {
                this.handleBroadcast(data.message);
            }
        };
    }
    
    async discoverAgents(capabilities: string[]) {
        const response = await fetch('/api/adk/agent2agent/discover', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify({
                capabilities,
                status: 'active'
            })
        });
        
        return response.json();
    }
    
    async negotiateTask(taskData: any) {
        const response = await fetch('/api/adk/agent2agent/negotiate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(taskData)
        });
        
        return response.json();
    }
}
```

## Benefits of Integration

1. **Autonomous Agent Coordination**: Agents can discover and coordinate with each other without central control
2. **Secure Communication**: All agent-to-agent messages are encrypted and authenticated
3. **Dynamic Capability Matching**: Tasks are automatically routed to the most capable agents
4. **Real-time Monitoring**: WebSocket integration provides live updates on agent activities
5. **Scalability**: Distributed discovery and negotiation support large agent networks
6. **ADK Native**: Fully integrated with Google's ADK framework

## Next Steps

1. Deploy Redis for distributed discovery
2. Configure SSL certificates for secure channels
3. Set up monitoring dashboards
4. Create agent templates for common tasks
5. Implement advanced negotiation strategies
6. Add federation support for cross-organization agents