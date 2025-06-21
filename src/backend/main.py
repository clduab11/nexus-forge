"""
Nexus Forge Main API Entry Point

Enhanced FastAPI application with Starri orchestration, real-time WebSocket coordination,
and comprehensive Supabase integration for the Google Cloud Multi-Agent Hackathon.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

# Starri orchestrator integration
from .agents.starri.orchestrator import StarriOrchestrator

# Security and middleware
from .api.middleware.rate_limiter import RateLimiter

# Import routers
from .api.routers import adk, auth, health, nexus_forge, subscription
from .core.exceptions import NexusForgeError

# Monitoring
from .core.monitoring import setup_monitoring

# State management
from .core.state import StateManager

# Supabase integration
from .integrations.supabase.coordination_client import SupabaseCoordinationClient

# WebSocket manager for real-time updates
from .websockets.manager import WebSocketManager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Enhanced application lifespan manager with Starri orchestration."""
    # Startup
    logger.info("Starting Nexus Forge API with Starri orchestration...")

    # Initialize database
    from .database import init_db

    await init_db()

    # Initialize monitoring
    await setup_monitoring()

    # Initialize state management
    app.state.state_manager = StateManager()

    # Initialize Supabase coordination client
    app.state.supabase_client = SupabaseCoordinationClient(
        url=os.getenv("SUPABASE_URL", "https://woywcqjbubgkjumqgcqj.supabase.co"),
        key=os.getenv(
            "SUPABASE_ANON_KEY",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndveXdjcWpidWJna2p1bXFnY3FqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAwMDk1MjUsImV4cCI6MjA2NTU4NTUyNX0.7sau-I1jYaJ4uhaA27O3uS4TlxzucknxvFFi9hU4q5Q",
        ),
        project_id="woywcqjbubgkjumqgcqj",
    )
    await app.state.supabase_client.connect()

    # Initialize WebSocket manager with Supabase integration
    app.state.websocket_manager = WebSocketManager(
        supabase_client=app.state.supabase_client
    )

    # Initialize Starri orchestrator
    app.state.starri_orchestrator = StarriOrchestrator(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID", "nexus-forge-dev"),
        supabase_url=os.getenv(
            "SUPABASE_URL", "https://woywcqjbubgkjumqgcqj.supabase.co"
        ),
        supabase_key=os.getenv(
            "SUPABASE_ANON_KEY",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndveXdjcWpidWJna2p1bXFnY3FqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAwMDk1MjUsImV4cCI6MjA2NTU4NTUyNX0.7sau-I1jYaJ4uhaA27O3uS4TlxzucknxvFFi9hU4q5Q",
        ),
        mem0_api_key=os.getenv("MEM0_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )
    await app.state.starri_orchestrator.initialize()

    logger.info("Nexus Forge API with Starri orchestration started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Nexus Forge API...")

    # Cleanup Starri orchestrator
    if hasattr(app.state, "starri_orchestrator"):
        await app.state.starri_orchestrator.shutdown()

    # Cleanup Supabase connection
    if hasattr(app.state, "supabase_client"):
        await app.state.supabase_client.disconnect()

    # Cleanup WebSocket connections
    if hasattr(app.state, "websocket_manager"):
        await app.state.websocket_manager.cleanup()

    logger.info("Nexus Forge API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Nexus Forge API",
    description="AI-Powered One-Shot App Builder with Multi-Agent Orchestration",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "forge",
            "description": "Core app building operations with Starri orchestration",
        },
        {"name": "auth", "description": "Authentication and user management"},
        {"name": "adk", "description": "Google Agent Development Kit integration"},
        {"name": "health", "description": "System health and monitoring"},
    ],
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=(
        ["*"]
        if os.getenv("ENVIRONMENT") == "development"
        else ["nexusforge.example.com", "api.nexusforge.example.com", "*.run.app"]
    ),
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://nexusforge.example.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting middleware
rate_limiter = RateLimiter()
app.add_middleware(rate_limiter.middleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(nexus_forge.router, prefix="/api/nexus-forge", tags=["forge"])
app.include_router(adk.router, prefix="/api/adk", tags=["adk"])
app.include_router(
    subscription.router, prefix="/api/subscription", tags=["subscription"]
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(NexusForgeError)
async def nexus_forge_exception_handler(request: Request, exc: NexusForgeError):
    """Handle custom Nexus Forge errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
        },
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nexus Forge API",
        "version": "1.0.0",
        "description": "AI-Powered One-Shot App Builder with Starri Orchestration",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "websocket": "/ws",
        "features": [
            "starri_orchestration",
            "real_time_coordination",
            "multi_agent_workflows",
            "supabase_integration",
        ],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time agent coordination and task updates.

    Provides:
    - Agent status updates
    - Task progress notifications
    - Workflow coordination events
    - Error and completion notifications
    """
    try:
        # Accept the connection
        await websocket.accept()

        # Get WebSocket manager
        ws_manager = app.state.websocket_manager

        # For now, use a placeholder user ID - in production, extract from JWT
        user_id = "demo_user"  # TODO: Extract from authentication

        # Connect to WebSocket manager
        session_id = await ws_manager.connect(websocket, user_id, "pro")

        if not session_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            # Listen for messages
            while True:
                try:
                    message = await websocket.receive_json()
                    await ws_manager.handle_message(session_id, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket message error: {e}")
                    await ws_manager.send_to_session(
                        session_id,
                        {
                            "type": "error",
                            "message": "Message processing failed",
                            "timestamp": str(datetime.now()),
                        },
                    )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            await ws_manager.disconnect(websocket, user_id, session_id)

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


@app.get("/api/orchestrator/status", tags=["orchestration"])
async def get_orchestrator_status():
    """Get current Starri orchestrator status and metrics."""
    try:
        orchestrator = app.state.starri_orchestrator
        status = await orchestrator.get_orchestrator_status()
        return status
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get orchestrator status")


@app.post("/api/orchestrator/agents/register", tags=["orchestration"])
async def register_agent(agent_data: dict):
    """Register a new agent with the orchestrator."""
    try:
        orchestrator = app.state.starri_orchestrator

        # Extract agent information
        agent_id = agent_data.get("agent_id")
        agent_type = agent_data.get("agent_type", "generic")
        capabilities_list = agent_data.get("capabilities", [])
        configuration = agent_data.get("configuration", {})

        # Convert capability strings to enum values
        from .agents.starri.orchestrator import AgentCapability

        capabilities = []
        for cap_str in capabilities_list:
            try:
                capabilities.append(AgentCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")

        await orchestrator.register_agent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            configuration=configuration,
        )

        return {
            "status": "success",
            "message": f"Agent {agent_id} registered successfully",
            "agent_id": agent_id,
        }

    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(
            status_code=500, detail=f"Agent registration failed: {str(e)}"
        )


@app.post("/api/orchestrator/tasks/decompose", tags=["orchestration"])
async def decompose_task(task_request: dict):
    """Decompose a complex task into subtasks using Starri's deep thinking."""
    try:
        orchestrator = app.state.starri_orchestrator

        task_description = task_request.get("description", "")
        requirements = task_request.get("requirements", [])
        constraints = task_request.get("constraints", {})

        if not task_description:
            raise HTTPException(status_code=400, detail="Task description is required")

        result = await orchestrator.decompose_task(
            task_description=task_description,
            requirements=requirements,
            constraints=constraints,
        )

        return result

    except Exception as e:
        logger.error(f"Error decomposing task: {e}")
        raise HTTPException(
            status_code=500, detail=f"Task decomposition failed: {str(e)}"
        )


@app.post("/api/orchestrator/workflows/{workflow_id}/execute", tags=["orchestration"])
async def execute_workflow(workflow_id: str, execution_config: dict = None):
    """Execute a workflow using agent coordination."""
    try:
        orchestrator = app.state.starri_orchestrator

        execution_mode = (
            execution_config.get("mode", "parallel") if execution_config else "parallel"
        )

        result = await orchestrator.coordinate_agents(
            workflow_id=workflow_id, execution_mode=execution_mode
        )

        return result

    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/api/coordination/events", tags=["coordination"])
async def get_coordination_events(limit: int = 50):
    """Get recent coordination events from Supabase."""
    try:
        supabase_client = app.state.supabase_client
        events = await supabase_client.get_recent_events(limit=limit)
        return {"events": events}

    except Exception as e:
        logger.error(f"Error getting coordination events: {e}")
        raise HTTPException(status_code=500, detail="Failed to get coordination events")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development",
    )
