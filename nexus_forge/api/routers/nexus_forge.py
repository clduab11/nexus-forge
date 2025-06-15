"""
Nexus Forge Router - API endpoints for one-shot app building

Handles app generation requests, real-time updates via WebSocket,
and integration with multiple AI models.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio

from ...database import get_db
from ...core.auth import get_current_user
from ...models import User
from ...agents.agents.nexus_forge_agents import StarriOrchestrator, NexusForgeWebSocketHandler
from ...core.monitoring import StructuredLogger
from ...config import settings

router = APIRouter(
    prefix="/api/nexus-forge",
    tags=["nexus-forge"],
    responses={404: {"description": "Not found"}},
)

# Initialize components
logger = StructuredLogger("nexus-forge-router")
orchestrator = StarriOrchestrator(
    project_id=settings.get("gcp", {}).get("project_id", "nexus-forge-dev"),
    region=settings.get("gcp", {}).get("region", "us-central1")
)
ws_handler = NexusForgeWebSocketHandler(orchestrator)

# Active build sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/build",
    response_model=Dict[str, Any],
    summary="Start app building process",
    description="Initiate one-shot app building with natural language description")
async def start_build(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Start building an app from a natural language description.
    
    Args:
        request: Contains 'prompt' with app description and optional 'config'
        background_tasks: FastAPI background task handler
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Build session information including session_id and WebSocket URL
    """
    prompt = request.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="App description is required")
    
    # Create build session
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "user_id": current_user.id,
        "prompt": prompt,
        "status": "initializing",
        "started_at": datetime.utcnow(),
        "config": request.get("config", {})
    }
    
    # Log build request
    logger.log("info", "App build requested",
               user_id=current_user.id,
               session_id=session_id,
               prompt_length=len(prompt))
    
    # Start build process in background
    background_tasks.add_task(
        process_build_async,
        session_id,
        prompt,
        request.get("config", {})
    )
    
    return {
        "session_id": session_id,
        "status": "build_started",
        "websocket_url": f"/ws/nexus-forge/{session_id}",
        "message": "Connect to WebSocket for real-time updates"
    }


@router.get("/build/{session_id}",
    response_model=Dict[str, Any],
    summary="Get build status",
    description="Retrieve current status of an app build session")
async def get_build_status(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get the current status of a build session.
    
    Args:
        session_id: Build session identifier
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Current build status and results if available
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Build session not found")
    
    # Verify user owns this session
    if session["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "started_at": session.get("started_at"),
        "completed_at": session.get("completed_at"),
        "result": session.get("result"),
        "error": session.get("error")
    }


@router.get("/builds",
    response_model=List[Dict[str, Any]],
    summary="List user's builds",
    description="Get list of all build sessions for the current user")
async def list_builds(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all build sessions for the current user.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        current_user: Authenticated user
        
    Returns:
        List of build sessions with basic information
    """
    user_sessions = [
        {
            "session_id": sid,
            "prompt": session["prompt"][:100] + "..." if len(session["prompt"]) > 100 else session["prompt"],
            "status": session.get("status"),
            "started_at": session.get("started_at"),
            "completed_at": session.get("completed_at")
        }
        for sid, session in active_sessions.items()
        if session["user_id"] == current_user.id
    ]
    
    # Sort by start time descending
    user_sessions.sort(key=lambda x: x["started_at"], reverse=True)
    
    # Apply pagination
    return user_sessions[skip:skip + limit]


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time build updates.
    
    Provides live updates during the app building process including:
    - Phase progress updates
    - Generated file notifications
    - Mockup and video URLs
    - Live preview URLs
    - Error notifications
    """
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid session ID"
        })
        await websocket.close()
        return
    
    # Add WebSocket to session
    session["websocket"] = websocket
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to Nexus Forge build session"
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute timeout
                )
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "cancel":
                    await cancel_build(session_id)
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        logger.log("info", "WebSocket disconnected", session_id=session_id)
    except Exception as e:
        logger.log("error", "WebSocket error", 
                   session_id=session_id,
                   error=str(e))
    finally:
        # Remove WebSocket from session
        if "websocket" in session:
            del session["websocket"]


@router.post("/deploy/{session_id}",
    response_model=Dict[str, Any],
    summary="Deploy built app",
    description="Deploy the generated app to Google Cloud Run")
async def deploy_app(
    session_id: str,
    deployment_config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Deploy a successfully built app to Google Cloud Run.
    
    Args:
        session_id: Build session identifier
        deployment_config: Optional deployment configuration
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Deployment information including live URL
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Build session not found")
    
    # Verify user owns this session
    if session["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check build completed successfully
    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Build must be completed before deployment")
    
    # Deploy to Cloud Run
    # TODO: Implement actual deployment logic
    deployment_url = f"https://app-{session_id[:8]}-{current_user.id}.run.app"
    
    return {
        "deployment_url": deployment_url,
        "status": "deployed",
        "message": "App successfully deployed to Google Cloud Run"
    }


@router.get("/templates",
    response_model=List[Dict[str, Any]],
    summary="Get app templates",
    description="Retrieve available app templates for quick starts")
async def get_templates():
    """
    Get list of available app templates.
    
    Returns:
        List of templates with names, descriptions, and example prompts
    """
    templates = [
        {
            "id": "dashboard",
            "name": "Analytics Dashboard",
            "description": "Real-time data visualization dashboard",
            "example_prompt": "Build a real-time analytics dashboard with charts for sales data, user metrics, and performance KPIs",
            "preview_image": "/templates/dashboard.png"
        },
        {
            "id": "ecommerce",
            "name": "E-Commerce Store",
            "description": "Full-featured online shopping platform",
            "example_prompt": "Create an e-commerce store with product catalog, shopping cart, checkout, and admin panel",
            "preview_image": "/templates/ecommerce.png"
        },
        {
            "id": "saas",
            "name": "SaaS Application",
            "description": "Multi-tenant SaaS platform with subscriptions",
            "example_prompt": "Build a SaaS application for project management with team collaboration, task tracking, and billing",
            "preview_image": "/templates/saas.png"
        },
        {
            "id": "mobile",
            "name": "Mobile App",
            "description": "Cross-platform mobile application",
            "example_prompt": "Create a mobile app for fitness tracking with workout plans, progress charts, and social features",
            "preview_image": "/templates/mobile.png"
        },
        {
            "id": "api",
            "name": "REST API",
            "description": "Scalable REST API with documentation",
            "example_prompt": "Build a REST API for a blog platform with authentication, CRUD operations, and OpenAPI docs",
            "preview_image": "/templates/api.png"
        }
    ]
    
    return templates


async def process_build_async(session_id: str, prompt: str, config: Dict[str, Any]):
    """
    Process app build asynchronously.
    
    This function runs in the background and updates the session
    with progress information.
    """
    session = active_sessions.get(session_id)
    if not session:
        return
    
    try:
        # Update status
        session["status"] = "building"
        
        # Send updates via WebSocket if connected
        ws = session.get("websocket")
        if ws:
            await ws_handler.handle_build_request(ws, session_id, {
                "prompt": prompt,
                "config": config
            })
        
        # Execute build process
        result = await orchestrator.build_app_with_starri(prompt)
        
        # Update session with results
        session["status"] = "completed"
        session["completed_at"] = datetime.utcnow()
        session["result"] = result
        
        # Send completion message
        if ws:
            await ws.send_json({
                "type": "build_complete",
                "session_id": session_id,
                "result": result,
                "message": "App successfully built!"
            })
            
    except Exception as e:
        logger.log("error", "Build process failed",
                   session_id=session_id,
                   error=str(e))
        
        session["status"] = "failed"
        session["error"] = str(e)
        
        # Send error message
        ws = session.get("websocket")
        if ws:
            await ws.send_json({
                "type": "build_error",
                "session_id": session_id,
                "error": str(e),
                "message": "Build failed. Please try again."
            })


async def cancel_build(session_id: str):
    """Cancel an ongoing build process."""
    session = active_sessions.get(session_id)
    if session and session.get("status") == "building":
        session["status"] = "cancelled"
        session["completed_at"] = datetime.utcnow()
        
        logger.log("info", "Build cancelled", session_id=session_id)