"""
ADK Integration Router for Parallax Pal

This module provides API endpoints for interacting with the ADK-based agent system,
including research requests, status checks, and configuration management for Starri.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import database and authentication dependencies
from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import User
from ..schemas.auth import UserOut
from ..services.adk_service import get_adk_service

# Import WebSocket manager and ADK service
from ..websocket_adk import adk_websocket_manager

# Removed unused imports - using ADK service layer instead




# Define request models for API endpoints
class ResearchRequestAPI(BaseModel):
    query: str
    continuous_mode: bool = False
    force_refresh: bool = False
    max_sources: Optional[int] = 20
    depth_level: str = "detailed"
    focus_areas: Optional[List[str]] = None


# Create router
router = APIRouter(
    prefix="/api/adk",
    tags=["adk"],
    responses={404: {"description": "Not found"}},
)

# Set up logging
logger = logging.getLogger(__name__)

# ADK configuration - no longer needed with service layer
# ADK_BASE_URL = os.getenv("ADK_ORCHESTRATOR_URL", "http://localhost:8080")


@router.post("/research")
async def start_research(
    request: ResearchRequestAPI,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Start a new research query using the ADK agent system.

    Args:
        request: The research request payload
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with request ID and initial status
    """
    try:
        # Validate query
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required and cannot be empty",
            )

        # Log research request
        logger.info(
            f"Starting research for user {current_user.id}, query: {request.query}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Start research
        request_id = await adk_service.start_research(
            query=request.query,
            user_id=current_user.id,
            continuous_mode=request.continuous_mode,
            force_refresh=request.force_refresh,
            max_sources=request.max_sources,
            depth_level=request.depth_level,
            focus_areas=request.focus_areas,
        )

        # Return response with request ID
        return {
            "request_id": request_id,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting research: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting research: {str(e)}",
        )


@router.get("/research/{request_id}")
async def get_research_results(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the results of a research request.

    Args:
        request_id: ID of the research request
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with research results
    """
    try:
        # Log request
        logger.info(
            f"Getting research results for request {request_id}, user {current_user.id}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Get research results
        results = await adk_service.get_research_results(request_id, current_user.id)

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research request {request_id} not found or not accessible",
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting research results: {str(e)}",
        )


@router.get("/research/{request_id}/status")
async def get_research_status(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the current status of a research request.

    Args:
        request_id: ID of the research request
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with research status and progress
    """
    try:
        # Log request
        logger.info(
            f"Getting research status for request {request_id}, user {current_user.id}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Get research status
        status_info = await adk_service.get_research_status(request_id, current_user.id)

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research request {request_id} not found or not accessible",
            )

        return status_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting research status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting research status: {str(e)}",
        )


@router.post("/research/{request_id}/cancel")
async def cancel_research(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Cancel an ongoing research request.

    Args:
        request_id: ID of the research request to cancel
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with cancellation status
    """
    try:
        # Log cancellation request
        logger.info(
            f"Cancelling research request {request_id} for user {current_user.id}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Cancel research
        result = await adk_service.cancel_research(request_id, current_user.id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research request {request_id} not found or not accessible",
            )

        # Return cancellation status
        return {
            "request_id": request_id,
            "status": "cancelled",
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling research: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling research: {str(e)}",
        )


@router.post("/research/{request_id}/followup")
async def generate_followup_questions(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate follow-up questions for a completed research session.

    Args:
        request_id: ID of the research request
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with follow-up questions
    """
    try:
        # Log follow-up request
        logger.info(
            f"Generating follow-up questions for request {request_id}, user {current_user.id}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Generate follow-up questions
        questions = await adk_service.generate_followup_questions(
            request_id, current_user.id
        )

        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research request {request_id} not found or follow-up questions not available",
            )

        # Return follow-up questions
        return {
            "request_id": request_id,
            "questions": questions,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating follow-up questions: {str(e)}",
        )


@router.get("/knowledge-graph/{request_id}")
async def get_knowledge_graph(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the knowledge graph for a research session.

    Args:
        request_id: ID of the research request
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with knowledge graph data
    """
    try:
        # Log graph request
        logger.info(
            f"Getting knowledge graph for request {request_id}, user {current_user.id}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Get knowledge graph
        graph_data = await adk_service.generate_knowledge_graph(
            request_id, current_user.id
        )

        if not graph_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge graph for request {request_id} not found or not available",
            )

        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting knowledge graph: {str(e)}",
        )


@router.get("/citations/{request_id}")
async def get_citations(
    request_id: str,
    style: str = "apa",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the citations for a research session.

    Args:
        request_id: ID of the research request
        style: Citation style (default: apa)
        db: Database session
        current_user: The authenticated user

    Returns:
        JSON response with citations and bibliography
    """
    try:
        # Log citation request
        logger.info(
            f"Getting citations for request {request_id}, user {current_user.id}, style {style}"
        )

        # Get ADK service
        adk_service = get_adk_service()

        # Get citations
        citation_data = await adk_service.get_citations(
            request_id, current_user.id, style
        )

        if not citation_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citations for request {request_id} not found or not available",
            )

        # Return citations
        return citation_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting citations: {str(e)}",
        )


@router.get("/health")
async def check_adk_health():
    """
    Check the health of the ADK system.

    Returns:
        JSON response with ADK health status
    """
    try:
        # Get ADK service
        adk_service = get_adk_service()

        # Check ADK health
        health_data = await adk_service.check_health()

        return {
            "status": "healthy",
            "adk_status": health_data.get("status", "operational"),
            "message": "ADK system is operational",
            "timestamp": datetime.now().isoformat(),
            "details": health_data,
        }
    except Exception as e:
        logger.error(f"ADK health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "adk_status": "unavailable",
            "message": f"ADK health check error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


# WebSocket endpoint
@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """WebSocket endpoint for ADK real-time updates"""
    # This function is defined in websocket_adk.py
    # Use the endpoint from there, with proper authentication
    return await websocket_adk.websocket_endpoint(websocket, db, current_user)
