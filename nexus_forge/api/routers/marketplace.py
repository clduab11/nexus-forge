"""
Marketplace API endpoints for agent discovery and distribution
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
import tempfile
import os

from ...marketplace import (
    AgentRegistry, MarketplaceSearchEngine,
    AgentManifest, AgentPackage, AgentStatus
)
from ...core.auth import get_current_user
from ...core.exceptions import ValidationError, NotFoundError
from ..schemas.auth import User


router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace"])

# Initialize services
registry = AgentRegistry()
search_engine = MarketplaceSearchEngine()


@router.post("/agents/publish", response_model=AgentPackage)
async def publish_agent(
    manifest: str = Form(..., description="Agent manifest JSON"),
    package: UploadFile = File(..., description="Agent package ZIP file"),
    current_user: User = Depends(get_current_user)
):
    """
    Publish a new agent to the marketplace
    
    Requires authentication. The agent will undergo security scanning
    and performance benchmarking before approval.
    """
    try:
        # Parse manifest
        import json
        manifest_data = json.loads(manifest)
        agent_manifest = AgentManifest(**manifest_data)
        
        # Read package file
        package_content = await package.read()
        
        # Publish agent
        agent_package = await registry.publish_agent(
            package_file=package_content,
            manifest=agent_manifest,
            author_id=current_user.id,
            author_email=current_user.email
        )
        
        return agent_package
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid manifest JSON")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish agent: {str(e)}")


@router.get("/agents/search", response_model=List[Dict[str, Any]])
async def search_agents(
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    author: Optional[str] = Query(None, description="Filter by author"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    min_downloads: Optional[int] = Query(None, ge=0, description="Minimum downloads"),
    sort_by: str = Query("relevance", regex="^(relevance|downloads|rating|newest)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Search for agents in the marketplace
    
    Supports semantic search, filtering, and various sort options.
    """
    filters = {}
    if category:
        filters["category"] = category
    if tags:
        filters["tags"] = tags
    if author:
        filters["author"] = author
    if min_rating is not None:
        filters["min_rating"] = min_rating
    if min_downloads is not None:
        filters["min_downloads"] = min_downloads
    
    results = await search_engine.search(
        query=q,
        filters=filters,
        limit=limit,
        offset=offset,
        sort_by=sort_by
    )
    
    return [result.dict() for result in results]


@router.get("/agents/trending", response_model=List[AgentPackage])
async def get_trending_agents(
    time_window: str = Query("week", regex="^(day|week|month)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get trending agents based on recent activity"""
    agents = await search_engine.get_trending(time_window, limit)
    return agents


@router.get("/agents/suggest")
async def get_suggestions(
    q: str = Query(..., min_length=2, description="Partial search query"),
    limit: int = Query(10, ge=1, le=20)
):
    """Get search suggestions for autocomplete"""
    suggestions = await search_engine.suggest(q, limit)
    return suggestions


@router.get("/agents/{name}", response_model=AgentPackage)
async def get_agent(
    name: str,
    version: Optional[str] = Query(None, description="Specific version (default: latest)")
):
    """Get agent details by name and optional version"""
    try:
        agent = await registry.get_agent(name, version)
        return agent
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/agents/{agent_id}/similar", response_model=List[AgentPackage])
async def get_similar_agents(
    agent_id: str,
    limit: int = Query(10, ge=1, le=20)
):
    """Find agents similar to the specified agent"""
    agents = await search_engine.get_similar(agent_id, limit)
    return agents


@router.post("/agents/{agent_id}/install")
async def install_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Install an agent for the current user
    
    Returns installation instructions and dependencies.
    """
    try:
        installation_info = await registry.install_agent(agent_id, current_user.id)
        return installation_info
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/agents/{agent_id}/download")
async def download_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download agent package file"""
    try:
        # Get agent
        agent = await registry._get_agent_by_id(agent_id)
        
        if not agent.package_url:
            raise HTTPException(status_code=404, detail="Package file not found")
        
        # Record download
        await registry._increment_downloads(agent_id)
        
        # Return file
        return FileResponse(
            agent.package_url,
            media_type="application/zip",
            filename=f"{agent.manifest.name}-{agent.manifest.version}.zip"
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.post("/agents/{agent_id}/rate")
async def rate_agent(
    agent_id: str,
    rating: int = Form(..., ge=1, le=5),
    review: Optional[str] = Form(None, max_length=1000),
    current_user: User = Depends(get_current_user)
):
    """Rate and review an agent"""
    try:
        await registry.rate_agent(
            agent_id=agent_id,
            user_id=current_user.id,
            rating=rating,
            review=review
        )
        return {"status": "success", "message": "Rating submitted"}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/categories")
async def get_categories():
    """Get list of available agent categories"""
    from ...marketplace.models import AgentCategory
    
    categories = [
        {
            "value": cat.value,
            "name": cat.value.replace("_", " ").title(),
            "description": f"Agents for {cat.value.replace('_', ' ')}"
        }
        for cat in AgentCategory
    ]
    
    return categories


@router.get("/stats")
async def get_marketplace_stats():
    """Get marketplace statistics"""
    # This would query aggregated stats from the database
    # For now, return sample data
    return {
        "total_agents": 42,
        "total_downloads": 1337,
        "total_authors": 15,
        "agents_by_category": {
            "natural_language": 12,
            "code_generation": 8,
            "data_processing": 10,
            "workflow_automation": 7,
            "analytics": 5
        },
        "popular_tags": [
            {"tag": "ai", "count": 35},
            {"tag": "automation", "count": 28},
            {"tag": "nlp", "count": 22},
            {"tag": "productivity", "count": 18},
            {"tag": "integration", "count": 15}
        ]
    }


# Admin endpoints

@router.put("/admin/agents/{agent_id}/status")
async def update_agent_status(
    agent_id: str,
    status: AgentStatus,
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Update agent status (admin only)
    
    Used for approving/rejecting agents after review.
    """
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        agent = await registry.update_agent_status(
            agent_id=agent_id,
            status=status,
            reviewer_id=current_user.id,
            notes=notes
        )
        return agent
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/admin/review-queue")
async def get_review_queue(
    current_user: User = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get agents pending review (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Query pending agents
    agents = await registry.search_agents(
        category=None,
        tags=None,
        author=None,
        limit=limit,
        offset=offset
    )
    
    # Filter for pending status
    pending = [a for a in agents if a.status == AgentStatus.PENDING]
    
    return pending