"""
Nexus Forge Main API Entry Point

Consolidated FastAPI application with ADK integration, multi-agent orchestration,
and comprehensive monitoring for the Google Cloud Multi-Agent Hackathon.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

# Import routers
from .api.routers import auth, nexus_forge, adk, health, subscription
from .api.dependencies.auth import get_current_user
from .models import User

# Security and middleware
from .api.middleware.rate_limiter import RateLimiter
from .core.exceptions import NexusForgeError

# WebSocket manager for real-time updates
from .websockets.manager import WebSocketManager

# State management
from .core.state import StateManager

# Monitoring
from .core.monitoring import structured_logger, setup_monitoring
from .integrations.google.monitoring import CloudMonitoringService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Nexus Forge API...")
    
    # Initialize database
    from .database import init_db
    await init_db()
    
    # Initialize monitoring
    await setup_monitoring()
    
    # Initialize state management
    app.state.state_manager = StateManager()
    
    # Initialize WebSocket manager
    app.state.websocket_manager = WebSocketManager()
    
    logger.info("Nexus Forge API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nexus Forge API...")
    
    # Cleanup connections
    if hasattr(app.state, 'websocket_manager'):
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
            "description": "Core app building operations with Starri orchestration"
        },
        {
            "name": "auth",
            "description": "Authentication and user management"
        },
        {
            "name": "adk",
            "description": "Google Agent Development Kit integration"
        },
        {
            "name": "health",
            "description": "System health and monitoring"
        }
    ]
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if os.getenv("ENVIRONMENT") == "development" else [
        "nexusforge.example.com",
        "api.nexusforge.example.com",
        "*.run.app"
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://nexusforge.example.com"
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
app.include_router(subscription.router, prefix="/api/subscription", tags=["subscription"])

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
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred"
        }
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nexus Forge API",
        "version": "1.0.0",
        "description": "AI-Powered One-Shot App Builder",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "nexus_forge.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development"
    )