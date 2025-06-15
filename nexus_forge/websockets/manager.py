"""
Unified WebSocket Manager for Nexus Forge

Consolidates WebSocket functionality with enhanced security, rate limiting,
and native ADK integration for multi-agent coordination.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from starlette.websockets import WebSocketState

# Core imports
from ..core.exceptions import NexusForgeError, ValidationError
from ..models import User

# Security and validation
from ..api.middleware.rate_limiter import WebSocketRateLimiter
from ..core.monitoring import structured_logger

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Unified WebSocket manager for Nexus Forge
    
    Provides real-time communication for:
    - App building progress updates
    - Multi-agent coordination status  
    - Error notifications
    - Deployment status
    """
    
    def __init__(self):
        """Initialize WebSocket manager"""
        
        # Connection tracking
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Build session tracking
        self.build_sessions: Dict[str, Dict] = {}
        
        # Background task management
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Rate limiting
        self.rate_limiter = WebSocketRateLimiter()
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        logger.info("Nexus Forge WebSocket manager initialized")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str,
        user_tier: str = "free"
    ) -> Optional[str]:
        """
        Connect new WebSocket client with validation and rate limiting
        
        Args:
            websocket: WebSocket connection
            user_id: Authenticated user ID
            user_tier: User subscription tier
            
        Returns:
            Session ID if successful, None if rejected
        """
        # Check rate limits
        allowed, reason = await self.rate_limiter.check_connection_limit(
            user_id, 
            max_connections=self._get_connection_limit(user_tier)
        )
        
        if not allowed:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=reason
            )
            return None
        
        # Accept connection
        await websocket.accept()
        
        async with self.lock:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Add to connections
            if user_id not in self.active_connections:
                self.active_connections[user_id] = {}
            
            self.active_connections[user_id][session_id] = websocket
            self.user_sessions[user_id] = session_id
            
            # Store metadata
            self.session_metadata[session_id] = {
                'user_id': user_id,
                'user_tier': user_tier,
                'connected_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'build_sessions': []
            }
            
            # Register connection
            await self.rate_limiter.register_connection(user_id, session_id)
            
            structured_logger.log(
                "info",
                "WebSocket connected",
                user_id=user_id,
                session_id=session_id
            )
            
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "features": self._get_tier_features(user_tier)
            })
            
            return session_id
    
    async def disconnect(self, websocket: WebSocket, user_id: str, session_id: Optional[str] = None):
        """Disconnect WebSocket client and cleanup"""
        
        async with self.lock:
            if user_id in self.active_connections:
                if session_id and session_id in self.active_connections[user_id]:
                    # Remove specific session
                    del self.active_connections[user_id][session_id]
                    if session_id in self.session_metadata:
                        del self.session_metadata[session_id]
                else:
                    # Find session by websocket reference
                    for sess_id, ws in list(self.active_connections[user_id].items()):
                        if ws == websocket:
                            del self.active_connections[user_id][sess_id]
                            if sess_id in self.session_metadata:
                                del self.session_metadata[sess_id]
                            session_id = sess_id
                            break
                
                # Clean up empty user entry
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                    if user_id in self.user_sessions:
                        del self.user_sessions[user_id]
                
                # Cleanup rate limiter
                if session_id:
                    await self.rate_limiter.unregister_connection(user_id, session_id)
                    await self._cleanup_session_tasks(session_id)
                
                structured_logger.log(
                    "info",
                    "WebSocket disconnected", 
                    user_id=user_id,
                    session_id=session_id
                )
    
    async def send_to_session(self, session_id: str, message: Dict):
        """Send message to specific session"""
        
        metadata = self.session_metadata.get(session_id)
        if not metadata:
            logger.warning(f"Session {session_id} not found")
            return
        
        user_id = metadata['user_id']
        websocket = self.active_connections.get(user_id, {}).get(session_id)
        
        if not websocket:
            logger.warning(f"WebSocket for session {session_id} not found")
            return
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
                
                # Update last activity
                metadata['last_activity'] = datetime.now().isoformat()
            else:
                await self.disconnect(websocket, user_id, session_id)
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")
            await self.disconnect(websocket, user_id, session_id)
    
    async def broadcast_to_user(self, user_id: str, message: Dict):
        """Broadcast message to all user sessions"""
        
        if user_id not in self.active_connections:
            return
        
        tasks = []
        for session_id in list(self.active_connections[user_id].keys()):
            tasks.append(self.send_to_session(session_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start_build_session(
        self, 
        user_id: str, 
        build_request: Dict
    ) -> str:
        """Start new app building session"""
        
        build_id = str(uuid.uuid4())
        
        # Create build session
        self.build_sessions[build_id] = {
            'user_id': user_id,
            'build_id': build_id,
            'request': build_request,
            'status': 'initializing',
            'progress': 0,
            'started_at': datetime.now().isoformat(),
            'current_phase': 'initialization',
            'agents_active': [],
            'results': {}
        }
        
        # Add to user's session metadata
        user_session_id = self.user_sessions.get(user_id)
        if user_session_id and user_session_id in self.session_metadata:
            self.session_metadata[user_session_id]['build_sessions'].append(build_id)
        
        # Send initial update
        await self.broadcast_to_user(user_id, {
            "type": "build_started",
            "build_id": build_id,
            "status": "initializing",
            "timestamp": datetime.now().isoformat()
        })
        
        return build_id
    
    async def update_build_progress(
        self,
        build_id: str,
        progress: int,
        phase: str,
        agent: Optional[str] = None,
        message: Optional[str] = None,
        results: Optional[Dict] = None
    ):
        """Update build session progress"""
        
        if build_id not in self.build_sessions:
            return
        
        session = self.build_sessions[build_id]
        session['progress'] = progress
        session['current_phase'] = phase
        
        if agent and agent not in session['agents_active']:
            session['agents_active'].append(agent)
        
        if results:
            session['results'].update(results)
        
        # Send progress update
        await self.broadcast_to_user(session['user_id'], {
            "type": "build_progress",
            "build_id": build_id,
            "progress": progress,
            "phase": phase,
            "agent": agent,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def complete_build_session(
        self,
        build_id: str,
        status: str,
        final_results: Dict
    ):
        """Complete build session"""
        
        if build_id not in self.build_sessions:
            return
        
        session = self.build_sessions[build_id]
        session['status'] = status
        session['completed_at'] = datetime.now().isoformat()
        session['results'].update(final_results)
        
        # Send completion message
        await self.broadcast_to_user(session['user_id'], {
            "type": "build_completed" if status == "completed" else "build_failed",
            "build_id": build_id,
            "status": status,
            "results": final_results,
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_message(self, session_id: str, message: Dict):
        """Handle incoming WebSocket message"""
        
        try:
            message_type = message.get("type", "")
            
            handlers = {
                "ping": self._handle_ping,
                "build_status": self._handle_build_status,
                "cancel_build": self._handle_cancel_build
            }
            
            handler = handlers.get(message_type)
            if handler:
                await handler(session_id, message)
            else:
                await self.send_to_session(session_id, {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_to_session(session_id, {
                "type": "error", 
                "error": "Message processing failed",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_ping(self, session_id: str, message: Dict):
        """Handle ping message"""
        
        await self.send_to_session(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_build_status(self, session_id: str, message: Dict):
        """Handle build status request"""
        
        build_id = message.get("build_id")
        if not build_id or build_id not in self.build_sessions:
            await self.send_to_session(session_id, {
                "type": "error",
                "error": "Build session not found",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        session = self.build_sessions[build_id]
        await self.send_to_session(session_id, {
            "type": "build_status",
            "build_id": build_id,
            "status": session['status'],
            "progress": session['progress'],
            "current_phase": session['current_phase'],
            "agents_active": session['agents_active'],
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_cancel_build(self, session_id: str, message: Dict):
        """Handle build cancellation"""
        
        build_id = message.get("build_id")
        if not build_id or build_id not in self.build_sessions:
            return
        
        session = self.build_sessions[build_id]
        session['status'] = 'cancelled'
        session['completed_at'] = datetime.now().isoformat()
        
        await self.send_to_session(session_id, {
            "type": "build_cancelled",
            "build_id": build_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_connection_limit(self, tier: str) -> int:
        """Get connection limit by tier"""
        limits = {
            'free': 2,
            'basic': 5, 
            'pro': 10,
            'enterprise': 50
        }
        return limits.get(tier, 2)
    
    def _get_tier_features(self, tier: str) -> Dict:
        """Get features by tier"""
        features = {
            'free': {
                'builds_per_hour': 3,
                'parallel_builds': 1,
                'ai_models': ['gemini_flash'],
                'export_formats': ['json']
            },
            'pro': {
                'builds_per_hour': 20,
                'parallel_builds': 3,
                'ai_models': ['gemini_pro', 'gemini_flash', 'imagen', 'veo'],
                'export_formats': ['json', 'zip', 'docker']
            }
        }
        return features.get(tier, features['free'])
    
    async def _cleanup_session_tasks(self, session_id: str):
        """Cleanup session background tasks"""
        
        # Cancel any active build sessions
        metadata = self.session_metadata.get(session_id, {})
        for build_id in metadata.get('build_sessions', []):
            if build_id in self.build_sessions:
                await self._handle_cancel_build(session_id, {'build_id': build_id})
    
    async def cleanup(self):
        """Cleanup manager resources"""
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("WebSocket manager cleanup complete")