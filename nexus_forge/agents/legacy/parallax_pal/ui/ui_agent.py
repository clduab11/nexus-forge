"""
UI Experience Agent for ParallaxMind

This agent manages the animated assistant interface, handles visualization requests,
and provides an engaging user experience with the Clippy-inspired character.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action

# Import schemas
from schemas.agent_messages import (
    UIRequest, UIResponse, ResearchResponse, KnowledgeGraphResponse
)

# Import config
from adk_config import get_config

# Set up logging
logger = logging.getLogger(__name__)

class UIAgent(Agent):
    """
    UI Experience Agent for ParallaxMind
    
    This agent is responsible for managing the animated assistant interface,
    controlling progressive information disclosure, and creating engaging
    visualizations based on research results.
    """
    
    def __init__(self):
        """Initialize UI agent with configuration."""
        super().__init__()
        self.config = get_config()
        self.logger = logging.getLogger("ui_agent")
        
        # Assistant state
        self.current_emotion = "neutral"
        self.animation_queue = []
        self.user_preferences = {}
    
    def initialize(self, context: AgentContext) -> None:
        """Initialize the agent with context."""
        self.context = context
        self.logger.info("UI agent initialized")
    
    @action
    def create_ui_components(self, request: Dict) -> Dict:
        """
        Create UI components based on the request.
        
        Args:
            request: The UI request dictionary
            
        Returns:
            Dict containing UI elements and animations
        """
        try:
            # Parse request
            ui_request = UIRequest(**request)
            request_id = ui_request.request_id
            request_type = ui_request.type
            content = ui_request.content
            
            self.logger.info(f"Creating UI components for request type: {request_type}, request_id: {request_id}")
            
            # Determine the appropriate UI elements based on request type
            if request_type == "research_complete":
                return self._create_research_ui(content)
            elif request_type == "knowledge_graph":
                return self._create_knowledge_graph_ui(content)
            elif request_type == "loading":
                return self._create_loading_ui(content)
            elif request_type == "error":
                return self._create_error_ui(content)
            elif request_type == "welcome":
                return self._create_welcome_ui(content)
            else:
                # Default UI
                return self._create_default_ui(content)
                
        except Exception as e:
            self.logger.error(f"Error creating UI components: {str(e)}")
            # Return basic error UI
            return {
                "request_id": request.get("request_id", str(uuid.uuid4())),
                "ui_elements": {
                    "component": "error_view",
                    "props": {
                        "message": f"Error creating UI: {str(e)}",
                        "showRetry": True
                    }
                },
                "animations": [
                    {
                        "character": "assistant",
                        "emotion": "confused",
                        "animation": "scratching_head"
                    }
                ],
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_research_ui(self, content: Dict) -> Dict:
        """
        Create UI for completed research.
        
        Args:
            content: The research content
            
        Returns:
            Dict containing UI elements for research
        """
        try:
            # Extract relevant information
            research = content.get("research", {})
            knowledge_graph = content.get("knowledge_graph", {})
            
            summary = research.get("summary", "No summary available")
            sources = research.get("sources", [])
            focus_areas = research.get("focus_areas", [])
            
            # Determine emotion based on research results
            emotion = "happy" if sources and summary else "neutral"
            animation = "presenting" if sources and summary else "thinking"
            
            # Create UI elements
            ui_elements = {
                "component": "research_results",
                "props": {
                    "summary": summary,
                    "sources": sources,
                    "focusAreas": focus_areas,
                    "showKnowledgeGraph": bool(knowledge_graph),
                    "knowledgeGraphData": knowledge_graph.get("graph", {}),
                    "layout": "tabbed",
                    "expandedSections": ["summary"]
                }
            }
            
            # Create animations
            animations = [
                {
                    "character": "assistant",
                    "emotion": emotion,
                    "animation": animation
                }
            ]
            
            return {
                "request_id": content.get("research", {}).get("request_id", str(uuid.uuid4())),
                "ui_elements": ui_elements,
                "animations": animations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating research UI: {str(e)}")
            return self._create_default_ui(content)
    
    def _create_knowledge_graph_ui(self, content: Dict) -> Dict:
        """
        Create UI for knowledge graph visualization.
        
        Args:
            content: The knowledge graph content
            
        Returns:
            Dict containing UI elements for knowledge graph
        """
        try:
            # Extract graph data
            graph = content.get("graph", {})
            summary = content.get("summary", "Knowledge graph visualization")
            
            # Determine emotion
            emotion = "excited" if graph.get("nodes") else "neutral"
            
            # Create UI elements
            ui_elements = {
                "component": "knowledge_graph_view",
                "props": {
                    "graphData": graph,
                    "summary": summary,
                    "interactiveControls": True,
                    "zoomEnabled": True,
                    "filterOptions": ["reliability", "relevance", "type"]
                }
            }
            
            # Create animations
            animations = [
                {
                    "character": "assistant",
                    "emotion": emotion,
                    "animation": "pointing"
                }
            ]
            
            return {
                "request_id": content.get("request_id", str(uuid.uuid4())),
                "ui_elements": ui_elements,
                "animations": animations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge graph UI: {str(e)}")
            return self._create_default_ui(content)
    
    def _create_loading_ui(self, content: Dict) -> Dict:
        """
        Create UI for loading state.
        
        Args:
            content: The loading content
            
        Returns:
            Dict containing UI elements for loading
        """
        try:
            # Extract loading information
            message = content.get("message", "Loading...")
            progress = content.get("progress", 0)
            
            # Create UI elements
            ui_elements = {
                "component": "loading_view",
                "props": {
                    "message": message,
                    "progress": progress,
                    "showProgressBar": True,
                    "showPercentage": True
                }
            }
            
            # Create animations
            animations = [
                {
                    "character": "assistant",
                    "emotion": "focused",
                    "animation": "searching"
                }
            ]
            
            return {
                "request_id": content.get("request_id", str(uuid.uuid4())),
                "ui_elements": ui_elements,
                "animations": animations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating loading UI: {str(e)}")
            return self._create_default_ui(content)
    
    def _create_error_ui(self, content: Dict) -> Dict:
        """
        Create UI for error state.
        
        Args:
            content: The error content
            
        Returns:
            Dict containing UI elements for error
        """
        try:
            # Extract error information
            error_message = content.get("error", "An error occurred")
            
            # Create UI elements
            ui_elements = {
                "component": "error_view",
                "props": {
                    "message": error_message,
                    "showRetry": True,
                    "showReport": True
                }
            }
            
            # Create animations
            animations = [
                {
                    "character": "assistant",
                    "emotion": "sad",
                    "animation": "apologizing"
                }
            ]
            
            return {
                "request_id": content.get("request_id", str(uuid.uuid4())),
                "ui_elements": ui_elements,
                "animations": animations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating error UI: {str(e)}")
            return self._create_default_ui(content)
    
    def _create_welcome_ui(self, content: Dict) -> Dict:
        """
        Create UI for welcome screen.
        
        Args:
            content: The welcome content
            
        Returns:
            Dict containing UI elements for welcome
        """
        try:
            # Extract welcome information
            user_name = content.get("user_name", "")
            
            # Create welcome message
            welcome_message = f"Welcome{' ' + user_name if user_name else ''}! I'm Starri, your Parallax Pal assistant. What would you like to research today?"
            
            # Create UI elements
            ui_elements = {
                "component": "welcome_view",
                "props": {
                    "message": welcome_message,
                    "showExamples": True,
                    "examples": [
                        "Research the latest developments in renewable energy",
                        "Explain the principles of quantum computing",
                        "Compare different machine learning algorithms for image recognition"
                    ],
                    "showHistory": bool(content.get("history"))
                }
            }
            
            # Create animations
            animations = [
                {
                    "character": "assistant",
                    "emotion": "happy",
                    "animation": "waving"
                }
            ]
            
            return {
                "request_id": content.get("request_id", str(uuid.uuid4())),
                "ui_elements": ui_elements,
                "animations": animations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating welcome UI: {str(e)}")
            return self._create_default_ui(content)
    
    def _create_default_ui(self, content: Dict) -> Dict:
        """
        Create default UI when specific type is not recognized.
        
        Args:
            content: The content
            
        Returns:
            Dict containing default UI elements
        """
        # Create UI elements
        ui_elements = {
            "component": "default_view",
            "props": {
                "content": content
            }
        }
        
        # Create animations
        animations = [
            {
                "character": "assistant",
                "emotion": "neutral",
                "animation": "idle"
            }
        ]
        
        return {
            "request_id": content.get("request_id", str(uuid.uuid4())),
            "ui_elements": ui_elements,
            "animations": animations,
            "timestamp": datetime.now().isoformat()
        }
    
    @action
    def generate_animation(self, emotion: str, context: Dict) -> Dict:
        """
        Generate appropriate animation based on emotion and context.
        
        Args:
            emotion: The desired emotion
            context: The context information
            
        Returns:
            Dict containing animation details
        """
        valid_emotions = [
            "neutral", "happy", "sad", "excited", "confused", 
            "focused", "surprised", "thoughtful"
        ]
        
        # Validate emotion
        if emotion not in valid_emotions:
            emotion = "neutral"
        
        # Map emotions to animations
        emotion_animations = {
            "neutral": ["idle", "nodding", "thinking"],
            "happy": ["smiling", "waving", "clapping"],
            "sad": ["drooping", "sighing", "apologizing"],
            "excited": ["jumping", "presenting", "celebrating"],
            "confused": ["scratching_head", "tilting_head", "shrugging"],
            "focused": ["searching", "examining", "typing"],
            "surprised": ["wide_eyes", "gasping", "stepping_back"],
            "thoughtful": ["chin_tapping", "looking_up", "nodding_slowly"]
        }
        
        # Select animation based on emotion and context
        available_animations = emotion_animations.get(emotion, ["idle"])
        
        # Determine best animation based on context
        if context.get("type") == "research":
            if context.get("stage") == "starting":
                preferred_animations = ["searching", "typing", "nodding"]
            elif context.get("stage") == "completing":
                preferred_animations = ["presenting", "smiling", "clapping"]
            else:
                preferred_animations = available_animations
        else:
            preferred_animations = available_animations
        
        # Find first matching animation or use first available
        animation = next((a for a in preferred_animations if a in available_animations), available_animations[0])
        
        # Create animation response
        animation_data = {
            "character": "assistant",
            "emotion": emotion,
            "animation": animation,
            "duration": 2.0,  # seconds
            "loop": context.get("loop", False)
        }
        
        return animation_data
    
    @action
    def update_user_preferences(self, preferences: Dict) -> Dict:
        """
        Update user interface preferences.
        
        Args:
            preferences: The user preferences
            
        Returns:
            Dict containing updated preferences
        """
        try:
            # Update stored preferences
            self.user_preferences.update(preferences)
            
            return {
                "status": "success",
                "preferences": self.user_preferences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @action
    def get_user_preferences(self) -> Dict:
        """
        Get current user interface preferences.
        
        Returns:
            Dict containing user preferences
        """
        return {
            "preferences": self.user_preferences,
            "timestamp": datetime.now().isoformat()
        }