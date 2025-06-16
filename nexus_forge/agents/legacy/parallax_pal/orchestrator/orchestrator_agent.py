"""
Orchestrator Agent for Parallax Pal

This is the main controller agent that delegates to specialized agents,
maintains conversation context, and coordinates the multi-agent system
to empower the Starri interface.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import ADK-specific libraries, fallback to local implementation if not available
try:
    from google.cloud.aiplatform import Model
    from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

# Import schemas
try:
    from schemas.agent_messages import (
        AgentActivity,
        AgentMessage,
        AssistantState,
        EmotionType,
        FocusArea,
        KnowledgeGraphData,
        KnowledgeGraphRequest,
        KnowledgeGraphResponse,
        ProgressUpdate,
        ResearchFocusArea,
        ResearchRequest,
        ResearchResponse,
        ResearchStatus,
        Source,
        UIRequest,
        UIResponse,
        UIState,
    )
except ImportError:
    # Fallback imports for local development
    from enum import Enum

    class ResearchStatus(str, Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    class EmotionType(str, Enum):
        NEUTRAL = "neutral"
        HAPPY = "happy"
        CONFUSED = "confused"
        EXCITED = "excited"
        SAD = "sad"

    class UIState(str, Enum):
        IDLE = "idle"
        THINKING = "thinking"
        PRESENTING = "presenting"
        ERROR = "error"

    class ResearchRequest:
        def __init__(
            self,
            query,
            continuous_mode=False,
            force_refresh=False,
            max_sources=None,
            depth_level="detailed",
            focus_areas=None,
            request_id=None,
        ):
            self.query = query
            self.continuous_mode = continuous_mode
            self.force_refresh = force_refresh
            self.max_sources = max_sources
            self.depth_level = depth_level
            self.focus_areas = focus_areas or []
            self.request_id = request_id or str(uuid.uuid4())

        def dict(self):
            return {
                "query": self.query,
                "continuous_mode": self.continuous_mode,
                "force_refresh": self.force_refresh,
                "max_sources": self.max_sources,
                "depth_level": self.depth_level,
                "focus_areas": self.focus_areas,
                "request_id": self.request_id,
            }

    class AssistantState:
        def __init__(
            self,
            emotion=EmotionType.NEUTRAL,
            state=UIState.IDLE,
            message=None,
            showBubble=False,
        ):
            self.emotion = emotion
            self.state = state
            self.message = message
            self.showBubble = showBubble

        def dict(self):
            return {
                "emotion": self.emotion,
                "state": self.state,
                "message": self.message,
                "showBubble": self.showBubble,
            }


# Import config (if available)
try:
    from adk_config import (
        AGENT_CONFIG,
        DEFAULT_MODEL,
        DEV_MODE,
        MODEL_CONFIG,
        PROJECT_ID,
        REGION,
        TIMEOUT,
    )
except ImportError:
    # Default config for local development
    DEV_MODE = True
    TIMEOUT = 300
    AGENT_CONFIG = {
        "orchestrator": {"delegation_threshold": 0.75, "max_delegation_depth": 3}
    }

# Set up logging
logging.basicConfig(
    level=logging.INFO if DEV_MODE else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator_agent")


class OrchestratorAgent:
    """
    Main controller agent that coordinates all specialized agents,
    manages the overall research process, and maintains context.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the orchestrator agent

        Args:
            config: Configuration parameters for the agent
        """
        self.config = config or AGENT_CONFIG.get("orchestrator", {})
        self.research_sessions: Dict[str, Dict[str, Any]] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}

        # Thresholds for delegation
        self.delegation_threshold = self.config.get("delegation_threshold", 0.75)
        self.max_delegation_depth = self.config.get("max_delegation_depth", 3)

        # For ADK integration
        self.context = None
        self.conversation_history = []

        logger.info(f"Orchestrator agent initialized with config: {self.config}")

    def initialize(self, context=None):
        """Initialize the agent with context (for ADK integration)"""
        self.context = context
        logger.info("Orchestrator agent initialized with context")

    async def handle_research_request(
        self, request, request_id: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Process a new research request

        Args:
            request: Research request parameters
            request_id: Unique ID for the research request
            user_id: ID of the user making the request

        Returns:
            Initial status of the research request
        """
        # Convert request to dict if needed
        request_dict = request if isinstance(request, dict) else request.dict()
        logger.info(f"Received research request: {request_dict}")

        # Create research session
        session = {
            "request_id": request_id,
            "user_id": user_id,
            "query": request_dict["query"],
            "continuous_mode": request_dict.get("continuous_mode", False),
            "force_refresh": request_dict.get("force_refresh", False),
            "max_sources": request_dict.get("max_sources", 20),
            "depth_level": request_dict.get("depth_level", "detailed"),
            "focus_areas": request_dict.get("focus_areas", []),
            "status": "pending",
            "progress": 0,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "sources": [],
            "summary": "",
            "agents": {},
            "error": None,
            "assistant_state": {
                "emotion": "excited",
                "state": "thinking",
                "message": "Starting research...",
                "showBubble": True,
            },
        }

        # Store session
        self.research_sessions[request_id] = session

        # Create task for processing research
        asyncio.create_task(self._process_research(request_id))

        # Return initial status
        return {
            "request_id": request_id,
            "status": "pending",
            "progress": 0,
            "message": "Research request received and queued for processing",
        }

    async def get_research_status(
        self, request_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Current status of the research request, or None if not found
        """
        logger.info(f"Getting status for research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return None

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return None

        # Return status
        return {
            "status": session["status"],
            "progress": session["progress"],
            "message": f"Research is {session['status']}",
            "agent_activities": self._get_agent_activities(request_id),
        }

    async def cancel_research(self, request_id: str, user_id: str) -> bool:
        """
        Cancel a research request

        Args:
            request_id: ID of the research request to cancel
            user_id: ID of the user making the request

        Returns:
            True if cancellation succeeded, False otherwise
        """
        logger.info(f"Cancelling research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return False

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return False

        # Cancel research
        session["status"] = "cancelled"
        session["end_time"] = datetime.utcnow().isoformat()
        session["assistant_state"] = {
            "emotion": "neutral",
            "state": "idle",
            "message": "Research has been cancelled.",
            "showBubble": True,
        }

        # Cancel any active agent tasks
        for agent_id, agent_info in session.get("agents", {}).items():
            if agent_info.get("status") == "working":
                agent_info["status"] = "cancelled"
                agent_info["end_time"] = datetime.utcnow().isoformat()

        return True

    async def get_research_results(
        self, request_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the results of a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Research results, or None if not found or not complete
        """
        logger.info(f"Getting results for research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return None

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return None

        # Convert session to ResearchResponse
        response = {
            "request_id": request_id,
            "query": session["query"],
            "status": session["status"],
            "progress": session["progress"],
            "focus_areas": session.get("focus_areas", []),
            "summary": session.get("summary", ""),
            "created_at": session["start_time"],
            "completed_at": session.get("end_time"),
            "error_message": session.get("error"),
        }

        return response

    async def generate_knowledge_graph(
        self, request_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a knowledge graph for a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Knowledge graph data, or None if not found or not complete
        """
        logger.info(f"Generating knowledge graph for research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return None

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return None

        # Check if research is complete or in progress
        if session["status"] not in ["completed", "in_progress"]:
            logger.warning(
                f"Research request {request_id} is not complete or in progress"
            )
            return None

        # Check if knowledge graph has already been generated
        if "knowledge_graph" in session:
            return session["knowledge_graph"]

        # Delegate to knowledge graph agent
        try:
            # In a real implementation, this would call the knowledge graph agent
            # For now, just create a placeholder graph
            knowledge_graph = await self._delegate_to_knowledge_graph_agent(request_id)

            # Store in session
            session["knowledge_graph"] = knowledge_graph

            return knowledge_graph
        except Exception as e:
            logger.error(f"Error generating knowledge graph: {str(e)}")
            return None

    async def generate_citations(
        self, request_id: str, style: str, user_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate citations for a research request

        Args:
            request_id: ID of the research request
            style: Citation style (apa, mla, chicago, ieee)
            user_id: ID of the user making the request

        Returns:
            List of citations, or None if not found or not complete
        """
        logger.info(f"Generating citations for research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return None

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return None

        # Check if research is complete or in progress
        if session["status"] not in ["completed", "in_progress"]:
            logger.warning(
                f"Research request {request_id} is not complete or in progress"
            )
            return None

        # Check if citations have already been generated
        if "citations" in session and style in session["citations"]:
            return session["citations"][style]

        # Delegate to citation agent
        try:
            # In a real implementation, this would call the citation agent
            # For now, just create placeholder citations
            citations = await self._delegate_to_citation_agent(request_id, style)

            # Store in session
            if "citations" not in session:
                session["citations"] = {}
            session["citations"][style] = citations

            return citations
        except Exception as e:
            logger.error(f"Error generating citations: {str(e)}")
            return None

    async def get_follow_up_questions(
        self, request_id: str, user_id: str
    ) -> Optional[List[str]]:
        """
        Generate follow-up questions for a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            List of follow-up questions, or None if not found or not complete
        """
        logger.info(
            f"Generating follow-up questions for research request: {request_id}"
        )

        if request_id not in self.research_sessions:
            logger.warning(f"Research request {request_id} not found")
            return None

        session = self.research_sessions[request_id]

        # Check user ownership
        if session["user_id"] != user_id:
            logger.warning(f"User {user_id} does not own research request {request_id}")
            return None

        # Check if research is complete
        if session["status"] not in ["completed"]:
            logger.warning(f"Research request {request_id} is not complete")
            return None

        # Check if follow-up questions have already been generated
        if "follow_up_questions" in session:
            return session["follow_up_questions"]

        # Delegate to analysis agent
        try:
            # In a real implementation, this would call the analysis agent
            # For now, just create placeholder questions
            questions = await self._delegate_to_analysis_agent_for_followups(request_id)

            # Store in session
            session["follow_up_questions"] = questions

            return questions
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return None

    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the orchestrator and its agents

        Returns:
            Health status of the system
        """
        # Check orchestrator health
        health = {
            "status": "healthy",
            "orchestrator": "online",
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {},
        }

        # In a real implementation, we would check the health of each agent
        agent_types = ["retrieval", "analysis", "knowledge_graph", "citation", "ui"]
        for agent_type in agent_types:
            health["agents"][agent_type] = "online"

        return health

    async def _process_research(self, request_id: str):
        """
        Process a research request

        Args:
            request_id: ID of the research request
        """
        logger.info(f"Processing research request: {request_id}")

        if request_id not in self.research_sessions:
            logger.error(f"Research request {request_id} not found")
            return

        session = self.research_sessions[request_id]

        try:
            # Update status to in_progress
            session["status"] = "in_progress"
            session["assistant_state"] = {
                "emotion": "excited",
                "state": "thinking",
                "message": "Researching your topic...",
                "showBubble": True,
            }

            # Process research task
            # 1. Analyze query to identify focus areas
            focus_areas = await self._analyze_query(request_id)

            # Update progress
            session["progress"] = 10

            # Update focus areas
            if not session.get("focus_areas"):
                session["focus_areas"] = focus_areas

            # 2. Retrieve information for each focus area
            if session["continuous_mode"]:
                # Process all focus areas
                for i, focus_area in enumerate(focus_areas):
                    if session["status"] == "cancelled":
                        logger.info(f"Research request {request_id} was cancelled")
                        break

                    # Update assistant state
                    session["assistant_state"] = {
                        "emotion": "excited",
                        "state": "thinking",
                        "message": f"Researching {focus_area['topic']}...",
                        "showBubble": True,
                    }

                    # Retrieve information
                    sources = await self._retrieve_information(
                        request_id, focus_area["topic"]
                    )

                    # Update focus area with sources
                    focus_area["sources"] = sources

                    # Analyze information
                    summary, key_points = await self._analyze_information(
                        request_id, focus_area["topic"], sources
                    )

                    # Update focus area with analysis
                    focus_area["summary"] = summary
                    focus_area["key_points"] = key_points
                    focus_area["completed"] = True

                    # Update progress
                    progress_increment = 80 / len(focus_areas)
                    session["progress"] = min(10 + (i + 1) * progress_increment, 90)
            else:
                # Process only the first focus area
                if focus_areas:
                    focus_area = focus_areas[0]

                    # Update assistant state
                    session["assistant_state"] = {
                        "emotion": "excited",
                        "state": "thinking",
                        "message": f"Researching {focus_area['topic']}...",
                        "showBubble": True,
                    }

                    # Retrieve information
                    sources = await self._retrieve_information(
                        request_id, focus_area["topic"]
                    )

                    # Update focus area with sources
                    focus_area["sources"] = sources

                    # Analyze information
                    summary, key_points = await self._analyze_information(
                        request_id, focus_area["topic"], sources
                    )

                    # Update focus area with analysis
                    focus_area["summary"] = summary
                    focus_area["key_points"] = key_points
                    focus_area["completed"] = True

                    # Update progress
                    session["progress"] = 80

            # 3. Generate summary of all focus areas
            if session["status"] != "cancelled":
                # Update assistant state
                session["assistant_state"] = {
                    "emotion": "excited",
                    "state": "thinking",
                    "message": "Generating final summary...",
                    "showBubble": True,
                }

                # Generate summary
                session["summary"] = await self._generate_summary(request_id)

                # Update progress
                session["progress"] = 90

                # 4. Generate knowledge graph
                session["assistant_state"] = {
                    "emotion": "excited",
                    "state": "thinking",
                    "message": "Creating knowledge graph...",
                    "showBubble": True,
                }

                session["knowledge_graph"] = (
                    await self._delegate_to_knowledge_graph_agent(request_id)
                )

                # 5. Generate follow-up questions
                session["assistant_state"] = {
                    "emotion": "excited",
                    "state": "thinking",
                    "message": "Generating follow-up questions...",
                    "showBubble": True,
                }

                session["follow_up_questions"] = (
                    await self._delegate_to_analysis_agent_for_followups(request_id)
                )

                # 6. Mark as completed
                session["status"] = "completed"
                session["progress"] = 100
                session["end_time"] = datetime.utcnow().isoformat()
                session["assistant_state"] = {
                    "emotion": "happy",
                    "state": "presenting",
                    "message": "Research complete! Here are the results.",
                    "showBubble": True,
                }

                logger.info(f"Completed research request: {request_id}")

        except Exception as e:
            logger.error(f"Error processing research request {request_id}: {str(e)}")
            session["status"] = "failed"
            session["error"] = str(e)
            session["end_time"] = datetime.utcnow().isoformat()
            session["assistant_state"] = {
                "emotion": "sad",
                "state": "error",
                "message": "Sorry, there was an error with the research.",
                "showBubble": True,
            }

    async def _analyze_query(self, request_id: str) -> List[Dict[str, Any]]:
        """
        Analyze the query to identify focus areas

        Args:
            request_id: ID of the research request

        Returns:
            List of focus areas
        """
        logger.info(f"Analyzing query for research request: {request_id}")

        session = self.research_sessions[request_id]
        query = session["query"]

        try:
            # Import and use the real analysis agent
            from agents.research.analysis_agent import analysis_agent

            # Use the real analysis agent to analyze the query
            focus_areas = await analysis_agent.analyze_query(query)

            # If user provided specific focus areas, merge them
            if session.get("focus_areas"):
                user_focus_areas = []
                for topic in session["focus_areas"]:
                    user_focus_areas.append(
                        {
                            "topic": topic,
                            "priority": 1.0,
                            "patterns": [],
                            "sources": [],
                            "summary": "",
                            "key_points": [],
                            "completed": False,
                        }
                    )

                # Combine user-provided with AI-analyzed focus areas
                focus_areas = user_focus_areas + focus_areas

                # Remove duplicates while preserving order
                seen_topics = set()
                unique_focus_areas = []
                for fa in focus_areas:
                    topic_lower = fa["topic"].lower()
                    if topic_lower not in seen_topics:
                        seen_topics.add(topic_lower)
                        unique_focus_areas.append(fa)

                focus_areas = unique_focus_areas

            return focus_areas

        except Exception as e:
            logger.error(f"Error with analysis agent: {e}")
            # Fallback to simple focus area extraction
            focus_areas = []

            if session.get("focus_areas"):
                # Use provided focus areas
                for topic in session["focus_areas"]:
                    focus_areas.append(
                        {
                            "topic": topic,
                            "sources": [],
                            "summary": "",
                            "key_points": [],
                            "completed": False,
                        }
                    )
            else:
                # Simple approach: split by commas or "and"
                import re

                topics = re.split(r",|\s+and\s+", query)

                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        focus_areas.append(
                            {
                                "topic": topic,
                                "sources": [],
                                "summary": "",
                                "key_points": [],
                                "completed": False,
                            }
                        )

                # If no focus areas were identified, use the whole query
                if not focus_areas:
                    focus_areas.append(
                        {
                            "topic": query,
                            "sources": [],
                            "summary": "",
                            "key_points": [],
                            "completed": False,
                        }
                    )

            return focus_areas

    async def _retrieve_information(
        self, request_id: str, topic: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve information for a focus area

        Args:
            request_id: ID of the research request
            topic: Topic to research

        Returns:
            List of sources
        """
        logger.info(
            f"Retrieving information for topic '{topic}' in research request: {request_id}"
        )

        # Register retrieval agent activity
        self._register_agent_activity(
            request_id,
            "retrieval_agent",
            "working",
            f"Searching for information on '{topic}'",
        )

        try:
            # Import and use the real retrieval agent
            from agents.research.retrieval_agent import retrieval_agent

            # Get session info for parameters
            session = self.research_sessions[request_id]
            max_sources = session.get("max_sources", 10)
            depth_level = session.get("depth_level", "detailed")
            force_refresh = session.get("force_refresh", False)

            # Use the real retrieval agent
            sources_objects = await retrieval_agent.enhanced_search(
                query=topic, max_sources=max_sources, depth_level=depth_level
            )

            # Convert Source objects to dictionaries
            sources = [source.to_dict() for source in sources_objects]

            # Update retrieval agent activity
            self._register_agent_activity(
                request_id,
                "retrieval_agent",
                "completed",
                f"Found {len(sources)} sources for '{topic}'",
            )

            return sources

        except Exception as e:
            logger.error(f"Error with retrieval agent: {e}")
            # Fallback to mock sources
            sources = [
                {
                    "url": f"https://example.com/source1_{topic.replace(' ', '_')}",
                    "title": f"Example Source 1 for {topic}",
                    "snippet": f"This is a simulated source snippet for {topic}.",
                    "reliability_score": 0.85,
                    "site_name": "example.com",
                    "is_primary": True,
                },
                {
                    "url": f"https://example.com/source2_{topic.replace(' ', '_')}",
                    "title": f"Example Source 2 for {topic}",
                    "snippet": f"Another simulated source snippet for {topic}.",
                    "reliability_score": 0.75,
                    "site_name": "example.com",
                    "is_primary": False,
                },
            ]

            # Update retrieval agent activity
            self._register_agent_activity(
                request_id,
                "retrieval_agent",
                "completed",
                f"Found {len(sources)} sources for '{topic}' (fallback)",
            )

            return sources

    async def _analyze_information(
        self, request_id: str, topic: str, sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Analyze information for a focus area

        Args:
            request_id: ID of the research request
            topic: Topic being researched
            sources: List of sources

        Returns:
            Summary and key points
        """
        logger.info(
            f"Analyzing information for topic '{topic}' in research request: {request_id}"
        )

        # Register analysis agent activity
        self._register_agent_activity(
            request_id,
            "analysis_agent",
            "working",
            f"Analyzing information on '{topic}'",
        )

        try:
            # Import and use the real analysis agent
            from agents.research.analysis_agent import analysis_agent

            # Get research patterns from the current session
            session = self.research_sessions[request_id]
            query = session["query"]

            # Identify patterns for this topic
            patterns = []
            # Try to find patterns from focus areas if available
            focus_areas = session.get("focus_areas", [])
            for fa in focus_areas:
                if fa.get("topic", "").lower() == topic.lower():
                    patterns = fa.get("patterns", [])
                    break

            # Use the real analysis agent to synthesize information
            summary, key_points = await analysis_agent.synthesize_information(
                topic=topic, sources=sources, patterns=patterns
            )

            # Update analysis agent activity
            self._register_agent_activity(
                request_id,
                "analysis_agent",
                "completed",
                f"Analysis complete for '{topic}'",
            )

            return summary, key_points

        except Exception as e:
            logger.error(f"Error with analysis agent: {e}")

            # Fallback to simple analysis
            summary = f"Research on {topic} based on {len(sources)} sources. "
            if sources:
                high_quality_sources = [
                    s for s in sources if s.get("reliability_score", 0) > 0.7
                ]
                if high_quality_sources:
                    summary += f"Analysis includes {len(high_quality_sources)} high-quality sources. "

                # Extract snippets for basic summary
                snippets = [
                    s.get("snippet", "") for s in sources[:3] if s.get("snippet")
                ]
                if snippets:
                    summary += "Key information includes: " + " ".join(snippets[:2])

            # Generate basic key points
            key_points = []
            for i, source in enumerate(sources[:5]):
                title = source.get("title", f"Source {i+1}")
                key_points.append(
                    f"Information from {title}: {source.get('snippet', 'Relevant content available.')[:100]}"
                )

            if not key_points:
                key_points = [
                    f"Key information about {topic} from research sources.",
                    f"Multiple perspectives on {topic} are available in the literature.",
                ]

            # Update analysis agent activity
            self._register_agent_activity(
                request_id,
                "analysis_agent",
                "completed",
                f"Analysis complete for '{topic}' (fallback)",
            )

            return summary, key_points

    async def _generate_summary(self, request_id: str) -> str:
        """
        Generate a summary of all focus areas

        Args:
            request_id: ID of the research request

        Returns:
            Summary of all focus areas
        """
        logger.info(f"Generating summary for research request: {request_id}")

        session = self.research_sessions[request_id]
        focus_areas = session.get("focus_areas", [])

        # In a real implementation, this would call the analysis agent
        # Register analysis agent activity
        self._register_agent_activity(
            request_id, "analysis_agent", "working", "Generating final summary"
        )

        # For simulation, create placeholder summary
        await asyncio.sleep(1)  # Simulate processing delay

        if len(focus_areas) == 1:
            summary = focus_areas[0].get("summary", "")
        else:
            topics = [area.get("topic", "") for area in focus_areas]
            topics_str = (
                ", ".join(topics[:-1]) + " and " + topics[-1]
                if len(topics) > 1
                else topics[0]
            )

            summary = f"This research covers {topics_str}. In a real implementation, this would be a comprehensive summary that ties together all the focus areas."

        # Update analysis agent activity
        self._register_agent_activity(
            request_id, "analysis_agent", "completed", "Final summary generated"
        )

        return summary

    async def _delegate_to_knowledge_graph_agent(
        self, request_id: str
    ) -> Dict[str, Any]:
        """
        Delegate to the knowledge graph agent to generate a knowledge graph

        Args:
            request_id: ID of the research request

        Returns:
            Knowledge graph data
        """
        logger.info(
            f"Delegating to knowledge graph agent for research request: {request_id}"
        )

        session = self.research_sessions[request_id]
        query = session["query"]
        focus_areas = session.get("focus_areas", [])

        # Register knowledge graph agent activity
        self._register_agent_activity(
            request_id, "knowledge_graph_agent", "working", "Generating knowledge graph"
        )

        try:
            # Import and use the real knowledge graph agent
            from agents.research.knowledge_graph_agent import KnowledgeGraphAgent

            kg_agent = KnowledgeGraphAgent()

            # Prepare research data for the knowledge graph
            research_data = {
                "sources": session.get("sources", []),
                "analysis": session.get("summary", ""),
                "focus_areas": focus_areas,
            }

            # Generate knowledge graph
            knowledge_graph = await kg_agent.build_knowledge_graph(research_data, query)

            # Update knowledge graph agent activity
            self._register_agent_activity(
                request_id,
                "knowledge_graph_agent",
                "completed",
                "Knowledge graph generated",
            )

            return knowledge_graph

        except Exception as e:
            logger.error(f"Error with knowledge graph agent: {e}")

            # Fallback to placeholder knowledge graph
            await asyncio.sleep(1)  # Simulate processing delay

            main_topic = query

        # Create nodes
        nodes = [
            {
                "id": "main",
                "label": main_topic,
                "type": "topic",
                "description": f"Main research topic: {main_topic}",
                "confidence": 1.0,
                "size": 20,
            }
        ]

        # Add focus area nodes
        for i, focus_area in enumerate(focus_areas):
            nodes.append(
                {
                    "id": f"focus_{i}",
                    "label": focus_area.get("topic", ""),
                    "type": "concept",
                    "description": focus_area.get("summary", ""),
                    "confidence": 0.9,
                    "size": 15,
                }
            )

            # Add key point nodes
            for j, key_point in enumerate(focus_area.get("key_points", [])):
                nodes.append(
                    {
                        "id": f"point_{i}_{j}",
                        "label": key_point,
                        "type": "entity",
                        "confidence": 0.8,
                        "size": 10,
                    }
                )

            # Add source nodes
            for j, source in enumerate(focus_area.get("sources", [])):
                nodes.append(
                    {
                        "id": f"source_{i}_{j}",
                        "label": source.get("title", ""),
                        "type": "source",
                        "description": source.get("snippet", ""),
                        "confidence": source.get("reliability_score", 0.7),
                        "size": 8,
                    }
                )

        # Create edges
        edges = []

        # Connect main topic to focus areas
        for i in range(len(focus_areas)):
            edges.append(
                {
                    "source": "main",
                    "target": f"focus_{i}",
                    "label": "includes",
                    "type": "relates_to",
                    "weight": 1.0,
                    "confidence": 0.9,
                }
            )

            # Connect focus areas to key points
            for j in range(len(focus_areas[i].get("key_points", []))):
                edges.append(
                    {
                        "source": f"focus_{i}",
                        "target": f"point_{i}_{j}",
                        "label": "contains",
                        "type": "relates_to",
                        "weight": 0.8,
                        "confidence": 0.8,
                    }
                )

            # Connect focus areas to sources
            for j in range(len(focus_areas[i].get("sources", []))):
                edges.append(
                    {
                        "source": f"focus_{i}",
                        "target": f"source_{i}_{j}",
                        "label": "cites",
                        "type": "cites",
                        "weight": 0.7,
                        "confidence": focus_areas[i]
                        .get("sources", [])[j]
                        .get("reliability_score", 0.7),
                    }
                )

        # Create knowledge graph
        knowledge_graph = {"nodes": nodes, "edges": edges, "main_topic": main_topic}

        # Update knowledge graph agent activity
        self._register_agent_activity(
            request_id,
            "knowledge_graph_agent",
            "completed",
            "Knowledge graph generated",
        )

        return knowledge_graph

    async def _delegate_to_citation_agent(
        self, request_id: str, style: str
    ) -> List[Dict[str, Any]]:
        """
        Delegate to the citation agent to generate citations

        Args:
            request_id: ID of the research request
            style: Citation style (apa, mla, chicago, ieee)

        Returns:
            List of citations
        """
        logger.info(
            f"Delegating to citation agent for research request: {request_id} with style: {style}"
        )

        session = self.research_sessions[request_id]
        focus_areas = session.get("focus_areas", [])

        # Register citation agent activity
        self._register_agent_activity(
            request_id, "citation_agent", "working", f"Generating {style} citations"
        )

        try:
            # Import and use the real citation agent
            from agents.research.citation_agent import CitationAgent

            citation_agent = CitationAgent()

            # Collect all sources from focus areas
            all_sources = []
            for focus_area in focus_areas:
                all_sources.extend(focus_area.get("sources", []))

            # Process sources with citation agent
            citation_result = await citation_agent.process_sources(
                all_sources, session["query"]
            )

            # Generate bibliography in requested style
            bibliography = await citation_agent.generate_bibliography(
                [citation["id"] for citation in citation_result.get("citations", [])],
                style,
            )

            # Convert to expected format
            citations = []
            for citation_entry in bibliography.get("bibliography", []):
                citations.append(
                    {
                        "source_id": citation_entry["id"],
                        "source_url": citation_entry["url"],
                        "citation_text": citation_entry["formatted_citation"],
                        "style": style,
                        "reliability_score": citation_entry["reliability_score"],
                        "reliability_category": citation_entry["reliability_category"],
                        "access_date": citation_entry["access_date"],
                    }
                )

            # Update citation agent activity
            self._register_agent_activity(
                request_id,
                "citation_agent",
                "completed",
                f"Generated {len(citations)} {style} citations",
            )

            return citations

        except Exception as e:
            logger.error(f"Error with citation agent: {e}")

            # Fallback to simplified citations
            await asyncio.sleep(1)  # Simulate processing delay

            citations = []

            # Generate citations for all sources
            for i, focus_area in enumerate(focus_areas):
                for j, source in enumerate(focus_area.get("sources", [])):
                    url = source.get("url", "")
                    title = source.get("title", "")
                    domain = source.get("domain", "")

                    citation_text = ""
                    if style == "apa":
                        citation_text = (
                            f"Author, A. (2024). {title}. Retrieved from {url}"
                        )
                    elif style == "mla":
                        citation_text = f'Author. "{title}." {domain}, 2024, {url}.'
                    elif style == "chicago":
                        citation_text = f"Author. {title}. {domain}, 2024. {url}."
                    elif style == "ieee":
                        citation_text = f'[{j+1}] Author, "{title}," {domain}, 2024. [Online]. Available: {url}.'
                    else:
                        citation_text = f"{title}. {url}"

                    citations.append(
                        {
                            "source_id": f"source_{i}_{j}",
                            "source_url": url,
                            "citation_text": citation_text,
                            "style": style,
                            "authors": ["Author"],
                            "title": title,
                            "published_date": "2024",
                            "publisher": domain,
                        }
                    )

            # Update citation agent activity
            self._register_agent_activity(
                request_id,
                "citation_agent",
                "completed",
                f"Generated {len(citations)} {style} citations (fallback)",
            )

            return citations

    async def _delegate_to_analysis_agent_for_followups(
        self, request_id: str
    ) -> List[str]:
        """
        Delegate to the analysis agent to generate follow-up questions

        Args:
            request_id: ID of the research request

        Returns:
            List of follow-up questions
        """
        logger.info(
            f"Delegating to analysis agent for follow-up questions for research request: {request_id}"
        )

        session = self.research_sessions[request_id]
        query = session["query"]
        focus_areas = session.get("focus_areas", [])

        # Register analysis agent activity
        self._register_agent_activity(
            request_id, "analysis_agent", "working", "Generating follow-up questions"
        )

        try:
            # Import and use the real analysis agent
            from agents.research.analysis_agent import analysis_agent

            # Get summary and collect data from focus areas
            summary = session.get("summary", "")
            all_key_points = []
            all_sources = []

            for fa in focus_areas:
                all_key_points.extend(fa.get("key_points", []))
                all_sources.extend(fa.get("sources", []))

            # Use the real analysis agent to generate follow-up questions
            questions = await analysis_agent.generate_followup_questions(
                topic=query,
                summary=summary,
                key_points=all_key_points,
                sources=all_sources,
            )

            # Update analysis agent activity
            self._register_agent_activity(
                request_id,
                "analysis_agent",
                "completed",
                f"Generated {len(questions)} follow-up questions",
            )

            return questions

        except Exception as e:
            logger.error(f"Error with analysis agent for follow-ups: {e}")

            # Fallback to simple follow-up questions
            questions = [
                f"What are the implications of {query}?",
                f"How does {query} compare to similar topics?",
                f"What are the practical applications of {query}?",
                f"What challenges exist with {query}?",
                f"What future trends are expected for {query}?",
            ]

            # Add focus area specific questions
            for focus_area in focus_areas:
                topic = focus_area.get("topic", "")
                if topic and topic != query:
                    questions.append(f"What are the key challenges in {topic}?")

            # Update analysis agent activity
            self._register_agent_activity(
                request_id,
                "analysis_agent",
                "completed",
                f"Generated {len(questions)} follow-up questions (fallback)",
            )

            return questions[:5]  # Limit to 5 questions

    def _register_agent_activity(
        self,
        request_id: str,
        agent_id: str,
        status: str,
        action: str,
        progress: float = None,
    ):
        """
        Register agent activity

        Args:
            request_id: ID of the research request
            agent_id: ID of the agent
            status: Status of the agent (working, completed, error)
            action: Current action being performed
            progress: Progress percentage (0-100)
        """
        if request_id not in self.research_sessions:
            logger.warning(
                f"Cannot register agent activity: Research request {request_id} not found"
            )
            return

        session = self.research_sessions[request_id]

        # Initialize agents dict if not exists
        if "agents" not in session:
            session["agents"] = {}

        # Get current time
        now = datetime.utcnow().isoformat()

        # Get agent type from ID
        agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id

        # Check if agent already exists
        if agent_id in session["agents"]:
            # Update existing agent
            agent = session["agents"][agent_id]
            agent["status"] = status
            agent["action"] = action

            if progress is not None:
                agent["progress"] = progress

            if status == "completed" or status == "error":
                agent["end_time"] = now
        else:
            # Create new agent activity
            agent = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "status": status,
                "action": action,
                "progress": progress or 0,
                "start_time": now,
                "end_time": now if status == "completed" or status == "error" else None,
            }

            session["agents"][agent_id] = agent

        logger.info(f"Registered agent activity: {agent_id} - {status} - {action}")

    def _get_agent_activities(self, request_id: str) -> List[Dict[str, Any]]:
        """
        Get current agent activities for a research request

        Args:
            request_id: ID of the research request

        Returns:
            List of agent activities
        """
        if request_id not in self.research_sessions:
            logger.warning(
                f"Cannot get agent activities: Research request {request_id} not found"
            )
            return []

        session = self.research_sessions[request_id]
        agents = session.get("agents", {})

        activities = []
        for agent_id, agent in agents.items():
            activities.append(
                {
                    "agent_id": agent_id,
                    "agent_type": agent.get("agent_type", "unknown"),
                    "status": agent.get("status", "unknown"),
                    "action": agent.get("action", ""),
                    "progress": agent.get("progress", 0),
                    "message": None,  # Not used in this implementation
                    "started_at": agent.get("start_time", ""),
                    "completed_at": agent.get("end_time"),
                }
            )

        return activities


# ADK Integration Hooks
if ADK_AVAILABLE:

    class OrchestratorAgentADK(Agent):
        """ADK wrapper for OrchestratorAgent"""

        def __init__(self):
            super().__init__()
            self.inner_agent = OrchestratorAgent()

        def initialize(self, context: AgentContext) -> None:
            self.context = context
            self.inner_agent.initialize(context)

        @action
        def start_research(
            self, query: str, continuous_mode: bool = False, force_refresh: bool = False
        ) -> Dict:
            """Start a research process"""
            import asyncio

            request = {
                "query": query,
                "continuous_mode": continuous_mode,
                "force_refresh": force_refresh,
            }
            request_id = str(uuid.uuid4())
            user_id = "adk_user"

            # Start research
            result = asyncio.run(
                self.inner_agent.handle_research_request(request, request_id, user_id)
            )

            # Wait for completion
            status = asyncio.run(
                self.inner_agent.get_research_status(request_id, user_id)
            )
            while status and status["status"] in ["pending", "in_progress"]:
                import time

                time.sleep(1)
                status = asyncio.run(
                    self.inner_agent.get_research_status(request_id, user_id)
                )

            # Get results
            results = asyncio.run(
                self.inner_agent.get_research_results(request_id, user_id)
            )

            # Get knowledge graph
            graph = asyncio.run(
                self.inner_agent.generate_knowledge_graph(request_id, user_id)
            )

            # Get citations
            citations = asyncio.run(
                self.inner_agent.generate_citations(request_id, "apa", user_id)
            )

            # Get follow-up questions
            questions = asyncio.run(
                self.inner_agent.get_follow_up_questions(request_id, user_id)
            )

            # Combine results
            return {
                "research": results,
                "knowledge_graph": graph,
                "citations": citations,
                "follow_up_questions": questions,
                "request_id": request_id,
            }

        @action
        def get_research_status(self, request_id: str) -> Dict:
            """Get the status of a research request"""
            import asyncio

            return asyncio.run(
                self.inner_agent.get_research_status(request_id, "adk_user")
            )

        @action
        def generate_followup_questions(
            self, query: str, research_result: Dict
        ) -> List[str]:
            """Generate follow-up questions for completed research"""
            # ADK specific implementation
            request_id = research_result.get("request_id")
            import asyncio

            questions = asyncio.run(
                self.inner_agent.get_follow_up_questions(request_id, "adk_user")
            )
            return questions or [
                f"What are the implications of {query}?",
                f"How does {query} compare to previous research?",
                f"What are the limitations of the current research on {query}?",
            ]

    # Expose ADK agent
    adk_agent = OrchestratorAgentADK
else:
    # Export functions for non-ADK usage
    def adk_init(config):
        """Initialize for ADK"""
        return OrchestratorAgent(config)

    def adk_process(agent, message, context):
        """Process message for ADK"""
        # TODO: Implement ADK processing
        return {"response": "Not yet implemented"}


# Direct implementation entry point (for testing)
if __name__ == "__main__":
    # Create orchestrator agent
    orchestrator = OrchestratorAgent()

    # Test with a sample request
    async def test():
        request = ResearchRequest(
            query="Artificial Intelligence Ethics", continuous_mode=True
        )

        request_id = "test_request_123"
        user_id = "test_user_456"

        # Start research
        result = await orchestrator.handle_research_request(
            request, request_id, user_id
        )
        print(f"Initial result: {result}")

        # Wait for processing
        print("Processing research...")
        await asyncio.sleep(5)

        # Get status
        status = await orchestrator.get_research_status(request_id, user_id)
        print(f"Status: {status}")

        # Wait for completion
        while status and status["status"] in ["pending", "in_progress"]:
            print(f"Progress: {status['progress']}%")
            await asyncio.sleep(1)
            status = await orchestrator.get_research_status(request_id, user_id)

        # Get results
        results = await orchestrator.get_research_results(request_id, user_id)
        print(f"Results: {results}")

        # Get knowledge graph
        graph = await orchestrator.generate_knowledge_graph(request_id, user_id)
        print(
            f"Knowledge graph nodes: {len(graph['nodes'])} edges: {len(graph['edges'])}"
        )

        # Get citations
        citations = await orchestrator.generate_citations(request_id, "apa", user_id)
        print(f"Citations: {citations}")

        # Get follow-up questions
        questions = await orchestrator.get_follow_up_questions(request_id, user_id)
        print(f"Follow-up questions: {questions}")

    # Run the test
    asyncio.run(test())
