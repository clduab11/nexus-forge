"""
ADK Service Integration for Parallax Pal

This service acts as a bridge between the FastAPI backend and the ADK agent system,
providing a clean interface for research operations and WebSocket management for Starri.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from adk_config import AGENT_CONFIG, DEV_MODE

# Import orchestrator agent
from agents.orchestrator.orchestrator_agent import OrchestratorAgent
from schemas.agent_messages import (
    Citation,
    KnowledgeGraphData,
    ResearchRequest,
    ResearchResponse,
    ResearchStatus,
)

# Import monitoring
from ..monitoring.cloud_monitoring import CloudMonitoringService
from ..monitoring.monitoring_middleware import AgentMonitoringMixin

# Set up logging
logging.basicConfig(
    level=logging.INFO if DEV_MODE else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adk_service")


class ADKService(AgentMonitoringMixin):
    """
    Service layer for ADK integration with Parallax Pal backend.

    This service manages the connection to ADK agents and provides
    methods for research operations through the Starri interface.
    """

    def __init__(self):
        """Initialize the ADK service"""
        super().__init__()  # Initialize monitoring mixin
        self.orchestrator = OrchestratorAgent()
        self.initialized = False
        self.monitoring = CloudMonitoringService()
        logger.info("ADK Service created")

    async def initialize(self):
        """Initialize the ADK service"""
        if not self.initialized:
            # Initialize orchestrator if needed
            self.orchestrator.initialize()
            self.initialized = True
            logger.info("ADK Service initialized")

    @property
    def monitored_start_research(self):
        return self.monitored_invoke(self._start_research)

    async def start_research(
        self,
        query: str,
        user_id: int,
        continuous_mode: bool = False,
        force_refresh: bool = False,
        max_sources: Optional[int] = None,
        depth_level: str = "detailed",
        focus_areas: Optional[List[str]] = None,
    ) -> str:
        """Start a research request with monitoring."""
        return await self.monitored_start_research(
            query,
            user_id,
            continuous_mode,
            force_refresh,
            max_sources,
            depth_level,
            focus_areas,
        )

    async def _start_research(
        self,
        query: str,
        user_id: int,
        continuous_mode: bool = False,
        force_refresh: bool = False,
        max_sources: Optional[int] = None,
        depth_level: str = "detailed",
        focus_areas: Optional[List[str]] = None,
    ) -> str:
        """
        Start a research request

        Args:
            query: Research query
            user_id: ID of the user making the request
            continuous_mode: Whether to research all focus areas
            force_refresh: Whether to ignore cache
            max_sources: Maximum number of sources to retrieve
            depth_level: Depth of research (basic, detailed, comprehensive)
            focus_areas: Specific focus areas to research

        Returns:
            Request ID for tracking the research
        """
        await self.initialize()

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Create research request
        request = {
            "query": query,
            "continuous_mode": continuous_mode,
            "force_refresh": force_refresh,
            "max_sources": max_sources or 20,
            "depth_level": depth_level,
            "focus_areas": focus_areas or [],
        }

        logger.info(f"Starting research for user {user_id}: {query}")

        # Record research query metrics
        self.monitoring.increment_counter(
            "research_queries_total",
            labels={
                "agent": "orchestrator",
                "status": "started",
                "user_tier": "standard",  # This should come from user data
            },
        )

        # Record query complexity
        complexity = "simple" if len(query.split()) < 10 else "complex"
        self.monitoring.observe_histogram(
            "research_query_duration",
            0.0,  # Will be updated when complete
            labels={"agent": "orchestrator", "complexity": complexity},
        )

        # Start research through orchestrator
        result = await self.orchestrator.handle_research_request(
            request, request_id, str(user_id)
        )

        return request_id

    async def get_research_status(
        self, request_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the status of a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Research status or None if not found
        """
        await self.initialize()

        logger.info(f"Getting research status for user {user_id}, request {request_id}")

        return await self.orchestrator.get_research_status(request_id, str(user_id))

    async def get_research_results(
        self, request_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the results of a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Research results or None if not found
        """
        await self.initialize()

        logger.info(
            f"Getting research results for user {user_id}, request {request_id}"
        )

        return await self.orchestrator.get_research_results(request_id, str(user_id))

    async def cancel_research(self, request_id: str, user_id: int) -> bool:
        """
        Cancel a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            True if cancellation succeeded, False otherwise
        """
        await self.initialize()

        logger.info(f"Cancelling research for user {user_id}, request {request_id}")

        return await self.orchestrator.cancel_research(request_id, str(user_id))

    async def generate_knowledge_graph(
        self, request_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a knowledge graph for a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            Knowledge graph data or None if not available
        """
        await self.initialize()

        logger.info(
            f"Generating knowledge graph for user {user_id}, request {request_id}"
        )

        return await self.orchestrator.generate_knowledge_graph(
            request_id, str(user_id)
        )

    async def generate_citations(
        self, request_id: str, style: str, user_id: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate citations for a research request

        Args:
            request_id: ID of the research request
            style: Citation style (apa, mla, chicago, ieee)
            user_id: ID of the user making the request

        Returns:
            List of citations or None if not available
        """
        await self.initialize()

        logger.info(
            f"Generating {style} citations for user {user_id}, request {request_id}"
        )

        return await self.orchestrator.generate_citations(
            request_id, style, str(user_id)
        )

    async def get_follow_up_questions(
        self, request_id: str, user_id: int
    ) -> Optional[List[str]]:
        """
        Get follow-up questions for a research request

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            List of follow-up questions or None if not available
        """
        await self.initialize()

        logger.info(
            f"Getting follow-up questions for user {user_id}, request {request_id}"
        )

        return await self.orchestrator.get_follow_up_questions(request_id, str(user_id))

    async def generate_followup_questions(
        self, request_id: str, user_id: int
    ) -> Optional[List[str]]:
        """
        Generate follow-up questions for a research request (alias for compatibility)

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request

        Returns:
            List of follow-up questions or None if not available
        """
        return await self.get_follow_up_questions(request_id, user_id)

    async def get_citations(
        self, request_id: str, user_id: int, style: str = "apa"
    ) -> Optional[Dict[str, Any]]:
        """
        Get citations for a research request (alias for compatibility)

        Args:
            request_id: ID of the research request
            user_id: ID of the user making the request
            style: Citation style (apa, mla, chicago, ieee)

        Returns:
            Citations data or None if not available
        """
        citations = await self.generate_citations(request_id, style, user_id)
        if citations:
            return {"citations": citations, "style": style}
        return None

    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the ADK system

        Returns:
            Health status of the ADK system
        """
        await self.initialize()

        logger.info("Checking ADK system health")

        return await self.orchestrator.check_health()

    def get_active_sessions(self) -> List[str]:
        """
        Get list of active research sessions

        Returns:
            List of active request IDs
        """
        if not self.initialized:
            return []

        return list(self.orchestrator.research_sessions.keys())

    def get_session_info(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session

        Args:
            request_id: ID of the research request

        Returns:
            Session information or None if not found
        """
        if (
            not self.initialized
            or request_id not in self.orchestrator.research_sessions
        ):
            return None

        session = self.orchestrator.research_sessions[request_id]

        # Return public session info
        return {
            "request_id": request_id,
            "query": session.get("query", ""),
            "status": session.get("status", "unknown"),
            "progress": session.get("progress", 0),
            "start_time": session.get("start_time", ""),
            "end_time": session.get("end_time"),
            "agents_count": len(session.get("agents", {})),
        }

    def cleanup_completed_sessions(self, max_age_minutes: int = 60):
        """
        Clean up completed research sessions older than max_age_minutes

        Args:
            max_age_minutes: Maximum age of completed sessions to keep
        """
        if not self.initialized:
            return

        current_time = datetime.utcnow()
        sessions_to_remove = []

        for request_id, session in self.orchestrator.research_sessions.items():
            if session.get("status") in ["completed", "failed", "cancelled"]:
                end_time_str = session.get("end_time")
                if end_time_str:
                    try:
                        end_time = datetime.fromisoformat(
                            end_time_str.replace("Z", "+00:00")
                        )
                        age_minutes = (current_time - end_time).total_seconds() / 60

                        if age_minutes > max_age_minutes:
                            sessions_to_remove.append(request_id)
                    except (ValueError, TypeError):
                        # If we can't parse the time, remove it anyway
                        sessions_to_remove.append(request_id)

        # Remove old sessions
        for request_id in sessions_to_remove:
            del self.orchestrator.research_sessions[request_id]
            logger.info(f"Cleaned up old session: {request_id}")

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old research sessions")


# Global ADK service instance
adk_service = None


def get_adk_service() -> ADKService:
    """
    Get the global ADK service instance

    Returns:
        The ADK service instance
    """
    global adk_service
    if adk_service is None:
        adk_service = ADKService()
    return adk_service


# Background task for cleanup
async def cleanup_task():
    """Background task to clean up old research sessions"""
    service = get_adk_service()

    while True:
        try:
            # Clean up sessions older than 1 hour
            service.cleanup_completed_sessions(max_age_minutes=60)

            # Sleep for 30 minutes before next cleanup
            await asyncio.sleep(30 * 60)
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            # Sleep for 5 minutes on error before retrying
            await asyncio.sleep(5 * 60)


# Start cleanup task if running as main module
if __name__ == "__main__":

    async def main():
        # Test the ADK service
        service = get_adk_service()

        # Start a test research
        request_id = await service.start_research(
            query="Machine Learning in Healthcare", user_id=123, continuous_mode=True
        )

        print(f"Started research with ID: {request_id}")

        # Monitor progress
        while True:
            status = await service.get_research_status(request_id, 123)
            if not status:
                print("Research not found")
                break

            print(f"Status: {status['status']}, Progress: {status['progress']}%")

            if status["status"] in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(2)

        # Get results if completed
        if status and status["status"] == "completed":
            results = await service.get_research_results(request_id, 123)
            print(f"Research completed: {results['summary']}")

            # Get knowledge graph
            graph = await service.generate_knowledge_graph(request_id, 123)
            print(
                f"Knowledge graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges"
            )

            # Get citations
            citations = await service.generate_citations(request_id, "apa", 123)
            print(f"Generated {len(citations)} citations")

            # Get follow-up questions
            questions = await service.get_follow_up_questions(request_id, 123)
            print(f"Follow-up questions: {questions}")

        # Check health
        health = await service.check_health()
        print(f"System health: {health}")

    asyncio.run(main())
