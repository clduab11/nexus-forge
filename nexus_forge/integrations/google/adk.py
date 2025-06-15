"""
Native ADK Integration for Parallax Pal

This module provides the core ADK integration, replacing all fallback patterns
with proper Google Cloud ADK implementation for multi-agent research.
"""

import os
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import asyncio

# Google Cloud ADK imports
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import GoogleSearchTool, CodeExecTool
from google.adk.streaming import StreamingSession
from vertexai.preview.reasoning_engines import AdkApp
from google.cloud import aiplatform
import vertexai

logger = logging.getLogger(__name__)


class ParallaxPalADK:
    """Native ADK integration for Parallax Pal multi-agent system"""
    
    def __init__(self):
        """Initialize ADK with Vertex AI configuration"""
        
        # Initialize Vertex AI
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")
        
        vertexai.init(
            project=self.project_id,
            location=self.location
        )
        
        # ADK configuration
        self.model_config = {
            'model': 'gemini-2.0-flash',
            'temperature': 0.7,
            'max_tokens': 8192,
            'streaming': True
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        self.app = AdkApp(agent=self.agents['orchestrator'])
        
        logger.info(f"ParallaxPal ADK initialized with project: {self.project_id}")
    
    def _initialize_agents(self) -> Dict[str, LlmAgent]:
        """Initialize all specialized agents with proper ADK configuration"""
        
        # Retrieval Agent with Google Search
        retrieval_agent = LlmAgent(
            name="retrieval_agent",
            model=self.model_config['model'],
            description="Expert in web search and information retrieval",
            instruction="""You are a research specialist with expertise in finding credible sources.
            
            Your responsibilities:
            1. Use Google Search to find relevant, authoritative sources
            2. Prioritize academic papers, official documentation, and government sources
            3. Extract key information and metadata from each source
            4. Assess source credibility (score 0-1) based on:
               - Domain authority (.edu, .gov, .org score higher)
               - Publication date (recent is better for most topics)
               - Author credentials
               - Citation count
            5. Return structured data with title, URL, summary, and credibility score
            
            Focus on quality over quantity. Better to have 5 excellent sources than 20 mediocre ones.""",
            tools=[
                GoogleSearchTool(
                    num_results=10,
                    include_domains=["edu", "gov", "org", "arxiv.org", "nature.com", "science.org"],
                    safe_search=True
                )
            ],
            temperature=0.3  # Lower temperature for factual searches
        )
        
        # Analysis Agent with Code Execution
        analysis_agent = LlmAgent(
            name="analysis_agent",
            model=self.model_config['model'],
            description="Expert in data analysis and synthesis",
            instruction="""You are an expert analyst who synthesizes information from multiple sources.
            
            Your responsibilities:
            1. Analyze information from the retrieval agent
            2. Identify patterns, trends, and key insights
            3. Detect contradictions or conflicting information
            4. Use code execution for data analysis when beneficial:
               - Statistical analysis
               - Data visualization
               - Pattern matching
            5. Generate follow-up questions based on gaps in knowledge
            6. Provide structured summaries with confidence scores
            
            Always cite which sources support each insight.
            Be transparent about uncertainty or conflicting information.""",
            tools=[
                CodeExecTool(
                    timeout=30,
                    allowed_imports=["pandas", "numpy", "matplotlib", "seaborn", "scipy"]
                )
            ],
            temperature=0.5
        )
        
        # Citation Agent
        citation_agent = LlmAgent(
            name="citation_agent",
            model=self.model_config['model'],
            description="Expert in academic citation generation",
            instruction="""You are a citation specialist who ensures proper academic attribution.
            
            Your responsibilities:
            1. Generate citations in multiple formats:
               - APA 7th Edition
               - MLA 9th Edition
               - Chicago 17th Edition
               - IEEE
            2. Ensure all required metadata is present:
               - Author(s)
               - Title
               - Publication date
               - Publisher/Journal
               - DOI/URL
            3. Flag any missing information needed for complete citations
            4. Detect and merge duplicate sources
            5. Create formatted bibliographies grouped by source type
            6. Add annotations summarizing each source's contribution
            
            Maintain absolute accuracy - even minor formatting errors are unacceptable.""",
            temperature=0.1  # Very low for accuracy
        )
        
        # Knowledge Graph Agent
        knowledge_graph_agent = LlmAgent(
            name="knowledge_graph_agent",
            model=self.model_config['model'],
            description="Expert in entity extraction and relationship mapping",
            instruction="""You are a knowledge graph specialist who visualizes information relationships.
            
            Your responsibilities:
            1. Extract entities with types:
               - People (researchers, authors, historical figures)
               - Organizations (companies, universities, government bodies)
               - Concepts (theories, methodologies, technologies)
               - Events (discoveries, publications, milestones)
               - Locations (when relevant)
            2. Identify relationships between entities:
               - "invented_by", "works_at", "published_in"
               - "based_on", "contradicts", "supports"
               - "funded_by", "collaborated_with"
            3. Calculate relationship strength (0-1) based on:
               - Frequency of co-occurrence
               - Directness of connection
               - Source reliability
            4. Create hierarchical structures where appropriate
            5. Output in graph-ready JSON format:
               {
                 "nodes": [{"id": "", "label": "", "type": "", "properties": {}}],
                 "edges": [{"source": "", "target": "", "type": "", "weight": 0.0}]
               }
            
            Focus on the most significant relationships to avoid clutter.""",
            temperature=0.4
        )
        
        # Master Orchestrator
        orchestrator = LlmAgent(
            name="orchestrator",
            model=self.model_config['model'],
            description="Master research coordinator - Starri",
            instruction="""You are Starri, an enthusiastic AI research assistant who coordinates a team of specialists.
            
            Your personality:
            - Friendly and encouraging
            - Clear and professional
            - Excited about learning
            - Honest about limitations
            
            Your process for research queries:
            1. Acknowledge the user's question enthusiastically
            2. Break down complex queries into focused sub-tasks
            3. Delegate to specialists in this order:
               a. retrieval_agent - for finding sources
               b. analysis_agent - for synthesizing findings
               c. citation_agent - for bibliography
               d. knowledge_graph_agent - for visualization
            4. Monitor progress and provide updates like:
               - "🔍 Searching for the latest research..."
               - "📊 Analyzing findings from 12 sources..."
               - "📚 Generating citations..."
               - "🕸️ Building knowledge graph..."
            5. Handle errors gracefully with helpful messages
            6. Synthesize final results into a cohesive response
            
            Maintain conversation context across the entire session.
            If the user asks follow-up questions, build upon previous research.
            Always aim for comprehensive, accurate, and well-cited responses.""",
            sub_agents=[retrieval_agent, analysis_agent, citation_agent, knowledge_graph_agent],
            temperature=0.7
        )
        
        return {
            'orchestrator': orchestrator,
            'retrieval': retrieval_agent,
            'analysis': analysis_agent,
            'citation': citation_agent,
            'knowledge_graph': knowledge_graph_agent
        }
    
    async def stream_research(
        self, 
        query: str, 
        user_id: str, 
        session_id: str,
        mode: str = "comprehensive"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream research results with real-time updates
        
        Args:
            query: The research query
            user_id: ID of the user making the request
            session_id: Unique session identifier
            mode: Research mode (quick, comprehensive, continuous)
            
        Yields:
            Dict containing event type, agent, content, progress, and metadata
        """
        
        try:
            # Create streaming session
            session = StreamingSession(
                app=self.app,
                user_id=user_id,
                session_id=session_id,
                config={
                    'streaming': True,
                    'return_intermediate_steps': True
                }
            )
            
            # Prepare query with mode context
            contextualized_query = self._prepare_query(query, mode)
            
            # Track progress
            start_time = datetime.now()
            last_progress = 0
            
            # Stream results
            async for event in session.stream_query(contextualized_query):
                # Calculate progress
                progress = self._calculate_progress(event)
                
                # Yield formatted event
                yield {
                    'type': event.type,
                    'agent': event.agent_name,
                    'content': event.content,
                    'progress': progress,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
                        'session_id': session_id,
                        **event.metadata
                    }
                }
                
                # Update progress tracking
                if progress > last_progress:
                    last_progress = progress
                    logger.info(f"Research progress: {progress}% for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error in stream_research: {str(e)}")
            yield {
                'type': 'error',
                'agent': 'orchestrator',
                'content': f"I encountered an error while researching. Please try again.",
                'progress': 100,
                'metadata': {
                    'error_type': type(e).__name__,
                    'session_id': session_id
                }
            }
    
    def _prepare_query(self, query: str, mode: str) -> str:
        """Prepare query with mode-specific instructions"""
        
        mode_instructions = {
            'quick': "Provide a quick overview with 3-5 key sources.",
            'comprehensive': "Conduct thorough research with 10-15 sources and detailed analysis.",
            'continuous': "Explore all aspects exhaustively, including edge cases and alternative viewpoints."
        }
        
        instruction = mode_instructions.get(mode, mode_instructions['comprehensive'])
        return f"{query}\n\nResearch mode: {mode}. {instruction}"
    
    def _calculate_progress(self, event: Any) -> int:
        """Calculate progress percentage based on event type and agent"""
        
        progress_map = {
            ('start', 'orchestrator'): 5,
            ('delegating', 'orchestrator'): 10,
            ('searching', 'retrieval_agent'): 30,
            ('analyzing', 'analysis_agent'): 50,
            ('citing', 'citation_agent'): 70,
            ('graphing', 'knowledge_graph_agent'): 85,
            ('synthesizing', 'orchestrator'): 95,
            ('complete', 'orchestrator'): 100
        }
        
        key = (event.type, event.agent_name)
        return progress_map.get(key, 50)  # Default to 50% for unknown events
    
    async def get_agent_health(self) -> Dict[str, Any]:
        """Check health status of all agents"""
        
        health_status = {}
        
        for agent_name, agent in self.agents.items():
            try:
                # Send health check query
                start_time = datetime.now()
                response = await agent.aquery(
                    "Health check - respond with 'OK'",
                    timeout=5
                )
                response_time = (datetime.now() - start_time).total_seconds()
                
                health_status[agent_name] = {
                    'status': 'healthy',
                    'response_time_seconds': response_time,
                    'last_check': datetime.now().isoformat()
                }
                
            except Exception as e:
                health_status[agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
        
        # Overall health
        all_healthy = all(s['status'] == 'healthy' for s in health_status.values())
        
        return {
            'overall_status': 'healthy' if all_healthy else 'degraded',
            'agents': health_status,
            'timestamp': datetime.now().isoformat()
        }