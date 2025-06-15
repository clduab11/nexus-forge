"""
Analysis Agent for ParallaxMind

This agent is responsible for analyzing queries, extracting research focus areas,
and synthesizing information from multiple sources into cohesive summaries.
"""

import json
import logging
import uuid
import re
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Try to import ADK-specific libraries, fallback if not available  
try:
    from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action
    from google.cloud.aiplatform import Model
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    # Fallback to simple class for local development
    class Agent:
        def __init__(self):
            pass

# Set up logging
logger = logging.getLogger("analysis_agent")

class AnalysisAgent(Agent):
    """
    Analysis Agent for ParallaxMind
    
    This agent is responsible for analyzing research queries to identify focus areas,
    determining research priorities, and synthesizing information from various sources
    into comprehensive summaries.
    """
    
    def __init__(self):
        """Initialize analysis agent with configuration."""
        if ADK_AVAILABLE:
            super().__init__()
        
        self.logger = logging.getLogger("analysis_agent")
        self.max_focus_areas = 5
        self.min_key_points = 2
        self.max_key_points = 8
        self.summary_max_length = 2000
        
        # Common research patterns and keywords
        self.research_patterns = {
            'comparison': ['vs', 'versus', 'compared to', 'difference between', 'compare'],
            'analysis': ['analyze', 'analysis', 'examine', 'evaluate', 'assess'],
            'trends': ['trend', 'trends', 'development', 'evolution', 'changes'],
            'impact': ['impact', 'effect', 'influence', 'consequence', 'result'],
            'methods': ['method', 'approach', 'technique', 'strategy', 'process'],
            'benefits': ['benefit', 'advantage', 'pros', 'positive'],
            'challenges': ['challenge', 'problem', 'issue', 'difficulty', 'cons'],
            'future': ['future', 'prediction', 'forecast', 'outlook', 'prospects']
        }
        
        self.logger.info("Analysis agent initialized")
    
    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Analyze a research query to identify focus areas
        
        Args:
            query: The research query to analyze
            context: Optional context information
            
        Returns:
            List of focus areas with topics and priorities
        """
        self.logger.info(f"Analyzing query: {query}")
        
        focus_areas = []
        
        try:
            # Extract key topics from the query
            topics = await self._extract_topics(query)
            
            # Identify research patterns
            patterns = self._identify_patterns(query)
            
            # Generate focus areas based on topics and patterns
            for i, topic in enumerate(topics):
                focus_area = {
                    "topic": topic,
                    "priority": 1.0 - (i * 0.1),  # Decreasing priority
                    "patterns": patterns,
                    "sources": [],
                    "summary": "",
                    "key_points": [],
                    "completed": False
                }
                focus_areas.append(focus_area)
                
                # Don't exceed max focus areas
                if len(focus_areas) >= self.max_focus_areas:
                    break
            
            # If no focus areas were identified, use the whole query
            if not focus_areas:
                focus_areas.append({
                    "topic": query,
                    "priority": 1.0,
                    "patterns": patterns,
                    "sources": [],
                    "summary": "",
                    "key_points": [],
                    "completed": False
                })
            
            self.logger.info(f"Identified {len(focus_areas)} focus areas")
            
        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}")
            # Fallback to simple focus area
            focus_areas = [{
                "topic": query,
                "priority": 1.0,
                "patterns": [],
                "sources": [],
                "summary": "",
                "key_points": [],
                "completed": False
            }]
        
        return focus_areas
    
    async def _extract_topics(self, query: str) -> List[str]:
        """
        Extract key topics from a query
        
        Args:
            query: Research query
            
        Returns:
            List of topic strings
        """
        topics = []
        
        # Simple approach: split by logical connectors
        connectors = [' and ', ' or ', ',', ';']
        current_topics = [query]
        
        for connector in connectors:
            new_topics = []
            for topic in current_topics:
                if connector in topic:
                    parts = [p.strip() for p in topic.split(connector)]
                    new_topics.extend(parts)
                else:
                    new_topics.append(topic)
            current_topics = new_topics
        
        # Clean up topics
        for topic in current_topics:
            topic = topic.strip()
            if topic and len(topic) > 3:  # Filter out very short topics
                topics.append(topic)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                unique_topics.append(topic)
        
        return unique_topics[:self.max_focus_areas]
    
    def _identify_patterns(self, query: str) -> List[str]:
        """
        Identify research patterns in the query
        
        Args:
            query: Research query
            
        Returns:
            List of identified patterns
        """
        patterns = []
        query_lower = query.lower()
        
        for pattern_type, keywords in self.research_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    patterns.append(pattern_type)
                    break  # Only add each pattern type once
        
        return patterns
    
    async def synthesize_information(self, topic: str, sources: List[Dict[str, Any]], 
                                   patterns: List[str] = None) -> Tuple[str, List[str]]:
        """
        Synthesize information from multiple sources into a summary
        
        Args:
            topic: The research topic
            sources: List of source dictionaries
            patterns: Research patterns to focus on
            
        Returns:
            Tuple of (summary, key_points)
        """
        self.logger.info(f"Synthesizing information for topic: {topic}")
        
        try:
            # Extract relevant content from sources
            content_pieces = []
            for source in sources:
                snippet = source.get('snippet', '')
                content = source.get('content', '')
                
                # Use content if available, otherwise use snippet
                text = content if content else snippet
                if text:
                    content_pieces.append({
                        'text': text,
                        'url': source.get('url', ''),
                        'title': source.get('title', ''),
                        'reliability': source.get('reliability_score', 0.5)
                    })
            
            # Generate summary
            summary = await self._generate_summary(topic, content_pieces, patterns)
            
            # Extract key points
            key_points = await self._extract_key_points(topic, content_pieces, patterns)
            
            self.logger.info(f"Generated summary with {len(key_points)} key points")
            
            return summary, key_points
            
        except Exception as e:
            self.logger.error(f"Error synthesizing information: {e}")
            # Fallback to basic summary
            summary = f"Research on {topic} based on {len(sources)} sources."
            key_points = [f"Key information about {topic} from available sources."]
            return summary, key_points
    
    async def _generate_summary(self, topic: str, content_pieces: List[Dict[str, Any]], 
                              patterns: List[str] = None) -> str:
        """
        Generate a comprehensive summary from content pieces
        
        Args:
            topic: Research topic
            content_pieces: List of content dictionaries
            patterns: Research patterns to focus on
            
        Returns:
            Generated summary
        """
        if not content_pieces:
            return f"No detailed information available for {topic}."
        
        # Simulate AI-driven analysis by creating structured summary
        summary_parts = []
        
        # Introduction
        summary_parts.append(f"Research on {topic} reveals several important aspects:")
        
        # Process high-reliability sources first
        high_rel_sources = [cp for cp in content_pieces if cp['reliability'] > 0.7]
        medium_rel_sources = [cp for cp in content_pieces if 0.4 <= cp['reliability'] <= 0.7]
        
        if high_rel_sources:
            summary_parts.append("High-reliability sources indicate that:")
            for i, source in enumerate(high_rel_sources[:3]):  # Limit to top 3
                text_sample = source['text'][:200] + "..." if len(source['text']) > 200 else source['text']
                summary_parts.append(f"• {text_sample}")
        
        if medium_rel_sources:
            summary_parts.append("Additional sources suggest that:")
            for i, source in enumerate(medium_rel_sources[:2]):  # Limit to top 2
                text_sample = source['text'][:150] + "..." if len(source['text']) > 150 else source['text']
                summary_parts.append(f"• {text_sample}")
        
        # Add pattern-specific insights if available
        if patterns:
            pattern_insights = self._generate_pattern_insights(topic, patterns, content_pieces)
            if pattern_insights:
                summary_parts.extend(pattern_insights)
        
        # Combine summary parts
        full_summary = "\n\n".join(summary_parts)
        
        # Truncate if too long
        if len(full_summary) > self.summary_max_length:
            full_summary = full_summary[:self.summary_max_length-3] + "..."
        
        return full_summary
    
    async def _extract_key_points(self, topic: str, content_pieces: List[Dict[str, Any]], 
                                patterns: List[str] = None) -> List[str]:
        """
        Extract key points from content pieces
        
        Args:
            topic: Research topic
            content_pieces: List of content dictionaries
            patterns: Research patterns to focus on
            
        Returns:
            List of key points
        """
        key_points = []
        
        # Extract sentences that contain topic keywords
        topic_keywords = topic.lower().split()
        
        for content in content_pieces:
            text = content['text']
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if sentence contains topic keywords
                sentence_lower = sentence.lower()
                relevance_score = sum(1 for keyword in topic_keywords if keyword in sentence_lower)
                
                if relevance_score > 0 and len(sentence) > 20:  # Minimum meaningful length
                    # Clean up the sentence
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    key_points.append(sentence)
                    
                    # Don't collect too many points
                    if len(key_points) >= self.max_key_points:
                        break
            
            if len(key_points) >= self.max_key_points:
                break
        
        # If no specific points found, create generic ones
        if not key_points:
            key_points = [
                f"Information about {topic} is available from multiple sources.",
                f"Research on {topic} shows varying perspectives and findings."
            ]
        
        # Ensure minimum number of key points
        while len(key_points) < self.min_key_points and len(key_points) < len(content_pieces):
            source_title = content_pieces[len(key_points)].get('title', f'Source {len(key_points)+1}')
            key_points.append(f"Relevant information found in: {source_title}")
        
        return key_points[:self.max_key_points]
    
    def _generate_pattern_insights(self, topic: str, patterns: List[str], 
                                 content_pieces: List[Dict[str, Any]]) -> List[str]:
        """
        Generate insights based on identified research patterns
        
        Args:
            topic: Research topic
            patterns: Identified research patterns
            content_pieces: Content to analyze
            
        Returns:
            List of pattern-specific insights
        """
        insights = []
        
        for pattern in patterns:
            if pattern == 'comparison':
                insights.append(f"Comparative analysis of {topic} shows distinct characteristics across different contexts.")
            elif pattern == 'trends':
                insights.append(f"Current trends in {topic} indicate ongoing developments and evolving perspectives.")
            elif pattern == 'impact':
                insights.append(f"The impact of {topic} extends across multiple domains and stakeholder groups.")
            elif pattern == 'methods':
                insights.append(f"Various methodological approaches to {topic} are documented in the literature.")
            elif pattern == 'challenges':
                insights.append(f"Several challenges and limitations regarding {topic} are highlighted in current research.")
            elif pattern == 'benefits':
                insights.append(f"Multiple benefits and advantages of {topic} are identified across different applications.")
        
        return insights
    
    async def generate_followup_questions(self, topic: str, summary: str, 
                                        key_points: List[str], sources: List[Dict[str, Any]]) -> List[str]:
        """
        Generate follow-up questions based on research results
        
        Args:
            topic: Original research topic
            summary: Generated summary
            key_points: Extracted key points
            sources: Research sources
            
        Returns:
            List of follow-up questions
        """
        self.logger.info(f"Generating follow-up questions for: {topic}")
        
        questions = []
        
        try:
            # Topic-based questions
            topic_words = topic.split()
            main_concept = topic_words[0] if topic_words else topic
            
            questions.extend([
                f"What are the latest developments in {topic}?",
                f"How does {topic} compare to similar concepts?",
                f"What are the practical applications of {topic}?",
                f"What challenges exist in implementing {topic}?",
                f"What future research directions exist for {topic}?"
            ])
            
            # Summary-based questions
            if "multiple" in summary.lower():
                questions.append(f"Which aspect of {topic} is most significant?")
            
            if "different" in summary.lower() or "various" in summary.lower():
                questions.append(f"What are the key differences in approaches to {topic}?")
            
            if "impact" in summary.lower():
                questions.append(f"What is the long-term impact of {topic}?")
            
            # Source-based questions
            if len(sources) > 3:
                questions.append(f"What do experts disagree about regarding {topic}?")
            
            # Pattern-based questions
            if any("trend" in kp.lower() for kp in key_points):
                questions.append(f"How have trends in {topic} evolved over time?")
            
            if any("method" in kp.lower() for kp in key_points):
                questions.append(f"What are the best practices for {topic}?")
            
            # Limit and randomize questions
            unique_questions = list(set(questions))  # Remove duplicates
            
            # Return up to 5 most relevant questions
            return unique_questions[:5]
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up questions: {e}")
            return [
                f"What are the main aspects of {topic}?",
                f"How can {topic} be applied in practice?",
                f"What are the current challenges with {topic}?"
            ]
    
    async def analyze_source_credibility(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the credibility and reliability of sources
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Credibility analysis results
        """
        if not sources:
            return {"average_reliability": 0.0, "high_quality_count": 0, "total_count": 0}
        
        reliabilities = [source.get('reliability_score', 0.5) for source in sources]
        average_reliability = sum(reliabilities) / len(reliabilities)
        high_quality_count = sum(1 for r in reliabilities if r > 0.7)
        
        return {
            "average_reliability": round(average_reliability, 2),
            "high_quality_count": high_quality_count,
            "total_count": len(sources),
            "reliability_distribution": {
                "high": sum(1 for r in reliabilities if r > 0.7),
                "medium": sum(1 for r in reliabilities if 0.4 <= r <= 0.7),
                "low": sum(1 for r in reliabilities if r < 0.4)
            }
        }
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about analysis operations
        
        Returns:
            Dictionary with analysis statistics
        """
        return {
            "max_focus_areas": self.max_focus_areas,
            "max_key_points": self.max_key_points,
            "summary_max_length": self.summary_max_length,
            "supported_patterns": list(self.research_patterns.keys()),
            "timestamp": datetime.now().isoformat()
        }

# Create singleton instance for easy import
analysis_agent = AnalysisAgent()