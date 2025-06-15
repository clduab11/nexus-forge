"""
Information Retrieval Agent for ParallaxMind

This agent is responsible for searching and retrieving information from various sources,
serving as a refactored and enhanced version of the previous search_engine.py.
"""

import json
import logging
import uuid
import time
import asyncio
import hashlib
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Try to import ADK-specific libraries, fallback if not available
try:
    from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    # Fallback to simple class for local development
    class Agent:
        def __init__(self):
            pass

# Import web scraping functionality
try:
    from web_scraper import MultiSearcher, WebScraperError
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("web_scraper not available, using mock functionality")

# Import ADK tools
try:
    from agents.tools.google_search_tool import google_search_tool
    GOOGLE_SEARCH_AVAILABLE = True
    logger.info("Google Search tool available for ADK integration")
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    google_search_tool = None
    logger.warning("Google Search tool not available")
    
    class WebScraperError(Exception):
        pass
    
    class MultiSearcher:
        def __init__(self):
            pass
        
        def search(self, query, num_results=10):
            return [
                {
                    'url': f"https://example.com/result_{i}",
                    'title': f"Example Result {i} for {query}",
                    'snippet': f"This is a mocked search result snippet for {query}.",
                    'domain': 'example.com'
                }
                for i in range(min(num_results, 3))
            ]

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logger = logging.getLogger("retrieval_agent")

@dataclass
class Source:
    """Represents a research source with citation information"""
    url: str
    title: str
    author: Optional[str] = None
    publication_date: Optional[str] = None
    site_name: Optional[str] = None
    content: str = ""
    snippet: str = ""
    access_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    citation_style: str = "apa"
    reliability_score: float = 0.0
    content_hash: str = ""
    is_primary: bool = False
    
    def __post_init__(self):
        # Generate content hash for verification if content exists
        if self.content and not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        
        # Try to extract site name from URL if not provided
        if not self.site_name and self.url:
            try:
                parsed_url = urllib.parse.urlparse(self.url)
                self.site_name = parsed_url.netloc
            except Exception:
                self.site_name = "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert source to dictionary format"""
        return {
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "site_name": self.site_name,
            "content": self.content,
            "snippet": self.snippet,
            "access_date": self.access_date,
            "citation_style": self.citation_style,
            "reliability_score": self.reliability_score,
            "content_hash": self.content_hash,
            "is_primary": self.is_primary
        }

class RetrievalAgent(Agent):
    """
    Information Retrieval Agent for ParallaxMind
    
    This agent is responsible for searching and retrieving information from the web,
    processing content, and providing structured research results.
    """
    
    def __init__(self):
        """Initialize retrieval agent with configuration."""
        if ADK_AVAILABLE:
            super().__init__()
        
        # Domain reliability scores
        self.domain_reliability = {
            'arxiv.org': 0.95,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'nature.com': 0.95,
            'science.org': 0.95,
            'ieee.org': 0.90,
            'acm.org': 0.90,
            'springer.com': 0.85,
            'wiley.com': 0.85,
            'elsevier.com': 0.85,
            'jstor.org': 0.85,
            'plos.org': 0.80,
            'mit.edu': 0.85,
            'stanford.edu': 0.85,
            'harvard.edu': 0.85,
            'princeton.edu': 0.85,
            'wikipedia.org': 0.70,  # Good for general info, but not primary source
            'reddit.com': 0.40,
            'twitter.com': 0.30,
            'facebook.com': 0.30
        }
        
        self.cache = {}
        self.seen_urls = set()
        self.max_searches_per_cycle = 5
        self.max_content_per_url = 50000
        self.snippet_length = 200
        self.logger = logging.getLogger("retrieval_agent")
        
        # Initialize searcher
        try:
            self.searcher = MultiSearcher()
        except Exception as e:
            self.logger.warning(f"Could not initialize MultiSearcher: {e}")
            self.searcher = MultiSearcher()  # Will use mock version
        
        self.logger.info("Retrieval agent initialized")
    
    async def search_information(self, query: str, max_sources: int = 10, 
                               force_refresh: bool = False) -> List[Source]:
        """
        Search for information on a given query
        
        Args:
            query: The search query
            max_sources: Maximum number of sources to return
            force_refresh: Whether to bypass cache
            
        Returns:
            List of Source objects containing search results
        """
        self.logger.info(f"Searching for information on: {query}")
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}_{max_sources}".encode()).hexdigest()
        if not force_refresh and cache_key in self.cache:
            self.logger.info("Using cached results")
            return self.cache[cache_key]
        
        sources = []
        
        try:
            # Perform search
            search_results = await self._perform_search(query, max_sources)
            
            # Process results
            for result in search_results:
                source = await self._process_search_result(result)
                if source and source.url not in self.seen_urls:
                    sources.append(source)
                    self.seen_urls.add(source.url)
                    
                    # Don't exceed max sources
                    if len(sources) >= max_sources:
                        break
            
            # Calculate reliability scores
            for source in sources:
                source.reliability_score = self._calculate_reliability(source)
            
            # Sort by reliability score
            sources.sort(key=lambda s: s.reliability_score, reverse=True)
            
            # Mark top sources as primary
            for i, source in enumerate(sources[:3]):  # Top 3 as primary
                source.is_primary = True
            
            # Cache results
            self.cache[cache_key] = sources
            
            self.logger.info(f"Found {len(sources)} sources for query: {query}")
            
        except Exception as e:
            self.logger.error(f"Error searching for information: {e}")
            # Return empty list on error
            sources = []
        
        return sources
    
    async def _perform_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform the actual search using available search tools
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search result dictionaries
        """
        try:
            # Try Google Search tool first (ADK integration)
            if GOOGLE_SEARCH_AVAILABLE and google_search_tool:
                self.logger.info("Using Google Search tool for enhanced search")
                search_result = await google_search_tool.search(query, num_results=max_results)
                
                if search_result.get("items"):
                    # Convert Google Search results to our format
                    results = []
                    for item in search_result["items"]:
                        results.append({
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "domain": item.get("display_url", ""),
                            "relevance_score": item.get("relevance_score", 0.5),
                            "source_type": item.get("source_type", "general"),
                            "domain_authority": item.get("domain_authority", 0.5)
                        })
                    return results
            
            # Fallback to original searcher
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.searcher.search(query, num_results=max_results)
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing search: {e}")
            return []
    
    async def _process_search_result(self, result: Dict[str, Any]) -> Optional[Source]:
        """
        Process a search result into a Source object
        
        Args:
            result: Search result dictionary
            
        Returns:
            Source object or None if processing failed
        """
        try:
            # Extract basic information
            url = result.get('url', '')
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', '')
            domain = result.get('domain', '')
            
            if not url:
                return None
            
            # Create source object
            source = Source(
                url=url,
                title=title,
                snippet=snippet[:self.snippet_length] if snippet else "",
                site_name=domain
            )
            
            return source
            
        except Exception as e:
            self.logger.error(f"Error processing search result: {e}")
            return None
    
    def _calculate_reliability(self, source: Source) -> float:
        """
        Calculate reliability score for a source
        
        Args:
            source: Source object
            
        Returns:
            Reliability score between 0 and 1
        """
        base_score = 0.5  # Default reliability
        
        # Check domain reliability
        if source.site_name:
            domain = source.site_name.lower()
            for reliable_domain, score in self.domain_reliability.items():
                if reliable_domain in domain:
                    base_score = score
                    break
        
        # Adjust based on URL characteristics
        url_lower = source.url.lower()
        
        # Academic indicators
        if any(indicator in url_lower for indicator in ['/pdf/', '.pdf', 'doi.org', 'research']):
            base_score += 0.1
        
        # News indicators
        if any(indicator in url_lower for indicator in ['news', 'article', 'report']):
            base_score += 0.05
        
        # Negative indicators
        if any(indicator in url_lower for indicator in ['blog', 'forum', 'comment']):
            base_score -= 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def retrieve_content(self, url: str) -> Optional[str]:
        """
        Retrieve full content from a URL
        
        Args:
            url: URL to retrieve content from
            
        Returns:
            Full content as string, or None if failed
        """
        try:
            self.logger.info(f"Retrieving content from: {url}")
            
            # This would be implemented with actual web scraping
            # For now, return mock content
            return f"Mock content from {url}. In a real implementation, this would contain the full scraped content."
            
        except Exception as e:
            self.logger.error(f"Error retrieving content from {url}: {e}")
            return None
    
    async def enhanced_search(self, query: str, max_sources: int = 10, 
                            depth_level: str = "detailed") -> List[Source]:
        """
        Enhanced search with multiple strategies
        
        Args:
            query: Search query
            max_sources: Maximum number of sources
            depth_level: Level of detail ("basic", "detailed", "comprehensive")
            
        Returns:
            List of enhanced Source objects
        """
        self.logger.info(f"Performing enhanced search for: {query} (depth: {depth_level})")
        
        # Adjust search parameters based on depth level
        if depth_level == "basic":
            search_sources = max_sources
        elif depth_level == "detailed":
            search_sources = max_sources * 2
        else:  # comprehensive
            search_sources = max_sources * 3
        
        # Perform initial search
        sources = await self.search_information(query, search_sources)
        
        # If detailed or comprehensive, enhance with content retrieval
        if depth_level in ["detailed", "comprehensive"] and sources:
            enhanced_sources = []
            for source in sources[:max_sources]:
                try:
                    # Retrieve full content for top sources
                    content = await self.retrieve_content(source.url)
                    if content:
                        source.content = content[:self.max_content_per_url]
                    enhanced_sources.append(source)
                except Exception as e:
                    self.logger.error(f"Error enhancing source {source.url}: {e}")
                    enhanced_sources.append(source)  # Add original anyway
            
            return enhanced_sources
        
        return sources[:max_sources]
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retrieved sources
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_urls_seen": len(self.seen_urls),
            "cache_size": len(self.cache),
            "high_reliability_domains": len([d for d, s in self.domain_reliability.items() if s > 0.8]),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear the search cache"""
        self.cache.clear()
        self.seen_urls.clear()
        self.logger.info("Search cache cleared")

# Create singleton instance for easy import
retrieval_agent = RetrievalAgent()