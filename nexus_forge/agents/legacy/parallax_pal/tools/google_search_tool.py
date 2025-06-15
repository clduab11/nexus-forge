"""
Google Search Tool for ADK Integration

This tool implements the Google Search API integration for use with
the Agent Development Kit (ADK) in ParallaxMind.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class GoogleSearchTool:
    """
    ADK-compatible Google Search tool for information retrieval.
    
    This tool provides enhanced search capabilities using Google's Custom Search API
    with proper formatting for ADK agent consumption.
    """
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """Initialize the Google Search tool."""
        self.api_key = api_key or os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Fallback configuration for development
        self.mock_mode = not (self.api_key and self.search_engine_id)
        
        if self.mock_mode:
            logger.warning("Google Search API credentials not found. Running in mock mode.")
        else:
            logger.info("Google Search tool initialized with API credentials")
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        site_restrict: Optional[str] = None,
        date_restrict: Optional[str] = None,
        safe_search: str = "active",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Perform a Google search and return structured results.
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 10 per request)
            site_restrict: Restrict results to specific site (e.g., "reddit.com")
            date_restrict: Date restriction (e.g., "d1" for past day, "w1" for past week)
            safe_search: Safe search setting ("active", "moderate", "off")
            language: Language for results
            
        Returns:
            Dictionary containing search results and metadata
        """
        if self.mock_mode:
            return await self._mock_search(query, num_results)
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Build search parameters
                params = {
                    "key": self.api_key,
                    "cx": self.search_engine_id,
                    "q": query,
                    "num": min(num_results, 10),  # Google Custom Search API limit
                    "safe": safe_search,
                    "lr": f"lang_{language}"
                }
                
                # Add optional parameters
                if site_restrict:
                    params["siteSearch"] = site_restrict
                
                if date_restrict:
                    params["dateRestrict"] = date_restrict
                
                # Make the search request
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                search_data = response.json()
                
                # Process and format results
                return await self._process_search_results(search_data, query)
                
        except Exception as e:
            logger.error(f"Google Search API error: {e}")
            # Fallback to mock results on error
            return await self._mock_search(query, num_results, error=str(e))
    
    async def search_academic(
        self,
        query: str,
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for academic sources using site restrictions.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Academic search results
        """
        # Modify query to focus on academic sources
        academic_query = f"{query} site:edu OR site:scholar.google.com OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
        
        return await self.search(
            query=academic_query,
            num_results=num_results,
            safe_search="active"
        )
    
    async def search_news(
        self,
        query: str,
        num_results: int = 10,
        recency: str = "w1"
    ) -> Dict[str, Any]:
        """
        Search for recent news articles.
        
        Args:
            query: Search query
            num_results: Number of results to return
            recency: How recent ("d1" = past day, "w1" = past week, "m1" = past month)
            
        Returns:
            News search results
        """
        return await self.search(
            query=query,
            num_results=num_results,
            date_restrict=recency,
            safe_search="moderate"
        )
    
    async def search_specific_sites(
        self,
        query: str,
        sites: List[str],
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search across specific websites.
        
        Args:
            query: Search query
            sites: List of sites to search (e.g., ["reddit.com", "stackoverflow.com"])
            num_results: Number of results to return
            
        Returns:
            Site-specific search results
        """
        all_results = []
        
        for site in sites:
            try:
                results = await self.search(
                    query=query,
                    num_results=min(num_results // len(sites) + 1, 10),
                    site_restrict=site
                )
                
                if results.get("items"):
                    all_results.extend(results["items"])
                    
            except Exception as e:
                logger.warning(f"Error searching {site}: {e}")
                continue
        
        # Sort by relevance score and limit results
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return {
            "query": query,
            "total_results": len(all_results),
            "search_time": datetime.now().isoformat(),
            "items": all_results[:num_results],
            "search_metadata": {
                "sites_searched": sites,
                "search_type": "multi_site"
            }
        }
    
    async def _process_search_results(self, search_data: Dict, query: str) -> Dict[str, Any]:
        """Process raw Google Search API results into structured format."""
        try:
            items = search_data.get("items", [])
            
            processed_items = []
            for item in items:
                processed_item = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_url": item.get("displayLink", ""),
                    "formatted_url": item.get("formattedUrl", ""),
                    "cache_id": item.get("cacheId", ""),
                    "page_map": item.get("pagemap", {}),
                    "relevance_score": self._calculate_relevance_score(item, query),
                    "source_type": self._classify_source_type(item.get("link", "")),
                    "extracted_date": self._extract_date_from_snippet(item.get("snippet", "")),
                    "domain_authority": self._estimate_domain_authority(item.get("displayLink", "")),
                }
                processed_items.append(processed_item)
            
            # Sort by relevance score
            processed_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "query": query,
                "total_results": search_data.get("searchInformation", {}).get("totalResults", 0),
                "search_time": search_data.get("searchInformation", {}).get("searchTime", 0),
                "items": processed_items,
                "search_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "search_type": "google_custom_search",
                    "safe_search": "active",
                    "spelling_suggestions": search_data.get("spelling", {}).get("correctedQuery", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return await self._mock_search(query, 10, error=str(e))
    
    def _calculate_relevance_score(self, item: Dict, query: str) -> float:
        """Calculate a relevance score for a search result."""
        try:
            score = 0.5  # Base score
            
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()
            query_lower = query.lower()
            query_words = query_lower.split()
            
            # Title relevance (higher weight)
            title_matches = sum(1 for word in query_words if word in title)
            score += (title_matches / len(query_words)) * 0.4
            
            # Snippet relevance
            snippet_matches = sum(1 for word in query_words if word in snippet)
            score += (snippet_matches / len(query_words)) * 0.2
            
            # Exact phrase bonus
            if query_lower in title:
                score += 0.3
            elif query_lower in snippet:
                score += 0.2
            
            # Source quality bonus
            url = item.get("link", "").lower()
            if any(domain in url for domain in ['.edu', '.gov', '.org']):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.5
    
    def _classify_source_type(self, url: str) -> str:
        """Classify the type of source based on URL."""
        url_lower = url.lower()
        
        if '.edu' in url_lower or 'scholar.google' in url_lower:
            return 'academic'
        elif '.gov' in url_lower:
            return 'government'
        elif any(news_site in url_lower for news_site in ['news', 'reuters', 'bbc', 'cnn', 'nytimes']):
            return 'news'
        elif '.org' in url_lower:
            return 'organization'
        elif any(social in url_lower for social in ['reddit', 'twitter', 'facebook', 'linkedin']):
            return 'social'
        elif 'wikipedia' in url_lower:
            return 'encyclopedia'
        else:
            return 'general'
    
    def _extract_date_from_snippet(self, snippet: str) -> Optional[str]:
        """Try to extract a date from the search snippet."""
        import re
        
        # Common date patterns
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{4}-\d{2}-\d{2})\b',      # YYYY-MM-DD
            r'\b(\w+ \d{1,2}, \d{4})\b',     # Month DD, YYYY
            r'\b(\d{1,2} \w+ \d{4})\b'       # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1)
        
        return None
    
    def _estimate_domain_authority(self, domain: str) -> float:
        """Estimate domain authority based on known high-authority domains."""
        high_authority_domains = {
            'wikipedia.org': 0.95,
            'edu': 0.9,
            'gov': 0.9,
            'nature.com': 0.9,
            'science.org': 0.9,
            'pubmed.ncbi.nlm.nih.gov': 0.9,
            'bbc.com': 0.8,
            'reuters.com': 0.8,
            'nytimes.com': 0.8,
            'washingtonpost.com': 0.8,
            'theguardian.com': 0.8,
            'stackoverflow.com': 0.8,
            'github.com': 0.8
        }
        
        domain_lower = domain.lower()
        
        # Check for exact matches
        for auth_domain, score in high_authority_domains.items():
            if auth_domain in domain_lower:
                return score
        
        # Default scores based on TLD
        if domain_lower.endswith('.edu'):
            return 0.85
        elif domain_lower.endswith('.gov'):
            return 0.85
        elif domain_lower.endswith('.org'):
            return 0.7
        else:
            return 0.5
    
    async def _mock_search(self, query: str, num_results: int, error: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock search results for development/testing."""
        mock_items = []
        
        # Generate realistic mock results based on query
        query_words = query.lower().split()
        
        for i in range(min(num_results, 8)):
            mock_items.append({
                "title": f"Mock Result {i+1}: {query} - Comprehensive Guide",
                "url": f"https://example{i+1}.com/article-about-{'-'.join(query_words)}",
                "snippet": f"This is a mock search result for '{query}'. It contains relevant information about {query} and related topics. This result demonstrates the structure of search results.",
                "display_url": f"example{i+1}.com",
                "formatted_url": f"https://example{i+1}.com/article-about-{'-'.join(query_words)}",
                "cache_id": f"mock_cache_{i}",
                "page_map": {},
                "relevance_score": 0.9 - (i * 0.1),
                "source_type": ["academic", "news", "general", "organization"][i % 4],
                "extracted_date": "2024-01-15",
                "domain_authority": 0.8 - (i * 0.05)
            })
        
        result = {
            "query": query,
            "total_results": len(mock_items),
            "search_time": 0.1,
            "items": mock_items,
            "search_metadata": {
                "timestamp": datetime.now().isoformat(),
                "search_type": "mock_search",
                "safe_search": "active",
                "spelling_suggestions": "",
                "mock_mode": True
            }
        }
        
        if error:
            result["search_metadata"]["fallback_reason"] = error
        
        return result

# Create global instance for ADK integration
google_search_tool = GoogleSearchTool()