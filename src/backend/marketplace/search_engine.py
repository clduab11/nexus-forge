"""
Marketplace search engine with semantic search capabilities
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.cache import RedisCache
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .models import AgentPackage, AgentSearchResult, AgentStatus


class MarketplaceSearchEngine:
    """Advanced search engine for agent marketplace"""

    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.embedding_cache = {}

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "relevance",
    ) -> List[AgentSearchResult]:
        """
        Search for agents with advanced ranking

        Args:
            query: Search query
            filters: Optional filters (category, tags, author, etc.)
            limit: Maximum results
            offset: Pagination offset
            sort_by: Sort order (relevance, downloads, rating, newest)

        Returns:
            List of search results with relevance scores
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, filters, limit, offset, sort_by)

        # Check cache
        cached_results = await self.cache.get(cache_key)
        if cached_results:
            return [AgentSearchResult(**result) for result in cached_results]

        # Perform search
        if query:
            # Semantic search
            results = await self._semantic_search(query, filters, limit * 2)
        else:
            # Browse without query
            results = await self._browse_agents(filters, limit * 2, offset)

        # Apply additional filtering
        filtered_results = self._apply_filters(results, filters)

        # Rank results
        ranked_results = self._rank_results(filtered_results, query, sort_by)

        # Apply pagination
        paginated_results = ranked_results[offset : offset + limit]

        # Create search results
        search_results = []
        for agent, score in paginated_results:
            search_result = AgentSearchResult(
                agent=agent,
                relevance_score=score,
                matched_fields=self._get_matched_fields(agent, query),
                highlight=self._generate_highlights(agent, query),
            )
            search_results.append(search_result)

        # Cache results
        await self.cache.set(
            cache_key,
            [result.dict() for result in search_results],
            ttl=300,  # 5 minutes
        )

        return search_results

    async def suggest(
        self, partial_query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Provide search suggestions based on partial query"""
        # Check cache
        cache_key = f"suggest:{partial_query}:{limit}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        suggestions = []

        # Agent name suggestions
        name_suggestions = await self._get_name_suggestions(partial_query, limit // 2)
        suggestions.extend(name_suggestions)

        # Tag suggestions
        tag_suggestions = await self._get_tag_suggestions(partial_query, limit // 2)
        suggestions.extend(tag_suggestions)

        # Sort by relevance
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        suggestions = suggestions[:limit]

        # Cache suggestions
        await self.cache.set(cache_key, suggestions, ttl=600)  # 10 minutes

        return suggestions

    async def get_trending(
        self, time_window: str = "week", limit: int = 10
    ) -> List[AgentPackage]:
        """Get trending agents based on recent activity"""
        cache_key = f"trending:{time_window}:{limit}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [AgentPackage(**agent) for agent in cached]

        # Calculate time window
        if time_window == "day":
            since = datetime.utcnow() - timedelta(days=1)
        elif time_window == "week":
            since = datetime.utcnow() - timedelta(days=7)
        elif time_window == "month":
            since = datetime.utcnow() - timedelta(days=30)
        else:
            since = datetime.utcnow() - timedelta(days=7)

        # Query trending agents based on recent downloads and ratings
        query = (
            self.supabase.client.table("agent_packages")
            .select("*")
            .eq("status", AgentStatus.APPROVED.value)
            .gte("updated_at", since.isoformat())
        )

        # Order by trending score (combination of downloads and ratings)
        # In production, this would use a more sophisticated algorithm
        result = await query.execute()

        agents = [AgentPackage(**data) for data in result.data]

        # Calculate trending score
        for agent in agents:
            recent_downloads = await self._get_recent_downloads(agent.id, since)
            recent_ratings = await self._get_recent_ratings(agent.id, since)

            # Simple trending score calculation
            agent.trending_score = recent_downloads * 0.7 + recent_ratings * 10 * 0.3

        # Sort by trending score
        agents.sort(key=lambda x: getattr(x, "trending_score", 0), reverse=True)
        trending = agents[:limit]

        # Cache results
        await self.cache.set(
            cache_key, [agent.dict() for agent in trending], ttl=3600  # 1 hour
        )

        return trending

    async def get_similar(self, agent_id: str, limit: int = 10) -> List[AgentPackage]:
        """Find similar agents based on features and usage patterns"""
        # Get reference agent
        reference = await self._get_agent_by_id(agent_id)
        if not reference:
            return []

        # Find similar agents
        similar = await self._find_similar_agents(reference, limit)

        return similar

    # Private helper methods

    async def _semantic_search(
        self, query: str, filters: Optional[Dict[str, Any]], limit: int
    ) -> List[AgentPackage]:
        """Perform semantic search using embeddings"""
        # For now, use keyword search
        # In production, this would use vector embeddings

        # Build database query
        db_query = (
            self.supabase.client.table("agent_packages")
            .select("*")
            .eq("status", AgentStatus.APPROVED.value)
        )

        # Add text search
        search_terms = query.lower().split()
        search_conditions = []

        for term in search_terms:
            conditions = [
                f"manifest->name.ilike.%{term}%",
                f"manifest->display_name.ilike.%{term}%",
                f"manifest->description.ilike.%{term}%",
                f"manifest->tags.cs.{{{term}}}",
            ]
            search_conditions.append(f"({','.join(conditions)})")

        if search_conditions:
            db_query = db_query.or_(",".join(search_conditions))

        # Apply basic filters
        if filters:
            if "category" in filters:
                db_query = db_query.eq("manifest->category", filters["category"])

            if "author" in filters:
                db_query = db_query.eq("manifest->author", filters["author"])

        # Execute query
        result = await db_query.limit(limit).execute()

        return [AgentPackage(**data) for data in result.data]

    async def _browse_agents(
        self, filters: Optional[Dict[str, Any]], limit: int, offset: int
    ) -> List[AgentPackage]:
        """Browse agents without search query"""
        db_query = (
            self.supabase.client.table("agent_packages")
            .select("*")
            .eq("status", AgentStatus.APPROVED.value)
        )

        # Apply filters
        if filters:
            if "category" in filters:
                db_query = db_query.eq("manifest->category", filters["category"])

            if "tags" in filters and filters["tags"]:
                db_query = db_query.contains("manifest->tags", filters["tags"])

            if "author" in filters:
                db_query = db_query.eq("manifest->author", filters["author"])

            if "min_rating" in filters:
                db_query = db_query.gte("rating", filters["min_rating"])

        # Default ordering
        db_query = db_query.order("downloads", desc=True)

        # Execute query
        result = await db_query.range(offset, offset + limit - 1).execute()

        return [AgentPackage(**data) for data in result.data]

    def _apply_filters(
        self, agents: List[AgentPackage], filters: Optional[Dict[str, Any]]
    ) -> List[AgentPackage]:
        """Apply additional filters to search results"""
        if not filters:
            return agents

        filtered = agents

        # Filter by minimum downloads
        if "min_downloads" in filters:
            filtered = [a for a in filtered if a.downloads >= filters["min_downloads"]]

        # Filter by date range
        if "published_after" in filters:
            date_threshold = datetime.fromisoformat(filters["published_after"])
            filtered = [
                a
                for a in filtered
                if a.published_at and a.published_at >= date_threshold
            ]

        # Filter by license
        if "license" in filters:
            filtered = [a for a in filtered if a.manifest.license == filters["license"]]

        return filtered

    def _rank_results(
        self, agents: List[AgentPackage], query: str, sort_by: str
    ) -> List[Tuple[AgentPackage, float]]:
        """Rank search results based on various factors"""
        ranked = []

        for agent in agents:
            if sort_by == "relevance" and query:
                score = self._calculate_relevance_score(agent, query)
            elif sort_by == "downloads":
                score = float(agent.downloads)
            elif sort_by == "rating":
                score = agent.rating or 0.0
            elif sort_by == "newest":
                score = agent.created_at.timestamp() if agent.created_at else 0.0
            else:
                score = 0.0

            ranked.append((agent, score))

        # Sort by score
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def _calculate_relevance_score(self, agent: AgentPackage, query: str) -> float:
        """Calculate relevance score for an agent"""
        score = 0.0
        query_lower = query.lower()
        terms = query_lower.split()

        # Name match (highest weight)
        name_lower = agent.manifest.name.lower()
        if query_lower == name_lower:
            score += 10.0  # Exact match
        elif query_lower in name_lower:
            score += 7.0  # Substring match
        else:
            for term in terms:
                if term in name_lower:
                    score += 3.0

        # Display name match
        display_lower = agent.manifest.display_name.lower()
        if query_lower in display_lower:
            score += 5.0
        else:
            for term in terms:
                if term in display_lower:
                    score += 2.0

        # Description match
        desc_lower = agent.manifest.description.lower()
        for term in terms:
            if term in desc_lower:
                score += 1.0

        # Tag match
        tags_lower = [tag.lower() for tag in agent.manifest.tags]
        for term in terms:
            if term in tags_lower:
                score += 2.5

        # Boost by popularity
        if agent.downloads > 1000:
            score *= 1.2
        elif agent.downloads > 100:
            score *= 1.1

        # Boost by rating
        if agent.rating:
            score *= 0.8 + agent.rating / 10  # Max 1.3x boost

        # Normalize score
        return min(score / 10.0, 1.0)

    def _get_matched_fields(self, agent: AgentPackage, query: str) -> List[str]:
        """Get list of fields that matched the query"""
        if not query:
            return []

        matched = []
        query_lower = query.lower()
        terms = query_lower.split()

        # Check name
        if any(term in agent.manifest.name.lower() for term in terms):
            matched.append("name")

        # Check display name
        if any(term in agent.manifest.display_name.lower() for term in terms):
            matched.append("display_name")

        # Check description
        if any(term in agent.manifest.description.lower() for term in terms):
            matched.append("description")

        # Check tags
        tags_lower = [tag.lower() for tag in agent.manifest.tags]
        if any(term in tags_lower for term in terms):
            matched.append("tags")

        return matched

    def _generate_highlights(
        self, agent: AgentPackage, query: str
    ) -> Optional[Dict[str, str]]:
        """Generate highlighted snippets for search results"""
        if not query:
            return None

        highlights = {}
        terms = query.lower().split()

        # Highlight description
        description = agent.manifest.description
        highlighted_desc = description

        for term in terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_desc = pattern.sub(f"<mark>{term}</mark>", highlighted_desc)

        if highlighted_desc != description:
            # Truncate around first match
            first_mark = highlighted_desc.find("<mark>")
            if first_mark > 50:
                start = max(0, first_mark - 50)
                highlighted_desc = "..." + highlighted_desc[start:]

            if len(highlighted_desc) > 200:
                highlighted_desc = highlighted_desc[:200] + "..."

            highlights["description"] = highlighted_desc

        return highlights if highlights else None

    def _generate_cache_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int,
        offset: int,
        sort_by: str,
    ) -> str:
        """Generate cache key for search results"""
        filter_str = ""
        if filters:
            filter_str = ":".join(f"{k}={v}" for k, v in sorted(filters.items()))

        return f"search:{query}:{filter_str}:{limit}:{offset}:{sort_by}"

    async def _get_name_suggestions(
        self, partial: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get agent name suggestions"""
        query = (
            self.supabase.client.table("agent_packages")
            .select("manifest")
            .eq("status", AgentStatus.APPROVED.value)
            .ilike("manifest->name", f"{partial}%")
            .limit(limit)
        )

        result = await query.execute()

        suggestions = []
        for data in result.data:
            manifest = data["manifest"]
            suggestions.append(
                {
                    "type": "agent",
                    "value": manifest["name"],
                    "display": manifest["display_name"],
                    "score": len(partial) / len(manifest["name"]),
                }
            )

        return suggestions

    async def _get_tag_suggestions(
        self, partial: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get tag suggestions"""
        # In production, this would query a tags index
        # For now, return common tags that match
        common_tags = [
            "ai",
            "nlp",
            "vision",
            "audio",
            "automation",
            "data",
            "analytics",
            "integration",
            "utility",
            "workflow",
        ]

        suggestions = []
        for tag in common_tags:
            if partial.lower() in tag:
                suggestions.append(
                    {
                        "type": "tag",
                        "value": tag,
                        "display": f"Tag: {tag}",
                        "score": len(partial) / len(tag),
                    }
                )

        return suggestions[:limit]

    async def _get_recent_downloads(self, agent_id: str, since: datetime) -> int:
        """Get download count since date"""
        query = (
            self.supabase.client.table("agent_installations")
            .select("id", count="exact")
            .eq("agent_id", agent_id)
            .gte("installed_at", since.isoformat())
        )

        result = await query.execute()
        return result.count or 0

    async def _get_recent_ratings(self, agent_id: str, since: datetime) -> int:
        """Get rating count since date"""
        query = (
            self.supabase.client.table("agent_ratings")
            .select("id", count="exact")
            .eq("agent_id", agent_id)
            .gte("created_at", since.isoformat())
        )

        result = await query.execute()
        return result.count or 0

    async def _get_agent_by_id(self, agent_id: str) -> Optional[AgentPackage]:
        """Get agent by ID"""
        result = (
            await self.supabase.client.table("agent_packages")
            .select("*")
            .eq("id", agent_id)
            .execute()
        )

        if result.data:
            return AgentPackage(**result.data[0])
        return None

    async def _find_similar_agents(
        self, reference: AgentPackage, limit: int
    ) -> List[AgentPackage]:
        """Find agents similar to reference"""
        # Query agents in same category
        query = (
            self.supabase.client.table("agent_packages")
            .select("*")
            .eq("status", AgentStatus.APPROVED.value)
            .eq("manifest->category", reference.manifest.category)
            .neq("id", reference.id)
            .limit(limit * 2)
        )

        result = await query.execute()
        candidates = [AgentPackage(**data) for data in result.data]

        # Calculate similarity scores
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_similarity(reference, candidate)
            scored_candidates.append((candidate, score))

        # Sort by similarity
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top similar agents
        return [agent for agent, _ in scored_candidates[:limit]]

    def _calculate_similarity(
        self, agent1: AgentPackage, agent2: AgentPackage
    ) -> float:
        """Calculate similarity score between two agents"""
        score = 0.0

        # Same category (already filtered)
        score += 0.3

        # Tag overlap
        tags1 = set(agent1.manifest.tags)
        tags2 = set(agent2.manifest.tags)
        if tags1 and tags2:
            overlap = len(tags1.intersection(tags2))
            total = len(tags1.union(tags2))
            score += 0.3 * (overlap / total)

        # Similar capabilities
        caps1 = set(agent1.manifest.capabilities)
        caps2 = set(agent2.manifest.capabilities)
        if caps1 and caps2:
            overlap = len(caps1.intersection(caps2))
            total = len(caps1.union(caps2))
            score += 0.2 * (overlap / total)

        # Similar rating
        if agent1.rating and agent2.rating:
            rating_diff = abs(agent1.rating - agent2.rating)
            score += 0.1 * (1 - rating_diff / 5.0)

        # Similar popularity
        if agent1.downloads and agent2.downloads:
            pop_ratio = min(agent1.downloads, agent2.downloads) / max(
                agent1.downloads, agent2.downloads
            )
            score += 0.1 * pop_ratio

        return score
