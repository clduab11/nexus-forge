"""
Enhanced Marketplace API for Agent/Tool Discovery and Distribution
Comprehensive implementation for Google ADK Hackathon
"""

import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import numpy as np
from fastapi import BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl, validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...core.cache import CacheManager
from ...core.exceptions import (
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from ...core.logging import get_logger
from ..ai.clients import GeminiClient
from .models import (
    AgentCategory,
    AgentManifest,
    AgentPackage,
    AgentStatus,
    PerformanceMetrics,
    SecurityReport,
)
from .mcp_integration import MCPMarketplaceClient, MCPTool, MCPToolType

logger = get_logger(__name__)


class MarketplaceAPIVersion(str):
    """API version management"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"  # Latest with full MCP integration


class PublishRequest(BaseModel):
    """Agent/Tool publishing request"""
    
    manifest: AgentManifest
    package_type: str = Field("agent", regex="^(agent|tool|workflow|template)$")
    visibility: str = Field("public", regex="^(public|private|unlisted)$")
    beta: bool = False
    pricing: Optional[Dict[str, Any]] = None
    mcp_integration: Optional[Dict[str, Any]] = None


class VersioningStrategy(BaseModel):
    """Versioning configuration for agents/tools"""
    
    strategy: str = Field("semantic", regex="^(semantic|date|custom)$")
    auto_increment: bool = True
    branch_support: bool = True
    rollback_enabled: bool = True
    max_versions_retained: int = 10


class DependencyResolver:
    """Advanced dependency resolution system"""
    
    def __init__(self):
        self.dependency_graph = {}
        self.conflict_resolver = ConflictResolver()
        self.version_matcher = VersionMatcher()
    
    async def resolve_dependencies(
        self,
        package: AgentPackage,
        target_environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve all dependencies with conflict resolution
        
        Returns:
            Dict containing resolved dependencies and installation order
        """
        # Build dependency tree
        tree = await self._build_dependency_tree(package)
        
        # Detect conflicts
        conflicts = await self.conflict_resolver.detect_conflicts(tree)
        
        # Resolve conflicts
        if conflicts:
            resolution = await self.conflict_resolver.resolve(conflicts)
            tree = await self._apply_resolution(tree, resolution)
        
        # Calculate installation order
        install_order = await self._topological_sort(tree)
        
        return {
            "dependencies": tree,
            "install_order": install_order,
            "conflicts_resolved": len(conflicts),
            "environment": target_environment,
        }
    
    async def _build_dependency_tree(
        self,
        package: AgentPackage,
        visited: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Recursively build dependency tree"""
        if visited is None:
            visited = set()
        
        if package.id in visited:
            return {"circular": True, "package_id": package.id}
        
        visited.add(package.id)
        tree = {
            "package": package,
            "dependencies": {},
        }
        
        for dep in package.manifest.dependencies:
            dep_package = await self._fetch_dependency(dep)
            if dep_package:
                tree["dependencies"][dep["name"]] = await self._build_dependency_tree(
                    dep_package, visited
                )
        
        return tree
    
    async def _fetch_dependency(self, dep_spec: Dict[str, str]) -> Optional[AgentPackage]:
        """Fetch dependency package information"""
        # Implementation would query marketplace
        return None
    
    async def _topological_sort(self, tree: Dict[str, Any]) -> List[str]:
        """Perform topological sort for installation order"""
        # Implementation of Kahn's algorithm
        result = []
        queue = []
        in_degree = defaultdict(int)
        
        # Build graph and calculate in-degrees
        # ... implementation ...
        
        return result
    
    async def _apply_resolution(
        self,
        tree: Dict[str, Any],
        resolution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply conflict resolution to dependency tree"""
        # Implementation would modify tree based on resolution
        return tree


class ConflictResolver:
    """Resolve dependency conflicts intelligently"""
    
    async def detect_conflicts(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect version conflicts in dependency tree"""
        conflicts = []
        version_requirements = defaultdict(list)
        
        # Traverse tree and collect version requirements
        # ... implementation ...
        
        return conflicts
    
    async def resolve(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve detected conflicts"""
        resolution = {}
        
        for conflict in conflicts:
            # Try different resolution strategies
            resolved = await self._try_resolution_strategies(conflict)
            resolution[conflict["package"]] = resolved
        
        return resolution
    
    async def _try_resolution_strategies(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Try various resolution strategies"""
        strategies = [
            self._resolve_by_latest_compatible,
            self._resolve_by_popularity,
            self._resolve_by_performance,
            self._resolve_by_security,
        ]
        
        for strategy in strategies:
            result = await strategy(conflict)
            if result["success"]:
                return result
        
        # If all strategies fail, return error
        return {"success": False, "reason": "No compatible version found"}
    
    async def _resolve_by_latest_compatible(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by finding latest compatible version"""
        # Implementation
        return {"success": True, "version": "1.0.0"}
    
    async def _resolve_by_popularity(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by choosing most popular version"""
        # Implementation
        return {"success": False}
    
    async def _resolve_by_performance(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by choosing best performing version"""
        # Implementation
        return {"success": False}
    
    async def _resolve_by_security(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by choosing most secure version"""
        # Implementation
        return {"success": False}


class VersionMatcher:
    """Match and compare version specifications"""
    
    def matches(self, version: str, spec: str) -> bool:
        """Check if version matches specification"""
        # Support various version specs: ^1.0.0, ~1.2.0, >=1.0.0, etc.
        # Implementation would parse and compare
        return True


class MarketplaceSearchEngine:
    """Advanced search engine with semantic capabilities"""
    
    def __init__(self):
        self.index = SearchIndex()
        self.embeddings = EmbeddingService()
        self.ranker = RelevanceRanker()
        self.cache = CacheManager()
    
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        search_mode: str = "hybrid",  # keyword, semantic, hybrid
        personalization: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple strategies
        
        Args:
            query: Search query
            filters: Filter criteria
            limit: Maximum results
            offset: Pagination offset
            search_mode: Search strategy
            personalization: User preferences for ranking
        
        Returns:
            Ranked search results
        """
        # Check cache
        cache_key = self._generate_cache_key(query, filters, personalization)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached[offset:offset + limit]
        
        results = []
        
        if search_mode in ["keyword", "hybrid"]:
            # Keyword search
            keyword_results = await self.index.search_keywords(query, filters)
            results.extend(keyword_results)
        
        if search_mode in ["semantic", "hybrid"]:
            # Semantic search
            query_embedding = await self.embeddings.encode(query)
            semantic_results = await self.index.search_semantic(
                query_embedding,
                filters,
                top_k=limit * 3,  # Get more for re-ranking
            )
            results.extend(semantic_results)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            if result["id"] not in seen:
                seen.add(result["id"])
                unique_results.append(result)
        
        # Rank results
        ranked_results = await self.ranker.rank(
            query=query,
            results=unique_results,
            personalization=personalization,
        )
        
        # Cache results
        await self.cache.set(cache_key, ranked_results, ttl=300)  # 5 minutes
        
        return ranked_results[offset:offset + limit]
    
    async def find_similar(
        self,
        reference_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find similar agents/tools"""
        # Get reference item
        reference = await self.index.get_by_id(reference_id)
        if not reference:
            raise NotFoundError(f"Reference item {reference_id} not found")
        
        # Get embedding
        ref_embedding = await self.embeddings.get_cached_embedding(reference_id)
        if ref_embedding is None:
            # Generate embedding from description
            ref_embedding = await self.embeddings.encode(
                f"{reference['name']} {reference['description']}"
            )
        
        # Find similar items
        similar = await self.index.find_similar_by_embedding(
            ref_embedding,
            exclude_ids=[reference_id],
            limit=limit * 2,
        )
        
        # Filter by threshold
        filtered = [
            item for item in similar
            if item["similarity"] >= similarity_threshold
        ]
        
        return filtered[:limit]
    
    async def get_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get personalized recommendations"""
        # Get user history and preferences
        user_profile = await self._get_user_profile(user_id)
        
        # Get context-aware recommendations
        recommendations = []
        
        # Collaborative filtering
        collab_recs = await self._collaborative_recommendations(user_profile, limit)
        recommendations.extend(collab_recs)
        
        # Content-based filtering
        content_recs = await self._content_based_recommendations(user_profile, limit)
        recommendations.extend(content_recs)
        
        # Context-aware filtering
        if context:
            context_recs = await self._context_aware_recommendations(context, limit)
            recommendations.extend(context_recs)
        
        # De-duplicate and rank
        final_recs = await self._merge_and_rank_recommendations(
            recommendations,
            user_profile,
            limit,
        )
        
        return final_recs
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps(args, sort_keys=True, default=str)
        return f"search:{hashlib.sha256(key_data.encode()).hexdigest()}"
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile for personalization"""
        # Implementation would fetch from database
        return {
            "user_id": user_id,
            "preferences": {},
            "history": [],
        }
    
    async def _collaborative_recommendations(
        self,
        user_profile: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations"""
        # Implementation would use user-item interactions
        return []
    
    async def _content_based_recommendations(
        self,
        user_profile: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Generate content-based recommendations"""
        # Implementation would use item features
        return []
    
    async def _context_aware_recommendations(
        self,
        context: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Generate context-aware recommendations"""
        # Implementation would consider current context
        return []
    
    async def _merge_and_rank_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Merge and rank recommendations from multiple sources"""
        # Implementation would combine and rank
        return recommendations[:limit]


class SearchIndex:
    """Inverted index for fast search"""
    
    def __init__(self):
        self.keyword_index = {}
        self.embedding_index = {}
        self.metadata_index = {}
    
    async def search_keywords(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search by keywords"""
        # Implementation would search inverted index
        return []
    
    async def search_semantic(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search by semantic similarity"""
        # Implementation would use vector similarity search
        return []
    
    async def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        # Implementation would fetch from index
        return None
    
    async def find_similar_by_embedding(
        self,
        embedding: np.ndarray,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find similar items by embedding"""
        # Implementation would use vector similarity
        return []


class EmbeddingService:
    """Generate and manage embeddings"""
    
    def __init__(self):
        self.model = "text-embedding-3-large"
        self.cache = {}
        self.gemini = GeminiClient()
    
    async def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        # Check cache
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = await self.gemini.generate_embedding(text)
        
        # Cache result
        self.cache[cache_key] = embedding
        
        return embedding
    
    async def get_cached_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for item"""
        # Implementation would check persistent cache
        return None


class RelevanceRanker:
    """Rank search results by relevance"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.ml_ranker = MLRanker()
    
    async def rank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        personalization: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank results by relevance"""
        # Extract features
        features = await self.feature_extractor.extract(query, results)
        
        # Apply ML ranking
        scores = await self.ml_ranker.predict_scores(features, personalization)
        
        # Sort by score
        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [result for result, _ in ranked]


class FeatureExtractor:
    """Extract features for ranking"""
    
    async def extract(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Extract ranking features"""
        # Implementation would extract various features
        return np.zeros((len(results), 10))


class MLRanker:
    """Machine learning based ranker"""
    
    async def predict_scores(
        self,
        features: np.ndarray,
        personalization: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Predict relevance scores"""
        # Implementation would use trained model
        return np.random.rand(features.shape[0])


class MarketplaceRatingSystem:
    """Advanced rating and review system"""
    
    def __init__(self):
        self.ratings_db = {}
        self.review_analyzer = ReviewAnalyzer()
        self.fraud_detector = FraudDetector()
    
    async def submit_rating(
        self,
        user_id: str,
        item_id: str,
        rating: int,
        review: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit rating with fraud detection"""
        # Check for fraud
        fraud_check = await self.fraud_detector.check_rating(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            review=review,
        )
        
        if fraud_check["is_fraudulent"]:
            raise ValidationError(f"Rating rejected: {fraud_check['reason']}")
        
        # Analyze review sentiment
        sentiment = None
        if review:
            sentiment = await self.review_analyzer.analyze_sentiment(review)
        
        # Store rating
        rating_id = str(uuid.uuid4())
        self.ratings_db[rating_id] = {
            "id": rating_id,
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "review": review,
            "sentiment": sentiment,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "verified": fraud_check["confidence"] > 0.8,
        }
        
        # Update item statistics
        await self._update_item_stats(item_id)
        
        return {
            "rating_id": rating_id,
            "status": "accepted",
            "verified": fraud_check["confidence"] > 0.8,
        }
    
    async def get_item_ratings(
        self,
        item_id: str,
        include_reviews: bool = True,
        verified_only: bool = False,
    ) -> Dict[str, Any]:
        """Get ratings for an item"""
        ratings = [
            r for r in self.ratings_db.values()
            if r["item_id"] == item_id
            and (not verified_only or r["verified"])
        ]
        
        if not ratings:
            return {
                "item_id": item_id,
                "average_rating": None,
                "total_ratings": 0,
                "ratings": [],
            }
        
        # Calculate statistics
        avg_rating = sum(r["rating"] for r in ratings) / len(ratings)
        
        # Get rating distribution
        distribution = defaultdict(int)
        for r in ratings:
            distribution[r["rating"]] += 1
        
        result = {
            "item_id": item_id,
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(ratings),
            "distribution": dict(distribution),
            "verified_percentage": sum(1 for r in ratings if r["verified"]) / len(ratings),
        }
        
        if include_reviews:
            result["reviews"] = [
                {
                    "rating": r["rating"],
                    "review": r["review"],
                    "sentiment": r["sentiment"],
                    "created_at": r["created_at"],
                    "verified": r["verified"],
                }
                for r in sorted(ratings, key=lambda x: x["created_at"], reverse=True)[:10]
            ]
        
        return result
    
    async def _update_item_stats(self, item_id: str):
        """Update item rating statistics"""
        # Implementation would update database
        pass


class ReviewAnalyzer:
    """Analyze review content"""
    
    async def analyze_sentiment(self, review: str) -> Dict[str, float]:
        """Analyze review sentiment"""
        # Implementation would use NLP
        return {
            "positive": 0.8,
            "negative": 0.1,
            "neutral": 0.1,
        }


class FraudDetector:
    """Detect fraudulent ratings"""
    
    async def check_rating(
        self,
        user_id: str,
        item_id: str,
        rating: int,
        review: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check rating for fraud indicators"""
        # Implementation would check various signals
        return {
            "is_fraudulent": False,
            "confidence": 0.95,
            "reason": None,
        }


class AgentVersionManager:
    """Manage agent versions and rollbacks"""
    
    def __init__(self):
        self.versions = defaultdict(list)
        self.version_strategy = VersioningStrategy()
    
    async def create_version(
        self,
        agent_id: str,
        manifest: AgentManifest,
        package_data: bytes,
        changelog: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new version of agent"""
        # Get current versions
        current_versions = self.versions.get(agent_id, [])
        
        # Determine new version
        if self.version_strategy.auto_increment and current_versions:
            new_version = self._increment_version(current_versions[-1]["version"])
        else:
            new_version = manifest.version
        
        # Check version conflicts
        if any(v["version"] == new_version for v in current_versions):
            raise ConflictError(f"Version {new_version} already exists")
        
        # Create version entry
        version_entry = {
            "version": new_version,
            "manifest": manifest,
            "package_checksum": hashlib.sha256(package_data).hexdigest(),
            "changelog": changelog,
            "created_at": datetime.utcnow(),
            "downloads": 0,
            "active": True,
        }
        
        # Store version
        self.versions[agent_id].append(version_entry)
        
        # Enforce retention policy
        if len(self.versions[agent_id]) > self.version_strategy.max_versions_retained:
            # Archive old versions
            await self._archive_old_versions(agent_id)
        
        return {
            "agent_id": agent_id,
            "version": new_version,
            "previous_version": current_versions[-1]["version"] if current_versions else None,
            "total_versions": len(self.versions[agent_id]),
        }
    
    async def rollback_version(
        self,
        agent_id: str,
        target_version: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Rollback to a previous version"""
        if not self.version_strategy.rollback_enabled:
            raise ValidationError("Rollback is not enabled for this agent")
        
        versions = self.versions.get(agent_id, [])
        target = None
        
        for v in versions:
            if v["version"] == target_version:
                target = v
                break
        
        if not target:
            raise NotFoundError(f"Version {target_version} not found")
        
        # Mark current version as inactive
        for v in versions:
            v["active"] = v["version"] == target_version
        
        # Record rollback
        rollback_entry = {
            "agent_id": agent_id,
            "from_version": self._get_active_version(agent_id),
            "to_version": target_version,
            "reason": reason,
            "timestamp": datetime.utcnow(),
        }
        
        return rollback_entry
    
    def _increment_version(self, version: str) -> str:
        """Increment semantic version"""
        parts = version.split(".")
        if len(parts) == 3:
            # Increment patch version
            parts[2] = str(int(parts[2]) + 1)
            return ".".join(parts)
        return version
    
    def _get_active_version(self, agent_id: str) -> Optional[str]:
        """Get currently active version"""
        versions = self.versions.get(agent_id, [])
        for v in versions:
            if v.get("active", False):
                return v["version"]
        return None
    
    async def _archive_old_versions(self, agent_id: str):
        """Archive old versions according to retention policy"""
        # Implementation would move old versions to cold storage
        pass


class CommunityContributionHub:
    """Manage community contributions and collaboration"""
    
    def __init__(self):
        self.contributions = {}
        self.peer_review = PeerReviewSystem()
        self.reward_system = ContributorRewardSystem()
    
    async def submit_contribution(
        self,
        contributor_id: str,
        contribution_type: str,  # agent, tool, template, workflow
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit community contribution"""
        # Validate contribution
        validation = await self._validate_contribution(contribution_type, content)
        if not validation["valid"]:
            raise ValidationError(validation["errors"])
        
        # Create contribution entry
        contribution_id = str(uuid.uuid4())
        contribution = {
            "id": contribution_id,
            "contributor_id": contributor_id,
            "type": contribution_type,
            "content": content,
            "metadata": metadata or {},
            "status": "pending_review",
            "submitted_at": datetime.utcnow(),
            "reviews": [],
            "score": 0,
        }
        
        self.contributions[contribution_id] = contribution
        
        # Start peer review process
        await self.peer_review.initiate_review(contribution_id)
        
        return {
            "contribution_id": contribution_id,
            "status": "submitted",
            "review_process": "initiated",
        }
    
    async def _validate_contribution(
        self,
        contribution_type: str,
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate contribution content"""
        # Implementation would validate based on type
        return {"valid": True, "errors": []}


class PeerReviewSystem:
    """Manage peer review process"""
    
    async def initiate_review(self, contribution_id: str):
        """Start peer review process"""
        # Implementation would assign reviewers
        pass


class ContributorRewardSystem:
    """Manage contributor rewards and recognition"""
    
    async def calculate_rewards(
        self,
        contributor_id: str,
        contribution_id: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate rewards for contribution"""
        # Implementation would calculate based on impact
        return {
            "points": 100,
            "badges": ["first_contribution"],
            "monetary_reward": 0,
        }