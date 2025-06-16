"""
Agent Registry Service
Handles agent package registration, versioning, and management
"""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .models import (
    AgentManifest,
    AgentPackage,
    AgentStatus,
    PerformanceMetrics,
    SecurityReport,
)
from .performance_benchmarker import PerformanceBenchmarker
from .security_scanner import SecurityScanner


class AgentRegistry:
    """Central registry for agent packages"""

    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.security_scanner = SecurityScanner()
        self.benchmarker = PerformanceBenchmarker()
        self.storage_path = os.getenv("AGENT_STORAGE_PATH", "/tmp/nexus-forge/agents")
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    async def publish_agent(
        self,
        package_file: bytes,
        manifest: AgentManifest,
        author_id: str,
        author_email: str,
    ) -> AgentPackage:
        """
        Publish a new agent to the marketplace

        Args:
            package_file: Agent package file content
            manifest: Agent manifest
            author_id: Author user ID
            author_email: Author email

        Returns:
            Published agent package
        """
        # Validate manifest
        await self._validate_manifest(manifest)

        # Check for existing version
        existing = await self._get_agent_version(manifest.name, manifest.version)
        if existing:
            raise ValidationError(
                f"Agent {manifest.name} version {manifest.version} already exists"
            )

        # Save package to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(package_file)
            temp_path = temp_file.name

        try:
            # Security scan
            security_report = await self.security_scanner.scan_package(temp_path)
            if not security_report.passed:
                raise ValidationError(
                    f"Security scan failed: Risk score {security_report.risk_score}"
                )

            # Performance benchmarking
            performance_metrics = await self.benchmarker.benchmark_agent(
                temp_path, manifest
            )
            if performance_metrics.performance_score < 60:
                raise ValidationError(
                    f"Performance below requirements: {performance_metrics.performance_score}"
                )

            # Create package entry
            agent_package = AgentPackage(
                manifest=manifest,
                author_id=author_id,
                author_email=author_email,
                security_report=security_report,
                performance_metrics=performance_metrics,
                package_size_bytes=len(package_file),
                status=AgentStatus.PENDING,
            )

            # Store package file
            package_path = await self._store_package_file(
                agent_package.id, manifest.name, manifest.version, temp_path
            )
            agent_package.package_url = package_path

            # Save to database
            await self._save_agent_package(agent_package)

            # Add to review queue
            await self._add_to_review_queue(agent_package)

            # Clear cache
            await self._clear_agent_cache(manifest.name)

            return agent_package

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def get_agent(self, name: str, version: Optional[str] = None) -> AgentPackage:
        """Get agent package by name and optional version"""
        # Check cache
        cache_key = f"agent:{name}:{version or 'latest'}"
        cached = await self.cache.get(cache_key)
        if cached:
            return AgentPackage(**cached)

        # Query database
        query = self.supabase.client.table("agent_packages").select("*")
        query = query.eq("manifest->name", name)

        if version:
            query = query.eq("manifest->version", version)
        else:
            # Get latest approved version
            query = query.eq("status", AgentStatus.APPROVED.value)
            query = query.order("created_at", desc=True)
            query = query.limit(1)

        result = await query.execute()

        if not result.data:
            raise NotFoundError(f"Agent {name} not found")

        agent_package = AgentPackage(**result.data[0])

        # Cache result
        await self.cache.set(cache_key, agent_package.dict(), ttl=3600)

        return agent_package

    async def search_agents(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[AgentPackage]:
        """Search for agents in the marketplace"""
        # Build cache key
        cache_key = f"search:{query}:{category}:{tags}:{author}:{limit}:{offset}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [AgentPackage(**pkg) for pkg in cached]

        # Build query
        db_query = self.supabase.client.table("agent_packages").select("*")
        db_query = db_query.eq("status", AgentStatus.APPROVED.value)

        if query:
            # Full-text search on name, display_name, and description
            db_query = db_query.or_(
                f"manifest->name.ilike.%{query}%,"
                f"manifest->display_name.ilike.%{query}%,"
                f"manifest->description.ilike.%{query}%"
            )

        if category:
            db_query = db_query.eq("manifest->category", category)

        if tags:
            # Search for any matching tags
            db_query = db_query.contains("manifest->tags", tags)

        if author:
            db_query = db_query.eq("manifest->author", author)

        # Order by downloads and rating
        db_query = db_query.order("downloads", desc=True)
        db_query = db_query.order("rating", desc=True)

        # Pagination
        db_query = db_query.range(offset, offset + limit - 1)

        result = await db_query.execute()

        agents = [AgentPackage(**pkg) for pkg in result.data]

        # Cache results
        await self.cache.set(cache_key, [pkg.dict() for pkg in agents], ttl=300)

        return agents

    async def install_agent(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Install an agent for a user"""
        # Get agent package
        agent = await self._get_agent_by_id(agent_id)

        if agent.status != AgentStatus.APPROVED:
            raise ValidationError("Agent is not approved for installation")

        # Check dependencies
        await self._check_dependencies(agent.manifest.dependencies)

        # Record installation
        await self._record_installation(agent_id, user_id)

        # Increment download count
        await self._increment_downloads(agent_id)

        # Return installation info
        return {
            "agent_id": agent_id,
            "name": agent.manifest.name,
            "version": agent.manifest.version,
            "package_url": agent.package_url,
            "main_class": agent.manifest.main_class,
            "config_schema": agent.manifest.config_schema,
            "dependencies": agent.manifest.dependencies,
        }

    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> AgentPackage:
        """Update agent status (for admin review)"""
        agent = await self._get_agent_by_id(agent_id)

        agent.status = status
        agent.review_status = status.value
        agent.review_notes = notes
        agent.reviewed_by = reviewer_id
        agent.reviewed_at = datetime.utcnow()

        if status == AgentStatus.APPROVED:
            agent.published_at = datetime.utcnow()

        # Update in database
        await self._update_agent_package(agent)

        # Clear cache
        await self._clear_agent_cache(agent.manifest.name)

        # Notify author
        await self._notify_author(agent, status, notes)

        return agent

    async def rate_agent(
        self, agent_id: str, user_id: str, rating: int, review: Optional[str] = None
    ) -> None:
        """Rate an agent"""
        if not 1 <= rating <= 5:
            raise ValidationError("Rating must be between 1 and 5")

        # Check if user has already rated
        existing = (
            await self.supabase.client.table("agent_ratings")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("user_id", user_id)
            .execute()
        )

        if existing.data:
            # Update existing rating
            await self.supabase.client.table("agent_ratings").update(
                {
                    "rating": rating,
                    "review": review,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            ).eq("id", existing.data[0]["id"]).execute()
        else:
            # Create new rating
            await self.supabase.client.table("agent_ratings").insert(
                {
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "rating": rating,
                    "review": review,
                }
            ).execute()

        # Update agent average rating
        await self._update_agent_rating(agent_id)

        # Clear cache
        agent = await self._get_agent_by_id(agent_id)
        await self._clear_agent_cache(agent.manifest.name)

    # Private helper methods

    async def _validate_manifest(self, manifest: AgentManifest) -> None:
        """Validate agent manifest"""
        # Check name availability
        existing = (
            await self.supabase.client.table("agent_packages")
            .select("id")
            .eq("manifest->name", manifest.name)
            .execute()
        )

        if existing.data:
            # Check if we're creating a new version
            versions = [pkg["manifest"]["version"] for pkg in existing.data]
            if manifest.version in versions:
                raise ValidationError(
                    f"Version {manifest.version} already exists for {manifest.name}"
                )

    async def _get_agent_version(
        self, name: str, version: str
    ) -> Optional[AgentPackage]:
        """Get specific agent version"""
        result = (
            await self.supabase.client.table("agent_packages")
            .select("*")
            .eq("manifest->name", name)
            .eq("manifest->version", version)
            .execute()
        )

        if result.data:
            return AgentPackage(**result.data[0])
        return None

    async def _store_package_file(
        self, agent_id: str, name: str, version: str, file_path: str
    ) -> str:
        """Store agent package file"""
        # Create storage path
        storage_dir = Path(self.storage_path) / name / version
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        dest_path = storage_dir / f"{name}-{version}.zip"
        shutil.copy2(file_path, dest_path)

        # In production, upload to object storage (S3, GCS)
        # For now, return local path
        return str(dest_path)

    async def _save_agent_package(self, package: AgentPackage) -> None:
        """Save agent package to database"""
        await self.supabase.client.table("agent_packages").insert(
            package.dict()
        ).execute()

    async def _update_agent_package(self, package: AgentPackage) -> None:
        """Update agent package in database"""
        package.updated_at = datetime.utcnow()
        await self.supabase.client.table("agent_packages").update(package.dict()).eq(
            "id", package.id
        ).execute()

    async def _add_to_review_queue(self, package: AgentPackage) -> None:
        """Add agent to review queue"""
        await self.supabase.client.table("review_queue").insert(
            {
                "agent_id": package.id,
                "agent_name": package.manifest.name,
                "agent_version": package.manifest.version,
                "author_id": package.author_id,
                "priority": "normal",
                "created_at": datetime.utcnow().isoformat(),
            }
        ).execute()

    async def _clear_agent_cache(self, name: str) -> None:
        """Clear all cache entries for an agent"""
        # Clear specific version caches
        pattern = f"agent:{name}:*"
        await self.cache.delete_pattern(pattern)

        # Clear search caches
        pattern = "search:*"
        await self.cache.delete_pattern(pattern)

    async def _get_agent_by_id(self, agent_id: str) -> AgentPackage:
        """Get agent by ID"""
        result = (
            await self.supabase.client.table("agent_packages")
            .select("*")
            .eq("id", agent_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Agent {agent_id} not found")

        return AgentPackage(**result.data[0])

    async def _check_dependencies(self, dependencies: List[Dict[str, str]]) -> None:
        """Check if all dependencies are available"""
        for dep in dependencies:
            dep_name = dep.get("name")
            dep_version = dep.get("version", "latest")

            try:
                await self.get_agent(dep_name, dep_version)
            except NotFoundError:
                raise ValidationError(f"Dependency {dep_name}@{dep_version} not found")

    async def _record_installation(self, agent_id: str, user_id: str) -> None:
        """Record agent installation"""
        await self.supabase.client.table("agent_installations").insert(
            {
                "agent_id": agent_id,
                "user_id": user_id,
                "installed_at": datetime.utcnow().isoformat(),
            }
        ).execute()

    async def _increment_downloads(self, agent_id: str) -> None:
        """Increment agent download count"""
        await self.supabase.client.rpc(
            "increment_downloads", {"agent_id": agent_id}
        ).execute()

    async def _update_agent_rating(self, agent_id: str) -> None:
        """Update agent average rating"""
        # Calculate average rating
        result = (
            await self.supabase.client.table("agent_ratings")
            .select("rating")
            .eq("agent_id", agent_id)
            .execute()
        )

        if result.data:
            ratings = [r["rating"] for r in result.data]
            avg_rating = sum(ratings) / len(ratings)
            rating_count = len(ratings)

            # Update agent package
            await self.supabase.client.table("agent_packages").update(
                {"rating": round(avg_rating, 2), "rating_count": rating_count}
            ).eq("id", agent_id).execute()

    async def _notify_author(
        self, agent: AgentPackage, status: AgentStatus, notes: Optional[str]
    ) -> None:
        """Notify author about status change"""
        # In production, send email or push notification
        # For now, just log
        print(f"Notification: Agent {agent.manifest.name} status changed to {status}")
        if notes:
            print(f"Review notes: {notes}")
