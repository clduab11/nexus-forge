"""
Region Manager
Handles region lifecycle, monitoring, and coordination
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor

from .models import (
    Region, EdgeLocation, RegionStatus, HealthStatus, 
    ResourceMetrics, LoadBalancingStrategy, RegionConfig,
    RegionMetrics, GlobalMetrics, GeographicLocation
)
from ..core.cache import RedisCache
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from ..core.exceptions import ValidationError, NotFoundError, ResourceError

logger = logging.getLogger(__name__)


class RegionManager:
    """Manages multi-region infrastructure and coordination"""
    
    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 10   # seconds
        self.unhealthy_threshold = 3     # consecutive failures
        
        # Monitoring configuration
        self.metrics_collection_interval = 60  # seconds
        self.metrics_retention_hours = 24
        
        # Background tasks
        self._health_check_task = None
        self._metrics_task = None
        self._running = False
    
    async def start(self):
        """Start background monitoring tasks"""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Region manager started")
    
    async def stop(self):
        """Stop background monitoring tasks"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._metrics_task:
            self._metrics_task.cancel()
        
        logger.info("Region manager stopped")
    
    # Region Management
    
    async def register_region(
        self, 
        region: Region,
        validate_connectivity: bool = True
    ) -> Region:
        """Register a new region"""
        logger.info(f"Registering region: {region.code} ({region.name})")
        
        # Validate region configuration
        await self._validate_region_config(region)
        
        # Check for duplicate region codes
        existing = await self._get_region_by_code(region.code)
        if existing:
            raise ValidationError(f"Region code {region.code} already exists")
        
        try:
            # Test connectivity if requested
            if validate_connectivity:
                connectivity_result = await self._test_region_connectivity(region)
                if not connectivity_result["success"]:
                    raise ValidationError(f"Region connectivity test failed: {connectivity_result['error']}")
            
            # Initialize region infrastructure
            await self._initialize_region_infrastructure(region)
            
            # Save region to database
            await self._save_region(region)
            
            # Cache region information
            await self._cache_region(region)
            
            # Start monitoring for this region
            await self._start_region_monitoring(region)
            
            logger.info(f"Successfully registered region: {region.code}")
            return region
            
        except Exception as e:
            logger.error(f"Failed to register region {region.code}: {e}")
            # Cleanup on failure
            await self._cleanup_failed_region(region)
            raise
    
    async def get_region(self, region_id: str) -> Region:
        """Get region by ID"""
        # Check cache first
        cache_key = f"region:id:{region_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return Region(**cached)
        
        # Query database
        result = await self.supabase.client.table("regions") \
            .select("*") \
            .eq("id", region_id) \
            .execute()
        
        if not result.data:
            raise NotFoundError(f"Region {region_id} not found")
        
        region = Region(**result.data[0])
        
        # Cache result
        await self.cache.set(cache_key, region.dict(), ttl=1800)
        
        return region
    
    async def get_region_by_code(self, region_code: str) -> Region:
        """Get region by code"""
        return await self._get_region_by_code(region_code)
    
    async def list_regions(
        self, 
        status: Optional[RegionStatus] = None,
        provider: Optional[str] = None
    ) -> List[Region]:
        """List all regions with optional filtering"""
        query = self.supabase.client.table("regions").select("*")
        
        if status:
            query = query.eq("status", status.value)
        
        if provider:
            query = query.eq("provider", provider)
        
        result = await query.execute()
        
        return [Region(**region_data) for region_data in result.data]
    
    async def update_region(self, region: Region) -> Region:
        """Update region configuration"""
        region.updated_at = datetime.utcnow()
        
        # Validate updated configuration
        await self._validate_region_config(region)
        
        # Save to database
        await self._update_region(region)
        
        # Clear cache
        await self._clear_region_cache(region.id)
        
        logger.info(f"Updated region: {region.code}")
        return region
    
    async def deregister_region(self, region_id: str) -> bool:
        """Deregister and cleanup a region"""
        region = await self.get_region(region_id)
        
        logger.info(f"Deregistering region: {region.code}")
        
        try:
            # Stop monitoring
            await self._stop_region_monitoring(region)
            
            # Cleanup infrastructure
            await self._cleanup_region_infrastructure(region)
            
            # Remove from database
            await self.supabase.client.table("regions") \
                .delete() \
                .eq("id", region_id) \
                .execute()
            
            # Clear cache
            await self._clear_region_cache(region_id)
            
            logger.info(f"Successfully deregistered region: {region.code}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister region {region.code}: {e}")
            return False
    
    # Edge Location Management
    
    async def register_edge_location(self, edge_location: EdgeLocation) -> EdgeLocation:
        """Register a new edge location"""
        logger.info(f"Registering edge location: {edge_location.code}")
        
        # Validate parent region exists
        await self.get_region(edge_location.parent_region_id)
        
        # Save to database
        await self._save_edge_location(edge_location)
        
        # Cache edge location
        await self._cache_edge_location(edge_location)
        
        # Start monitoring
        await self._start_edge_monitoring(edge_location)
        
        logger.info(f"Successfully registered edge location: {edge_location.code}")
        return edge_location
    
    async def list_edge_locations(
        self, 
        parent_region_id: Optional[str] = None
    ) -> List[EdgeLocation]:
        """List edge locations"""
        query = self.supabase.client.table("edge_locations").select("*")
        
        if parent_region_id:
            query = query.eq("parent_region_id", parent_region_id)
        
        result = await query.execute()
        
        return [EdgeLocation(**edge_data) for edge_data in result.data]
    
    # Health Monitoring
    
    async def check_region_health(self, region_id: str) -> Dict[str, Any]:
        """Perform health check for a specific region"""
        region = await self.get_region(region_id)
        
        health_result = {
            "region_id": region_id,
            "region_code": region.code,
            "timestamp": datetime.utcnow().isoformat(),
            "status": HealthStatus.UNKNOWN,
            "checks": {},
            "metrics": None
        }
        
        try:
            # Connectivity check
            connectivity = await self._check_region_connectivity(region)
            health_result["checks"]["connectivity"] = connectivity
            
            # Service health check
            services = await self._check_region_services(region)
            health_result["checks"]["services"] = services
            
            # Resource check
            resources = await self._check_region_resources(region)
            health_result["checks"]["resources"] = resources
            
            # Determine overall status
            overall_status = self._calculate_health_status(health_result["checks"])
            health_result["status"] = overall_status
            
            # Update region status if changed
            if region.health_status != overall_status:
                region.health_status = overall_status
                region.last_health_check = datetime.utcnow()
                await self._update_region(region)
            
            # Collect metrics
            metrics = await self._collect_region_metrics(region)
            health_result["metrics"] = metrics.dict() if metrics else None
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed for region {region.code}: {e}")
            health_result["status"] = HealthStatus.CRITICAL
            health_result["error"] = str(e)
            return health_result
    
    async def get_global_health(self) -> Dict[str, Any]:
        """Get global health status across all regions"""
        regions = await self.list_regions()
        
        global_health = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_regions": len(regions),
            "healthy_regions": 0,
            "warning_regions": 0,
            "critical_regions": 0,
            "offline_regions": 0,
            "regions": []
        }
        
        # Check health for each region
        health_tasks = [
            self.check_region_health(region.id) 
            for region in regions
        ]
        
        region_health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for i, result in enumerate(region_health_results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for region {regions[i].code}: {result}")
                continue
            
            global_health["regions"].append(result)
            
            # Count by status
            status = result["status"]
            if status == HealthStatus.HEALTHY:
                global_health["healthy_regions"] += 1
            elif status == HealthStatus.WARNING:
                global_health["warning_regions"] += 1
            elif status == HealthStatus.CRITICAL:
                global_health["critical_regions"] += 1
            else:
                global_health["offline_regions"] += 1
        
        # Calculate global status
        total = global_health["total_regions"]
        if total == 0:
            global_health["overall_status"] = HealthStatus.UNKNOWN
        elif global_health["critical_regions"] > total * 0.5:
            global_health["overall_status"] = HealthStatus.CRITICAL
        elif global_health["warning_regions"] + global_health["critical_regions"] > total * 0.3:
            global_health["overall_status"] = HealthStatus.WARNING
        else:
            global_health["overall_status"] = HealthStatus.HEALTHY
        
        return global_health
    
    # Metrics and Analytics
    
    async def get_region_metrics(
        self, 
        region_id: str,
        hours: int = 24
    ) -> List[RegionMetrics]:
        """Get historical metrics for a region"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        result = await self.supabase.client.table("region_metrics") \
            .select("*") \
            .eq("region_id", region_id) \
            .gte("timestamp", start_time.isoformat()) \
            .lte("timestamp", end_time.isoformat()) \
            .order("timestamp", desc=False) \
            .execute()
        
        return [RegionMetrics(**metric) for metric in result.data]
    
    async def get_global_metrics(self, hours: int = 24) -> GlobalMetrics:
        """Get global metrics across all regions"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get region metrics
        result = await self.supabase.client.table("region_metrics") \
            .select("*") \
            .gte("timestamp", start_time.isoformat()) \
            .lte("timestamp", end_time.isoformat()) \
            .execute()
        
        region_metrics = [RegionMetrics(**metric) for metric in result.data]
        
        # Calculate global aggregates
        global_metrics = GlobalMetrics(
            total_regions=len(set(m.region_id for m in region_metrics)),
            region_metrics=region_metrics
        )
        
        # Aggregate metrics
        if region_metrics:
            global_metrics.global_requests_per_second = sum(
                m.total_requests for m in region_metrics
            ) / len(region_metrics)
            
            global_metrics.global_avg_latency_ms = sum(
                m.avg_response_time_ms for m in region_metrics
            ) / len(region_metrics)
            
            global_metrics.total_hourly_cost = sum(
                m.hourly_cost for m in region_metrics
            )
            
            global_metrics.total_monthly_cost = global_metrics.total_hourly_cost * 24 * 30
        
        return global_metrics
    
    # Failover Management
    
    async def trigger_failover(
        self, 
        failed_region_id: str, 
        target_region_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger failover from failed region to target region"""
        logger.warning(f"Triggering failover from region {failed_region_id}")
        
        failed_region = await self.get_region(failed_region_id)
        
        # Find target region if not specified
        if not target_region_id:
            target_region_id = await self._find_best_failover_region(failed_region_id)
        
        target_region = await self.get_region(target_region_id)
        
        failover_result = {
            "failed_region": failed_region.code,
            "target_region": target_region.code,
            "started_at": datetime.utcnow().isoformat(),
            "steps": [],
            "success": False
        }
        
        try:
            # Step 1: Update DNS routing
            step1 = await self._update_dns_routing(failed_region, target_region)
            failover_result["steps"].append({"dns_routing": step1})
            
            # Step 2: Update load balancer configuration
            step2 = await self._update_load_balancer_failover(failed_region, target_region)
            failover_result["steps"].append({"load_balancer": step2})
            
            # Step 3: Migrate active connections
            step3 = await self._migrate_active_connections(failed_region, target_region)
            failover_result["steps"].append({"connection_migration": step3})
            
            # Step 4: Update region status
            failed_region.status = RegionStatus.OFFLINE
            target_region.priority = min(target_region.priority, failed_region.priority)
            
            await self._update_region(failed_region)
            await self._update_region(target_region)
            
            failover_result["success"] = True
            failover_result["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info(f"Failover completed: {failed_region.code} -> {target_region.code}")
            
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            failover_result["error"] = str(e)
            failover_result["failed_at"] = datetime.utcnow().isoformat()
        
        # Log failover event
        await self._log_failover_event(failover_result)
        
        return failover_result
    
    # Private helper methods
    
    async def _validate_region_config(self, region: Region) -> None:
        """Validate region configuration"""
        # Check required fields
        if not region.compute.cluster_name:
            raise ValidationError("Cluster name is required")
        
        if region.compute.min_nodes > region.compute.max_nodes:
            raise ValidationError("Min nodes cannot be greater than max nodes")
        
        # Validate geographic location
        if not (-90 <= region.location.latitude <= 90):
            raise ValidationError("Invalid latitude")
        
        if not (-180 <= region.location.longitude <= 180):
            raise ValidationError("Invalid longitude")
    
    async def _get_region_by_code(self, region_code: str) -> Optional[Region]:
        """Get region by code"""
        # Check cache first
        cache_key = f"region:code:{region_code}"
        cached = await self.cache.get(cache_key)
        if cached:
            return Region(**cached)
        
        # Query database
        result = await self.supabase.client.table("regions") \
            .select("*") \
            .eq("code", region_code) \
            .execute()
        
        if not result.data:
            return None
        
        region = Region(**result.data[0])
        
        # Cache result
        await self.cache.set(cache_key, region.dict(), ttl=1800)
        
        return region
    
    async def _test_region_connectivity(self, region: Region) -> Dict[str, Any]:
        """Test connectivity to region"""
        # This would implement actual connectivity testing
        # For now, return success
        return {"success": True, "latency_ms": 50.0}
    
    async def _initialize_region_infrastructure(self, region: Region) -> None:
        """Initialize region infrastructure"""
        # This would implement actual infrastructure provisioning
        logger.info(f"Initializing infrastructure for region {region.code}")
    
    async def _save_region(self, region: Region) -> None:
        """Save region to database"""
        await self.supabase.client.table("regions").insert(
            region.dict()
        ).execute()
    
    async def _update_region(self, region: Region) -> None:
        """Update region in database"""
        await self.supabase.client.table("regions") \
            .update(region.dict()) \
            .eq("id", region.id) \
            .execute()
    
    async def _cache_region(self, region: Region) -> None:
        """Cache region information"""
        await self.cache.set(f"region:id:{region.id}", region.dict(), ttl=1800)
        await self.cache.set(f"region:code:{region.code}", region.dict(), ttl=1800)
    
    async def _clear_region_cache(self, region_id: str) -> None:
        """Clear region cache"""
        region = await self.get_region(region_id)
        await self.cache.delete(f"region:id:{region_id}")
        await self.cache.delete(f"region:code:{region.code}")
    
    async def _cleanup_failed_region(self, region: Region) -> None:
        """Cleanup resources for failed region registration"""
        try:
            await self._cleanup_region_infrastructure(region)
        except Exception as e:
            logger.error(f"Failed to cleanup region {region.code}: {e}")
    
    async def _cleanup_region_infrastructure(self, region: Region) -> None:
        """Cleanup region infrastructure"""
        logger.info(f"Cleaning up infrastructure for region {region.code}")
    
    async def _start_region_monitoring(self, region: Region) -> None:
        """Start monitoring for region"""
        logger.info(f"Starting monitoring for region {region.code}")
    
    async def _stop_region_monitoring(self, region: Region) -> None:
        """Stop monitoring for region"""
        logger.info(f"Stopping monitoring for region {region.code}")
    
    async def _save_edge_location(self, edge_location: EdgeLocation) -> None:
        """Save edge location to database"""
        await self.supabase.client.table("edge_locations").insert(
            edge_location.dict()
        ).execute()
    
    async def _cache_edge_location(self, edge_location: EdgeLocation) -> None:
        """Cache edge location"""
        await self.cache.set(
            f"edge_location:{edge_location.id}", 
            edge_location.dict(), 
            ttl=1800
        )
    
    async def _start_edge_monitoring(self, edge_location: EdgeLocation) -> None:
        """Start monitoring for edge location"""
        logger.info(f"Starting monitoring for edge location {edge_location.code}")
    
    # Background task methods
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(10)  # Short delay on error
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(10)  # Short delay on error
    
    async def _perform_health_checks(self):
        """Perform health checks for all regions"""
        regions = await self.list_regions()
        
        # Perform health checks in parallel
        health_tasks = [
            self.check_region_health(region.id) 
            for region in regions
        ]
        
        await asyncio.gather(*health_tasks, return_exceptions=True)
    
    async def _collect_metrics(self):
        """Collect metrics for all regions"""
        regions = await self.list_regions()
        
        for region in regions:
            try:
                metrics = await self._collect_region_metrics(region)
                if metrics:
                    await self._save_region_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to collect metrics for region {region.code}: {e}")
    
    async def _collect_region_metrics(self, region: Region) -> Optional[ResourceMetrics]:
        """Collect metrics for a specific region"""
        # This would implement actual metrics collection
        # For now, return dummy metrics
        return ResourceMetrics(
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            storage_usage_percent=40.0,
            network_bandwidth_mbps=100.0,
            active_connections=150,
            requests_per_second=25.0,
            avg_response_time_ms=120.0,
            p95_response_time_ms=200.0,
            p99_response_time_ms=350.0,
            error_rate_percent=1.5
        )
    
    async def _save_region_metrics(self, metrics: RegionMetrics) -> None:
        """Save region metrics to database"""
        await self.supabase.client.table("region_metrics").insert(
            metrics.dict()
        ).execute()
    
    async def _check_region_connectivity(self, region: Region) -> Dict[str, Any]:
        """Check region connectivity"""
        return {"status": "healthy", "latency_ms": 45.0}
    
    async def _check_region_services(self, region: Region) -> Dict[str, Any]:
        """Check region services health"""
        return {"api_service": "healthy", "database": "healthy", "cache": "healthy"}
    
    async def _check_region_resources(self, region: Region) -> Dict[str, Any]:
        """Check region resource utilization"""
        return {"cpu_ok": True, "memory_ok": True, "storage_ok": True}
    
    def _calculate_health_status(self, checks: Dict[str, Any]) -> HealthStatus:
        """Calculate overall health status from individual checks"""
        # Simple logic - can be made more sophisticated
        if all(check.get("status") == "healthy" for check in checks.values()):
            return HealthStatus.HEALTHY
        elif any(check.get("status") == "critical" for check in checks.values()):
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.WARNING
    
    async def _find_best_failover_region(self, failed_region_id: str) -> str:
        """Find the best region for failover"""
        regions = await self.list_regions(status=RegionStatus.ACTIVE)
        
        # Filter out the failed region
        available_regions = [r for r in regions if r.id != failed_region_id]
        
        if not available_regions:
            raise ResourceError("No available regions for failover")
        
        # Sort by priority (lower number = higher priority)
        available_regions.sort(key=lambda r: r.priority)
        
        return available_regions[0].id
    
    async def _update_dns_routing(self, failed_region: Region, target_region: Region) -> Dict[str, Any]:
        """Update DNS routing for failover"""
        # Implementation would update actual DNS records
        return {"success": True, "updated_records": ["api.nexusforge.ai"]}
    
    async def _update_load_balancer_failover(self, failed_region: Region, target_region: Region) -> Dict[str, Any]:
        """Update load balancer for failover"""
        # Implementation would update load balancer configuration
        return {"success": True, "target_region": target_region.code}
    
    async def _migrate_active_connections(self, failed_region: Region, target_region: Region) -> Dict[str, Any]:
        """Migrate active connections during failover"""
        # Implementation would handle connection migration
        return {"success": True, "migrated_connections": 150}
    
    async def _log_failover_event(self, failover_result: Dict[str, Any]) -> None:
        """Log failover event for audit purposes"""
        await self.supabase.client.table("failover_events").insert(
            failover_result
        ).execute()