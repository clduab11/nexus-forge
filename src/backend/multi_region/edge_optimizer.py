"""
Edge Optimizer
Optimizes edge locations and content delivery
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .load_balancer import GlobalLoadBalancer
from .models import (
    EdgeLocation,
    HealthStatus,
)
from .region_manager import RegionManager

logger = logging.getLogger(__name__)


class EdgeOptimizer:
    """Optimizes edge infrastructure and content delivery"""

    def __init__(
        self,
        region_manager: Optional[RegionManager] = None,
        load_balancer: Optional[GlobalLoadBalancer] = None,
    ):
        self.region_manager = region_manager or RegionManager()
        self.load_balancer = load_balancer or GlobalLoadBalancer()
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()

        # Optimization parameters
        self.target_cache_hit_rate = 0.85  # 85%
        self.max_edge_latency_ms = 150  # 150ms
        self.min_edge_utilization = 0.20  # 20%
        self.max_edge_utilization = 0.80  # 80%

        # Content optimization
        self.popular_content_threshold = 100  # requests/hour
        self.cache_ttl_multipliers = {
            "static": 3600,  # 1 hour
            "api": 300,  # 5 minutes
            "dynamic": 60,  # 1 minute
            "real-time": 5,  # 5 seconds
        }

        # Background tasks
        self._running = False
        self._optimization_task = None
        self._content_analysis_task = None

    async def start(self):
        """Start edge optimization service"""
        if self._running:
            return

        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._content_analysis_task = asyncio.create_task(self._content_analysis_loop())

        logger.info("Edge optimizer started")

    async def stop(self):
        """Stop edge optimization service"""
        self._running = False

        if self._optimization_task:
            self._optimization_task.cancel()

        if self._content_analysis_task:
            self._content_analysis_task.cancel()

        logger.info("Edge optimizer stopped")

    # Edge Location Optimization

    async def optimize_edge_placement(
        self, traffic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize edge location placement based on traffic patterns"""
        logger.info("Starting edge placement optimization")

        # Get current edge locations
        current_edges = await self.region_manager.list_edge_locations()

        # Analyze traffic patterns
        if not traffic_data:
            traffic_data = await self._analyze_traffic_patterns()

        # Identify optimization opportunities
        optimization_plan = {
            "current_edges": len(current_edges),
            "recommendations": [],
            "cost_impact": {},
            "performance_impact": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check for underutilized edges
        underutilized = await self._find_underutilized_edges(
            current_edges, traffic_data
        )
        for edge in underutilized:
            optimization_plan["recommendations"].append(
                {
                    "type": "consolidate_or_remove",
                    "edge_id": edge.id,
                    "edge_code": edge.code,
                    "utilization": edge.metrics.cpu_usage_percent,
                    "reason": "Low utilization",
                    "estimated_savings": edge.monthly_cost,
                }
            )

        # Check for coverage gaps
        coverage_gaps = await self._identify_coverage_gaps(traffic_data)
        for gap in coverage_gaps:
            optimization_plan["recommendations"].append(
                {
                    "type": "add_edge_location",
                    "location": gap["location"],
                    "traffic_volume": gap["traffic_volume"],
                    "avg_latency": gap["avg_latency"],
                    "reason": "Coverage gap",
                    "estimated_cost": gap["estimated_cost"],
                }
            )

        # Check for overloaded edges
        overloaded = await self._find_overloaded_edges(current_edges)
        for edge in overloaded:
            optimization_plan["recommendations"].append(
                {
                    "type": "scale_up_or_replicate",
                    "edge_id": edge.id,
                    "edge_code": edge.code,
                    "utilization": edge.metrics.cpu_usage_percent,
                    "reason": "High utilization",
                    "estimated_cost": edge.monthly_cost * 0.5,  # Assume 50% increase
                }
            )

        # Calculate overall impact
        total_savings = sum(
            rec.get("estimated_savings", 0)
            for rec in optimization_plan["recommendations"]
            if rec["type"] in ["consolidate_or_remove"]
        )

        total_costs = sum(
            rec.get("estimated_cost", 0)
            for rec in optimization_plan["recommendations"]
            if rec["type"] in ["add_edge_location", "scale_up_or_replicate"]
        )

        optimization_plan["cost_impact"] = {
            "potential_savings": total_savings,
            "additional_costs": total_costs,
            "net_impact": total_savings - total_costs,
        }

        return optimization_plan

    async def optimize_cache_strategy(
        self, edge_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize caching strategy for edge locations"""
        logger.info(f"Optimizing cache strategy for edge: {edge_id or 'all'}")

        # Get edge locations to optimize
        if edge_id:
            edges = [await self._get_edge_location(edge_id)]
        else:
            edges = await self.region_manager.list_edge_locations()

        optimization_results = {
            "optimized_edges": [],
            "cache_recommendations": [],
            "performance_improvements": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        for edge in edges:
            edge_optimization = await self._optimize_edge_cache(edge)
            optimization_results["optimized_edges"].append(edge_optimization)

        # Global cache recommendations
        global_recommendations = await self._generate_global_cache_recommendations()
        optimization_results["cache_recommendations"] = global_recommendations

        return optimization_results

    async def analyze_content_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze content delivery performance"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get content delivery metrics
        metrics = await self._get_content_delivery_metrics(start_time, end_time)

        analysis = {
            "time_window_hours": hours,
            "total_requests": metrics.get("total_requests", 0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0),
            "avg_response_time_ms": metrics.get("avg_response_time", 0),
            "popular_content": [],
            "slow_content": [],
            "cache_misses": [],
            "bandwidth_usage": metrics.get("bandwidth_gb", 0),
            "geographic_distribution": {},
        }

        # Analyze popular content
        popular_content = await self._analyze_popular_content(start_time, end_time)
        analysis["popular_content"] = popular_content[:10]  # Top 10

        # Analyze slow content
        slow_content = await self._analyze_slow_content(start_time, end_time)
        analysis["slow_content"] = slow_content[:10]  # Top 10

        # Analyze cache misses
        cache_misses = await self._analyze_cache_misses(start_time, end_time)
        analysis["cache_misses"] = cache_misses[:10]  # Top 10

        # Geographic distribution
        geographic_dist = await self._analyze_geographic_distribution(
            start_time, end_time
        )
        analysis["geographic_distribution"] = geographic_dist

        return analysis

    # CDN Optimization

    async def optimize_cdn_configuration(self) -> Dict[str, Any]:
        """Optimize CDN configuration across all edge locations"""
        logger.info("Optimizing CDN configuration")

        # Get current CDN performance
        cdn_metrics = await self._get_cdn_metrics()

        optimization = {
            "current_performance": cdn_metrics,
            "optimizations": [],
            "estimated_improvements": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Optimize cache rules
        cache_optimizations = await self._optimize_cache_rules()
        optimization["optimizations"].extend(cache_optimizations)

        # Optimize compression settings
        compression_optimizations = await self._optimize_compression()
        optimization["optimizations"].extend(compression_optimizations)

        # Optimize origin settings
        origin_optimizations = await self._optimize_origin_settings()
        optimization["optimizations"].extend(origin_optimizations)

        # Optimize geographic routing
        routing_optimizations = await self._optimize_geographic_routing()
        optimization["optimizations"].extend(routing_optimizations)

        # Calculate estimated improvements
        optimization["estimated_improvements"] = {
            "cache_hit_rate_improvement": 5.0,  # 5% improvement
            "response_time_reduction_ms": 25,  # 25ms reduction
            "bandwidth_savings_percent": 15,  # 15% bandwidth savings
            "cost_reduction_percent": 10,  # 10% cost reduction
        }

        return optimization

    async def update_edge_configuration(
        self, edge_id: str, config_updates: Dict[str, Any]
    ) -> bool:
        """Update edge location configuration"""
        logger.info(f"Updating edge configuration: {edge_id}")

        try:
            # Get current edge configuration
            edge = await self._get_edge_location(edge_id)

            # Validate configuration updates
            await self._validate_edge_config_updates(edge, config_updates)

            # Apply configuration updates
            success = await self._apply_edge_config_updates(edge, config_updates)

            if success:
                # Update database
                await self.supabase.client.table("edge_locations").update(
                    {
                        "config": {**edge.network.dict(), **config_updates},
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                ).eq("id", edge_id).execute()

                # Clear cache
                await self.cache.delete(f"edge_location:{edge_id}")

                logger.info(f"Updated edge configuration: {edge_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update edge configuration {edge_id}: {e}")
            return False

    # Performance Analysis

    async def analyze_edge_performance(
        self, edge_id: str, hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze performance for specific edge location"""
        edge = await self._get_edge_location(edge_id)

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get performance metrics
        metrics = await self._get_edge_performance_metrics(
            edge_id, start_time, end_time
        )

        analysis = {
            "edge_id": edge_id,
            "edge_code": edge.code,
            "edge_name": edge.name,
            "time_window_hours": hours,
            "performance_summary": {
                "avg_response_time_ms": metrics.get("avg_response_time", 0),
                "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                "error_rate": metrics.get("error_rate", 0),
                "uptime_percent": metrics.get("uptime_percent", 0),
                "requests_per_second": metrics.get("requests_per_second", 0),
            },
            "resource_utilization": {
                "cpu_avg": metrics.get("cpu_avg", 0),
                "memory_avg": metrics.get("memory_avg", 0),
                "bandwidth_avg": metrics.get("bandwidth_avg", 0),
                "storage_used_percent": metrics.get("storage_used_percent", 0),
            },
            "geographic_coverage": {},
            "content_analysis": {},
            "recommendations": [],
        }

        # Analyze geographic coverage
        coverage = await self._analyze_edge_geographic_coverage(
            edge_id, start_time, end_time
        )
        analysis["geographic_coverage"] = coverage

        # Analyze content performance
        content_perf = await self._analyze_edge_content_performance(
            edge_id, start_time, end_time
        )
        analysis["content_analysis"] = content_perf

        # Generate recommendations
        recommendations = await self._generate_edge_recommendations(edge, analysis)
        analysis["recommendations"] = recommendations

        return analysis

    async def get_global_edge_health(self) -> Dict[str, Any]:
        """Get global health status of all edge locations"""
        edges = await self.region_manager.list_edge_locations()

        global_health = {
            "total_edges": len(edges),
            "healthy_edges": 0,
            "warning_edges": 0,
            "critical_edges": 0,
            "overall_performance": {},
            "edge_details": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Analyze each edge
        total_response_time = 0
        total_cache_hit_rate = 0
        total_requests = 0

        for edge in edges:
            edge_health = {
                "edge_id": edge.id,
                "edge_code": edge.code,
                "health_status": edge.health_status.value,
                "response_time_ms": edge.metrics.avg_response_time_ms,
                "cache_hit_rate": 0,  # Would be calculated from metrics
                "uptime_percent": 99.5,  # Would be calculated from metrics
                "requests_per_second": edge.metrics.requests_per_second,
            }

            # Count by health status
            if edge.health_status == HealthStatus.HEALTHY:
                global_health["healthy_edges"] += 1
            elif edge.health_status == HealthStatus.WARNING:
                global_health["warning_edges"] += 1
            else:
                global_health["critical_edges"] += 1

            # Accumulate for averages
            total_response_time += edge.metrics.avg_response_time_ms
            total_requests += edge.metrics.requests_per_second

            global_health["edge_details"].append(edge_health)

        # Calculate global averages
        if edges:
            global_health["overall_performance"] = {
                "avg_response_time_ms": total_response_time / len(edges),
                "total_requests_per_second": total_requests,
                "global_cache_hit_rate": 0.82,  # Would be calculated from actual metrics
                "global_uptime_percent": 99.8,
            }

        return global_health

    # Private helper methods

    async def _analyze_traffic_patterns(self) -> Dict[str, Any]:
        """Analyze global traffic patterns"""
        # Get traffic data from load balancer logs
        traffic_data = await self.load_balancer.get_traffic_distribution(
            hours=168
        )  # 1 week

        return {
            "total_requests": traffic_data.get("total_requests", 0),
            "geographic_distribution": traffic_data.get("by_country", {}),
            "peak_hours": [],  # Would be calculated from hourly data
            "growth_trends": {},  # Would be calculated from historical data
        }

    async def _find_underutilized_edges(
        self, edges: List[EdgeLocation], traffic_data: Dict[str, Any]
    ) -> List[EdgeLocation]:
        """Find underutilized edge locations"""
        underutilized = []

        for edge in edges:
            if edge.metrics.cpu_usage_percent < self.min_edge_utilization * 100:
                underutilized.append(edge)

        return underutilized

    async def _identify_coverage_gaps(
        self, traffic_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify geographic coverage gaps"""
        gaps = []

        # Analyze traffic by country/region
        geographic_dist = traffic_data.get("geographic_distribution", {})

        # Find high-traffic areas without nearby edges
        for country, request_count in geographic_dist.items():
            if request_count > 1000:  # High traffic threshold
                # Check if there's an edge location nearby
                has_nearby_edge = await self._has_nearby_edge_location(country)

                if not has_nearby_edge:
                    gaps.append(
                        {
                            "location": country,
                            "traffic_volume": request_count,
                            "avg_latency": 200,  # Estimated high latency
                            "estimated_cost": 2000,  # Monthly cost estimate
                        }
                    )

        return gaps

    async def _find_overloaded_edges(
        self, edges: List[EdgeLocation]
    ) -> List[EdgeLocation]:
        """Find overloaded edge locations"""
        overloaded = []

        for edge in edges:
            if edge.metrics.cpu_usage_percent > self.max_edge_utilization * 100:
                overloaded.append(edge)

        return overloaded

    async def _optimize_edge_cache(self, edge: EdgeLocation) -> Dict[str, Any]:
        """Optimize cache configuration for specific edge"""
        optimization = {
            "edge_id": edge.id,
            "edge_code": edge.code,
            "current_cache_size_gb": edge.cache_size_gb,
            "optimizations": [],
            "estimated_improvements": {},
        }

        # Analyze cache hit rate
        current_hit_rate = 0.75  # Would get from actual metrics

        if current_hit_rate < self.target_cache_hit_rate:
            # Recommend cache size increase
            optimization["optimizations"].append(
                {
                    "type": "increase_cache_size",
                    "current_size_gb": edge.cache_size_gb,
                    "recommended_size_gb": edge.cache_size_gb * 1.5,
                    "reason": "Low cache hit rate",
                }
            )

        # Analyze cache TTL settings
        optimization["optimizations"].append(
            {
                "type": "optimize_ttl_settings",
                "recommendations": [
                    {
                        "content_type": "static",
                        "current_ttl": 3600,
                        "recommended_ttl": 7200,
                    },
                    {"content_type": "api", "current_ttl": 300, "recommended_ttl": 600},
                ],
            }
        )

        return optimization

    async def _generate_global_cache_recommendations(self) -> List[Dict[str, Any]]:
        """Generate global cache optimization recommendations"""
        return [
            {
                "type": "content_type_optimization",
                "description": "Optimize TTL for static assets",
                "impact": "15% cache hit rate improvement",
            },
            {
                "type": "compression_optimization",
                "description": "Enable Brotli compression for text content",
                "impact": "30% bandwidth reduction",
            },
            {
                "type": "prefetch_optimization",
                "description": "Implement intelligent prefetching for popular content",
                "impact": "20% response time improvement",
            },
        ]

    async def _get_content_delivery_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get content delivery metrics for time window"""
        # Mock metrics - would get from actual monitoring system
        return {
            "total_requests": 1000000,
            "cache_hit_rate": 0.78,
            "avg_response_time": 125.0,
            "bandwidth_gb": 500.0,
        }

    async def _analyze_popular_content(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze most popular content"""
        # Mock data - would analyze actual request logs
        return [
            {"path": "/api/agents", "requests": 50000, "cache_hit_rate": 0.95},
            {"path": "/static/app.js", "requests": 45000, "cache_hit_rate": 0.98},
            {"path": "/api/workflows", "requests": 40000, "cache_hit_rate": 0.85},
        ]

    async def _analyze_slow_content(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze slowest content"""
        # Mock data - would analyze actual performance metrics
        return [
            {"path": "/api/analytics/report", "avg_response_ms": 850, "requests": 5000},
            {"path": "/api/training/models", "avg_response_ms": 750, "requests": 3000},
            {"path": "/api/execution/logs", "avg_response_ms": 650, "requests": 8000},
        ]

    async def _analyze_cache_misses(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze content with high cache miss rates"""
        # Mock data - would analyze actual cache metrics
        return [
            {"path": "/api/real-time/status", "miss_rate": 0.95, "requests": 10000},
            {"path": "/api/user/preferences", "miss_rate": 0.80, "requests": 15000},
            {"path": "/api/dynamic/content", "miss_rate": 0.75, "requests": 20000},
        ]

    async def _analyze_geographic_distribution(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Analyze geographic distribution of requests"""
        # Mock data - would analyze actual geographic data
        return {"US": 400000, "EU": 300000, "APAC": 200000, "Other": 100000}

    async def _get_cdn_metrics(self) -> Dict[str, Any]:
        """Get current CDN performance metrics"""
        return {
            "global_cache_hit_rate": 0.82,
            "avg_response_time_ms": 145,
            "bandwidth_usage_gb": 1200,
            "error_rate_percent": 0.05,
            "edge_locations_active": 15,
        }

    async def _optimize_cache_rules(self) -> List[Dict[str, Any]]:
        """Optimize cache rules across edge locations"""
        return [
            {
                "type": "cache_rule_optimization",
                "description": "Increase TTL for static assets",
                "current_ttl": 3600,
                "recommended_ttl": 7200,
                "impact": "10% cache hit rate improvement",
            }
        ]

    async def _optimize_compression(self) -> List[Dict[str, Any]]:
        """Optimize compression settings"""
        return [
            {
                "type": "compression_optimization",
                "description": "Enable Brotli compression",
                "impact": "25% bandwidth savings",
            }
        ]

    async def _optimize_origin_settings(self) -> List[Dict[str, Any]]:
        """Optimize origin server settings"""
        return [
            {
                "type": "origin_optimization",
                "description": "Optimize origin connection pooling",
                "impact": "15% origin response time improvement",
            }
        ]

    async def _optimize_geographic_routing(self) -> List[Dict[str, Any]]:
        """Optimize geographic routing rules"""
        return [
            {
                "type": "routing_optimization",
                "description": "Update geographic routing for APAC",
                "impact": "30ms latency reduction for APAC users",
            }
        ]

    async def _get_edge_location(self, edge_id: str) -> EdgeLocation:
        """Get edge location by ID"""
        # Check cache first
        cache_key = f"edge_location:{edge_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return EdgeLocation(**cached)

        # Query database
        result = (
            await self.supabase.client.table("edge_locations")
            .select("*")
            .eq("id", edge_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Edge location {edge_id} not found")

        edge = EdgeLocation(**result.data[0])

        # Cache result
        await self.cache.set(cache_key, edge.dict(), ttl=1800)

        return edge

    async def _has_nearby_edge_location(self, country: str) -> bool:
        """Check if there's an edge location near the specified country"""
        # Simplified check - would use actual geographic calculations
        edges = await self.region_manager.list_edge_locations()

        for edge in edges:
            if edge.location.country_name == country:
                return True

        return False

    async def _validate_edge_config_updates(
        self, edge: EdgeLocation, config_updates: Dict[str, Any]
    ) -> None:
        """Validate edge configuration updates"""
        # Validate cache size
        if "cache_size_gb" in config_updates:
            if config_updates["cache_size_gb"] < 1:
                raise ValidationError("Cache size must be at least 1GB")

        # Validate TTL settings
        if "max_cache_ttl_seconds" in config_updates:
            if config_updates["max_cache_ttl_seconds"] < 60:
                raise ValidationError("Max cache TTL must be at least 60 seconds")

    async def _apply_edge_config_updates(
        self, edge: EdgeLocation, config_updates: Dict[str, Any]
    ) -> bool:
        """Apply configuration updates to edge location"""
        # In production, this would update actual edge infrastructure
        logger.info(f"Applying config updates to edge {edge.code}: {config_updates}")
        return True

    async def _get_edge_performance_metrics(
        self, edge_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get performance metrics for edge location"""
        # Mock metrics - would get from actual monitoring
        return {
            "avg_response_time": 120.0,
            "cache_hit_rate": 0.85,
            "error_rate": 0.02,
            "uptime_percent": 99.8,
            "requests_per_second": 150.0,
            "cpu_avg": 45.0,
            "memory_avg": 60.0,
            "bandwidth_avg": 500.0,
            "storage_used_percent": 70.0,
        }

    async def _analyze_edge_geographic_coverage(
        self, edge_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Analyze geographic coverage for edge location"""
        # Mock analysis - would analyze actual request origins
        return {
            "primary_countries": ["US", "CA", "MX"],
            "coverage_radius_km": 2000,
            "avg_distance_km": 800,
            "requests_by_country": {"US": 10000, "CA": 3000, "MX": 1000},
        }

    async def _analyze_edge_content_performance(
        self, edge_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Analyze content performance for edge location"""
        # Mock analysis - would analyze actual content metrics
        return {
            "top_cached_content": [
                {"path": "/static/app.js", "hit_rate": 0.98, "requests": 5000},
                {"path": "/api/agents", "hit_rate": 0.85, "requests": 4000},
            ],
            "cache_efficiency": 0.82,
            "storage_efficiency": 0.75,
        }

    async def _generate_edge_recommendations(
        self, edge: EdgeLocation, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations for edge"""
        recommendations = []

        performance = analysis["performance_summary"]

        # Response time recommendations
        if performance["avg_response_time_ms"] > self.max_edge_latency_ms:
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "high",
                    "description": "Response time exceeds target",
                    "action": "Increase cache size or optimize cache rules",
                    "expected_improvement": "30ms response time reduction",
                }
            )

        # Cache hit rate recommendations
        if performance["cache_hit_rate"] < self.target_cache_hit_rate:
            recommendations.append(
                {
                    "type": "cache",
                    "priority": "medium",
                    "description": "Cache hit rate below target",
                    "action": "Optimize cache TTL settings",
                    "expected_improvement": "10% cache hit rate increase",
                }
            )

        return recommendations

    # Background task methods

    async def _optimization_loop(self):
        """Background optimization loop"""
        while self._running:
            try:
                # Run optimization analysis every hour
                await self.optimize_edge_placement()
                await asyncio.sleep(3600)  # 1 hour
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _content_analysis_loop(self):
        """Background content analysis loop"""
        while self._running:
            try:
                # Analyze content performance every 30 minutes
                await self.analyze_content_performance()
                await asyncio.sleep(1800)  # 30 minutes
            except Exception as e:
                logger.error(f"Content analysis loop error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
