"""
Global Load Balancer
Handles intelligent traffic routing across regions
"""

import asyncio
import logging
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import geoip2.database
import geoip2.errors

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ResourceError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .models import (
    EdgeLocation,
    HealthStatus,
    LoadBalancerConfig,
    LoadBalancingStrategy,
    Region,
    ResourceMetrics,
    TrafficRouting,
)
from .region_manager import RegionManager

logger = logging.getLogger(__name__)


class GlobalLoadBalancer:
    """Global load balancer for multi-region traffic routing"""

    def __init__(self, region_manager: Optional[RegionManager] = None):
        self.region_manager = region_manager or RegionManager()
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()

        # Load balancing configuration
        self.default_strategy = LoadBalancingStrategy.GEOGRAPHIC
        self.health_check_weight = 0.3  # Weight of health in routing decisions
        self.latency_weight = 0.4  # Weight of latency in routing decisions
        self.capacity_weight = 0.3  # Weight of capacity in routing decisions

        # Caching configuration
        self.routing_cache_ttl = 300  # 5 minutes
        self.metrics_cache_ttl = 60  # 1 minute

        # Geographic routing
        self.geoip_db_path = "/opt/geoip/GeoLite2-City.mmdb"
        self.geoip_reader = None

        # Initialize GeoIP if available
        self._init_geoip()

    def _init_geoip(self):
        """Initialize GeoIP database reader"""
        try:
            self.geoip_reader = geoip2.database.Reader(self.geoip_db_path)
            logger.info("GeoIP database initialized")
        except Exception as e:
            logger.warning(f"GeoIP database not available: {e}")
            self.geoip_reader = None

    async def route_request(
        self,
        client_ip: str,
        request_path: str = "/",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Route request to optimal region"""

        routing_start = datetime.utcnow()

        # Get client location
        client_location = await self._get_client_location(client_ip)

        # Get available regions
        available_regions = await self._get_available_regions()

        if not available_regions:
            raise ResourceError("No available regions for routing")

        # Check for session affinity
        if session_id:
            pinned_region = await self._get_session_affinity(session_id)
            if pinned_region and pinned_region in [r.id for r in available_regions]:
                region = next(r for r in available_regions if r.id == pinned_region)
                return await self._create_routing_response(
                    region, client_location, "session_affinity", routing_start
                )

        # Apply routing strategy
        strategy = await self._get_routing_strategy(request_path)

        if strategy == LoadBalancingStrategy.GEOGRAPHIC:
            region = await self._route_geographic(available_regions, client_location)
        elif strategy == LoadBalancingStrategy.LATENCY_BASED:
            region = await self._route_latency_based(available_regions, client_ip)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            region = await self._route_least_connections(available_regions)
        elif strategy == LoadBalancingStrategy.WEIGHTED:
            region = await self._route_weighted(available_regions)
        else:  # ROUND_ROBIN
            region = await self._route_round_robin(available_regions)

        # Set session affinity if enabled
        if session_id:
            await self._set_session_affinity(session_id, region.id)

        return await self._create_routing_response(
            region, client_location, strategy.value, routing_start
        )

    async def update_load_balancer_config(
        self, config: LoadBalancerConfig
    ) -> LoadBalancerConfig:
        """Update load balancer configuration"""
        # Validate configuration
        await self._validate_lb_config(config)

        # Save to database
        await self.supabase.client.table("load_balancer_configs").upsert(
            config.dict()
        ).execute()

        # Clear routing cache
        await self._clear_routing_cache()

        logger.info(f"Updated load balancer config: {config.name}")
        return config

    async def get_traffic_distribution(self, hours: int = 24) -> Dict[str, Any]:
        """Get traffic distribution statistics"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get routing logs
        result = (
            await self.supabase.client.table("routing_logs")
            .select("*")
            .gte("timestamp", start_time.isoformat())
            .lte("timestamp", end_time.isoformat())
            .execute()
        )

        routing_logs = result.data

        # Calculate distribution
        distribution = {
            "total_requests": len(routing_logs),
            "by_region": defaultdict(int),
            "by_strategy": defaultdict(int),
            "by_country": defaultdict(int),
            "avg_routing_time_ms": 0,
            "health_routing_percentage": 0,
        }

        total_routing_time = 0
        health_routings = 0

        for log in routing_logs:
            region_id = log.get("region_id")
            strategy = log.get("strategy")
            country = log.get("client_country")
            routing_time = log.get("routing_time_ms", 0)

            if region_id:
                distribution["by_region"][region_id] += 1

            if strategy:
                distribution["by_strategy"][strategy] += 1

            if country:
                distribution["by_country"][country] += 1

            total_routing_time += routing_time

            if strategy == "health_based":
                health_routings += 1

        if routing_logs:
            distribution["avg_routing_time_ms"] = total_routing_time / len(routing_logs)
            distribution["health_routing_percentage"] = (
                health_routings / len(routing_logs)
            ) * 100

        # Convert defaultdicts to regular dicts
        distribution["by_region"] = dict(distribution["by_region"])
        distribution["by_strategy"] = dict(distribution["by_strategy"])
        distribution["by_country"] = dict(distribution["by_country"])

        return distribution

    async def simulate_failover(
        self, failed_region_id: str, client_requests: int = 1000
    ) -> Dict[str, Any]:
        """Simulate traffic distribution after region failover"""

        # Get available regions (excluding failed one)
        all_regions = await self.region_manager.list_regions()
        available_regions = [r for r in all_regions if r.id != failed_region_id]

        if not available_regions:
            return {"error": "No available regions for failover simulation"}

        simulation_results = {
            "failed_region_id": failed_region_id,
            "available_regions": len(available_regions),
            "simulated_requests": client_requests,
            "redistribution": {},
            "capacity_impact": {},
            "estimated_latency_impact": {},
        }

        # Simulate request redistribution
        redistribution = defaultdict(int)

        for _ in range(client_requests):
            # Simulate random client locations
            client_ip = self._generate_random_ip()
            client_location = await self._get_client_location(client_ip)

            # Route using geographic strategy
            selected_region = await self._route_geographic(
                available_regions, client_location
            )
            redistribution[selected_region.id] += 1

        simulation_results["redistribution"] = dict(redistribution)

        # Calculate capacity impact
        for region_id, request_count in redistribution.items():
            region = next(r for r in available_regions if r.id == region_id)

            # Estimate capacity impact (simplified)
            current_usage = region.metrics.cpu_usage_percent
            additional_load = (
                request_count / client_requests
            ) * 20  # Assume 20% additional load
            projected_usage = current_usage + additional_load

            simulation_results["capacity_impact"][region_id] = {
                "current_cpu_usage": current_usage,
                "projected_cpu_usage": min(100, projected_usage),
                "additional_requests": request_count,
                "capacity_warning": projected_usage > 80,
            }

        return simulation_results

    # Geographic routing methods

    async def _get_client_location(self, client_ip: str) -> Dict[str, Any]:
        """Get client geographic location from IP"""
        # Check cache first
        cache_key = f"geoip:{client_ip}"
        cached_location = await self.cache.get(cache_key)
        if cached_location:
            return cached_location

        location = {
            "ip": client_ip,
            "country": "US",
            "country_code": "US",
            "region": "Unknown",
            "city": "Unknown",
            "latitude": 39.0458,  # Default to center of US
            "longitude": -76.6413,
            "timezone": "UTC",
        }

        if self.geoip_reader:
            try:
                response = self.geoip_reader.city(client_ip)
                location.update(
                    {
                        "country": response.country.name or "Unknown",
                        "country_code": response.country.iso_code or "US",
                        "region": response.subdivisions.most_specific.name or "Unknown",
                        "city": response.city.name or "Unknown",
                        "latitude": float(response.location.latitude or 39.0458),
                        "longitude": float(response.location.longitude or -76.6413),
                        "timezone": response.location.time_zone or "UTC",
                    }
                )
            except geoip2.errors.AddressNotFoundError:
                logger.debug(f"IP {client_ip} not found in GeoIP database")
            except Exception as e:
                logger.warning(f"GeoIP lookup failed for {client_ip}: {e}")

        # Cache location
        await self.cache.set(cache_key, location, ttl=3600)  # 1 hour

        return location

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two geographic points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    # Routing strategy implementations

    async def _route_geographic(
        self, regions: List[Region], client_location: Dict[str, Any]
    ) -> Region:
        """Route based on geographic proximity"""

        client_lat = client_location.get("latitude", 39.0458)
        client_lon = client_location.get("longitude", -76.6413)

        best_region = None
        best_score = float("inf")

        for region in regions:
            # Calculate geographic distance
            distance = self._calculate_distance(
                client_lat,
                client_lon,
                region.location.latitude,
                region.location.longitude,
            )

            # Apply health and capacity weighting
            health_penalty = 0
            if region.health_status == HealthStatus.WARNING:
                health_penalty = 1000  # km penalty
            elif region.health_status == HealthStatus.CRITICAL:
                continue  # Skip critical regions

            capacity_penalty = 0
            if region.metrics.cpu_usage_percent > 80:
                capacity_penalty = 500  # km penalty for high CPU usage

            # Calculate weighted score
            total_score = distance + health_penalty + capacity_penalty

            if total_score < best_score:
                best_score = total_score
                best_region = region

        return best_region or regions[0]  # Fallback to first region

    async def _route_latency_based(
        self, regions: List[Region], client_ip: str
    ) -> Region:
        """Route based on measured latency"""

        # Check cached latency measurements
        latency_scores = {}

        for region in regions:
            cache_key = f"latency:{client_ip}:{region.id}"
            cached_latency = await self.cache.get(cache_key)

            if cached_latency:
                latency_scores[region.id] = cached_latency
            else:
                # Estimate latency based on distance and region metrics
                estimated_latency = region.metrics.avg_response_time_ms
                latency_scores[region.id] = estimated_latency

        # Find region with lowest latency
        best_region_id = min(latency_scores.keys(), key=lambda k: latency_scores[k])
        return next(r for r in regions if r.id == best_region_id)

    async def _route_least_connections(self, regions: List[Region]) -> Region:
        """Route to region with least active connections"""

        # Sort by active connections (ascending)
        sorted_regions = sorted(regions, key=lambda r: r.metrics.active_connections)

        return sorted_regions[0]

    async def _route_weighted(self, regions: List[Region]) -> Region:
        """Route based on region weights"""

        # Filter healthy regions
        healthy_regions = [
            r
            for r in regions
            if r.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        ]

        if not healthy_regions:
            healthy_regions = regions  # Fallback to all regions

        # Calculate total weight
        total_weight = sum(r.weight for r in healthy_regions)

        if total_weight == 0:
            return random.choice(healthy_regions)

        # Weighted random selection
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0

        for region in healthy_regions:
            cumulative_weight += region.weight
            if random_value <= cumulative_weight:
                return region

        return healthy_regions[-1]  # Fallback

    async def _route_round_robin(self, regions: List[Region]) -> Region:
        """Route using round-robin strategy"""

        # Get current round-robin counter
        counter_key = "lb:round_robin:counter"
        counter = await self.cache.get(counter_key) or 0

        # Filter healthy regions
        healthy_regions = [
            r
            for r in regions
            if r.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        ]

        if not healthy_regions:
            healthy_regions = regions  # Fallback

        # Select region based on counter
        selected_region = healthy_regions[counter % len(healthy_regions)]

        # Increment counter
        await self.cache.set(counter_key, counter + 1, ttl=3600)

        return selected_region

    # Helper methods

    async def _get_available_regions(self) -> List[Region]:
        """Get list of available regions"""
        cache_key = "lb:available_regions"
        cached_regions = await self.cache.get(cache_key)

        if cached_regions:
            return [Region(**region) for region in cached_regions]

        # Get from region manager
        all_regions = await self.region_manager.list_regions()

        # Filter out offline regions
        available_regions = [
            r
            for r in all_regions
            if r.status != "offline" and r.health_status != HealthStatus.CRITICAL
        ]

        # Cache available regions
        await self.cache.set(
            cache_key, [r.dict() for r in available_regions], ttl=self.metrics_cache_ttl
        )

        return available_regions

    async def _get_routing_strategy(self, request_path: str) -> LoadBalancingStrategy:
        """Get routing strategy for request path"""

        # Check for path-specific routing rules
        cache_key = f"lb:strategy:{request_path}"
        cached_strategy = await self.cache.get(cache_key)

        if cached_strategy:
            return LoadBalancingStrategy(cached_strategy)

        # Default strategy
        return self.default_strategy

    async def _get_session_affinity(self, session_id: str) -> Optional[str]:
        """Get pinned region for session"""
        cache_key = f"session_affinity:{session_id}"
        return await self.cache.get(cache_key)

    async def _set_session_affinity(self, session_id: str, region_id: str) -> None:
        """Set session affinity to region"""
        cache_key = f"session_affinity:{session_id}"
        await self.cache.set(cache_key, region_id, ttl=3600)  # 1 hour

    async def _create_routing_response(
        self,
        region: Region,
        client_location: Dict[str, Any],
        strategy: str,
        routing_start: datetime,
    ) -> Dict[str, Any]:
        """Create routing response"""

        routing_time_ms = (datetime.utcnow() - routing_start).total_seconds() * 1000

        response = {
            "region_id": region.id,
            "region_code": region.code,
            "region_name": region.name,
            "endpoint": f"https://{region.code}.api.nexusforge.ai",
            "strategy": strategy,
            "routing_time_ms": routing_time_ms,
            "client_location": client_location,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log routing decision
        await self._log_routing_decision(response)

        return response

    async def _log_routing_decision(self, routing_response: Dict[str, Any]) -> None:
        """Log routing decision for analytics"""
        try:
            await self.supabase.client.table("routing_logs").insert(
                routing_response
            ).execute()
        except Exception as e:
            logger.error(f"Failed to log routing decision: {e}")

    async def _validate_lb_config(self, config: LoadBalancerConfig) -> None:
        """Validate load balancer configuration"""
        if config.health_check_interval_seconds < 5:
            raise ValidationError("Health check interval must be at least 5 seconds")

        if config.health_check_timeout_seconds >= config.health_check_interval_seconds:
            raise ValidationError("Health check timeout must be less than interval")

    async def _clear_routing_cache(self) -> None:
        """Clear routing-related cache entries"""
        pattern = "lb:*"
        keys = await self.cache.scan(pattern)
        if keys:
            await self.cache.delete(*keys)

    def _generate_random_ip(self) -> str:
        """Generate random IP for simulation"""
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

    # Edge location routing

    async def route_to_edge(
        self,
        client_ip: str,
        content_type: str = "static",
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route request to optimal edge location"""

        client_location = await self._get_client_location(client_ip)

        # Get available edge locations
        edge_locations = await self.region_manager.list_edge_locations()

        if not edge_locations:
            # Fallback to region routing
            return await self.route_request(client_ip)

        # Find closest edge location
        best_edge = None
        best_distance = float("inf")

        client_lat = client_location.get("latitude", 39.0458)
        client_lon = client_location.get("longitude", -76.6413)

        for edge in edge_locations:
            if edge.health_status == HealthStatus.CRITICAL:
                continue

            distance = self._calculate_distance(
                client_lat, client_lon, edge.location.latitude, edge.location.longitude
            )

            if distance < best_distance:
                best_distance = distance
                best_edge = edge

        if not best_edge:
            # Fallback to region routing
            return await self.route_request(client_ip)

        return {
            "edge_id": best_edge.id,
            "edge_code": best_edge.code,
            "edge_name": best_edge.name,
            "endpoint": f"https://{best_edge.code}.edge.nexusforge.ai",
            "distance_km": best_distance,
            "cache_enabled": True,
            "client_location": client_location,
            "timestamp": datetime.utcnow().isoformat(),
        }
