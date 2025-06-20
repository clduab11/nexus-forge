"""
Enterprise Scalability Framework
Implements global load balancing, auto-scaling, and distributed system patterns
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import aioredis
import httpx
from pydantic import BaseModel, Field

from ...core.exceptions import ScalabilityError, ServiceUnavailableError
from ...core.monitoring import get_logger

logger = get_logger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"
    GEOGRAPHIC = "geographic"
    ADAPTIVE = "adaptive"


class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_SIZE = "queue_size"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"


class CacheStrategy(Enum):
    """Distributed caching strategies"""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"
    REFRESH_AHEAD = "refresh_ahead"


class ConsistencyModel(Enum):
    """Data consistency models"""
    STRONG = "strong"
    EVENTUAL = "eventual"
    BOUNDED_STALENESS = "bounded_staleness"
    SESSION = "session"
    MONOTONIC_READ = "monotonic_read"


@dataclass
class ServiceEndpoint:
    """Service endpoint definition"""
    endpoint_id: str
    region: str
    zone: str
    host: str
    port: int
    protocol: str  # http, https, grpc
    health_check_url: str
    weight: float = 1.0
    capacity: int = 100
    current_connections: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    last_health_check: Optional[datetime] = None
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardConfig:
    """Database shard configuration"""
    shard_id: str
    shard_key_range: Tuple[int, int]  # Start and end of hash range
    master_endpoint: str
    replica_endpoints: List[str]
    region: str
    capacity_units: int
    current_size_gb: float
    max_size_gb: float
    replication_lag_ms: float = 0.0
    healthy: bool = True


@dataclass
class CacheNode:
    """Distributed cache node"""
    node_id: str
    host: str
    port: int
    memory_mb: int
    used_memory_mb: int = 0
    hit_rate: float = 0.0
    eviction_rate: float = 0.0
    connection_count: int = 0
    healthy: bool = True


@dataclass
class ScalingEvent:
    """Auto-scaling event"""
    event_id: str
    timestamp: datetime
    scaling_policy: ScalingPolicy
    trigger_metric: str
    trigger_value: float
    threshold: float
    action: str  # scale_up, scale_down
    instances_before: int
    instances_after: int
    region: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None


class GlobalLoadBalancer:
    """Global load balancing with geographic awareness"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.endpoints: Dict[str, List[ServiceEndpoint]] = {}
        self.round_robin_counters: Dict[str, int] = {}
        self.health_check_interval = 30  # seconds
        self.unhealthy_threshold = 3
        self.healthy_threshold = 2
        self.endpoint_health_counts: Dict[str, int] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
    
    async def register_endpoint(self, service: str, endpoint: ServiceEndpoint):
        """Register a service endpoint"""
        if service not in self.endpoints:
            self.endpoints[service] = []
            self.round_robin_counters[service] = 0
        
        self.endpoints[service].append(endpoint)
        
        # Start health checking
        if service not in self._health_check_tasks:
            task = asyncio.create_task(self._health_check_loop(service))
            self._health_check_tasks[service] = task
        
        logger.info(
            f"Registered endpoint {endpoint.endpoint_id} for service {service} "
            f"in region {endpoint.region}"
        )
    
    async def select_endpoint(
        self,
        service: str,
        client_ip: Optional[str] = None,
        client_region: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[ServiceEndpoint]:
        """Select optimal endpoint based on strategy"""
        if service not in self.endpoints or not self.endpoints[service]:
            return None
        
        # Filter healthy endpoints
        healthy_endpoints = [
            ep for ep in self.endpoints[service] if ep.healthy
        ]
        
        if not healthy_endpoints:
            logger.error(f"No healthy endpoints available for service {service}")
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(service, healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service, healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash_select(healthy_endpoints, client_ip)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self._geographic_select(healthy_endpoints, client_region)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_select(
                service, healthy_endpoints, client_ip, client_region, session_id
            )
        
        return healthy_endpoints[0]  # Fallback
    
    def _round_robin_select(
        self, service: str, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Round-robin selection"""
        counter = self.round_robin_counters[service]
        selected = endpoints[counter % len(endpoints)]
        self.round_robin_counters[service] = (counter + 1) % len(endpoints)
        return selected
    
    def _least_connections_select(
        self, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select endpoint with least connections"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _weighted_round_robin_select(
        self, service: str, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Weighted round-robin selection"""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        target = random.uniform(0, total_weight)
        current = 0
        
        for endpoint in endpoints:
            current += endpoint.weight
            if current >= target:
                return endpoint
        
        return endpoints[-1]
    
    def _ip_hash_select(
        self, endpoints: List[ServiceEndpoint], client_ip: Optional[str]
    ) -> ServiceEndpoint:
        """IP hash-based selection for session affinity"""
        if not client_ip:
            return endpoints[0]
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return endpoints[hash_value % len(endpoints)]
    
    def _least_response_time_select(
        self, endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select endpoint with lowest response time"""
        return min(endpoints, key=lambda ep: ep.response_time_ms)
    
    def _geographic_select(
        self, endpoints: List[ServiceEndpoint], client_region: Optional[str]
    ) -> ServiceEndpoint:
        """Select geographically closest endpoint"""
        if not client_region:
            return self._least_response_time_select(endpoints)
        
        # Prefer same region
        same_region = [ep for ep in endpoints if ep.region == client_region]
        if same_region:
            return self._least_connections_select(same_region)
        
        # Otherwise, use geographic distance (simplified)
        return self._get_closest_region_endpoint(endpoints, client_region)
    
    async def _adaptive_select(
        self,
        service: str,
        endpoints: List[ServiceEndpoint],
        client_ip: Optional[str],
        client_region: Optional[str],
        session_id: Optional[str]
    ) -> ServiceEndpoint:
        """Adaptive selection based on multiple factors"""
        scores = {}
        
        for endpoint in endpoints:
            score = 100.0  # Base score
            
            # Factor in current load
            load_ratio = endpoint.current_connections / endpoint.capacity
            score -= load_ratio * 30
            
            # Factor in response time
            if endpoint.response_time_ms > 0:
                score -= min(endpoint.response_time_ms / 10, 20)
            
            # Factor in error rate
            score -= endpoint.error_rate * 50
            
            # Geographic bonus
            if client_region and endpoint.region == client_region:
                score += 20
            
            # Session affinity bonus
            if session_id and endpoint.endpoint_id == self._get_session_endpoint(session_id):
                score += 15
            
            scores[endpoint] = max(score, 0)
        
        # Select endpoint with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _get_closest_region_endpoint(
        self, endpoints: List[ServiceEndpoint], client_region: str
    ) -> ServiceEndpoint:
        """Get endpoint from closest region (simplified)"""
        # Region distance matrix (simplified)
        region_distances = {
            ("us-east", "us-west"): 1,
            ("us-east", "eu-west"): 2,
            ("us-west", "asia-pacific"): 2,
            ("eu-west", "asia-pacific"): 3,
        }
        
        closest_endpoint = None
        min_distance = float('inf')
        
        for endpoint in endpoints:
            distance = self._get_region_distance(client_region, endpoint.region, region_distances)
            if distance < min_distance:
                min_distance = distance
                closest_endpoint = endpoint
        
        return closest_endpoint or endpoints[0]
    
    def _get_region_distance(
        self, region1: str, region2: str, distances: Dict[Tuple[str, str], int]
    ) -> int:
        """Get distance between regions"""
        if region1 == region2:
            return 0
        
        key1 = (region1, region2)
        key2 = (region2, region1)
        
        if key1 in distances:
            return distances[key1]
        elif key2 in distances:
            return distances[key2]
        else:
            return 4  # Unknown distance
    
    def _get_session_endpoint(self, session_id: str) -> str:
        """Get endpoint ID for session (for affinity)"""
        # Simple hash-based mapping
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return f"endpoint-{hash_value % 10}"
    
    async def _health_check_loop(self, service: str):
        """Continuous health checking loop"""
        while True:
            try:
                for endpoint in self.endpoints.get(service, []):
                    await self._check_endpoint_health(endpoint)
                
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error for service {service}: {e}")
                await asyncio.sleep(5)
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint):
        """Check individual endpoint health"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}{endpoint.health_check_url}"
                )
                
                if response.status_code == 200:
                    # Update metrics
                    endpoint.response_time_ms = response.elapsed.total_seconds() * 1000
                    endpoint.last_health_check = datetime.now(timezone.utc)
                    
                    # Mark as healthy
                    if not endpoint.healthy:
                        health_count = self.endpoint_health_counts.get(endpoint.endpoint_id, 0)
                        health_count += 1
                        
                        if health_count >= self.healthy_threshold:
                            endpoint.healthy = True
                            self.endpoint_health_counts[endpoint.endpoint_id] = 0
                            logger.info(f"Endpoint {endpoint.endpoint_id} marked as healthy")
                        else:
                            self.endpoint_health_counts[endpoint.endpoint_id] = health_count
                    else:
                        self.endpoint_health_counts[endpoint.endpoint_id] = 0
                else:
                    await self._mark_unhealthy(endpoint, f"HTTP {response.status_code}")
        
        except Exception as e:
            await self._mark_unhealthy(endpoint, str(e))
    
    async def _mark_unhealthy(self, endpoint: ServiceEndpoint, reason: str):
        """Mark endpoint as unhealthy"""
        if endpoint.healthy:
            health_count = self.endpoint_health_counts.get(endpoint.endpoint_id, 0)
            health_count += 1
            
            if health_count >= self.unhealthy_threshold:
                endpoint.healthy = False
                self.endpoint_health_counts[endpoint.endpoint_id] = 0
                logger.warning(
                    f"Endpoint {endpoint.endpoint_id} marked as unhealthy: {reason}"
                )
            else:
                self.endpoint_health_counts[endpoint.endpoint_id] = health_count
    
    async def get_service_health(self, service: str) -> Dict[str, Any]:
        """Get health status for a service"""
        if service not in self.endpoints:
            return {"status": "unknown", "message": "Service not registered"}
        
        endpoints = self.endpoints[service]
        healthy_count = sum(1 for ep in endpoints if ep.healthy)
        total_count = len(endpoints)
        
        # Calculate aggregate metrics
        total_connections = sum(ep.current_connections for ep in endpoints)
        avg_response_time = sum(ep.response_time_ms for ep in endpoints) / total_count if total_count > 0 else 0
        avg_error_rate = sum(ep.error_rate for ep in endpoints) / total_count if total_count > 0 else 0
        
        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "healthy_endpoints": healthy_count,
            "total_endpoints": total_count,
            "availability": (healthy_count / total_count * 100) if total_count > 0 else 0,
            "total_connections": total_connections,
            "avg_response_time_ms": avg_response_time,
            "avg_error_rate": avg_error_rate,
            "endpoints": [
                {
                    "id": ep.endpoint_id,
                    "region": ep.region,
                    "healthy": ep.healthy,
                    "connections": ep.current_connections,
                    "response_time_ms": ep.response_time_ms
                }
                for ep in endpoints
            ]
        }


class DatabaseShardManager:
    """Database sharding and partition management"""
    
    def __init__(self, shard_count: int = 16):
        self.shard_count = shard_count
        self.shards: Dict[str, ShardConfig] = {}
        self.shard_key_function = self._default_shard_key_function
        self.rebalancing = False
        self.migration_queue: List[Dict[str, Any]] = []
    
    def _default_shard_key_function(self, key: str) -> int:
        """Default sharding function using consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.shard_count
    
    def register_shard(self, shard: ShardConfig):
        """Register a database shard"""
        self.shards[shard.shard_id] = shard
        logger.info(
            f"Registered shard {shard.shard_id} for range "
            f"{shard.shard_key_range[0]}-{shard.shard_key_range[1]}"
        )
    
    def get_shard_for_key(self, key: str) -> Optional[ShardConfig]:
        """Get shard for a given key"""
        shard_index = self.shard_key_function(key)
        
        for shard in self.shards.values():
            if shard.shard_key_range[0] <= shard_index <= shard.shard_key_range[1]:
                return shard
        
        return None
    
    async def execute_query(
        self,
        query: str,
        key: str,
        consistency: ConsistencyModel = ConsistencyModel.EVENTUAL
    ) -> Any:
        """Execute query on appropriate shard"""
        shard = self.get_shard_for_key(key)
        if not shard:
            raise ScalabilityError(f"No shard found for key: {key}")
        
        if not shard.healthy:
            # Try to use replica
            if shard.replica_endpoints and consistency != ConsistencyModel.STRONG:
                return await self._execute_on_replica(query, shard, consistency)
            else:
                raise ServiceUnavailableError(f"Shard {shard.shard_id} is unhealthy")
        
        # Execute on master for writes or strong consistency
        if self._is_write_query(query) or consistency == ConsistencyModel.STRONG:
            return await self._execute_on_master(query, shard)
        
        # For reads, can use replica based on consistency model
        if consistency == ConsistencyModel.EVENTUAL:
            # Randomly choose between master and replicas
            if random.random() < 0.7 and shard.replica_endpoints:  # 70% to replicas
                return await self._execute_on_replica(query, shard, consistency)
        
        return await self._execute_on_master(query, shard)
    
    def _is_write_query(self, query: str) -> bool:
        """Check if query is a write operation"""
        write_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        query_upper = query.upper().strip()
        return any(query_upper.startswith(keyword) for keyword in write_keywords)
    
    async def _execute_on_master(self, query: str, shard: ShardConfig) -> Any:
        """Execute query on shard master"""
        # In production, this would use actual database connection
        logger.debug(f"Executing on master {shard.master_endpoint}: {query[:50]}...")
        return {"result": "success", "shard": shard.shard_id, "endpoint": "master"}
    
    async def _execute_on_replica(
        self, query: str, shard: ShardConfig, consistency: ConsistencyModel
    ) -> Any:
        """Execute query on shard replica"""
        # Check replication lag for bounded staleness
        if consistency == ConsistencyModel.BOUNDED_STALENESS:
            max_lag_ms = 1000  # 1 second max lag
            if shard.replication_lag_ms > max_lag_ms:
                # Fall back to master
                return await self._execute_on_master(query, shard)
        
        # Select replica
        replica = random.choice(shard.replica_endpoints)
        
        logger.debug(f"Executing on replica {replica}: {query[:50]}...")
        return {"result": "success", "shard": shard.shard_id, "endpoint": "replica"}
    
    async def rebalance_shards(self, target_distribution: Dict[str, int]):
        """Rebalance data across shards"""
        if self.rebalancing:
            logger.warning("Rebalancing already in progress")
            return
        
        self.rebalancing = True
        logger.info("Starting shard rebalancing")
        
        try:
            # Calculate current distribution
            current_distribution = await self._calculate_shard_distribution()
            
            # Plan migrations
            migration_plan = self._plan_migrations(current_distribution, target_distribution)
            
            # Execute migrations
            for migration in migration_plan:
                await self._execute_migration(migration)
            
            logger.info(f"Shard rebalancing completed. {len(migration_plan)} migrations executed")
        
        finally:
            self.rebalancing = False
    
    async def _calculate_shard_distribution(self) -> Dict[str, int]:
        """Calculate current data distribution across shards"""
        distribution = {}
        
        for shard_id, shard in self.shards.items():
            # In production, query actual data size
            distribution[shard_id] = int(shard.current_size_gb * 1024)  # Convert to MB
        
        return distribution
    
    def _plan_migrations(
        self, current: Dict[str, int], target: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Plan data migrations to achieve target distribution"""
        migrations = []
        
        # Calculate deltas
        deltas = {}
        for shard_id in current:
            current_size = current.get(shard_id, 0)
            target_size = target.get(shard_id, current_size)
            deltas[shard_id] = target_size - current_size
        
        # Plan migrations from overloaded to underloaded shards
        sources = [(s, -d) for s, d in deltas.items() if d < 0]
        destinations = [(s, d) for s, d in deltas.items() if d > 0]
        
        sources.sort(key=lambda x: x[1], reverse=True)
        destinations.sort(key=lambda x: x[1], reverse=True)
        
        for source_id, source_excess in sources:
            for i, (dest_id, dest_need) in enumerate(destinations):
                if dest_need <= 0:
                    continue
                
                migration_size = min(source_excess, dest_need)
                if migration_size > 0:
                    migrations.append({
                        "source_shard": source_id,
                        "destination_shard": dest_id,
                        "size_mb": migration_size,
                        "key_range": self._calculate_migration_key_range(
                            source_id, dest_id, migration_size
                        )
                    })
                    
                    source_excess -= migration_size
                    destinations[i] = (dest_id, dest_need - migration_size)
                
                if source_excess <= 0:
                    break
        
        return migrations
    
    def _calculate_migration_key_range(
        self, source_id: str, dest_id: str, size_mb: int
    ) -> Tuple[int, int]:
        """Calculate key range to migrate"""
        # Simplified calculation
        source_shard = self.shards[source_id]
        total_range = source_shard.shard_key_range[1] - source_shard.shard_key_range[0]
        
        # Calculate proportion of range to migrate
        proportion = size_mb / (source_shard.current_size_gb * 1024)
        range_size = int(total_range * proportion)
        
        # Return range from end of source shard's range
        start = source_shard.shard_key_range[1] - range_size
        end = source_shard.shard_key_range[1]
        
        return (start, end)
    
    async def _execute_migration(self, migration: Dict[str, Any]):
        """Execute a shard migration"""
        logger.info(
            f"Migrating {migration['size_mb']}MB from "
            f"{migration['source_shard']} to {migration['destination_shard']}"
        )
        
        # In production, this would:
        # 1. Set up replication from source to destination
        # 2. Copy data in chunks
        # 3. Verify data integrity
        # 4. Update shard key ranges
        # 5. Clean up source data
        
        await asyncio.sleep(1)  # Simulate migration time
        
        # Update shard configurations
        source = self.shards[migration['source_shard']]
        dest = self.shards[migration['destination_shard']]
        
        source.current_size_gb -= migration['size_mb'] / 1024
        dest.current_size_gb += migration['size_mb'] / 1024
    
    async def add_shard(
        self, region: str, capacity_units: int = 100
    ) -> ShardConfig:
        """Add a new shard to the cluster"""
        new_shard_id = f"shard-{uuid4().hex[:8]}"
        
        # Calculate key range for new shard
        existing_shards = len(self.shards)
        range_size = self.shard_count // (existing_shards + 1)
        start_range = existing_shards * range_size
        end_range = start_range + range_size - 1
        
        new_shard = ShardConfig(
            shard_id=new_shard_id,
            shard_key_range=(start_range, end_range),
            master_endpoint=f"db-master-{new_shard_id}.{region}.db",
            replica_endpoints=[
                f"db-replica-{new_shard_id}-{i}.{region}.db" for i in range(2)
            ],
            region=region,
            capacity_units=capacity_units,
            current_size_gb=0.0,
            max_size_gb=capacity_units * 10.0  # 10GB per capacity unit
        )
        
        self.register_shard(new_shard)
        
        # Trigger rebalancing
        await self._rebalance_after_add(new_shard_id)
        
        return new_shard
    
    async def _rebalance_after_add(self, new_shard_id: str):
        """Rebalance data after adding a shard"""
        # Calculate target distribution (equal distribution)
        total_size = sum(s.current_size_gb for s in self.shards.values())
        target_size_per_shard = total_size / len(self.shards)
        
        target_distribution = {
            shard_id: int(target_size_per_shard * 1024)
            for shard_id in self.shards
        }
        
        await self.rebalance_shards(target_distribution)


class DistributedCacheManager:
    """Distributed cache management with multiple strategies"""
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE,
        default_ttl: int = 3600
    ):
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache_nodes: List[CacheNode] = []
        self.consistent_hash_ring: Dict[int, CacheNode] = {}
        self.virtual_nodes = 150  # Virtual nodes for consistent hashing
        self._rebuild_hash_ring()
    
    def add_cache_node(self, node: CacheNode):
        """Add a cache node to the cluster"""
        self.cache_nodes.append(node)
        self._rebuild_hash_ring()
        logger.info(f"Added cache node {node.node_id} with {node.memory_mb}MB memory")
    
    def remove_cache_node(self, node_id: str):
        """Remove a cache node from the cluster"""
        self.cache_nodes = [n for n in self.cache_nodes if n.node_id != node_id]
        self._rebuild_hash_ring()
        logger.info(f"Removed cache node {node_id}")
    
    def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring"""
        self.consistent_hash_ring = {}
        
        for node in self.cache_nodes:
            if not node.healthy:
                continue
            
            # Add virtual nodes for better distribution
            for i in range(self.virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self.consistent_hash_ring[hash_value] = node
    
    def _get_cache_node(self, key: str) -> Optional[CacheNode]:
        """Get cache node for a key using consistent hashing"""
        if not self.consistent_hash_ring:
            return None
        
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find the first node with hash >= key_hash
        node_hashes = sorted(self.consistent_hash_ring.keys())
        for node_hash in node_hashes:
            if node_hash >= key_hash:
                return self.consistent_hash_ring[node_hash]
        
        # Wrap around to first node
        return self.consistent_hash_ring[node_hashes[0]]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        node = self._get_cache_node(key)
        if not node:
            return None
        
        try:
            # In production, connect to actual cache node
            async with aioredis.from_url(
                f"redis://{node.host}:{node.port}",
                decode_responses=True
            ) as redis:
                value = await redis.get(key)
                
                if value:
                    # Update hit rate (simplified)
                    node.hit_rate = min(node.hit_rate + 0.01, 1.0)
                else:
                    node.hit_rate = max(node.hit_rate - 0.01, 0.0)
                
                return value
        
        except Exception as e:
            logger.error(f"Cache get error for key {key} on node {node.node_id}: {e}")
            node.healthy = False
            self._rebuild_hash_ring()
            return None
    
    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in distributed cache"""
        node = self._get_cache_node(key)
        if not node:
            return False
        
        ttl = ttl or self.default_ttl
        
        try:
            async with aioredis.from_url(
                f"redis://{node.host}:{node.port}",
                decode_responses=True
            ) as redis:
                await redis.setex(key, ttl, value)
                
                # Update memory usage (simplified)
                node.used_memory_mb = min(
                    node.used_memory_mb + 0.001,  # Assume 1KB per entry
                    node.memory_mb
                )
                
                return True
        
        except Exception as e:
            logger.error(f"Cache set error for key {key} on node {node.node_id}: {e}")
            node.healthy = False
            self._rebuild_hash_ring()
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from distributed cache"""
        node = self._get_cache_node(key)
        if not node:
            return False
        
        try:
            async with aioredis.from_url(
                f"redis://{node.host}:{node.port}",
                decode_responses=True
            ) as redis:
                result = await redis.delete(key)
                
                # Update memory usage
                if result > 0:
                    node.used_memory_mb = max(node.used_memory_mb - 0.001, 0)
                
                return result > 0
        
        except Exception as e:
            logger.error(f"Cache delete error for key {key} on node {node.node_id}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        total_deleted = 0
        
        for node in self.cache_nodes:
            if not node.healthy:
                continue
            
            try:
                async with aioredis.from_url(
                    f"redis://{node.host}:{node.port}",
                    decode_responses=True
                ) as redis:
                    # Find matching keys
                    keys = await redis.keys(pattern)
                    
                    if keys:
                        deleted = await redis.delete(*keys)
                        total_deleted += deleted
                        
                        # Update memory usage
                        node.used_memory_mb = max(
                            node.used_memory_mb - (deleted * 0.001), 0
                        )
            
            except Exception as e:
                logger.error(
                    f"Pattern invalidation error on node {node.node_id}: {e}"
                )
        
        return total_deleted
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get distributed cache statistics"""
        total_memory = sum(n.memory_mb for n in self.cache_nodes)
        used_memory = sum(n.used_memory_mb for n in self.cache_nodes)
        healthy_nodes = sum(1 for n in self.cache_nodes if n.healthy)
        
        avg_hit_rate = (
            sum(n.hit_rate for n in self.cache_nodes) / len(self.cache_nodes)
            if self.cache_nodes else 0
        )
        
        return {
            "total_nodes": len(self.cache_nodes),
            "healthy_nodes": healthy_nodes,
            "total_memory_mb": total_memory,
            "used_memory_mb": used_memory,
            "memory_usage_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
            "avg_hit_rate": avg_hit_rate * 100,
            "cache_strategy": self.strategy.value,
            "nodes": [
                {
                    "id": n.node_id,
                    "healthy": n.healthy,
                    "memory_usage_percent": (n.used_memory_mb / n.memory_mb * 100) if n.memory_mb > 0 else 0,
                    "hit_rate": n.hit_rate * 100,
                    "connections": n.connection_count
                }
                for n in self.cache_nodes
            ]
        }
    
    async def rebalance_cache(self):
        """Rebalance cache data across nodes"""
        logger.info("Starting cache rebalancing")
        
        # Calculate average memory usage
        total_used = sum(n.used_memory_mb for n in self.cache_nodes)
        avg_used = total_used / len(self.cache_nodes) if self.cache_nodes else 0
        
        # Identify overloaded and underloaded nodes
        overloaded = [n for n in self.cache_nodes if n.used_memory_mb > avg_used * 1.2]
        underloaded = [n for n in self.cache_nodes if n.used_memory_mb < avg_used * 0.8]
        
        if not overloaded or not underloaded:
            logger.info("Cache is already balanced")
            return
        
        # In production, this would migrate data between nodes
        # For now, just log the plan
        for over_node in overloaded:
            excess = over_node.used_memory_mb - avg_used
            logger.info(
                f"Node {over_node.node_id} is overloaded by {excess:.1f}MB"
            )
        
        for under_node in underloaded:
            deficit = avg_used - under_node.used_memory_mb
            logger.info(
                f"Node {under_node.node_id} can accept {deficit:.1f}MB more data"
            )


class AutoScaler:
    """Auto-scaling manager for dynamic resource allocation"""
    
    def __init__(self):
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.scaling_thresholds: Dict[str, Dict[str, float]] = {}
        self.cooldown_periods: Dict[str, int] = {}  # seconds
        self.last_scaling_times: Dict[str, datetime] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.min_instances: Dict[str, int] = {}
        self.max_instances: Dict[str, int] = {}
        self.current_instances: Dict[str, int] = {}
    
    def configure_auto_scaling(
        self,
        service: str,
        policy: ScalingPolicy,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 20.0,
        cooldown_seconds: int = 300
    ):
        """Configure auto-scaling for a service"""
        self.scaling_policies[service] = policy
        self.min_instances[service] = min_instances
        self.max_instances[service] = max_instances
        self.current_instances[service] = min_instances
        self.cooldown_periods[service] = cooldown_seconds
        
        self.scaling_thresholds[service] = {
            "scale_up": scale_up_threshold,
            "scale_down": scale_down_threshold
        }
        
        logger.info(
            f"Configured {policy.value} auto-scaling for {service}: "
            f"{min_instances}-{max_instances} instances"
        )
    
    async def evaluate_scaling(
        self, service: str, metrics: Dict[str, float]
    ) -> Optional[ScalingEvent]:
        """Evaluate if scaling is needed based on metrics"""
        if service not in self.scaling_policies:
            return None
        
        # Check cooldown period
        if not self._check_cooldown(service):
            return None
        
        policy = self.scaling_policies[service]
        thresholds = self.scaling_thresholds[service]
        
        # Get relevant metric
        metric_value = self._get_policy_metric(policy, metrics)
        if metric_value is None:
            return None
        
        # Determine scaling action
        current = self.current_instances[service]
        action = None
        new_count = current
        
        if metric_value > thresholds["scale_up"]:
            if current < self.max_instances[service]:
                action = "scale_up"
                new_count = min(
                    current + self._calculate_scale_increment(current),
                    self.max_instances[service]
                )
        elif metric_value < thresholds["scale_down"]:
            if current > self.min_instances[service]:
                action = "scale_down"
                new_count = max(
                    current - self._calculate_scale_decrement(current),
                    self.min_instances[service]
                )
        
        if action:
            # Create scaling event
            event = ScalingEvent(
                event_id=f"SCALE-{uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                scaling_policy=policy,
                trigger_metric=self._get_metric_name(policy),
                trigger_value=metric_value,
                threshold=thresholds["scale_up" if action == "scale_up" else "scale_down"],
                action=action,
                instances_before=current,
                instances_after=new_count,
                region="global",  # Could be region-specific
                success=False,  # Will be updated after execution
                duration_seconds=0.0
            )
            
            # Execute scaling
            success = await self._execute_scaling(service, new_count, event)
            
            if success:
                self.current_instances[service] = new_count
                self.last_scaling_times[service] = datetime.now(timezone.utc)
                event.success = True
                logger.info(
                    f"Auto-scaled {service} from {current} to {new_count} instances"
                )
            
            self.scaling_history.append(event)
            return event
        
        return None
    
    def _check_cooldown(self, service: str) -> bool:
        """Check if cooldown period has passed"""
        if service not in self.last_scaling_times:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_scaling_times[service]).total_seconds()
        return elapsed >= self.cooldown_periods[service]
    
    def _get_policy_metric(
        self, policy: ScalingPolicy, metrics: Dict[str, float]
    ) -> Optional[float]:
        """Get metric value based on scaling policy"""
        metric_mapping = {
            ScalingPolicy.CPU_BASED: "cpu_usage",
            ScalingPolicy.MEMORY_BASED: "memory_usage",
            ScalingPolicy.REQUEST_RATE: "request_rate",
            ScalingPolicy.RESPONSE_TIME: "response_time_ms",
            ScalingPolicy.QUEUE_SIZE: "queue_size"
        }
        
        metric_name = metric_mapping.get(policy)
        if metric_name:
            return metrics.get(metric_name)
        
        return None
    
    def _get_metric_name(self, policy: ScalingPolicy) -> str:
        """Get human-readable metric name"""
        return policy.value.replace("_based", "").replace("_", " ").title()
    
    def _calculate_scale_increment(self, current: int) -> int:
        """Calculate how many instances to add"""
        # Scale up by 20% or minimum 1
        return max(int(current * 0.2), 1)
    
    def _calculate_scale_decrement(self, current: int) -> int:
        """Calculate how many instances to remove"""
        # Scale down by 10% or minimum 1
        return max(int(current * 0.1), 1)
    
    async def _execute_scaling(
        self, service: str, new_count: int, event: ScalingEvent
    ) -> bool:
        """Execute the scaling action"""
        start_time = time.time()
        
        try:
            # In production, this would:
            # 1. Call cloud provider API to adjust instances
            # 2. Wait for instances to be ready
            # 3. Update load balancer configuration
            # 4. Verify health of new instances
            
            await asyncio.sleep(2)  # Simulate scaling time
            
            event.duration_seconds = time.time() - start_time
            return True
        
        except Exception as e:
            logger.error(f"Scaling failed for {service}: {e}")
            event.error = str(e)
            event.duration_seconds = time.time() - start_time
            return False
    
    async def predict_scaling_needs(
        self, service: str, historical_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict future scaling needs using historical data"""
        if len(historical_metrics) < 24:  # Need at least 24 data points
            return {"prediction": "insufficient_data"}
        
        # Simple prediction based on trends
        policy = self.scaling_policies.get(service)
        if not policy:
            return {"prediction": "no_policy"}
        
        metric_name = self._get_metric_name(policy).lower().replace(" ", "_")
        
        # Extract metric values
        values = [m.get(metric_name, 0) for m in historical_metrics]
        
        # Calculate trend
        recent_avg = sum(values[-6:]) / 6  # Last 6 periods
        overall_avg = sum(values) / len(values)
        
        trend = "increasing" if recent_avg > overall_avg * 1.1 else "decreasing" if recent_avg < overall_avg * 0.9 else "stable"
        
        # Predict scaling needs
        thresholds = self.scaling_thresholds[service]
        
        prediction = {
            "trend": trend,
            "current_value": values[-1],
            "recent_average": recent_avg,
            "overall_average": overall_avg,
            "scale_up_likelihood": "high" if recent_avg > thresholds["scale_up"] * 0.8 else "low",
            "scale_down_likelihood": "high" if recent_avg < thresholds["scale_down"] * 1.2 else "low",
            "recommended_instances": self._calculate_recommended_instances(
                service, recent_avg, trend
            )
        }
        
        return prediction
    
    def _calculate_recommended_instances(
        self, service: str, metric_avg: float, trend: str
    ) -> int:
        """Calculate recommended instance count"""
        current = self.current_instances[service]
        thresholds = self.scaling_thresholds[service]
        
        # Target 60% utilization
        target_utilization = 60.0
        
        if metric_avg > 0:
            recommended = int(current * (metric_avg / target_utilization))
            
            # Adjust for trend
            if trend == "increasing":
                recommended = int(recommended * 1.2)  # Add 20% buffer
            elif trend == "decreasing":
                recommended = int(recommended * 0.9)  # Remove 10%
            
            # Apply bounds
            recommended = max(
                self.min_instances[service],
                min(recommended, self.max_instances[service])
            )
            
            return recommended
        
        return current
    
    def get_scaling_history(
        self, service: Optional[str] = None, hours: int = 24
    ) -> List[ScalingEvent]:
        """Get scaling history for analysis"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        history = [
            event for event in self.scaling_history
            if event.timestamp > cutoff
        ]
        
        if service:
            # Filter by service (would need service info in ScalingEvent)
            pass
        
        return history