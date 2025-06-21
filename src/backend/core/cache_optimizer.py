"""
Advanced Cache Optimization System
Multi-level caching with intelligent eviction and predictive pre-loading
"""

import asyncio
import logging
import pickle
import statistics
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    PREDICTIVE = "predictive"  # Predictive based on usage patterns


class CacheLevel(Enum):
    """Cache level types"""

    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_REDIS = "l2_redis"  # Redis cache
    L3_DISK = "l3_disk"  # Disk-based cache


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).seconds > self.ttl_seconds

    @property
    def age_seconds(self) -> int:
        """Get age in seconds"""
        return (datetime.utcnow() - self.created_at).seconds

    @property
    def last_access_seconds(self) -> int:
        """Get seconds since last access"""
        return (datetime.utcnow() - self.accessed_at).seconds


@dataclass
class CacheMetrics:
    """Cache performance metrics"""

    cache_level: CacheLevel
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def calculate_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class IntelligentCacheLayer:
    """Intelligent cache layer with adaptive strategies"""

    def __init__(
        self,
        cache_level: CacheLevel,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        max_entries: int = 10000,
        default_ttl: int = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    ):

        self.cache_level = cache_level
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.strategy = strategy

        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.metrics = CacheMetrics(cache_level)

        # Adaptive strategy parameters
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.prediction_window = 300  # 5 minutes

        # Compression settings
        self.compression_enabled = True
        self.compression_threshold = 1024  # 1KB

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()

        try:
            if key not in self.cache:
                self.metrics.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired:
                await self._evict_entry(key)
                self.metrics.misses += 1
                return None

            # Update access metadata
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1

            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            # Record access pattern
            self._record_access_pattern(key)

            self.metrics.hits += 1

            # Decompress if needed
            value = await self._decompress_value(entry.value)

            return value

        finally:
            access_time = (time.time() - start_time) * 1000
            self._update_access_time(access_time)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Set value in cache"""
        try:
            # Serialize and compress value
            compressed_value, size_bytes = await self._compress_value(value)

            # Check if we need to make space
            if not await self._ensure_space(size_bytes):
                logger.warning(f"Could not make space for cache entry {key}")
                return False

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl,
                tags=tags or [],
            )

            # Remove old entry if exists
            if key in self.cache:
                await self._evict_entry(key)

            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)

            # Update metrics
            self.metrics.size_bytes += size_bytes
            self.metrics.entry_count += 1

            logger.debug(
                f"Cached {key} ({size_bytes} bytes) in {self.cache_level.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            await self._evict_entry(key)
            return True
        return False

    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        self.access_patterns.clear()
        self.metrics.size_bytes = 0
        self.metrics.entry_count = 0
        logger.info(f"Cleared all entries from {self.cache_level.value} cache")

    async def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure there's enough space for new entry"""
        # Check entry count limit
        while len(self.cache) >= self.max_entries:
            if not await self._evict_least_valuable():
                return False

        # Check size limit
        while self.metrics.size_bytes + required_bytes > self.max_size_bytes:
            if not await self._evict_least_valuable():
                return False

        return True

    async def _evict_least_valuable(self) -> bool:
        """Evict the least valuable entry based on strategy"""
        if not self.cache:
            return False

        if self.strategy == CacheStrategy.LRU:
            key_to_evict = self._get_lru_key()
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = self._get_lfu_key()
        elif self.strategy == CacheStrategy.TTL:
            key_to_evict = self._get_expired_key()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = await self._get_adaptive_eviction_key()
        elif self.strategy == CacheStrategy.PREDICTIVE:
            key_to_evict = await self._get_predictive_eviction_key()
        else:
            key_to_evict = self._get_lru_key()  # Default to LRU

        if key_to_evict:
            await self._evict_entry(key_to_evict)
            return True

        return False

    def _get_lru_key(self) -> Optional[str]:
        """Get least recently used key"""
        return self.access_order[0] if self.access_order else None

    def _get_lfu_key(self) -> Optional[str]:
        """Get least frequently used key"""
        if not self.cache:
            return None
        return min(self.cache.keys(), key=lambda k: self.cache[k].access_count)

    def _get_expired_key(self) -> Optional[str]:
        """Get an expired key"""
        for key, entry in self.cache.items():
            if entry.is_expired:
                return key
        # If no expired keys, fall back to LRU
        return self._get_lru_key()

    async def _get_adaptive_eviction_key(self) -> Optional[str]:
        """Get key to evict using adaptive strategy"""
        if not self.cache:
            return None

        # Score each entry based on multiple factors
        scores = {}
        current_time = datetime.utcnow()

        for key, entry in self.cache.items():
            # Recency score (higher = more recent)
            recency_score = 1.0 / (1 + entry.last_access_seconds)

            # Frequency score
            frequency_score = entry.access_count / 100.0  # Normalize

            # Size penalty (larger entries are more expensive)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB

            # TTL urgency (closer to expiration = higher urgency to keep)
            if entry.ttl_seconds:
                time_to_expire = entry.ttl_seconds - entry.age_seconds
                ttl_urgency = max(0, time_to_expire / entry.ttl_seconds)
            else:
                ttl_urgency = 0.5

            # Combined score (lower = better candidate for eviction)
            scores[key] = (
                recency_score * 0.3
                + frequency_score * 0.3
                + ttl_urgency * 0.3
                - size_penalty * 0.1
            )

        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])

    async def _get_predictive_eviction_key(self) -> Optional[str]:
        """Get key to evict using predictive strategy"""
        # Predict which entries are least likely to be accessed soon
        predictions = await self._predict_access_likelihood()

        if predictions:
            # Return key with lowest predicted access likelihood
            return min(predictions.keys(), key=lambda k: predictions[k])

        # Fall back to adaptive if prediction fails
        return await self._get_adaptive_eviction_key()

    async def _predict_access_likelihood(self) -> Dict[str, float]:
        """Predict likelihood of access for each cache entry"""
        predictions = {}
        current_time = datetime.utcnow()

        for key in self.cache.keys():
            if key in self.access_patterns:
                accesses = self.access_patterns[key]

                # Remove old access records
                recent_accesses = [
                    access
                    for access in accesses
                    if (current_time - access).seconds < self.prediction_window
                ]

                if recent_accesses:
                    # Calculate access frequency
                    access_frequency = len(recent_accesses) / self.prediction_window

                    # Calculate time since last access
                    time_since_last = (current_time - max(recent_accesses)).seconds

                    # Predict likelihood (higher frequency, lower time since last = higher likelihood)
                    likelihood = access_frequency * (
                        1.0 / (1 + time_since_last / 60)
                    )  # Decay over minutes
                    predictions[key] = likelihood
                else:
                    predictions[key] = 0.0
            else:
                predictions[key] = 0.0

        return predictions

    async def _evict_entry(self, key: str):
        """Evict a specific entry"""
        if key in self.cache:
            entry = self.cache[key]
            self.metrics.size_bytes -= entry.size_bytes
            self.metrics.entry_count -= 1
            self.metrics.evictions += 1

            del self.cache[key]

            if key in self.access_order:
                self.access_order.remove(key)

            if key in self.access_patterns:
                del self.access_patterns[key]

            logger.debug(f"Evicted {key} from {self.cache_level.value} cache")

    def _record_access_pattern(self, key: str):
        """Record access pattern for predictive caching"""
        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(datetime.utcnow())

        # Keep only recent access patterns
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.prediction_window)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key] if access > cutoff_time
        ]

    async def _compress_value(self, value: Any) -> Tuple[Any, int]:
        """Compress value if beneficial"""
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            original_size = len(serialized)

            # Compress if above threshold and compression enabled
            if self.compression_enabled and original_size > self.compression_threshold:
                compressed = zlib.compress(serialized)
                compressed_size = len(compressed)

                # Only use compression if it saves significant space
                if compressed_size < original_size * 0.8:
                    return {"compressed": True, "data": compressed}, compressed_size

            return {"compressed": False, "data": serialized}, original_size

        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value, len(str(value).encode())

    async def _decompress_value(self, compressed_value: Any) -> Any:
        """Decompress value if needed"""
        try:
            if isinstance(compressed_value, dict) and "compressed" in compressed_value:
                if compressed_value["compressed"]:
                    decompressed = zlib.decompress(compressed_value["data"])
                    return pickle.loads(decompressed)
                else:
                    return pickle.loads(compressed_value["data"])
            else:
                return compressed_value

        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return compressed_value

    def _update_access_time(self, access_time_ms: float):
        """Update average access time metric"""
        alpha = 0.1  # Smoothing factor for exponential moving average
        self.metrics.avg_access_time_ms = (
            1 - alpha
        ) * self.metrics.avg_access_time_ms + alpha * access_time_ms

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self.metrics.calculate_hit_rate()

        return {
            "cache_level": self.cache_level.value,
            "strategy": self.strategy.value,
            "entry_count": len(self.cache),
            "max_entries": self.max_entries,
            "size_bytes": self.metrics.size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "hit_rate": self.metrics.hit_rate,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "evictions": self.metrics.evictions,
            "avg_access_time_ms": self.metrics.avg_access_time_ms,
            "size_utilization": self.metrics.size_bytes / self.max_size_bytes,
            "entry_utilization": len(self.cache) / self.max_entries,
        }


class MultiLevelCacheOptimizer:
    """Multi-level cache system with intelligent optimization"""

    def __init__(self):
        self.cache_layers: Dict[CacheLevel, IntelligentCacheLayer] = {}
        self.global_metrics = {
            "total_requests": 0,
            "total_hits": 0,
            "total_misses": 0,
            "cache_cascade_hits": 0,
        }

        # Initialize cache layers
        self._initialize_cache_layers()

    def _initialize_cache_layers(self):
        """Initialize all cache layers"""
        # L1: Fast in-memory cache
        self.cache_layers[CacheLevel.L1_MEMORY] = IntelligentCacheLayer(
            cache_level=CacheLevel.L1_MEMORY,
            max_size_bytes=50 * 1024 * 1024,  # 50MB
            max_entries=5000,
            default_ttl=300,  # 5 minutes
            strategy=CacheStrategy.ADAPTIVE,
        )

        # L2: Redis-like distributed cache (simulated)
        self.cache_layers[CacheLevel.L2_REDIS] = IntelligentCacheLayer(
            cache_level=CacheLevel.L2_REDIS,
            max_size_bytes=500 * 1024 * 1024,  # 500MB
            max_entries=50000,
            default_ttl=3600,  # 1 hour
            strategy=CacheStrategy.PREDICTIVE,
        )

        # L3: Disk-based persistent cache (simulated)
        self.cache_layers[CacheLevel.L3_DISK] = IntelligentCacheLayer(
            cache_level=CacheLevel.L3_DISK,
            max_size_bytes=2 * 1024 * 1024 * 1024,  # 2GB
            max_entries=100000,
            default_ttl=86400,  # 24 hours
            strategy=CacheStrategy.LFU,
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        self.global_metrics["total_requests"] += 1

        # Try each cache level in order
        for cache_level in [
            CacheLevel.L1_MEMORY,
            CacheLevel.L2_REDIS,
            CacheLevel.L3_DISK,
        ]:
            cache_layer = self.cache_layers[cache_level]
            value = await cache_layer.get(key)

            if value is not None:
                self.global_metrics["total_hits"] += 1

                # Cache cascade: promote to higher levels
                await self._promote_to_higher_levels(key, value, cache_level)

                if cache_level != CacheLevel.L1_MEMORY:
                    self.global_metrics["cache_cascade_hits"] += 1

                return value

        self.global_metrics["total_misses"] += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Set value in appropriate cache levels"""
        success = True

        # Set in all appropriate levels based on value characteristics
        cache_levels = await self._determine_cache_levels(key, value)

        for cache_level in cache_levels:
            cache_layer = self.cache_layers[cache_level]
            level_success = await cache_layer.set(key, value, ttl, tags)
            if not level_success:
                success = False

        return success

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        success = True

        for cache_layer in self.cache_layers.values():
            if not await cache_layer.delete(key):
                success = False

        return success

    async def _promote_to_higher_levels(
        self, key: str, value: Any, found_level: CacheLevel
    ):
        """Promote cache entry to higher levels"""
        levels_to_promote = []

        if found_level == CacheLevel.L3_DISK:
            levels_to_promote = [CacheLevel.L2_REDIS, CacheLevel.L1_MEMORY]
        elif found_level == CacheLevel.L2_REDIS:
            levels_to_promote = [CacheLevel.L1_MEMORY]

        for level in levels_to_promote:
            cache_layer = self.cache_layers[level]
            await cache_layer.set(key, value)

    async def _determine_cache_levels(self, key: str, value: Any) -> List[CacheLevel]:
        """Determine which cache levels to use for a value"""
        # Simple heuristic based on key patterns and value size
        value_size = len(str(value).encode())

        levels = []

        # Always try L1 for small values
        if value_size < 10 * 1024:  # 10KB
            levels.append(CacheLevel.L1_MEMORY)

        # L2 for medium-sized values
        if value_size < 1024 * 1024:  # 1MB
            levels.append(CacheLevel.L2_REDIS)

        # L3 for all values (persistent storage)
        levels.append(CacheLevel.L3_DISK)

        return levels

    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Run comprehensive cache optimization"""
        logger.info("🚀 Starting cache performance optimization")

        optimization_results = {
            "optimization_start": datetime.utcnow().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
        }

        # Collect baseline metrics
        baseline_stats = await self._collect_cache_stats()

        # Optimization 1: Adjust cache sizes based on utilization
        size_optimizations = await self._optimize_cache_sizes()
        optimization_results["optimizations_applied"].extend(size_optimizations)

        # Optimization 2: Tune eviction strategies
        strategy_optimizations = await self._optimize_eviction_strategies()
        optimization_results["optimizations_applied"].extend(strategy_optimizations)

        # Optimization 3: Optimize TTL settings
        ttl_optimizations = await self._optimize_ttl_settings()
        optimization_results["optimizations_applied"].extend(ttl_optimizations)

        # Optimization 4: Predictive preloading
        preload_optimizations = await self._optimize_predictive_preloading()
        optimization_results["optimizations_applied"].extend(preload_optimizations)

        # Collect post-optimization metrics
        optimized_stats = await self._collect_cache_stats()

        # Calculate improvements
        improvements = self._calculate_cache_improvements(
            baseline_stats, optimized_stats
        )
        optimization_results["performance_improvements"] = improvements

        optimization_results["optimization_end"] = datetime.utcnow().isoformat()

        logger.info(
            f"🎉 Cache optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied"
        )

        return optimization_results

    async def _collect_cache_stats(self) -> Dict[str, Any]:
        """Collect comprehensive cache statistics"""
        stats = {"global_metrics": self.global_metrics.copy(), "cache_layers": {}}

        for level, cache_layer in self.cache_layers.items():
            stats["cache_layers"][level.value] = cache_layer.get_cache_stats()

        # Calculate overall hit rate
        total_requests = self.global_metrics["total_requests"]
        total_hits = self.global_metrics["total_hits"]
        stats["overall_hit_rate"] = (
            total_hits / total_requests if total_requests > 0 else 0
        )

        return stats

    async def _optimize_cache_sizes(self) -> List[Dict[str, Any]]:
        """Optimize cache sizes based on utilization patterns"""
        optimizations = []

        for level, cache_layer in self.cache_layers.items():
            stats = cache_layer.get_cache_stats()

            # If utilization is consistently high, consider increasing size
            if stats["size_utilization"] > 0.9 and stats["hit_rate"] > 0.8:
                new_size = int(cache_layer.max_size_bytes * 1.2)  # Increase by 20%
                cache_layer.max_size_bytes = new_size

                optimizations.append(
                    {
                        "type": "size_increase",
                        "cache_level": level.value,
                        "old_size_mb": stats["max_size_bytes"] / (1024 * 1024),
                        "new_size_mb": new_size / (1024 * 1024),
                        "reason": "high_utilization_high_hit_rate",
                    }
                )

            # If utilization is low and hit rate is low, consider decreasing size
            elif stats["size_utilization"] < 0.3 and stats["hit_rate"] < 0.5:
                new_size = int(cache_layer.max_size_bytes * 0.8)  # Decrease by 20%
                cache_layer.max_size_bytes = new_size

                optimizations.append(
                    {
                        "type": "size_decrease",
                        "cache_level": level.value,
                        "old_size_mb": stats["max_size_bytes"] / (1024 * 1024),
                        "new_size_mb": new_size / (1024 * 1024),
                        "reason": "low_utilization_low_hit_rate",
                    }
                )

        return optimizations

    async def _optimize_eviction_strategies(self) -> List[Dict[str, Any]]:
        """Optimize eviction strategies based on access patterns"""
        optimizations = []

        for level, cache_layer in self.cache_layers.items():
            stats = cache_layer.get_cache_stats()
            current_strategy = cache_layer.strategy

            # Recommend strategy changes based on performance
            recommended_strategy = None

            if stats["hit_rate"] < 0.6:
                if current_strategy != CacheStrategy.ADAPTIVE:
                    recommended_strategy = CacheStrategy.ADAPTIVE
                elif current_strategy != CacheStrategy.PREDICTIVE:
                    recommended_strategy = CacheStrategy.PREDICTIVE

            if recommended_strategy:
                cache_layer.strategy = recommended_strategy

                optimizations.append(
                    {
                        "type": "strategy_change",
                        "cache_level": level.value,
                        "old_strategy": current_strategy.value,
                        "new_strategy": recommended_strategy.value,
                        "reason": f"improve_hit_rate_from_{stats['hit_rate']:.2f}",
                    }
                )

        return optimizations

    async def _optimize_ttl_settings(self) -> List[Dict[str, Any]]:
        """Optimize TTL settings based on access patterns"""
        optimizations = []

        for level, cache_layer in self.cache_layers.items():
            stats = cache_layer.get_cache_stats()

            # Analyze access patterns to optimize TTL
            if (
                stats["hit_rate"] > 0.8
                and stats["evictions"] > stats["entry_count"] * 0.5
            ):
                # High hit rate but many evictions - increase TTL
                new_ttl = int(cache_layer.default_ttl * 1.5)
                cache_layer.default_ttl = new_ttl

                optimizations.append(
                    {
                        "type": "ttl_increase",
                        "cache_level": level.value,
                        "new_ttl_seconds": new_ttl,
                        "reason": "high_hit_rate_high_evictions",
                    }
                )

            elif (
                stats["hit_rate"] < 0.5
                and stats["evictions"] < stats["entry_count"] * 0.1
            ):
                # Low hit rate and few evictions - decrease TTL
                new_ttl = int(cache_layer.default_ttl * 0.7)
                cache_layer.default_ttl = new_ttl

                optimizations.append(
                    {
                        "type": "ttl_decrease",
                        "cache_level": level.value,
                        "new_ttl_seconds": new_ttl,
                        "reason": "low_hit_rate_few_evictions",
                    }
                )

        return optimizations

    async def _optimize_predictive_preloading(self) -> List[Dict[str, Any]]:
        """Optimize predictive preloading based on access patterns"""
        optimizations = []

        # Analyze global access patterns
        for cache_layer in self.cache_layers.values():
            if cache_layer.strategy in [
                CacheStrategy.PREDICTIVE,
                CacheStrategy.ADAPTIVE,
            ]:
                # Enable more aggressive prediction for frequently accessed patterns
                prediction_optimizations = await self._enhance_prediction_accuracy(
                    cache_layer
                )
                optimizations.extend(prediction_optimizations)

        return optimizations

    async def _enhance_prediction_accuracy(
        self, cache_layer: IntelligentCacheLayer
    ) -> List[Dict[str, Any]]:
        """Enhance prediction accuracy for a cache layer"""
        optimizations = []

        # Increase prediction window if access patterns are consistent
        if len(cache_layer.access_patterns) > 100:  # Enough data
            # Analyze pattern consistency
            consistent_patterns = 0
            for key, accesses in cache_layer.access_patterns.items():
                if len(accesses) > 5:  # Multiple accesses
                    intervals = []
                    for i in range(1, len(accesses)):
                        interval = (accesses[i] - accesses[i - 1]).seconds
                        intervals.append(interval)

                    if (
                        intervals and statistics.stdev(intervals) < 60
                    ):  # Low deviation (consistent)
                        consistent_patterns += 1

            consistency_ratio = consistent_patterns / len(cache_layer.access_patterns)

            if consistency_ratio > 0.7:  # 70% of patterns are consistent
                cache_layer.prediction_window = int(cache_layer.prediction_window * 1.5)

                optimizations.append(
                    {
                        "type": "prediction_enhancement",
                        "cache_level": cache_layer.cache_level.value,
                        "new_prediction_window": cache_layer.prediction_window,
                        "consistency_ratio": consistency_ratio,
                        "reason": "high_pattern_consistency",
                    }
                )

        return optimizations

    def _calculate_cache_improvements(
        self, baseline: Dict[str, Any], optimized: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate cache performance improvements"""
        improvements = {}

        # Overall hit rate improvement
        baseline_hit_rate = baseline.get("overall_hit_rate", 0)
        optimized_hit_rate = optimized.get("overall_hit_rate", 0)
        improvements["hit_rate_improvement"] = optimized_hit_rate - baseline_hit_rate

        # Per-level improvements
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]:
            level_key = level.value

            if (
                level_key in baseline["cache_layers"]
                and level_key in optimized["cache_layers"]
            ):
                baseline_level = baseline["cache_layers"][level_key]
                optimized_level = optimized["cache_layers"][level_key]

                improvements[f"{level_key}_hit_rate_improvement"] = (
                    optimized_level["hit_rate"] - baseline_level["hit_rate"]
                )

                improvements[f"{level_key}_access_time_improvement"] = (
                    baseline_level["avg_access_time_ms"]
                    - optimized_level["avg_access_time_ms"]
                )

        return improvements

    async def get_comprehensive_cache_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report"""
        stats = await self._collect_cache_stats()

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "cache_performance": stats,
            "recommendations": await self._generate_cache_recommendations(),
            "optimization_opportunities": await self._identify_optimization_opportunities(),
        }

        return report

    async def _generate_cache_recommendations(self) -> List[Dict[str, str]]:
        """Generate cache optimization recommendations"""
        recommendations = []

        for level, cache_layer in self.cache_layers.items():
            stats = cache_layer.get_cache_stats()

            if stats["hit_rate"] < 0.7:
                recommendations.append(
                    {
                        "level": level.value,
                        "type": "performance",
                        "recommendation": "Consider tuning eviction strategy or increasing cache size",
                        "priority": "high",
                    }
                )

            if stats["size_utilization"] > 0.95:
                recommendations.append(
                    {
                        "level": level.value,
                        "type": "capacity",
                        "recommendation": "Cache is near capacity - consider increasing size or optimizing TTL",
                        "priority": "medium",
                    }
                )

            if stats["avg_access_time_ms"] > 10:
                recommendations.append(
                    {
                        "level": level.value,
                        "type": "latency",
                        "recommendation": "High access time detected - consider optimizing data structures",
                        "priority": "medium",
                    }
                )

        return recommendations

    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []

        # Check for cascade opportunities
        l1_stats = self.cache_layers[CacheLevel.L1_MEMORY].get_cache_stats()
        l2_stats = self.cache_layers[CacheLevel.L2_REDIS].get_cache_stats()

        if l1_stats["hit_rate"] < 0.8 and l2_stats["hit_rate"] > 0.9:
            opportunities.append(
                {
                    "type": "cache_promotion",
                    "description": "L2 has high hit rate but L1 is low - consider promoting more entries to L1",
                    "impact": "medium",
                }
            )

        # Check for memory usage optimization
        total_size = sum(
            layer.get_cache_stats()["size_bytes"]
            for layer in self.cache_layers.values()
        )

        if total_size > 1024 * 1024 * 1024:  # 1GB total
            opportunities.append(
                {
                    "type": "memory_optimization",
                    "description": "High total cache memory usage - consider compression or size limits",
                    "impact": "high",
                }
            )

        return opportunities


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    async def demo_cache_optimizer():
        """Demo the cache optimizer functionality"""
        optimizer = MultiLevelCacheOptimizer()

        logger.info("🚀 Starting cache optimization demo")

        # Simulate cache usage
        for i in range(100):
            key = f"test_key_{i % 20}"  # Some overlap for realistic patterns
            value = {"data": f"test_data_{i}", "size": i * 100}

            # Set some values
            if i < 50:
                await optimizer.set(key, value)

            # Get some values (mix of hits and misses)
            result = await optimizer.get(key)

            if i % 20 == 0:
                stats = await optimizer._collect_cache_stats()
                logger.info(
                    f"Iteration {i}: Overall hit rate {stats['overall_hit_rate']:.2f}"
                )

        # Run optimization
        optimization_results = await optimizer.optimize_cache_performance()

        # Generate final report
        final_report = await optimizer.get_comprehensive_cache_report()

        logger.info("🎉 Cache optimization demo completed")
        logger.info(
            f"Optimizations applied: {len(optimization_results['optimizations_applied'])}"
        )
        logger.info(
            f"Final overall hit rate: {final_report['cache_performance']['overall_hit_rate']:.2f}"
        )

    asyncio.run(demo_cache_optimizer())
