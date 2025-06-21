"""
Integrated Cache Optimization Module
Combines Redis cache with intelligent multi-level caching for maximum performance
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from .cache import RedisCache, CacheStrategy
from .cache_optimizer import MultiLevelCacheOptimizer, CacheLevel

logger = logging.getLogger(__name__)


class IntegratedCacheSystem:
    """
    Unified cache system combining Redis and multi-level intelligent caching
    for optimal database query result caching
    """
    
    def __init__(self):
        self.redis_cache = RedisCache()
        self.ml_optimizer = MultiLevelCacheOptimizer()
        
        # Cache configuration for different data types
        self.cache_configs = {
            "user_data": {
                "ttl": 300,  # 5 minutes
                "strategy": CacheStrategy.COMPRESSED,
                "levels": [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS],
                "tags": ["user"],
            },
            "research_task": {
                "ttl": 600,  # 10 minutes
                "strategy": CacheStrategy.COMPRESSED,
                "levels": [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK],
                "tags": ["research"],
            },
            "model_result": {
                "ttl": 3600,  # 1 hour
                "strategy": CacheStrategy.COMPRESSED,
                "levels": [CacheLevel.L2_REDIS, CacheLevel.L3_DISK],
                "tags": ["ai", "model"],
            },
            "analytics": {
                "ttl": 1800,  # 30 minutes
                "strategy": CacheStrategy.SIMPLE,
                "levels": [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS],
                "tags": ["analytics"],
            },
            "subscription": {
                "ttl": 900,  # 15 minutes
                "strategy": CacheStrategy.SIMPLE,
                "levels": [CacheLevel.L1_MEMORY],
                "tags": ["billing"],
            },
        }
        
        # Query result cache with automatic invalidation
        self.query_cache_invalidation_rules = {
            "users": ["user_data", "subscription"],
            "research_tasks": ["research_task", "analytics"],
            "model_results": ["model_result", "analytics"],
            "subscriptions": ["subscription", "user_data"],
        }
        
        # Performance tracking
        self.cache_performance = {
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "invalidations": 0,
            "avg_retrieval_time": 0,
        }
    
    async def get_cached_query(
        self, 
        query_key: str, 
        data_type: str = "general",
        fallback: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get cached query result with multi-level cache support
        
        Args:
            query_key: Unique key for the query
            data_type: Type of data for cache configuration
            fallback: Async function to call if cache miss
        """
        start_time = time.time()
        
        try:
            # Try multi-level cache first
            result = await self.ml_optimizer.get(query_key)
            
            if result is not None:
                self.cache_performance["query_cache_hits"] += 1
                self._update_avg_retrieval_time(time.time() - start_time)
                return result
            
            # Try Redis cache as backup
            result = self.redis_cache.get(query_key)
            
            if result is not None:
                # Promote to multi-level cache
                config = self.cache_configs.get(data_type, self.cache_configs["general"])
                await self.ml_optimizer.set(
                    query_key, 
                    result, 
                    ttl=config["ttl"],
                    tags=config["tags"]
                )
                
                self.cache_performance["query_cache_hits"] += 1
                self._update_avg_retrieval_time(time.time() - start_time)
                return result
            
            # Cache miss
            self.cache_performance["query_cache_misses"] += 1
            
            # Execute fallback if provided
            if fallback:
                result = await fallback()
                if result is not None:
                    await self.set_cached_query(query_key, result, data_type)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error for {query_key}: {e}")
            return None
    
    async def set_cached_query(
        self, 
        query_key: str, 
        value: Any, 
        data_type: str = "general",
        custom_ttl: Optional[int] = None
    ) -> bool:
        """
        Set query result in cache with intelligent distribution
        """
        try:
            config = self.cache_configs.get(data_type, {
                "ttl": custom_ttl or 600,
                "strategy": CacheStrategy.SIMPLE,
                "levels": [CacheLevel.L2_REDIS],
                "tags": [data_type],
            })
            
            # Set in multi-level cache
            ml_success = await self.ml_optimizer.set(
                query_key,
                value,
                ttl=custom_ttl or config["ttl"],
                tags=config["tags"]
            )
            
            # Also set in Redis for redundancy
            redis_success = self.redis_cache.set(
                query_key,
                value,
                timeout=custom_ttl or config["ttl"],
                strategy=config["strategy"]
            )
            
            return ml_success and redis_success
            
        except Exception as e:
            logger.error(f"Cache set error for {query_key}: {e}")
            return False
    
    async def invalidate_related_caches(self, table_name: str, record_id: Optional[int] = None):
        """
        Invalidate caches related to a database table update
        """
        self.cache_performance["invalidations"] += 1
        
        # Get data types to invalidate
        data_types = self.query_cache_invalidation_rules.get(table_name, [])
        
        for data_type in data_types:
            # Invalidate by tag in Redis
            self.redis_cache.invalidate_by_tag(data_type)
            
            # Clear related entries in multi-level cache
            # This is a simplified approach - in production, implement tag-based invalidation
            if record_id:
                pattern_key = f"{table_name}:{record_id}:*"
                await self.ml_optimizer.delete(pattern_key)
    
    def _update_avg_retrieval_time(self, retrieval_time: float):
        """Update average retrieval time with exponential moving average"""
        alpha = 0.1
        self.cache_performance["avg_retrieval_time"] = (
            (1 - alpha) * self.cache_performance["avg_retrieval_time"] + 
            alpha * retrieval_time * 1000  # Convert to ms
        )
    
    async def optimize_cache_performance(self):
        """Run cache optimization and return results"""
        optimization_results = await self.ml_optimizer.optimize_cache_performance()
        
        # Add integrated system metrics
        optimization_results["integrated_metrics"] = {
            "query_cache_hit_rate": (
                self.cache_performance["query_cache_hits"] / 
                max(1, self.cache_performance["query_cache_hits"] + self.cache_performance["query_cache_misses"])
            ) * 100,
            "avg_retrieval_time_ms": self.cache_performance["avg_retrieval_time"],
            "total_invalidations": self.cache_performance["invalidations"],
        }
        
        return optimization_results
    
    async def get_cache_report(self) -> Dict[str, Any]:
        """Get comprehensive cache report"""
        ml_report = await self.ml_optimizer.get_comprehensive_cache_report()
        redis_stats = self.redis_cache.get_cache_stats()
        
        return {
            "multi_level_cache": ml_report,
            "redis_cache": redis_stats,
            "integrated_performance": self.cache_performance,
            "recommendations": await self._generate_integrated_recommendations(),
        }
    
    async def _generate_integrated_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for integrated cache system"""
        recommendations = []
        
        hit_rate = (
            self.cache_performance["query_cache_hits"] / 
            max(1, self.cache_performance["query_cache_hits"] + self.cache_performance["query_cache_misses"])
        ) * 100
        
        if hit_rate < 70:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "recommendation": "Low query cache hit rate - consider reviewing cache keys and TTL settings",
            })
        
        if self.cache_performance["avg_retrieval_time"] > 50:  # ms
            recommendations.append({
                "type": "latency",
                "priority": "medium",
                "recommendation": "High average retrieval time - consider moving hot data to L1 cache",
            })
        
        if self.cache_performance["invalidations"] > 1000:
            recommendations.append({
                "type": "invalidation",
                "priority": "low",
                "recommendation": "High invalidation count - consider implementing more granular cache keys",
            })
        
        return recommendations


def cached_query(data_type: str = "general", ttl: Optional[int] = None):
    """
    Decorator for caching database query results
    
    Usage:
        @cached_query(data_type="user_data", ttl=300)
        async def get_user_by_id(user_id: int):
            # Database query here
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hashlib.md5(f'{args}{kwargs}'.encode()).hexdigest()}"
            
            # Get cache instance
            cache_system = IntegratedCacheSystem()
            
            # Define fallback function
            async def fallback():
                return await func(*args, **kwargs)
            
            # Try cache first
            result = await cache_system.get_cached_query(
                cache_key, 
                data_type=data_type,
                fallback=fallback
            )
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache_on_update(table_name: str):
    """
    Decorator to invalidate related caches after database update
    
    Usage:
        @invalidate_cache_on_update("users")
        async def update_user(user_id: int, data: dict):
            # Update user in database
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the update
            result = await func(*args, **kwargs)
            
            # Invalidate related caches
            cache_system = IntegratedCacheSystem()
            
            # Try to extract record ID from args/kwargs
            record_id = None
            if args and isinstance(args[0], int):
                record_id = args[0]
            elif "id" in kwargs:
                record_id = kwargs["id"]
            elif hasattr(result, "id"):
                record_id = result.id
            
            await cache_system.invalidate_related_caches(table_name, record_id)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
integrated_cache = IntegratedCacheSystem()


# Convenience functions
async def get_from_cache(key: str, data_type: str = "general") -> Optional[Any]:
    """Get value from integrated cache"""
    return await integrated_cache.get_cached_query(key, data_type)


async def set_in_cache(key: str, value: Any, data_type: str = "general", ttl: Optional[int] = None) -> bool:
    """Set value in integrated cache"""
    return await integrated_cache.set_cached_query(key, value, data_type, ttl)


async def invalidate_cache(table_name: str, record_id: Optional[int] = None):
    """Invalidate related caches"""
    await integrated_cache.invalidate_related_caches(table_name, record_id)


async def optimize_caches():
    """Run cache optimization"""
    return await integrated_cache.optimize_cache_performance()


async def get_cache_health():
    """Get cache health report"""
    return await integrated_cache.get_cache_report()
