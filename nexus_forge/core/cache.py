import redis
from typing import Optional, Any, Dict, List, Union
import json
import logging
from functools import wraps
import os
import hashlib
import zlib
import pickle
import time
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Different caching strategies for AI responses"""
    SIMPLE = "simple"  # Basic key-value caching
    COMPRESSED = "compressed"  # Compressed for large responses
    SEMANTIC = "semantic"  # Semantic similarity-based caching
    TIERED = "tiered"  # Multiple TTL tiers based on content type

class CacheMetrics:
    """Track cache performance metrics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.cache_size = 0
        self.last_reset = time.time()
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def record_set(self):
        self.sets += 1
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def reset(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.last_reset = time.time()

class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.client = redis.StrictRedis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self.client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
        
        self.default_timeout = 3600  # 1 hour default
        self.compression_threshold = 1024  # Compress if > 1KB
        self.max_cache_size = 100 * 1024 * 1024  # 100MB max cache size
        self.metrics = CacheMetrics()
        
        # Cache warming configurations
        self.warm_cache_keys = [
            "popular_gemini_prompts",
            "common_code_patterns",
            "design_templates"
        ]
        
        # AI-specific TTL configurations
        self.ai_ttl_config = {
            "gemini_code": 86400,  # 24 hours for code generation
            "gemini_text": 7200,   # 2 hours for text generation
            "jules_specs": 43200,  # 12 hours for app specifications
            "imagen_design": 604800,  # 1 week for image designs
            "veo_metadata": 3600,  # 1 hour for video metadata
            "chat_session": 1800   # 30 minutes for chat sessions
        }

    def get(self, key: str, strategy: CacheStrategy = CacheStrategy.SIMPLE) -> Optional[Any]:
        """Get value from cache with intelligent deserialization"""
        try:
            value = self.client.get(key)
            if value:
                self.metrics.record_hit()
                return self._deserialize_value(value, strategy)
            else:
                self.metrics.record_miss()
                return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {str(e)}")
            self.metrics.record_miss()
            return None

    def set(self, key: str, value: Any, timeout: int = None, strategy: CacheStrategy = CacheStrategy.SIMPLE) -> bool:
        """Set value in cache with intelligent serialization"""
        try:
            serialized_value = self._serialize_value(value, strategy)
            ttl = timeout or self.default_timeout
            
            # Check cache size before setting
            if self._should_evict(serialized_value):
                self._evict_lru_items()
            
            result = self.client.set(key, serialized_value, ex=ttl)
            if result:
                self.metrics.record_set()
                logger.debug(f"Cached key {key} with TTL {ttl}s using {strategy.value} strategy")
            return result
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            result = self.client.delete(key)
            if result:
                logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {str(e)}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Redis DELETE PATTERN error for pattern {pattern}: {str(e)}")
            return 0

    def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.client.flushall()
            self.metrics.reset()
            logger.info("Cache cleared and metrics reset")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHALL error: {str(e)}")
            return False
    
    def _serialize_value(self, value: Any, strategy: CacheStrategy) -> bytes:
        """Serialize value based on caching strategy"""
        if strategy == CacheStrategy.COMPRESSED:
            # Use pickle for complex objects, then compress
            serialized = pickle.dumps(value)
            if len(serialized) > self.compression_threshold:
                return zlib.compress(serialized)
            return serialized
        elif strategy == CacheStrategy.SEMANTIC:
            # Store with semantic metadata for similarity matching
            semantic_data = {
                "content": value,
                "hash": self._generate_semantic_hash(value),
                "timestamp": time.time()
            }
            return pickle.dumps(semantic_data)
        else:
            # Simple JSON serialization
            return json.dumps(value).encode('utf-8')
    
    def _deserialize_value(self, value: bytes, strategy: CacheStrategy) -> Any:
        """Deserialize value based on caching strategy"""
        try:
            if strategy == CacheStrategy.COMPRESSED:
                # Try decompression first, fallback to direct unpickling
                try:
                    decompressed = zlib.decompress(value)
                    return pickle.loads(decompressed)
                except zlib.error:
                    return pickle.loads(value)
            elif strategy == CacheStrategy.SEMANTIC:
                semantic_data = pickle.loads(value)
                return semantic_data["content"]
            else:
                # Simple JSON deserialization
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Deserialization error with strategy {strategy.value}: {str(e)}")
            # Fallback: try all strategies
            for fallback_strategy in CacheStrategy:
                try:
                    return self._deserialize_value(value, fallback_strategy)
                except:
                    continue
            raise
    
    def _generate_semantic_hash(self, value: Any) -> str:
        """Generate semantic hash for similarity matching"""
        if isinstance(value, dict) and "content" in value:
            content = str(value["content"])
        else:
            content = str(value)
        
        # Create hash based on normalized content
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _should_evict(self, value: bytes) -> bool:
        """Check if cache eviction is needed"""
        try:
            info = self.client.info('memory')
            used_memory = info.get('used_memory', 0)
            return used_memory + len(value) > self.max_cache_size
        except:
            return False
    
    def _evict_lru_items(self, target_free_bytes: int = 10 * 1024 * 1024):
        """Evict least recently used items"""
        try:
            # Get all keys with their TTL
            keys = self.client.keys("*")
            if not keys:
                return
            
            # Sort by TTL (items with shorter TTL are evicted first)
            key_ttls = [(key, self.client.ttl(key)) for key in keys]
            key_ttls.sort(key=lambda x: x[1] if x[1] > 0 else float('inf'))
            
            # Evict items until we have enough space
            freed_bytes = 0
            for key, ttl in key_ttls:
                if freed_bytes >= target_free_bytes:
                    break
                
                try:
                    # Estimate size and delete
                    key_size = len(self.client.get(key) or b'')
                    if self.client.delete(key):
                        freed_bytes += key_size
                        self.metrics.evictions += 1
                except:
                    continue
            
            logger.info(f"Evicted cache items, freed {freed_bytes} bytes")
        except Exception as e:
            logger.error(f"Cache eviction error: {str(e)}")
    
    def warm_cache(self, warm_data: Dict[str, Any]):
        """Warm cache with popular/common data"""
        logger.info("Starting cache warming process")
        
        for key, data in warm_data.items():
            try:
                # Determine appropriate TTL and strategy
                if "code" in key:
                    ttl = self.ai_ttl_config["gemini_code"]
                    strategy = CacheStrategy.COMPRESSED
                elif "design" in key:
                    ttl = self.ai_ttl_config["imagen_design"]
                    strategy = CacheStrategy.SIMPLE
                else:
                    ttl = self.default_timeout
                    strategy = CacheStrategy.SIMPLE
                
                self.set(f"warm:{key}", data, ttl, strategy)
            except Exception as e:
                logger.error(f"Failed to warm cache for key {key}: {str(e)}")
        
        logger.info(f"Cache warming completed for {len(warm_data)} items")
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with a specific tag"""
        pattern = f"*:{tag}:*"
        deleted = self.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted} cache entries with tag: {tag}")
        return deleted
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            redis_info = self.client.info()
            return {
                "hit_rate": round(self.metrics.get_hit_rate(), 2),
                "total_hits": self.metrics.hits,
                "total_misses": self.metrics.misses,
                "total_sets": self.metrics.sets,
                "evictions": self.metrics.evictions,
                "redis_memory_used": redis_info.get('used_memory_human', 'Unknown'),
                "redis_connected_clients": redis_info.get('connected_clients', 0),
                "redis_total_commands": redis_info.get('total_commands_processed', 0),
                "uptime_seconds": time.time() - self.metrics.last_reset
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Test set
            set_result = self.set(test_key, test_value, 60)
            
            # Test get
            get_result = self.get(test_key)
            
            # Test delete
            delete_result = self.delete(test_key)
            
            latency = time.time() - start_time
            
            return {
                "status": "healthy" if set_result and get_result and delete_result else "unhealthy",
                "latency_ms": round(latency * 1000, 2),
                "operations_successful": {
                    "set": set_result,
                    "get": bool(get_result),
                    "delete": delete_result
                },
                "cache_stats": self.get_cache_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": None
            }

# Advanced cache validation and health checks
def validate_cache_integrity():
    """Validate cache data integrity and performance"""
    cache = RedisCache()
    
    issues = []
    recommendations = []
    
    try:
        # Test basic operations
        test_key = "cache_validation_test"
        test_value = {"test": True, "timestamp": time.time()}
        
        # Test set operation
        if not cache.set(test_key, test_value, 60):
            issues.append("Cache SET operation failed")
        
        # Test get operation
        retrieved = cache.get(test_key)
        if not retrieved or retrieved.get("test") != True:
            issues.append("Cache GET operation failed or data corrupted")
        
        # Test delete operation
        if not cache.delete(test_key):
            issues.append("Cache DELETE operation failed")
        
        # Check cache statistics
        stats = cache.get_cache_stats()
        hit_rate = stats.get("hit_rate", 0)
        
        if hit_rate < 30:
            issues.append(f"Very low cache hit rate: {hit_rate}%")
            recommendations.append("Review cache TTL values and key generation strategy")
        elif hit_rate < 60:
            recommendations.append("Consider optimizing cache strategy for better hit rates")
        
        # Check memory usage
        memory_info = stats.get("redis_memory_used", "")
        if "GB" in memory_info:
            gb_used = float(memory_info.replace("GB", "").strip())
            if gb_used > 2:
                recommendations.append("High memory usage detected - consider cache cleanup")
        
        # Check for cache warming effectiveness
        warm_keys = ["warm:popular_python_patterns", "warm:react_component_templates", "warm:common_design_patterns"]
        missing_warm_keys = []
        
        for key in warm_keys:
            if not cache.get(key):
                missing_warm_keys.append(key)
        
        if missing_warm_keys:
            recommendations.append(f"Re-warm cache with missing keys: {missing_warm_keys}")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
            "recommendations": recommendations,
            "statistics": stats,
            "validation_time": time.time()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "validation_time": time.time()
        }

# Cache decorator for API endpoints
def cache_response(timeout: int = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache first
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value
            
            # If not in cache, execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, timeout)
            logger.debug(f"Cache miss for key: {cache_key}")
            return result
            
        return wrapper
    return decorator

# Initialize Redis cache
cache = RedisCache()