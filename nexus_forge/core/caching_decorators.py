"""
Advanced Caching Decorators for AI Services
Provides intelligent caching strategies for different types of AI responses.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from .cache import CacheStrategy, RedisCache

logger = logging.getLogger(__name__)

# Global cache instance
_cache_instance = None


def get_cache_instance() -> RedisCache:
    """Get or create cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def generate_cache_key(
    func_name: str, args: tuple, kwargs: dict, include_timestamp: bool = False
) -> str:
    """Generate a consistent cache key from function arguments"""
    # Create a deterministic key from function name and arguments
    key_data = {
        "function": func_name,
        "args": args,
        "kwargs": {k: v for k, v in kwargs.items() if k not in ["self", "cls"]},
    }

    # Add timestamp for time-sensitive caching
    if include_timestamp:
        # Round to nearest hour for grouping similar requests
        hour_timestamp = int(time.time() // 3600) * 3600
        key_data["timestamp"] = hour_timestamp

    # Create hash of the key data
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return f"{func_name}:{key_hash}"


def cache_ai_response(
    ttl: Optional[int] = None,
    strategy: CacheStrategy = CacheStrategy.SIMPLE,
    cache_tag: Optional[str] = None,
    invalidate_patterns: Optional[List[str]] = None,
    ignore_args: Optional[List[str]] = None,
):
    """
    Cache AI service responses with intelligent strategies.

    Args:
        ttl: Time to live in seconds (uses AI-specific defaults if None)
        strategy: Caching strategy to use
        cache_tag: Tag for bulk invalidation
        invalidate_patterns: Patterns to invalidate when this cache is set
        ignore_args: Argument names to ignore when generating cache key
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_instance()

            # Filter ignored arguments
            filtered_kwargs = kwargs.copy()
            if ignore_args:
                for arg in ignore_args:
                    filtered_kwargs.pop(arg, None)

            # Generate cache key
            cache_key = generate_cache_key(func.__name__, args, filtered_kwargs)

            # Add cache tag if specified
            if cache_tag:
                cache_key = f"{cache_tag}:{cache_key}"

            # Try to get from cache
            cached_result = cache.get(cache_key, strategy)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")

                # Add cache metadata
                if isinstance(cached_result, dict):
                    cached_result["_cache_metadata"] = {
                        "hit": True,
                        "key": cache_key,
                        "timestamp": time.time(),
                    }

                return cached_result

            # Cache miss - execute function
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Determine TTL based on function type and result
                actual_ttl = ttl or _determine_ai_ttl(func.__name__, result)

                # Add cache metadata to result
                if isinstance(result, dict):
                    result["_cache_metadata"] = {
                        "hit": False,
                        "key": cache_key,
                        "timestamp": time.time(),
                        "execution_time": execution_time,
                        "ttl": actual_ttl,
                    }

                # Cache the result
                cache.set(cache_key, result, actual_ttl, strategy)

                # Invalidate related patterns if specified
                if invalidate_patterns:
                    for pattern in invalidate_patterns:
                        cache.delete_pattern(pattern)

                logger.debug(
                    f"Cached result for {func.__name__} with TTL {actual_ttl}s"
                )
                return result

            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Convert to async for consistent handling
            return asyncio.run(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_with_ttl(ttl: int, strategy: CacheStrategy = CacheStrategy.SIMPLE):
    """
    Simple cache decorator with explicit TTL.

    Args:
        ttl: Time to live in seconds
        strategy: Caching strategy to use
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_instance()
            cache_key = generate_cache_key(func.__name__, args, kwargs)

            # Try cache first
            cached_result = cache.get(cache_key, strategy)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl, strategy)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache_instance()
            cache_key = generate_cache_key(func.__name__, args, kwargs)

            # Try cache first
            cached_result = cache.get(cache_key, strategy)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl, strategy)
            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def invalidate_cache(
    patterns: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    specific_keys: Optional[List[str]] = None,
):
    """
    Invalidate cache entries after function execution.

    Args:
        patterns: Patterns to delete (e.g., ["gemini:*", "jules:*"])
        tags: Tags to invalidate (e.g., ["user_123", "project_456"])
        specific_keys: Specific cache keys to delete
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function first
            result = await func(*args, **kwargs)

            # Then invalidate cache
            cache = get_cache_instance()

            if patterns:
                for pattern in patterns:
                    cache.delete_pattern(pattern)
                    logger.debug(f"Invalidated cache pattern: {pattern}")

            if tags:
                for tag in tags:
                    cache.invalidate_by_tag(tag)
                    logger.debug(f"Invalidated cache tag: {tag}")

            if specific_keys:
                for key in specific_keys:
                    cache.delete(key)
                    logger.debug(f"Invalidated cache key: {key}")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Execute function first
            result = func(*args, **kwargs)

            # Then invalidate cache
            cache = get_cache_instance()

            if patterns:
                for pattern in patterns:
                    cache.delete_pattern(pattern)

            if tags:
                for tag in tags:
                    cache.invalidate_by_tag(tag)

            if specific_keys:
                for key in specific_keys:
                    cache.delete(key)

            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def semantic_cache(similarity_threshold: float = 0.8, ttl: Optional[int] = None):
    """
    Cache based on semantic similarity of inputs.
    Useful for similar prompts that should return similar results.

    Args:
        similarity_threshold: Minimum similarity score to consider a cache hit
        ttl: Time to live in seconds
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_instance()

            # Extract prompt/content for similarity matching
            prompt = _extract_prompt_from_args(args, kwargs)
            if not prompt:
                # Fallback to regular caching if no prompt found
                return await cache_ai_response(ttl=ttl)(func)(*args, **kwargs)

            # Generate semantic key
            semantic_hash = hashlib.md5(prompt.lower().strip().encode()).hexdigest()
            cache_key = f"semantic:{func.__name__}:{semantic_hash}"

            # Check for semantically similar cached results
            similar_result = _find_similar_cached_result(
                cache, func.__name__, prompt, similarity_threshold
            )
            if similar_result:
                logger.debug(f"Semantic cache hit for {func.__name__}")
                return similar_result

            # Execute function and cache with semantic strategy
            result = await func(*args, **kwargs)
            actual_ttl = ttl or _determine_ai_ttl(func.__name__, result)

            cache.set(
                cache_key,
                {"result": result, "prompt": prompt, "timestamp": time.time()},
                actual_ttl,
                CacheStrategy.SEMANTIC,
            )

            return result

        # Return async wrapper (semantic caching is async-only for now)
        return async_wrapper

    return decorator


def conditional_cache(
    condition_func: Callable[[Any], bool],
    ttl: Optional[int] = None,
    strategy: CacheStrategy = CacheStrategy.SIMPLE,
):
    """
    Cache only when condition is met.

    Args:
        condition_func: Function that takes the result and returns True if it should be cached
        ttl: Time to live in seconds
        strategy: Caching strategy
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_instance()
            cache_key = generate_cache_key(func.__name__, args, kwargs)

            # Try cache first
            cached_result = cache.get(cache_key, strategy)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache only if condition is met
            if condition_func(result):
                actual_ttl = ttl or _determine_ai_ttl(func.__name__, result)
                cache.set(cache_key, result, actual_ttl, strategy)
                logger.debug(f"Conditionally cached result for {func.__name__}")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache_instance()
            cache_key = generate_cache_key(func.__name__, args, kwargs)

            # Try cache first
            cached_result = cache.get(cache_key, strategy)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache only if condition is met
            if condition_func(result):
                actual_ttl = ttl or _determine_ai_ttl(func.__name__, result)
                cache.set(cache_key, result, actual_ttl, strategy)

            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Helper functions


def _determine_ai_ttl(func_name: str, result: Any) -> int:
    """Determine appropriate TTL based on function name and result"""
    cache = get_cache_instance()

    # Check function name patterns
    if "code" in func_name.lower() or "generate_code" in func_name:
        return cache.ai_ttl_config["gemini_code"]
    elif "text" in func_name.lower() or "generate_content" in func_name:
        return cache.ai_ttl_config["gemini_text"]
    elif "jules" in func_name.lower() or "app_spec" in func_name.lower():
        return cache.ai_ttl_config["jules_specs"]
    elif "image" in func_name.lower() or "design" in func_name.lower():
        return cache.ai_ttl_config["imagen_design"]
    elif "video" in func_name.lower() or "veo" in func_name.lower():
        return cache.ai_ttl_config["veo_metadata"]
    elif "chat" in func_name.lower() or "session" in func_name.lower():
        return cache.ai_ttl_config["chat_session"]

    # Check result content for additional hints
    if isinstance(result, dict):
        if "code" in str(result).lower():
            return cache.ai_ttl_config["gemini_code"]
        elif "generation_type" in result:
            gen_type = result["generation_type"]
            if gen_type == "code":
                return cache.ai_ttl_config["gemini_code"]
            elif gen_type == "app_specification":
                return cache.ai_ttl_config["jules_specs"]

    # Default fallback
    return cache.default_timeout


def _extract_prompt_from_args(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract prompt text from function arguments"""
    # Check kwargs first
    for key in ["prompt", "text", "content", "message", "requirements"]:
        if key in kwargs and isinstance(kwargs[key], str):
            return kwargs[key]

    # Check positional args
    for arg in args:
        if isinstance(arg, str) and len(arg) > 10:  # Likely a prompt
            return arg
        elif isinstance(arg, dict) and "prompt" in arg:
            return arg["prompt"]

    return None


def _find_similar_cached_result(
    cache: RedisCache, func_name: str, prompt: str, threshold: float
) -> Optional[Any]:
    """Find semantically similar cached results"""
    try:
        # Get all semantic cache keys for this function
        pattern = f"semantic:{func_name}:*"
        keys = cache.client.keys(pattern)

        prompt_words = set(prompt.lower().split())

        for key in keys:
            try:
                cached_data = cache.get(
                    key.decode() if isinstance(key, bytes) else key,
                    CacheStrategy.SEMANTIC,
                )
                if not cached_data or "prompt" not in cached_data:
                    continue

                cached_prompt = cached_data["prompt"]
                cached_words = set(cached_prompt.lower().split())

                # Simple Jaccard similarity
                intersection = len(prompt_words & cached_words)
                union = len(prompt_words | cached_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    logger.debug(
                        f"Found similar cached result with similarity {similarity:.2f}"
                    )
                    return cached_data["result"]
            except Exception as e:
                logger.debug(f"Error checking similarity for key {key}: {str(e)}")
                continue

        return None
    except Exception as e:
        logger.error(f"Error in semantic similarity search: {str(e)}")
        return None


# Cache warming utilities


async def warm_ai_cache():
    """Warm cache with popular AI responses"""
    cache = get_cache_instance()

    # Common prompts and responses for warming
    warm_data = {
        "common_python_patterns": {
            "content": "# Common Python patterns\nclass APIClient:\n    def __init__(self):\n        pass",
            "language": "python",
            "type": "code_pattern",
        },
        "react_component_template": {
            "content": "import React from 'react';\n\nconst Component = () => {\n  return <div></div>;\n};\n\nexport default Component;",
            "language": "typescript",
            "type": "react_template",
        },
        "api_error_handling": {
            "content": "try:\n    response = await api_call()\n    return response\nexcept Exception as e:\n    logger.error(f'API error: {e}')\n    raise",
            "language": "python",
            "type": "error_handling",
        },
    }

    cache.warm_cache(warm_data)
    logger.info("AI cache warming completed")


# Cache monitoring utilities


def get_cache_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive cache performance metrics"""
    cache = get_cache_instance()
    stats = cache.get_cache_stats()

    # Add additional performance metrics
    stats.update(
        {
            "cache_strategies_used": {
                "simple": "JSON serialization",
                "compressed": "Pickle + compression",
                "semantic": "Semantic similarity",
                "tiered": "Multi-TTL tiers",
            },
            "ai_specific_ttls": cache.ai_ttl_config,
            "recommendations": _generate_cache_recommendations(stats),
        }
    )

    return stats


def _generate_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate cache optimization recommendations"""
    recommendations = []

    hit_rate = stats.get("hit_rate", 0)
    if hit_rate < 50:
        recommendations.append(
            "Consider increasing TTL values or improving cache key generation"
        )
    elif hit_rate > 90:
        recommendations.append(
            "Excellent cache performance! Consider expanding caching to more functions"
        )

    evictions = stats.get("evictions", 0)
    if evictions > 100:
        recommendations.append(
            "High eviction rate detected. Consider increasing cache size or optimizing data storage"
        )

    return recommendations
