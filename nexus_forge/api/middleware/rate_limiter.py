"""
Rate limiting middleware for Parallax Pal API

Provides Redis-based rate limiting with configurable windows and limits.
"""

from fastapi import Request, HTTPException
from typing import Callable, Optional
import redis.asyncio as redis
import time
import logging
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter with sliding window"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize rate limiter
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if not self._redis:
            self._redis = await redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        return self._redis
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier
        
        Priority:
        1. Authenticated user ID
        2. API key
        3. IP address + User-Agent hash
        """
        # Check for authenticated user
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:16]}"  # Use prefix only for privacy
        
        # Fall back to IP + User-Agent
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Create hash for privacy
        identifier = f"{client_ip}:{user_agent}"
        id_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        
        return f"ip:{id_hash}"
    
    async def check_rate_limit(
        self, 
        key: str, 
        max_requests: int = 60,
        window_seconds: int = 60,
        burst_size: Optional[int] = None
    ) -> tuple[bool, dict]:
        """
        Check if rate limit is exceeded using sliding window
        
        Args:
            key: Rate limit key
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            burst_size: Optional burst allowance
            
        Returns:
            Tuple of (allowed, metadata)
        """
        redis_client = await self.get_redis()
        
        now = time.time()
        window_start = now - window_seconds
        
        # Sliding window key
        window_key = f"rate_limit:{key}:{int(now // window_seconds)}"
        
        try:
            # Remove old entries
            await redis_client.zremrangebyscore(window_key, 0, window_start)
            
            # Count requests in current window
            request_count = await redis_client.zcard(window_key)
            
            # Check burst if configured
            if burst_size:
                burst_key = f"burst:{key}"
                burst_count = await redis_client.get(burst_key)
                if burst_count and int(burst_count) >= burst_size:
                    return False, {
                        "limit": max_requests,
                        "remaining": 0,
                        "reset": int(now + window_seconds),
                        "burst_exceeded": True
                    }
            
            # Check rate limit
            if request_count >= max_requests:
                # Calculate when the oldest request will expire
                oldest_request = await redis_client.zrange(
                    window_key, 0, 0, withscores=True
                )
                if oldest_request:
                    reset_time = oldest_request[0][1] + window_seconds
                else:
                    reset_time = now + window_seconds
                
                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "reset": int(reset_time)
                }
            
            # Add current request
            await redis_client.zadd(window_key, {str(now): now})
            await redis_client.expire(window_key, window_seconds * 2)
            
            # Update burst counter if needed
            if burst_size:
                burst_key = f"burst:{key}"
                await redis_client.incr(burst_key)
                await redis_client.expire(burst_key, 1)  # 1 second burst window
            
            remaining = max_requests - request_count - 1
            
            return True, {
                "limit": max_requests,
                "remaining": remaining,
                "reset": int(now + window_seconds)
            }
            
        except redis.RedisError as e:
            # Log error but don't block requests if Redis is down
            logger.error(f"Redis error in rate limiting: {e}")
            return True, {
                "limit": max_requests,
                "remaining": -1,
                "reset": -1,
                "redis_error": True
            }
    
    def middleware(
        self, 
        max_requests: int = 60,
        window_seconds: int = 60,
        burst_size: Optional[int] = None,
        endpoints: Optional[list] = None
    ):
        """
        Create rate limiting middleware
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            burst_size: Optional burst allowance
            endpoints: Optional list of endpoints to rate limit
        """
        async def rate_limit_middleware(request: Request, call_next: Callable):
            # Skip rate limiting for health checks
            if request.url.path in ["/health", "/metrics"]:
                return await call_next(request)
            
            # Check if endpoint should be rate limited
            if endpoints and request.url.path not in endpoints:
                return await call_next(request)
            
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Create rate limit key with endpoint
            endpoint_key = request.url.path.replace("/", "_")
            key = f"{client_id}:{endpoint_key}"
            
            # Check rate limit
            allowed, metadata = await self.check_rate_limit(
                key, max_requests, window_seconds, burst_size
            )
            
            if not allowed:
                # Add rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(metadata["limit"]),
                    "X-RateLimit-Remaining": str(metadata["remaining"]),
                    "X-RateLimit-Reset": str(metadata["reset"]),
                    "Retry-After": str(metadata["reset"] - int(time.time()))
                }
                
                logger.warning(
                    f"Rate limit exceeded for {client_id} on {request.url.path}"
                )
                
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later.",
                    headers=headers
                )
            
            # Add rate limit headers to response
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
            response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
            response.headers["X-RateLimit-Reset"] = str(metadata["reset"])
            
            return response
        
        return rate_limit_middleware


class WebSocketRateLimiter:
    """Rate limiter specifically for WebSocket connections"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def check_websocket_limit(
        self,
        user_id: str,
        max_connections: int = 5,
        max_messages_per_minute: int = 100
    ) -> tuple[bool, str]:
        """
        Check WebSocket-specific rate limits
        
        Args:
            user_id: User identifier
            max_connections: Maximum concurrent connections
            max_messages_per_minute: Maximum messages per minute
            
        Returns:
            Tuple of (allowed, reason)
        """
        redis_client = await self.rate_limiter.get_redis()
        
        # Check concurrent connections
        conn_key = f"ws_connections:{user_id}"
        current_connections = await redis_client.get(conn_key)
        
        if current_connections and int(current_connections) >= max_connections:
            return False, f"Maximum {max_connections} concurrent connections exceeded"
        
        # Check message rate
        allowed, _ = await self.rate_limiter.check_rate_limit(
            f"ws_messages:{user_id}",
            max_messages_per_minute,
            60  # 1 minute window
        )
        
        if not allowed:
            return False, "Message rate limit exceeded"
        
        return True, "OK"
    
    async def register_connection(self, user_id: str, session_id: str):
        """Register a new WebSocket connection"""
        redis_client = await self.rate_limiter.get_redis()
        
        conn_key = f"ws_connections:{user_id}"
        sessions_key = f"ws_sessions:{user_id}"
        
        # Increment connection count
        await redis_client.incr(conn_key)
        await redis_client.expire(conn_key, 3600)  # 1 hour expiry
        
        # Track session
        await redis_client.sadd(sessions_key, session_id)
        await redis_client.expire(sessions_key, 3600)
    
    async def unregister_connection(self, user_id: str, session_id: str):
        """Unregister a WebSocket connection"""
        redis_client = await self.rate_limiter.get_redis()
        
        conn_key = f"ws_connections:{user_id}"
        sessions_key = f"ws_sessions:{user_id}"
        
        # Decrement connection count
        count = await redis_client.decr(conn_key)
        if count <= 0:
            await redis_client.delete(conn_key)
        
        # Remove session
        await redis_client.srem(sessions_key, session_id)


# Specialized rate limiters for different operations
class OperationRateLimiter:
    """Specialized rate limits for different operations"""
    
    # Rate limit configurations
    LIMITS = {
        "research_query": {
            "free": (10, 3600),      # 10 per hour
            "basic": (50, 3600),     # 50 per hour
            "pro": (200, 3600),      # 200 per hour
            "enterprise": (1000, 3600)  # 1000 per hour
        },
        "export": {
            "free": (5, 86400),      # 5 per day
            "basic": (20, 86400),    # 20 per day
            "pro": (100, 86400),     # 100 per day
            "enterprise": (500, 86400)  # 500 per day
        },
        "knowledge_graph": {
            "free": (20, 3600),      # 20 per hour
            "basic": (100, 3600),    # 100 per hour
            "pro": (500, 3600),      # 500 per hour
            "enterprise": (2000, 3600)  # 2000 per hour
        }
    }
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def check_operation_limit(
        self,
        user_id: str,
        operation: str,
        tier: str = "free"
    ) -> tuple[bool, dict]:
        """Check if user can perform operation based on their tier"""
        
        if operation not in self.LIMITS:
            return True, {"allowed": True}
        
        if tier not in self.LIMITS[operation]:
            tier = "free"
        
        limit, window = self.LIMITS[operation][tier]
        
        key = f"op:{operation}:{user_id}"
        allowed, metadata = await self.rate_limiter.check_rate_limit(
            key, limit, window
        )
        
        metadata["operation"] = operation
        metadata["tier"] = tier
        
        return allowed, metadata