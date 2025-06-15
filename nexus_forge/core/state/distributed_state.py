"""
Distributed state management for Parallax Pal

Provides Redis and Cloud Firestore integration for managing distributed
state across multiple instances of the application.
"""

from google.cloud import firestore
import redis.asyncio as redis
from typing import Optional, Dict, Any, List, AsyncGenerator
import json
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DistributedStateManager:
    """Manage distributed state across multiple instances"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        firestore_project: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize distributed state manager
        
        Args:
            redis_url: Redis connection URL
            firestore_project: Google Cloud project ID
            cache_ttl: Default cache TTL in seconds
        """
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        
        # Initialize Redis
        self._redis: Optional[redis.Redis] = None
        
        # Initialize Firestore
        if firestore_project:
            self.firestore = firestore.AsyncClient(project=firestore_project)
        else:
            self.firestore = firestore.AsyncClient()
        
        # Pub/sub for real-time updates
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscription_tasks: Dict[str, asyncio.Task] = {}
    
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
        """Close all connections"""
        # Cancel subscription tasks
        for task in self._subscription_tasks.values():
            task.cancel()
        
        # Close pub/sub
        if self._pubsub:
            await self._pubsub.close()
        
        # Close Redis
        if self._redis:
            await self._redis.close()
    
    # Session Management
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session state from Redis with Firestore fallback
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state dict or None
        """
        redis_client = await self.get_redis()
        
        # Try Redis first (fast)
        try:
            state_json = await redis_client.get(f"session:{session_id}")
            if state_json:
                return json.loads(state_json)
        except redis.RedisError as e:
            logger.warning(f"Redis error getting session: {e}")
        
        # Fallback to Firestore (persistent)
        try:
            doc = await self.firestore.collection('sessions').document(session_id).get()
            if doc.exists:
                state = doc.to_dict()
                
                # Cache in Redis for next time
                try:
                    await redis_client.setex(
                        f"session:{session_id}",
                        self.cache_ttl,
                        json.dumps(state)
                    )
                except redis.RedisError:
                    pass  # Continue even if caching fails
                
                return state
        except Exception as e:
            logger.error(f"Firestore error getting session: {e}")
        
        return None
    
    async def update_session_state(
        self, 
        session_id: str, 
        updates: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Update session state in both Redis and Firestore
        
        Args:
            session_id: Session identifier
            updates: State updates to apply
            ttl: Optional TTL override
        """
        redis_client = await self.get_redis()
        
        # Get current state
        current_state = await self.get_session_state(session_id) or {}
        
        # Apply updates
        current_state.update(updates)
        current_state['updated_at'] = datetime.now().isoformat()
        
        # Update Redis
        try:
            await redis_client.setex(
                f"session:{session_id}",
                ttl or self.cache_ttl,
                json.dumps(current_state)
            )
        except redis.RedisError as e:
            logger.warning(f"Redis error updating session: {e}")
        
        # Update Firestore
        try:
            await self.firestore.collection('sessions').document(session_id).set(
                current_state, merge=True
            )
        except Exception as e:
            logger.error(f"Firestore error updating session: {e}")
            raise
    
    async def delete_session(self, session_id: str):
        """Delete session from both stores"""
        redis_client = await self.get_redis()
        
        # Delete from Redis
        try:
            await redis_client.delete(f"session:{session_id}")
        except redis.RedisError:
            pass
        
        # Delete from Firestore
        try:
            await self.firestore.collection('sessions').document(session_id).delete()
        except Exception:
            pass
    
    # Research Task Management
    
    async def create_research_task(
        self,
        user_id: str,
        query: str,
        mode: str = "comprehensive"
    ) -> str:
        """
        Create a new research task
        
        Returns:
            Task ID
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id[:8]}"
        
        task_data = {
            'id': task_id,
            'user_id': user_id,
            'query': query,
            'mode': mode,
            'status': 'created',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'results': {},
            'agents_status': {}
        }
        
        # Store in Firestore
        await self.firestore.collection('research_tasks').document(task_id).set(task_data)
        
        # Cache in Redis
        redis_client = await self.get_redis()
        try:
            await redis_client.setex(
                f"task:{task_id}",
                86400,  # 24 hours
                json.dumps(task_data)
            )
        except redis.RedisError:
            pass
        
        return task_id
    
    async def update_task_progress(
        self,
        task_id: str,
        progress: int,
        agent: Optional[str] = None,
        status: Optional[str] = None,
        partial_results: Optional[Dict] = None
    ):
        """Update research task progress"""
        redis_client = await self.get_redis()
        
        updates = {
            'progress': progress,
            'updated_at': datetime.now().isoformat()
        }
        
        if status:
            updates['status'] = status
        
        if agent:
            updates[f'agents_status.{agent}'] = {
                'status': status or 'working',
                'progress': progress,
                'updated_at': datetime.now().isoformat()
            }
        
        if partial_results:
            updates[f'results.{agent}'] = partial_results
        
        # Update Redis
        try:
            task_json = await redis_client.get(f"task:{task_id}")
            if task_json:
                task_data = json.loads(task_json)
                task_data.update(updates)
                await redis_client.setex(
                    f"task:{task_id}",
                    86400,
                    json.dumps(task_data)
                )
        except redis.RedisError:
            pass
        
        # Update Firestore
        await self.firestore.collection('research_tasks').document(task_id).update(updates)
        
        # Publish progress event
        await self.publish_event(f"task:{task_id}", {
            'type': 'progress',
            'task_id': task_id,
            'progress': progress,
            'agent': agent,
            'status': status
        })
    
    # Event Publishing and Subscription
    
    async def publish_event(self, channel: str, event: Dict[str, Any]):
        """
        Publish event for multi-instance coordination
        
        Args:
            channel: Channel to publish to
            event: Event data
        """
        redis_client = await self.get_redis()
        
        event['timestamp'] = datetime.now().isoformat()
        
        try:
            await redis_client.publish(channel, json.dumps(event))
        except redis.RedisError as e:
            logger.error(f"Error publishing event: {e}")
    
    @asynccontextmanager
    async def subscribe_to_events(self, channels: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Subscribe to events from other instances
        
        Args:
            channels: List of channels to subscribe to
            
        Yields:
            Event dictionaries
        """
        redis_client = await self.get_redis()
        pubsub = redis_client.pubsub()
        
        try:
            # Subscribe to channels
            await pubsub.subscribe(*channels)
            
            # Yield events
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event = json.loads(message['data'])
                        event['channel'] = message['channel']
                        yield event
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in message: {message['data']}")
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()
    
    # Distributed Locking
    
    @asynccontextmanager
    async def distributed_lock(
        self,
        resource: str,
        timeout: int = 10,
        blocking: bool = True
    ):
        """
        Acquire a distributed lock
        
        Args:
            resource: Resource to lock
            timeout: Lock timeout in seconds
            blocking: Whether to wait for lock
            
        Yields:
            Lock context
        """
        redis_client = await self.get_redis()
        lock_key = f"lock:{resource}"
        lock_value = f"{datetime.now().timestamp()}"
        
        # Try to acquire lock
        acquired = False
        retry_count = 0
        max_retries = timeout if blocking else 1
        
        while retry_count < max_retries and not acquired:
            acquired = await redis_client.set(
                lock_key,
                lock_value,
                nx=True,  # Only set if not exists
                ex=timeout
            )
            
            if not acquired and blocking:
                await asyncio.sleep(1)
                retry_count += 1
        
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for {resource}")
        
        try:
            yield
        finally:
            # Release lock if we still hold it
            current_value = await redis_client.get(lock_key)
            if current_value == lock_value:
                await redis_client.delete(lock_key)
    
    # Caching utilities
    
    async def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value"""
        redis_client = await self.get_redis()
        
        try:
            value = await redis_client.get(f"cache:{key}")
            if value:
                return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError):
            pass
        
        return None
    
    async def set_cached(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set cached value"""
        redis_client = await self.get_redis()
        
        try:
            await redis_client.setex(
                f"cache:{key}",
                ttl or self.cache_ttl,
                json.dumps(value)
            )
        except (redis.RedisError, json.JSONEncodeError) as e:
            logger.warning(f"Cache set error: {e}")
    
    async def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        redis_client = await self.get_redis()
        
        try:
            async for key in redis_client.scan_iter(match=f"cache:{pattern}"):
                await redis_client.delete(key)
        except redis.RedisError as e:
            logger.warning(f"Cache invalidation error: {e}")
    
    # Metrics and Analytics
    
    async def increment_metric(self, metric: str, value: int = 1):
        """Increment a metric counter"""
        redis_client = await self.get_redis()
        
        # Daily metric key
        today = datetime.now().strftime('%Y%m%d')
        key = f"metric:{metric}:{today}"
        
        try:
            await redis_client.incrby(key, value)
            await redis_client.expire(key, 7 * 86400)  # Keep for 7 days
        except redis.RedisError:
            pass
    
    async def get_metrics(self, metric: str, days: int = 7) -> Dict[str, int]:
        """Get metrics for the last N days"""
        redis_client = await self.get_redis()
        
        metrics = {}
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            key = f"metric:{metric}:{date}"
            
            try:
                value = await redis_client.get(key)
                metrics[date] = int(value) if value else 0
            except redis.RedisError:
                metrics[date] = 0
        
        return metrics
    
    # User Session Management
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all active sessions for a user"""
        redis_client = await self.get_redis()
        
        try:
            sessions = await redis_client.smembers(f"user_sessions:{user_id}")
            return list(sessions)
        except redis.RedisError:
            return []
    
    async def add_user_session(self, user_id: str, session_id: str):
        """Add session to user's active sessions"""
        redis_client = await self.get_redis()
        
        try:
            await redis_client.sadd(f"user_sessions:{user_id}", session_id)
            await redis_client.expire(f"user_sessions:{user_id}", 86400)  # 24 hours
        except redis.RedisError:
            pass
    
    async def remove_user_session(self, user_id: str, session_id: str):
        """Remove session from user's active sessions"""
        redis_client = await self.get_redis()
        
        try:
            await redis_client.srem(f"user_sessions:{user_id}", session_id)
        except redis.RedisError:
            pass