"""
Memory Optimization Module
Advanced memory management and optimization for AI agents and database operations
"""

import asyncio
import gc
import logging
import os
import psutil
import sys
import tracemalloc
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import threading

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    heap_size_mb: float = 0.0
    gc_stats: Dict[str, Any] = field(default_factory=dict)
    largest_objects: List[Tuple[str, float]] = field(default_factory=list)
    memory_leaks: List[Dict[str, Any]] = field(default_factory=list)


class MemoryOptimizer:
    """
    Advanced memory optimization system for high-performance applications
    """
    
    def __init__(self, 
                 enable_profiling: bool = True,
                 memory_limit_mb: float = 2048,
                 gc_threshold_mb: float = 512):
        
        self.enable_profiling = enable_profiling
        self.memory_limit_mb = memory_limit_mb
        self.gc_threshold_mb = gc_threshold_mb
        
        # Memory tracking
        self.process = psutil.Process()
        self.memory_snapshots: List[MemoryMetrics] = []
        self.object_registry = weakref.WeakValueDictionary()
        self.large_object_cache = {}
        
        # Memory pools for frequently allocated objects
        self.memory_pools = {
            "query_results": ObjectPool(max_size=1000, object_size_limit=1024 * 1024),  # 1MB
            "ai_responses": ObjectPool(max_size=500, object_size_limit=5 * 1024 * 1024),  # 5MB
            "analytics_data": ObjectPool(max_size=200, object_size_limit=10 * 1024 * 1024),  # 10MB
        }
        
        # Leak detection
        self.allocation_tracker = defaultdict(list)
        self.leak_threshold = 100  # Number of allocations before checking for leaks
        
        # GC optimization
        self._optimize_gc_settings()
        
        # Start profiling if enabled
        if self.enable_profiling:
            tracemalloc.start(10)  # Keep 10 frames of traceback
            
        # Background monitoring
        self.monitoring_task = None
        self.monitoring_interval = 60  # seconds
        
        logger.info(f"Memory Optimizer initialized with {memory_limit_mb}MB limit")
    
    def _optimize_gc_settings(self):
        """Optimize garbage collection settings for performance"""
        # Get current thresholds
        gen0, gen1, gen2 = gc.get_threshold()
        
        # Increase thresholds for better performance (less frequent GC)
        gc.set_threshold(gen0 * 2, gen1 * 2, gen2 * 2)
        
        # Disable GC during critical operations (will be re-enabled after)
        gc.collect()  # Full collection before optimization
        
        logger.info(f"GC thresholds optimized: {gc.get_threshold()}")
    
    async def start_monitoring(self):
        """Start background memory monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_memory_loop())
            logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop background memory monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Memory monitoring stopped")
    
    async def _monitor_memory_loop(self):
        """Background memory monitoring loop"""
        while True:
            try:
                metrics = await self.collect_memory_metrics()
                self.memory_snapshots.append(metrics)
                
                # Keep only last 24 hours of snapshots
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.memory_snapshots = [
                    s for s in self.memory_snapshots 
                    if s.timestamp > cutoff_time
                ]
                
                # Check for memory issues
                if metrics.process_memory_mb > self.memory_limit_mb:
                    logger.warning(f"Memory usage ({metrics.process_memory_mb}MB) exceeds limit ({self.memory_limit_mb}MB)")
                    await self.optimize_memory()
                
                elif metrics.process_memory_mb > self.gc_threshold_mb:
                    # Trigger garbage collection
                    gc.collect()
                    logger.info(f"GC triggered at {metrics.process_memory_mb}MB")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def collect_memory_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics"""
        metrics = MemoryMetrics()
        
        # System memory
        memory = psutil.virtual_memory()
        metrics.total_memory_mb = memory.total / 1024 / 1024
        metrics.available_memory_mb = memory.available / 1024 / 1024
        
        # Process memory
        process_memory = self.process.memory_info()
        metrics.process_memory_mb = process_memory.rss / 1024 / 1024
        metrics.process_memory_percent = self.process.memory_percent()
        metrics.heap_size_mb = process_memory.vms / 1024 / 1024
        
        # Garbage collection stats
        metrics.gc_stats = {
            "collections": gc.get_count(),
            "collected": gc.collect(),
            "uncollectable": len(gc.garbage),
            "thresholds": gc.get_threshold(),
        }
        
        # Memory profiling if enabled
        if self.enable_profiling and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            metrics.largest_objects = [
                (str(stat), stat.size / 1024 / 1024)
                for stat in top_stats
            ]
            
            # Detect potential memory leaks
            metrics.memory_leaks = await self._detect_memory_leaks(snapshot)
        
        return metrics
    
    async def _detect_memory_leaks(self, snapshot) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        leaks = []
        
        # Group allocations by traceback
        allocation_groups = defaultdict(list)
        for stat in snapshot.statistics('traceback'):
            key = str(stat.traceback)
            allocation_groups[key].append(stat)
        
        # Find suspicious patterns
        for traceback_key, stats in allocation_groups.items():
            if len(stats) > self.leak_threshold:
                total_size = sum(stat.size for stat in stats)
                if total_size > 10 * 1024 * 1024:  # 10MB threshold
                    leaks.append({
                        "traceback": traceback_key[:200],  # Truncate for readability
                        "count": len(stats),
                        "total_size_mb": total_size / 1024 / 1024,
                    })
        
        return leaks
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Run comprehensive memory optimization"""
        logger.info("🚀 Starting memory optimization")
        
        optimization_results = {
            "start_time": datetime.utcnow(),
            "initial_memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "optimizations_applied": [],
        }
        
        # Step 1: Force garbage collection
        gc_collected = gc.collect()
        optimization_results["optimizations_applied"].append({
            "type": "garbage_collection",
            "objects_collected": gc_collected,
        })
        
        # Step 2: Clear large object cache
        cleared_objects = 0
        cleared_size = 0
        for key in list(self.large_object_cache.keys()):
            obj = self.large_object_cache.get(key)
            if obj and sys.getsizeof(obj) > 1024 * 1024:  # Clear objects > 1MB
                cleared_size += sys.getsizeof(obj)
                del self.large_object_cache[key]
                cleared_objects += 1
        
        optimization_results["optimizations_applied"].append({
            "type": "large_object_cache_clear",
            "objects_cleared": cleared_objects,
            "size_cleared_mb": cleared_size / 1024 / 1024,
        })
        
        # Step 3: Optimize memory pools
        pool_stats = {}
        for pool_name, pool in self.memory_pools.items():
            stats = pool.optimize()
            pool_stats[pool_name] = stats
        
        optimization_results["optimizations_applied"].append({
            "type": "memory_pool_optimization",
            "pool_stats": pool_stats,
        })
        
        # Step 4: Clear weak references
        self.object_registry = weakref.WeakValueDictionary()
        
        # Step 5: Trim process working set (platform-specific)
        try:
            if hasattr(self.process, "memory_maps"):
                # This is platform-specific and may not work everywhere
                self.process.memory_info()  # Refresh memory info
        except Exception as e:
            logger.debug(f"Could not trim working set: {e}")
        
        # Final memory state
        optimization_results["final_memory_mb"] = self.process.memory_info().rss / 1024 / 1024
        optimization_results["memory_freed_mb"] = (
            optimization_results["initial_memory_mb"] - 
            optimization_results["final_memory_mb"]
        )
        optimization_results["end_time"] = datetime.utcnow()
        
        logger.info(f"🎉 Memory optimization completed: {optimization_results['memory_freed_mb']:.2f}MB freed")
        
        return optimization_results
    
    def register_large_object(self, name: str, obj: Any):
        """Register a large object for tracking"""
        self.large_object_cache[name] = obj
        self.object_registry[name] = obj
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        current_metrics = MemoryMetrics()
        
        # Collect current metrics synchronously
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        current_metrics.total_memory_mb = memory.total / 1024 / 1024
        current_metrics.available_memory_mb = memory.available / 1024 / 1024
        current_metrics.process_memory_mb = process_memory.rss / 1024 / 1024
        current_metrics.process_memory_percent = self.process.memory_percent()
        
        # Memory trends
        trends = self._calculate_memory_trends()
        
        # Pool statistics
        pool_stats = {}
        for name, pool in self.memory_pools.items():
            pool_stats[name] = pool.get_stats()
        
        return {
            "current_metrics": {
                "process_memory_mb": current_metrics.process_memory_mb,
                "process_memory_percent": current_metrics.process_memory_percent,
                "available_system_memory_mb": current_metrics.available_memory_mb,
                "heap_size_mb": process_memory.vms / 1024 / 1024,
            },
            "memory_pools": pool_stats,
            "trends": trends,
            "large_objects": len(self.large_object_cache),
            "gc_stats": {
                "collections": gc.get_count(),
                "thresholds": gc.get_threshold(),
                "garbage_count": len(gc.garbage),
            },
            "recommendations": self._generate_memory_recommendations(),
        }
    
    def _calculate_memory_trends(self) -> Dict[str, float]:
        """Calculate memory usage trends"""
        if len(self.memory_snapshots) < 2:
            return {}
        
        # Calculate trends over last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_snapshots = [s for s in self.memory_snapshots if s.timestamp > one_hour_ago]
        
        if len(recent_snapshots) < 2:
            return {}
        
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds() / 3600
        memory_diff = last_snapshot.process_memory_mb - first_snapshot.process_memory_mb
        
        return {
            "memory_growth_mb_per_hour": memory_diff / time_diff if time_diff > 0 else 0,
            "average_memory_mb": sum(s.process_memory_mb for s in recent_snapshots) / len(recent_snapshots),
            "peak_memory_mb": max(s.process_memory_mb for s in recent_snapshots),
            "min_memory_mb": min(s.process_memory_mb for s in recent_snapshots),
        }
    
    def _generate_memory_recommendations(self) -> List[Dict[str, str]]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        if current_memory > self.memory_limit_mb * 0.9:
            recommendations.append({
                "priority": "high",
                "type": "memory_limit",
                "recommendation": "Process is near memory limit. Consider scaling horizontally or optimizing memory usage.",
            })
        
        if len(gc.garbage) > 100:
            recommendations.append({
                "priority": "medium",
                "type": "garbage_collection",
                "recommendation": f"Found {len(gc.garbage)} uncollectable objects. Review circular references.",
            })
        
        # Check pool efficiency
        for name, pool in self.memory_pools.items():
            stats = pool.get_stats()
            if stats["efficiency"] < 0.5:
                recommendations.append({
                    "priority": "low",
                    "type": "memory_pool",
                    "recommendation": f"Memory pool '{name}' has low efficiency ({stats['efficiency']:.2f}). Consider resizing.",
                })
        
        return recommendations


class ObjectPool:
    """Object pool for efficient memory management"""
    
    def __init__(self, max_size: int = 1000, object_size_limit: int = 1024 * 1024):
        self.max_size = max_size
        self.object_size_limit = object_size_limit
        self.pool: List[Any] = []
        self.in_use: Set[int] = set()
        self.stats = {
            "allocations": 0,
            "reuses": 0,
            "evictions": 0,
        }
        self.lock = threading.Lock()
    
    def acquire(self) -> Optional[Any]:
        """Acquire an object from the pool"""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.in_use.add(id(obj))
                self.stats["reuses"] += 1
                return obj
            else:
                self.stats["allocations"] += 1
                return None
    
    def release(self, obj: Any):
        """Release an object back to the pool"""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                
                # Check size limit
                try:
                    if sys.getsizeof(obj) <= self.object_size_limit and len(self.pool) < self.max_size:
                        self.pool.append(obj)
                    else:
                        self.stats["evictions"] += 1
                except:
                    # If we can't determine size, don't pool it
                    self.stats["evictions"] += 1
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize the pool by clearing old objects"""
        with self.lock:
            initial_size = len(self.pool)
            
            # Clear half of the pool if it's more than 50% full
            if len(self.pool) > self.max_size * 0.5:
                self.pool = self.pool[:self.max_size // 2]
            
            cleared = initial_size - len(self.pool)
            
            return {
                "cleared_objects": cleared,
                "remaining_objects": len(self.pool),
                "in_use_objects": len(self.in_use),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            total_operations = self.stats["allocations"] + self.stats["reuses"]
            efficiency = self.stats["reuses"] / total_operations if total_operations > 0 else 0
            
            return {
                "pool_size": len(self.pool),
                "in_use": len(self.in_use),
                "allocations": self.stats["allocations"],
                "reuses": self.stats["reuses"],
                "evictions": self.stats["evictions"],
                "efficiency": efficiency,
            }


# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()


# Decorator for memory-intensive operations
def memory_managed(pool_name: str = "query_results"):
    """
    Decorator for memory-intensive operations
    
    Usage:
        @memory_managed(pool_name="ai_responses")
        async def process_large_data():
            # Memory-intensive operation
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Monitor memory before
            initial_memory = memory_optimizer.process.memory_info().rss / 1024 / 1024
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Check if result should be pooled
                if pool_name in memory_optimizer.memory_pools:
                    pool = memory_optimizer.memory_pools[pool_name]
                    if result and sys.getsizeof(result) < pool.object_size_limit:
                        memory_optimizer.register_large_object(
                            f"{func.__name__}_{id(result)}", 
                            result
                        )
                
                return result
                
            finally:
                # Monitor memory after
                final_memory = memory_optimizer.process.memory_info().rss / 1024 / 1024
                memory_delta = final_memory - initial_memory
                
                if memory_delta > 100:  # More than 100MB increase
                    logger.warning(
                        f"{func.__name__} increased memory by {memory_delta:.2f}MB"
                    )
                    
                    # Trigger optimization if needed
                    if final_memory > memory_optimizer.gc_threshold_mb:
                        asyncio.create_task(memory_optimizer.optimize_memory())
        
        return wrapper
    return decorator


# Convenience functions
async def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    metrics = await memory_optimizer.collect_memory_metrics()
    return {
        "process_mb": metrics.process_memory_mb,
        "available_mb": metrics.available_memory_mb,
        "percent": metrics.process_memory_percent,
    }


async def optimize_memory_now():
    """Trigger immediate memory optimization"""
    return await memory_optimizer.optimize_memory()


def get_memory_health() -> Dict[str, Any]:
    """Get memory health report"""
    return memory_optimizer.get_memory_report()