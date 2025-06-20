"""
Optimized Swarm Execution Engine
High-performance execution engine with advanced parallelization and resource optimization
"""

import asyncio
import aiohttp
import json
import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np
import psutil
from asyncio_throttle import Throttler

from nexus_forge.core.cache import CacheStrategy, RedisCache
from nexus_forge.core.exceptions import (
    AgentError,
    CoordinationError,
    ExecutionError,
    ResourceError,
)
from nexus_forge.core.monitoring import get_logger

from .swarm_intelligence import (
    SwarmAgent,
    SwarmMessage,
    SwarmObjective,
    SwarmTask,
    SwarmResult,
    CommunicationType,
)

logger = get_logger(__name__)


# Performance optimization constants
MAX_CONCURRENT_TASKS = 100
MAX_WORKERS_PER_AGENT = 10
CONNECTION_POOL_SIZE = 50
BATCH_SIZE = 25
CACHE_TTL = 300  # 5 minutes
THROTTLE_RATE = 100  # requests per second


class ConnectionPool:
    """HTTP connection pool for API calls"""
    
    def __init__(self, size: int = CONNECTION_POOL_SIZE):
        self.size = size
        self._connector = None
        self._session = None
        
    async def __aenter__(self):
        self._connector = aiohttp.TCPConnector(
            limit=self.size,
            limit_per_host=self.size // 4,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={'User-Agent': 'NexusForge/1.0'}
        )
        return self._session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()


class TaskQueue:
    """Priority queue with batch processing support"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues = defaultdict(asyncio.PriorityQueue)  # priority -> queue
        self._batch_buffer = defaultdict(list)
        self._size = 0
        self._lock = asyncio.Lock()
        
    async def put(self, task: SwarmTask, priority: Optional[int] = None):
        """Add task to queue"""
        async with self._lock:
            if self._size >= self.max_size:
                raise ResourceError("Task queue is full")
            
            priority = priority or task.priority
            await self._queues[priority].put((time.time(), task))
            self._size += 1
    
    async def get_batch(self, batch_size: int = BATCH_SIZE) -> List[SwarmTask]:
        """Get batch of tasks with same priority"""
        async with self._lock:
            batch = []
            
            # Find highest priority queue with tasks
            for priority in sorted(self._queues.keys(), reverse=True):
                queue = self._queues[priority]
                
                while not queue.empty() and len(batch) < batch_size:
                    try:
                        _, task = await asyncio.wait_for(queue.get(), timeout=0.1)
                        batch.append(task)
                        self._size -= 1
                    except asyncio.TimeoutError:
                        break
                
                if len(batch) >= batch_size:
                    break
            
            return batch
    
    async def get(self) -> Optional[SwarmTask]:
        """Get single highest priority task"""
        batch = await self.get_batch(1)
        return batch[0] if batch else None
    
    def size(self) -> int:
        """Get current queue size"""
        return self._size


class OptimizedAgentExecutor:
    """Optimized task executor for individual agents"""
    
    def __init__(self, agent: SwarmAgent, max_workers: int = MAX_WORKERS_PER_AGENT):
        self.agent = agent
        self.max_workers = max_workers
        self.task_queue = TaskQueue()
        self.active_tasks: Set[str] = set()
        self.completed_count = 0
        self.failed_count = 0
        self.metrics = defaultdict(float)
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
        
        # Performance tracking
        self.execution_times = deque(maxlen=100)
        self.last_optimization = time.time()
        
    async def execute_tasks(self, tasks: List[SwarmTask], context: Any) -> Dict[str, Any]:
        """Execute multiple tasks efficiently"""
        results = {}
        
        # Add tasks to queue
        for task in tasks:
            await self.task_queue.put(task)
        
        # Create worker coroutines
        workers = [
            self._worker(context, results, worker_id)
            for worker_id in range(self.max_workers)
        ]
        
        # Run workers
        await asyncio.gather(*workers, return_exceptions=True)
        
        return results
    
    async def _worker(self, context: Any, results: Dict[str, Any], worker_id: int):
        """Worker coroutine for task execution"""
        throttler = Throttler(rate_limit=THROTTLE_RATE)
        
        while True:
            # Get batch of tasks
            batch = await self.task_queue.get_batch()
            if not batch:
                break
            
            # Execute batch
            async with throttler:
                batch_results = await self._execute_batch(batch, context, worker_id)
                
                # Store results
                for task, result in zip(batch, batch_results):
                    results[task.id] = result
                    
                    if result.get("success"):
                        self.completed_count += 1
                    else:
                        self.failed_count += 1
    
    async def _execute_batch(
        self, 
        batch: List[SwarmTask], 
        context: Any,
        worker_id: int
    ) -> List[Dict[str, Any]]:
        """Execute a batch of tasks"""
        start_time = time.time()
        
        # Group by task type for optimization
        task_groups = defaultdict(list)
        for task in batch:
            task_groups[task.type].append(task)
        
        results = []
        
        for task_type, group_tasks in task_groups.items():
            if task_type == "compute":
                # CPU-bound tasks go to process pool
                group_results = await self._execute_compute_batch(group_tasks, context)
            elif task_type == "io":
                # I/O-bound tasks use async
                group_results = await self._execute_io_batch(group_tasks, context)
            else:
                # Default async execution
                group_results = await self._execute_async_batch(group_tasks, context)
            
            results.extend(group_results)
        
        # Track metrics
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        self.metrics[f"worker_{worker_id}_time"] += execution_time
        self.metrics[f"worker_{worker_id}_tasks"] += len(batch)
        
        return results
    
    async def _execute_compute_batch(
        self, 
        tasks: List[SwarmTask], 
        context: Any
    ) -> List[Dict[str, Any]]:
        """Execute CPU-bound tasks in process pool"""
        loop = asyncio.get_event_loop()
        
        # Prepare tasks for multiprocessing
        task_data = [
            (task.id, task.data, task.parameters)
            for task in tasks
        ]
        
        # Execute in process pool
        func = partial(self._compute_task_worker, context=context)
        futures = [
            loop.run_in_executor(self.process_pool, func, data)
            for data in task_data
        ]
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Format results
        formatted_results = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "success": False,
                    "error": str(result),
                    "task_id": task.id
                })
            else:
                formatted_results.append({
                    "success": True,
                    "result": result,
                    "task_id": task.id
                })
        
        return formatted_results
    
    async def _execute_io_batch(
        self, 
        tasks: List[SwarmTask], 
        context: Any
    ) -> List[Dict[str, Any]]:
        """Execute I/O-bound tasks with connection pooling"""
        async with ConnectionPool() as session:
            # Execute all I/O tasks concurrently
            coroutines = [
                self._execute_io_task(task, session, context)
                for task in tasks
            ]
            
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Format results
            formatted_results = []
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    formatted_results.append({
                        "success": False,
                        "error": str(result),
                        "task_id": task.id
                    })
                else:
                    formatted_results.append({
                        "success": True,
                        "result": result,
                        "task_id": task.id
                    })
            
            return formatted_results
    
    async def _execute_async_batch(
        self, 
        tasks: List[SwarmTask], 
        context: Any
    ) -> List[Dict[str, Any]]:
        """Execute general async tasks"""
        coroutines = [
            self._execute_async_task(task, context)
            for task in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Format results
        formatted_results = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "success": False,
                    "error": str(result),
                    "task_id": task.id
                })
            else:
                formatted_results.append({
                    "success": True,
                    "result": result,
                    "task_id": task.id
                })
        
        return formatted_results
    
    @staticmethod
    def _compute_task_worker(task_data: Tuple, context: Any) -> Any:
        """Worker function for compute tasks (runs in separate process)"""
        task_id, data, parameters = task_data
        
        # Simulate compute-intensive work
        result = {
            "task_id": task_id,
            "computation": sum(range(parameters.get("iterations", 1000000))),
            "data_size": len(str(data))
        }
        
        return result
    
    async def _execute_io_task(
        self, 
        task: SwarmTask, 
        session: aiohttp.ClientSession,
        context: Any
    ) -> Any:
        """Execute I/O task with connection reuse"""
        # Example: API call
        if task.parameters.get("api_endpoint"):
            async with session.get(task.parameters["api_endpoint"]) as response:
                return await response.json()
        
        # Example: Database query (would use actual DB connection pool)
        return {
            "task_id": task.id,
            "query_result": f"Simulated DB result for {task.id}"
        }
    
    async def _execute_async_task(self, task: SwarmTask, context: Any) -> Any:
        """Execute general async task"""
        # Real task execution based on task type
        result = {
            "task_id": task.id,
            "agent": self.agent.id,
            "output": f"Processed {task.type} task with data: {task.data}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Simulate some async work
        await asyncio.sleep(0.01)  # Minimal delay for demonstration
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics"""
        avg_execution_time = (
            np.mean(self.execution_times) if self.execution_times else 0
        )
        
        return {
            "completed_tasks": self.completed_count,
            "failed_tasks": self.failed_count,
            "success_rate": self.completed_count / (self.completed_count + self.failed_count)
            if (self.completed_count + self.failed_count) > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "queue_size": self.task_queue.size(),
            "active_tasks": len(self.active_tasks),
            "worker_metrics": dict(self.metrics)
        }
    
    async def shutdown(self):
        """Gracefully shutdown executor"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class OptimizedSwarmExecutionEngine:
    """High-performance swarm execution engine"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine_id = f"opt_engine_{uuid4().hex[:8]}"
        
        # Cache for results
        self.cache = RedisCache()
        
        # Agent executors
        self.agent_executors: Dict[str, OptimizedAgentExecutor] = {}
        
        # Global task queue for load balancing
        self.global_queue = TaskQueue(max_size=10000)
        
        # Metrics
        self.start_time = time.time()
        self.total_tasks_executed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        logger.info(f"Optimized Swarm Execution Engine initialized: {self.engine_id}")
    
    async def execute_swarm(
        self,
        context: Any,
        monitoring_callback: Optional[Callable] = None
    ) -> SwarmResult:
        """Execute swarm with optimized performance"""
        execution_id = f"opt_exec_{uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Initialize agent executors
            for agent in context.agents:
                if agent.id not in self.agent_executors:
                    # Adjust workers based on available resources
                    max_workers = self._calculate_optimal_workers(agent)
                    self.agent_executors[agent.id] = OptimizedAgentExecutor(
                        agent, max_workers
                    )
            
            # Distribute tasks to agents with load balancing
            task_distribution = await self._distribute_tasks_optimized(
                context.tasks,
                context.agents
            )
            
            # Execute tasks in parallel across all agents
            agent_results = await self._execute_distributed_tasks(
                task_distribution,
                context,
                monitoring_callback
            )
            
            # Compile results
            result = self._compile_results(
                agent_results,
                context,
                execution_id,
                time.time() - start_time
            )
            
            # Update metrics
            self.total_tasks_executed += len(context.tasks)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized execution {execution_id} failed: {e}")
            
            return SwarmResult(
                objective_id=context.objective.id if context.objective else "",
                status="failed",
                results={"error": str(e)},
                execution_time=time.time() - start_time
            )
    
    def _calculate_optimal_workers(self, agent: SwarmAgent) -> int:
        """Calculate optimal number of workers based on resources"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base workers on CPU and memory
        base_workers = min(cpu_count * 2, int(memory_gb * 2))
        
        # Adjust based on agent capabilities
        if "high_performance" in agent.capabilities:
            return min(base_workers * 2, MAX_WORKERS_PER_AGENT)
        elif "low_latency" in agent.capabilities:
            return min(base_workers, MAX_WORKERS_PER_AGENT // 2)
        else:
            return min(base_workers, MAX_WORKERS_PER_AGENT)
    
    async def _distribute_tasks_optimized(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent]
    ) -> Dict[str, List[SwarmTask]]:
        """Distribute tasks with optimization algorithms"""
        distribution = defaultdict(list)
        
        # Cache lookup for previously executed tasks
        cached_tasks = []
        uncached_tasks = []
        
        for task in tasks:
            cache_key = f"task_result:{task.id}:{hash(str(task.data))}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                cached_tasks.append((task, cached_result))
                self.cache_hits += 1
            else:
                uncached_tasks.append(task)
                self.cache_misses += 1
        
        # Group tasks by affinity for better cache locality
        task_groups = self._group_tasks_by_affinity(uncached_tasks)
        
        # Distribute groups to agents
        agent_loads = {agent.id: 0 for agent in agents}
        
        for group_key, group_tasks in task_groups.items():
            # Find best agent for this group
            best_agent = min(
                agents,
                key=lambda a: agent_loads[a.id] / (a.performance_score + 0.1)
            )
            
            distribution[best_agent.id].extend(group_tasks)
            agent_loads[best_agent.id] += len(group_tasks)
        
        return dict(distribution)
    
    def _group_tasks_by_affinity(
        self, 
        tasks: List[SwarmTask]
    ) -> Dict[str, List[SwarmTask]]:
        """Group tasks that should be executed together"""
        groups = defaultdict(list)
        
        for task in tasks:
            # Group by task type and data similarity
            group_key = f"{task.type}:{task.data.get('category', 'default')}"
            groups[group_key].append(task)
        
        return dict(groups)
    
    async def _execute_distributed_tasks(
        self,
        distribution: Dict[str, List[SwarmTask]],
        context: Any,
        monitoring_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute distributed tasks across agents"""
        # Create execution coroutines
        execution_coroutines = []
        
        for agent_id, tasks in distribution.items():
            if agent_id in self.agent_executors:
                executor = self.agent_executors[agent_id]
                coro = executor.execute_tasks(tasks, context)
                execution_coroutines.append(coro)
        
        # Monitor execution if callback provided
        if monitoring_callback:
            monitor_task = asyncio.create_task(
                self._monitor_execution(distribution, monitoring_callback)
            )
        
        # Execute all agent tasks in parallel
        agent_results = await asyncio.gather(
            *execution_coroutines,
            return_exceptions=True
        )
        
        # Cancel monitoring
        if monitoring_callback:
            monitor_task.cancel()
        
        # Combine results
        combined_results = {}
        for results in agent_results:
            if isinstance(results, dict):
                combined_results.update(results)
        
        # Cache successful results
        for task_id, result in combined_results.items():
            if result.get("success"):
                cache_key = f"task_result:{task_id}:{hash(str(result))}"
                await self.cache.set(
                    cache_key,
                    result,
                    timeout=CACHE_TTL,
                    strategy=CacheStrategy.STANDARD
                )
        
        return combined_results
    
    async def _monitor_execution(
        self,
        distribution: Dict[str, List[SwarmTask]],
        callback: Callable
    ):
        """Monitor execution progress"""
        while True:
            try:
                # Collect metrics from all executors
                metrics = {}
                
                for agent_id, executor in self.agent_executors.items():
                    metrics[agent_id] = executor.get_performance_metrics()
                
                # Add system metrics
                metrics["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0 else 0,
                    "total_tasks": self.total_tasks_executed,
                    "uptime": time.time() - self.start_time
                }
                
                await callback(metrics)
                await asyncio.sleep(1)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _compile_results(
        self,
        agent_results: Dict[str, Any],
        context: Any,
        execution_id: str,
        execution_time: float
    ) -> SwarmResult:
        """Compile final results with performance metrics"""
        completed_tasks = []
        failed_tasks = []
        
        for task_id, result in agent_results.items():
            if result.get("success"):
                completed_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)
        
        # Calculate performance metrics
        total_tasks = len(completed_tasks) + len(failed_tasks)
        success_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
        throughput = total_tasks / execution_time if execution_time > 0 else 0
        
        # Get agent utilization
        agent_utilization = {}
        for agent_id, executor in self.agent_executors.items():
            metrics = executor.get_performance_metrics()
            agent_utilization[agent_id] = metrics["success_rate"]
        
        return SwarmResult(
            objective_id=context.objective.id if context.objective else "",
            status="completed" if len(failed_tasks) == 0 else "partial",
            results=agent_results,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            execution_time=execution_time,
            agent_utilization=agent_utilization,
            performance_metrics={
                "throughput": throughput,
                "success_rate": success_rate,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0,
                "avg_task_time": execution_time / total_tasks if total_tasks > 0 else 0
            },
            confidence=success_rate
        )
    
    async def shutdown(self):
        """Gracefully shutdown engine"""
        logger.info(f"Shutting down optimized execution engine {self.engine_id}")
        
        # Shutdown all agent executors
        shutdown_tasks = [
            executor.shutdown()
            for executor in self.agent_executors.values()
        ]
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Save final metrics
        logger.info(f"Final metrics: Tasks executed: {self.total_tasks_executed}, "
                   f"Cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0:.2f}")


class ResourceMonitor:
    """Monitor system resources for optimization"""
    
    def __init__(self):
        self.history = deque(maxlen=60)  # 1 minute of history
        self.alerts = []
    
    async def monitor(self):
        """Continuous resource monitoring"""
        while True:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters(),
                    "network_io": psutil.net_io_counters()
                }
                
                self.history.append(metrics)
                
                # Check for resource alerts
                if metrics["cpu_percent"] > 90:
                    self.alerts.append({
                        "type": "high_cpu",
                        "value": metrics["cpu_percent"],
                        "timestamp": metrics["timestamp"]
                    })
                
                if metrics["memory_percent"] > 85:
                    self.alerts.append({
                        "type": "high_memory",
                        "value": metrics["memory_percent"],
                        "timestamp": metrics["timestamp"]
                    })
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on monitoring"""
        recommendations = []
        
        if self.history:
            avg_cpu = np.mean([m["cpu_percent"] for m in self.history])
            avg_memory = np.mean([m["memory_percent"] for m in self.history])
            
            if avg_cpu > 80:
                recommendations.append("Consider scaling horizontally - high CPU usage")
            
            if avg_memory > 75:
                recommendations.append("Consider increasing memory or optimizing memory usage")
            
            if len([a for a in self.alerts if a["type"] == "high_cpu"]) > 10:
                recommendations.append("Frequent CPU spikes detected - review task distribution")
        
        return recommendations