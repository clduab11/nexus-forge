"""
Optimized Swarm Coordination Patterns
High-performance coordination with <50ms latency targets
"""

import asyncio
import time
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .swarm_intelligence import (
    SwarmAgent,
    SwarmCoordinator,
    SwarmMessage,
    SwarmObjective,
    SwarmTask,
    CommunicationType,
)
from ..protocols.agent2agent.fast_coordination import (
    OptimizedSwarmCommunication,
    FastHierarchicalCoordinator,
    ParallelCoordinator,
    FastMessageRouter,
    ZeroCopyMessagePool,
)
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class UltraFastSwarmCoordinator(SwarmCoordinator):
    """
    Ultra-fast swarm coordinator with <50ms coordination latency
    Optimized for Google ADK Hackathon performance requirements
    """
    
    def __init__(self, swarm):
        super().__init__(swarm)
        self.comm = OptimizedSwarmCommunication(f"swarm_{uuid4().hex[:8]}")
        self.parallel_coordinator = ParallelCoordinator()
        self.message_pool = ZeroCopyMessagePool()
        
        # Performance tracking
        self.latency_samples = deque(maxlen=1000)
        self.coordination_metrics = {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "throughput_msg_s": 0.0,
            "success_rate": 1.0
        }
        
        # Optimization flags
        self.enable_predictive_routing = True
        self.enable_task_prefetching = True
        self.enable_adaptive_batching = True
        
    async def execute(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute with ultra-low latency coordination"""
        start_time = time.perf_counter()
        
        try:
            # Start communication layer
            await self.comm.start()
            
            # Register all agents for O(1) lookups
            for agent in agents:
                self.comm.register_agent(agent.id, agent.capabilities)
                
            # Parallel task decomposition
            tasks = await self._parallel_decompose_objective(objective, agents)
            
            # Create dependency graph
            await self._build_parallel_task_graph(tasks)
            
            # Execute with maximum parallelism
            results = await self._execute_ultra_fast(agents, tasks, objective)
            
            # Calculate metrics
            execution_time = (time.perf_counter() - start_time) * 1000  # ms
            self.latency_samples.append(execution_time)
            self._update_metrics()
            
            return {
                "status": "completed",
                "results": results,
                "execution_time_ms": execution_time,
                "coordination_metrics": self.coordination_metrics.copy(),
                "parallelism_achieved": self._calculate_parallelism(tasks),
                "message_count": self.comm.router.metrics["messages_routed"]
            }
            
        except Exception as e:
            logger.error(f"Ultra-fast coordination failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000
            }
        finally:
            await self.comm.stop()
            
    async def _parallel_decompose_objective(
        self,
        objective: SwarmObjective,
        agents: List[SwarmAgent]
    ) -> List[SwarmTask]:
        """Decompose objective in parallel using multiple agents"""
        # Select top agents for decomposition
        decomposer_count = min(3, len(agents))
        decomposers = sorted(agents, key=lambda a: a.performance_score, reverse=True)[:decomposer_count]
        
        # Parallel decomposition
        decompose_tasks = []
        for agent in decomposers:
            task = asyncio.create_task(
                self._agent_decompose_fast(agent, objective)
            )
            decompose_tasks.append(task)
            
        # Wait for all decompositions
        proposals = await asyncio.gather(*decompose_tasks)
        
        # Merge proposals efficiently
        return self._merge_task_proposals(proposals)
        
    async def _agent_decompose_fast(
        self,
        agent: SwarmAgent,
        objective: SwarmObjective
    ) -> List[SwarmTask]:
        """Fast task decomposition by single agent"""
        # Pre-computed task templates based on objective type
        task_count = 5 + (objective.priority // 2)
        
        tasks = []
        for i in range(task_count):
            task = SwarmTask(
                description=f"{objective.strategy.value} task {i+1}",
                priority=objective.priority,
                required_capabilities=self._get_task_capabilities(objective, i),
                estimated_duration=None
            )
            
            # Add dependencies for complex objectives
            if i > 0 and objective.priority > 7:
                task.dependencies.append(tasks[i-1].id)
                
            tasks.append(task)
            
        return tasks
        
    def _get_task_capabilities(self, objective: SwarmObjective, index: int) -> List[str]:
        """Get required capabilities for task"""
        # Fast capability assignment based on strategy
        capability_map = {
            "research": ["web_search", "analysis"],
            "development": ["code_generation", "testing"],
            "analysis": ["data_processing", "visualization"],
            "optimization": ["performance_analysis", "tuning"],
        }
        
        base_caps = capability_map.get(objective.strategy.value, ["processing"])
        return base_caps[:1] if index % 2 == 0 else base_caps
        
    def _merge_task_proposals(self, proposals: List[List[SwarmTask]]) -> List[SwarmTask]:
        """Merge task proposals from multiple agents"""
        if not proposals:
            return []
            
        # Use first proposal as base
        merged = proposals[0]
        
        # Add unique tasks from other proposals
        seen_descriptions = {t.description for t in merged}
        
        for proposal in proposals[1:]:
            for task in proposal:
                if task.description not in seen_descriptions:
                    merged.append(task)
                    seen_descriptions.add(task.description)
                    
        return merged
        
    async def _build_parallel_task_graph(self, tasks: List[SwarmTask]):
        """Build task dependency graph in parallel coordinator"""
        for task in tasks:
            await self.parallel_coordinator.add_task(task.id, task.dependencies)
            self.swarm.tasks[task.id] = task
            
    async def _execute_ultra_fast(
        self,
        agents: List[SwarmAgent],
        tasks: List[SwarmTask],
        objective: SwarmObjective
    ) -> Dict[str, Any]:
        """Execute tasks with ultra-low latency"""
        results = {}
        completed_count = 0
        
        # Create agent executors
        executors = []
        for agent in agents:
            executor = asyncio.create_task(
                self._agent_executor_loop(agent, objective)
            )
            executors.append(executor)
            
        # Monitor completion
        monitor = asyncio.create_task(
            self._monitor_execution_fast(len(tasks))
        )
        
        # Wait for all tasks to complete or timeout
        try:
            await asyncio.wait_for(monitor, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Execution timeout reached")
            
        # Cancel executors
        for executor in executors:
            executor.cancel()
            
        # Collect results
        for task_id, task in self.swarm.tasks.items():
            if task.status == "completed" and task.result:
                results[task_id] = task.result
                completed_count += 1
                
        return results
        
    async def _agent_executor_loop(self, agent: SwarmAgent, objective: SwarmObjective):
        """Agent execution loop with minimal overhead"""
        while True:
            try:
                # Get batch of ready tasks
                ready_tasks = await self.parallel_coordinator.get_ready_tasks(3)
                
                if not ready_tasks:
                    # No tasks ready, brief pause
                    await asyncio.sleep(0.001)  # 1ms
                    continue
                    
                # Execute tasks in parallel
                execute_tasks = []
                for task_id in ready_tasks:
                    if task_id in self.swarm.tasks:
                        task = self.swarm.tasks[task_id]
                        
                        # Check capability match
                        if self._can_execute_task(agent, task):
                            task.agent_id = agent.id
                            task.status = "running"
                            
                            execute_task = asyncio.create_task(
                                self._execute_task_fast(agent, task)
                            )
                            execute_tasks.append(execute_task)
                            
                # Wait for executions
                if execute_tasks:
                    await asyncio.gather(*execute_tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent {agent.id} executor error: {e}")
                await asyncio.sleep(0.01)  # 10ms pause on error
                
    def _can_execute_task(self, agent: SwarmAgent, task: SwarmTask) -> bool:
        """Check if agent can execute task"""
        if not task.required_capabilities:
            return True
            
        agent_caps = set(agent.capabilities)
        required_caps = set(task.required_capabilities)
        
        return required_caps.issubset(agent_caps)
        
    async def _execute_task_fast(self, agent: SwarmAgent, task: SwarmTask):
        """Execute single task with minimal latency"""
        start_time = time.perf_counter()
        
        try:
            task.started_at = datetime.now(timezone.utc)
            
            # Simulate fast task execution
            # In production, this would call actual task logic
            await asyncio.sleep(0.01)  # 10ms simulated work
            
            # Mark complete
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            task.result = {
                "output": f"Fast result for {task.description}",
                "agent": agent.id,
                "duration_ms": (time.perf_counter() - start_time) * 1000
            }
            
            # Update agent state
            agent.completed_tasks.append(task.id)
            agent.load = max(0, agent.load - 0.1)
            
            # Notify coordinator
            await self.parallel_coordinator.complete_task(task.id, task.result)
            
            # Send completion message (batched)
            msg = self.message_pool.acquire()
            msg.id = str(uuid4())
            msg.type = CommunicationType.BROADCAST
            msg.sender = agent.id
            msg.payload = {
                "type": "task_completed",
                "task_id": task.id,
                "duration_ms": task.result["duration_ms"]
            }
            msg.timestamp = time.time()
            
            await self.comm.broadcast_message(msg)
            self.message_pool.release(msg)
            
        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")
            task.status = "failed"
            await self.parallel_coordinator.complete_task(task.id, None)
            
    async def _monitor_execution_fast(self, total_tasks: int):
        """Fast execution monitoring"""
        completed = 0
        last_check = time.time()
        
        while completed < total_tasks:
            # Count completed tasks
            completed = sum(
                1 for task in self.swarm.tasks.values()
                if task.status == "completed"
            )
            
            # Brief pause
            await asyncio.sleep(0.005)  # 5ms check interval
            
            # Timeout check
            if time.time() - last_check > 30:
                logger.warning("Execution monitor timeout")
                break
                
    def _calculate_parallelism(self, tasks: List[SwarmTask]) -> float:
        """Calculate achieved parallelism"""
        if not tasks:
            return 0.0
            
        # Group tasks by execution time
        time_slots = defaultdict(int)
        
        for task in tasks:
            if task.started_at and task.completed_at:
                slot = int(task.started_at.timestamp())
                time_slots[slot] += 1
                
        # Max concurrent tasks
        max_concurrent = max(time_slots.values()) if time_slots else 1
        
        return max_concurrent / len(tasks)
        
    def _update_metrics(self):
        """Update coordination metrics"""
        if not self.latency_samples:
            return
            
        samples = list(self.latency_samples)
        samples.sort()
        
        self.coordination_metrics["avg_latency_ms"] = np.mean(samples)
        self.coordination_metrics["p50_latency_ms"] = np.percentile(samples, 50)
        self.coordination_metrics["p99_latency_ms"] = np.percentile(samples, 99)
        
        # Calculate throughput
        if len(samples) > 1:
            duration_s = (samples[-1] - samples[0]) / 1000.0
            if duration_s > 0:
                msg_count = self.comm.router.metrics["messages_routed"]
                self.coordination_metrics["throughput_msg_s"] = msg_count / duration_s


class LockFreeMessageQueue:
    """Lock-free message queue for ultra-low latency"""
    
    def __init__(self, size: int = 10000):
        self.buffer = [None] * size
        self.size = size
        self.head = 0
        self.tail = 0
        
    def enqueue(self, message: Any) -> bool:
        """Add message to queue (lock-free)"""
        next_tail = (self.tail + 1) % self.size
        
        if next_tail == self.head:
            return False  # Queue full
            
        self.buffer[self.tail] = message
        self.tail = next_tail
        return True
        
    def dequeue(self) -> Optional[Any]:
        """Remove message from queue (lock-free)"""
        if self.head == self.tail:
            return None  # Queue empty
            
        message = self.buffer[self.head]
        self.head = (self.head + 1) % self.size
        return message
        
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.head == self.tail
        
    def is_full(self) -> bool:
        """Check if queue is full"""
        return (self.tail + 1) % self.size == self.head


class AdaptiveBatchProcessor:
    """Adaptive batch processor for optimal throughput"""
    
    def __init__(self):
        self.batch_sizes = deque([10, 10, 10], maxlen=10)
        self.processing_times = deque([0.01, 0.01, 0.01], maxlen=10)
        self.optimal_batch_size = 10
        
    def update_metrics(self, batch_size: int, processing_time: float):
        """Update batch processing metrics"""
        self.batch_sizes.append(batch_size)
        self.processing_times.append(processing_time)
        
        # Calculate optimal batch size
        throughputs = [
            size / time 
            for size, time in zip(self.batch_sizes, self.processing_times)
        ]
        
        # Find batch size with best throughput
        best_idx = throughputs.index(max(throughputs))
        self.optimal_batch_size = self.batch_sizes[best_idx]
        
        # Constrain to reasonable range
        self.optimal_batch_size = max(5, min(50, self.optimal_batch_size))
        
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.optimal_batch_size


class PredictiveTaskScheduler:
    """Predictive task scheduler using ML-like heuristics"""
    
    def __init__(self):
        self.task_history = deque(maxlen=1000)
        self.agent_performance = defaultdict(lambda: {"success": 0, "total": 0})
        
    def predict_task_duration(self, task: SwarmTask, agent: SwarmAgent) -> float:
        """Predict task duration based on history"""
        # Simple heuristic based on task priority and agent performance
        base_duration = 0.01 * (11 - task.priority)  # 10-100ms based on priority
        
        # Adjust based on agent performance
        agent_stats = self.agent_performance[agent.id]
        if agent_stats["total"] > 0:
            success_rate = agent_stats["success"] / agent_stats["total"]
            base_duration *= (2 - success_rate)  # Faster for better agents
            
        return base_duration
        
    def update_history(self, task_id: str, agent_id: str, duration: float, success: bool):
        """Update task execution history"""
        self.task_history.append({
            "task_id": task_id,
            "agent_id": agent_id,
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })
        
        # Update agent performance
        self.agent_performance[agent_id]["total"] += 1
        if success:
            self.agent_performance[agent_id]["success"] += 1
            
    def get_best_agent_for_task(self, task: SwarmTask, available_agents: List[SwarmAgent]) -> Optional[SwarmAgent]:
        """Get predicted best agent for task"""
        if not available_agents:
            return None
            
        # Score each agent
        scores = []
        for agent in available_agents:
            # Capability match
            cap_score = len(set(task.required_capabilities) & set(agent.capabilities))
            
            # Performance score
            agent_stats = self.agent_performance[agent.id]
            perf_score = 0.5
            if agent_stats["total"] > 0:
                perf_score = agent_stats["success"] / agent_stats["total"]
                
            # Load score (inverse of load)
            load_score = 1.0 - agent.load
            
            # Combined score
            total_score = cap_score * 0.5 + perf_score * 0.3 + load_score * 0.2
            scores.append((agent, total_score))
            
        # Return best agent
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]