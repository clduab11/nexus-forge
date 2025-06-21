"""
Swarm Execution Engine
Core engine for executing swarm tasks with dynamic load balancing and fault tolerance
"""

import asyncio
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np
import networkx as nx

from nexus_forge.core.cache import CacheStrategy, RedisCache
from nexus_forge.core.monitoring import get_logger

from .swarm_intelligence import (
import random
    SwarmAgent,
    SwarmObjective,
    SwarmTask,
    SwarmResult,
)

logger = get_logger(__name__)


# Core Execution Classes
@dataclass
class ExecutionContext:
    """Context for task execution"""
    id: str = field(default_factory=lambda: f"ctx_{uuid4().hex[:8]}")
    objective: Optional[SwarmObjective] = None
    agents: List[SwarmAgent] = field(default_factory=list)
    tasks: List[SwarmTask] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan for swarm tasks"""
    id: str = field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    context: Optional[ExecutionContext] = None
    task_graph: Optional[nx.DiGraph] = None
    agent_assignments: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> task_ids
    priority_queue: deque = field(default_factory=deque)
    estimated_duration: Optional[timedelta] = None
    parallelism_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadMetrics:
    """Load metrics for an agent"""
    agent_id: str = ""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    queue_length: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskScheduler:
    """Advanced task scheduler with multiple scheduling algorithms"""
    
    def __init__(self):
        self.scheduling_algorithms = {
            "round_robin": self._round_robin_schedule,
            "least_loaded": self._least_loaded_schedule,
            "capability_match": self._capability_match_schedule,
            "priority_based": self._priority_based_schedule,
            "predictive": self._predictive_schedule,
        }
        self.current_algorithm = "capability_match"
        self.schedule_history = deque(maxlen=100)
        
    async def create_execution_plan(
        self,
        context: ExecutionContext,
        algorithm: Optional[str] = None
    ) -> ExecutionPlan:
        """Create execution plan for tasks"""
        if algorithm:
            self.current_algorithm = algorithm
            
        plan = ExecutionPlan(context=context)
        
        # Build task dependency graph
        plan.task_graph = self._build_task_graph(context.tasks)
        
        # Determine parallelism level
        plan.parallelism_level = self._calculate_parallelism_level(
            plan.task_graph,
            len(context.agents)
        )
        
        # Schedule tasks to agents
        scheduling_func = self.scheduling_algorithms.get(
            self.current_algorithm,
            self._capability_match_schedule
        )
        
        plan.agent_assignments = await scheduling_func(
            context.tasks,
            context.agents,
            plan.task_graph
        )
        
        # Build priority queue
        plan.priority_queue = self._build_priority_queue(
            context.tasks,
            plan.task_graph
        )
        
        # Estimate duration
        plan.estimated_duration = await self._estimate_execution_duration(
            plan,
            context.agents
        )
        
        # Record scheduling decision
        self.schedule_history.append({
            "timestamp": datetime.now(timezone.utc),
            "algorithm": self.current_algorithm,
            "task_count": len(context.tasks),
            "agent_count": len(context.agents),
            "parallelism": plan.parallelism_level
        })
        
        return plan
    
    def _build_task_graph(self, tasks: List[SwarmTask]) -> nx.DiGraph:
        """Build directed graph of task dependencies"""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in tasks:
            graph.add_node(
                task.id,
                task=task,
                priority=task.priority,
                status=task.status
            )
        
        # Add dependency edges
        for task in tasks:
            for dep_id in task.dependencies:
                if graph.has_node(dep_id):
                    graph.add_edge(dep_id, task.id)
        
        return graph
    
    def _calculate_parallelism_level(
        self,
        task_graph: nx.DiGraph,
        agent_count: int
    ) -> int:
        """Calculate optimal parallelism level"""
        if not task_graph:
            return 1
            
        # Find maximum width of the graph (max parallel tasks)
        levels = {}
        for node in nx.topological_sort(task_graph):
            predecessors = list(task_graph.predecessors(node))
            if not predecessors:
                levels[node] = 0
            else:
                levels[node] = max(levels[pred] for pred in predecessors) + 1
        
        # Count tasks at each level
        level_counts = defaultdict(int)
        for node, level in levels.items():
            level_counts[level] += 1
        
        max_width = max(level_counts.values()) if level_counts else 1
        
        # Optimal parallelism is min of max width and agent count
        return min(max_width, agent_count)
    
    async def _round_robin_schedule(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent],
        task_graph: nx.DiGraph
    ) -> Dict[str, List[str]]:
        """Simple round-robin scheduling"""
        assignments = defaultdict(list)
        
        agent_index = 0
        for task in tasks:
            agent = agents[agent_index]
            assignments[agent.id].append(task.id)
            agent_index = (agent_index + 1) % len(agents)
        
        return dict(assignments)
    
    async def _least_loaded_schedule(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent],
        task_graph: nx.DiGraph
    ) -> Dict[str, List[str]]:
        """Schedule tasks to least loaded agents"""
        assignments = defaultdict(list)
        agent_loads = {agent.id: agent.load for agent in agents}
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            # Find least loaded agent
            least_loaded_agent = min(agent_loads.keys(), key=agent_loads.get)
            
            assignments[least_loaded_agent].append(task.id)
            
            # Update load estimate
            agent_loads[least_loaded_agent] += 0.1  # Simple load increment
        
        return dict(assignments)
    
    async def _capability_match_schedule(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent],
        task_graph: nx.DiGraph
    ) -> Dict[str, List[str]]:
        """Schedule based on capability matching"""
        assignments = defaultdict(list)
        agent_loads = {agent.id: agent.load for agent in agents}
        
        # Create agent capability index
        agent_capabilities = {
            agent.id: set(agent.capabilities)
            for agent in agents
        }
        
        # Sort tasks by priority and capability requirements
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.priority, len(t.required_capabilities)),
            reverse=True
        )
        
        for task in sorted_tasks:
            best_agent = None
            best_score = -1
            
            for agent in agents:
                # Calculate capability match score
                required = set(task.required_capabilities)
                available = agent_capabilities[agent.id]
                
                if not required:
                    # No specific requirements, use load balancing
                    score = 1.0 / (agent_loads[agent.id] + 1)
                else:
                    # Capability match + load consideration
                    match_ratio = len(required & available) / len(required)
                    load_factor = 1.0 / (agent_loads[agent.id] + 1)
                    score = match_ratio * 0.7 + load_factor * 0.3
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                assignments[best_agent.id].append(task.id)
                agent_loads[best_agent.id] += 0.1
        
        return dict(assignments)
    
    async def _priority_based_schedule(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent],
        task_graph: nx.DiGraph
    ) -> Dict[str, List[str]]:
        """Schedule high-priority tasks to best agents"""
        assignments = defaultdict(list)
        
        # Sort agents by performance score
        sorted_agents = sorted(agents, key=lambda a: a.performance_score, reverse=True)
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Assign high-priority tasks to best agents
        for i, task in enumerate(sorted_tasks):
            agent = sorted_agents[i % len(sorted_agents)]
            assignments[agent.id].append(task.id)
        
        return dict(assignments)
    
    async def _predictive_schedule(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent],
        task_graph: nx.DiGraph
    ) -> Dict[str, List[str]]:
        """Predictive scheduling based on historical performance"""
        # For now, fall back to capability match
        # In production, this would use ML models to predict execution times
        return await self._capability_match_schedule(tasks, agents, task_graph)
    
    def _build_priority_queue(
        self,
        tasks: List[SwarmTask],
        task_graph: nx.DiGraph
    ) -> deque:
        """Build priority queue respecting dependencies"""
        # Topological sort for dependency order
        try:
            topo_order = list(nx.topological_sort(task_graph))
        except nx.NetworkXError:
            # Graph has cycles, use simple priority sort
            topo_order = [t.id for t in sorted(tasks, key=lambda t: t.priority, reverse=True)]
        
        # Create queue with tasks that have no dependencies first
        queue = deque()
        
        for task_id in topo_order:
            task = next((t for t in tasks if t.id == task_id), None)
            if task:
                queue.append(task)
        
        return queue
    
    async def _estimate_execution_duration(
        self,
        plan: ExecutionPlan,
        agents: List[SwarmAgent]
    ) -> timedelta:
        """Estimate total execution duration"""
        if not plan.task_graph or not agents:
            return timedelta(minutes=5)  # Default estimate
        
        # Calculate critical path length
        critical_path_length = 0
        
        try:
            # Find longest path in DAG
            for path in nx.all_simple_paths(plan.task_graph, 
                                           source=None,  # Will find all paths
                                           target=None):
                path_length = len(path)
                critical_path_length = max(critical_path_length, path_length)
        except:
            critical_path_length = len(plan.task_graph)
        
        # Estimate based on critical path and parallelism
        avg_task_time = 30  # seconds per task (estimated)
        total_time = (critical_path_length * avg_task_time) / plan.parallelism_level
        
        return timedelta(seconds=total_time)


class LoadBalancer:
    """Dynamic load balancer for swarm agents"""
    
    def __init__(self):
        self.agent_metrics: Dict[str, LoadMetrics] = {}
        self.load_threshold_high = 0.8
        self.load_threshold_low = 0.2
        self.rebalance_interval = 10  # seconds
        self.last_rebalance = time.time()
        
    async def update_agent_metrics(self, agent: SwarmAgent, metrics: Dict[str, Any]):
        """Update metrics for an agent"""
        agent_id = agent.id
        
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = LoadMetrics(agent_id=agent_id)
        
        load_metrics = self.agent_metrics[agent_id]
        
        # Update metrics
        load_metrics.cpu_usage = metrics.get("cpu_usage", 0.0)
        load_metrics.memory_usage = metrics.get("memory_usage", 0.0)
        load_metrics.active_tasks = metrics.get("active_tasks", 0)
        load_metrics.completed_tasks = len(agent.completed_tasks)
        load_metrics.queue_length = metrics.get("queue_length", 0)
        load_metrics.last_updated = datetime.now(timezone.utc)
        
        # Calculate average task time
        if load_metrics.completed_tasks > 0:
            load_metrics.average_task_time = metrics.get("avg_task_time", 30.0)
        
        # Update agent load
        agent.load = self._calculate_agent_load(load_metrics)
    
    def _calculate_agent_load(self, metrics: LoadMetrics) -> float:
        """Calculate overall agent load"""
        # Weighted combination of metrics
        cpu_weight = 0.3
        memory_weight = 0.2
        task_weight = 0.3
        queue_weight = 0.2
        
        # Normalize task count (assume max 10 concurrent tasks)
        task_load = min(metrics.active_tasks / 10, 1.0)
        
        # Normalize queue length (assume max 20 queued tasks)
        queue_load = min(metrics.queue_length / 20, 1.0)
        
        total_load = (
            metrics.cpu_usage * cpu_weight +
            metrics.memory_usage * memory_weight +
            task_load * task_weight +
            queue_load * queue_weight
        )
        
        return min(total_load, 1.0)
    
    async def should_rebalance(self, agents: List[SwarmAgent]) -> bool:
        """Check if load rebalancing is needed"""
        # Check time since last rebalance
        if time.time() - self.last_rebalance < self.rebalance_interval:
            return False
        
        if not agents:
            return False
        
        # Calculate load statistics
        loads = [agent.load for agent in agents]
        avg_load = np.mean(loads)
        std_load = np.std(loads)
        
        # Check for imbalance
        if std_load > 0.2:  # High variance in loads
            return True
        
        # Check for overloaded agents
        overloaded = sum(1 for load in loads if load > self.load_threshold_high)
        if overloaded > 0:
            return True
        
        # Check for underutilized agents
        underutilized = sum(1 for load in loads if load < self.load_threshold_low)
        if underutilized > len(agents) * 0.3:  # More than 30% underutilized
            return True
        
        return False
    
    async def rebalance_tasks(
        self,
        agents: List[SwarmAgent],
        active_tasks: Dict[str, SwarmTask],
        plan: ExecutionPlan
    ) -> Dict[str, List[str]]:
        """Rebalance tasks across agents"""
        self.last_rebalance = time.time()
        
        # Identify overloaded and underloaded agents
        overloaded_agents = [a for a in agents if a.load > self.load_threshold_high]
        underloaded_agents = [a for a in agents if a.load < self.load_threshold_low]
        
        if not overloaded_agents or not underloaded_agents:
            return {}  # No rebalancing needed
        
        # Tasks to redistribute
        tasks_to_move = {}
        
        for overloaded in overloaded_agents:
            # Get movable tasks (not currently running)
            agent_tasks = plan.agent_assignments.get(overloaded.id, [])
            movable_tasks = [
                task_id for task_id in agent_tasks
                if task_id in active_tasks and active_tasks[task_id].status == "pending"
            ]
            
            # Move some tasks
            num_to_move = max(1, len(movable_tasks) // 3)  # Move 1/3 of pending tasks
            tasks_to_move[overloaded.id] = movable_tasks[:num_to_move]
        
        # Redistribute to underloaded agents
        new_assignments = {}
        task_pool = []
        
        for agent_id, tasks in tasks_to_move.items():
            task_pool.extend(tasks)
            # Remove from original assignments
            for task_id in tasks:
                plan.agent_assignments[agent_id].remove(task_id)
        
        # Assign to underloaded agents
        for i, task_id in enumerate(task_pool):
            target_agent = underloaded_agents[i % len(underloaded_agents)]
            
            if target_agent.id not in new_assignments:
                new_assignments[target_agent.id] = []
            
            new_assignments[target_agent.id].append(task_id)
            plan.agent_assignments[target_agent.id].append(task_id)
        
        logger.info(
            f"Rebalanced {len(task_pool)} tasks from "
            f"{len(overloaded_agents)} overloaded agents to "
            f"{len(underloaded_agents)} underloaded agents"
        )
        
        return new_assignments


class FaultToleranceManager:
    """Manages fault tolerance and recovery for swarm execution"""
    
    def __init__(self):
        self.failure_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.recovery_strategies = {
            "retry": self._retry_strategy,
            "reassign": self._reassign_strategy,
            "checkpoint": self._checkpoint_strategy,
            "replicate": self._replicate_strategy,
        }
        self.max_retries = 3
        self.checkpoint_interval = 30  # seconds
        
    async def handle_task_failure(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        error: Exception,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Handle task failure with appropriate recovery strategy"""
        failure_info = {
            "task_id": task.id,
            "agent_id": agent.id,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc),
            "retry_count": len([
                f for f in self.failure_history[task.id]
                if f["type"] == "retry"
            ])
        }
        
        self.failure_history[task.id].append(failure_info)
        
        # Determine recovery strategy
        strategy = await self._select_recovery_strategy(task, agent, failure_info)
        
        # Execute recovery
        recovery_func = self.recovery_strategies.get(strategy, self._retry_strategy)
        result = await recovery_func(task, agent, context, failure_info)
        
        return result
    
    async def _select_recovery_strategy(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        failure_info: Dict[str, Any]
    ) -> str:
        """Select appropriate recovery strategy"""
        retry_count = failure_info["retry_count"]
        
        # Check retry limit
        if retry_count >= self.max_retries:
            # Try different agent
            return "reassign"
        
        # Check if error is transient
        error_msg = failure_info["error"].lower()
        transient_errors = ["timeout", "connection", "temporary"]
        
        if any(err in error_msg for err in transient_errors):
            return "retry"
        
        # Check task priority
        if task.priority > 8:
            return "replicate"  # High priority tasks get replicated
        
        return "retry"  # Default strategy
    
    async def _retry_strategy(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        context: ExecutionContext,
        failure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retry task execution"""
        retry_count = failure_info["retry_count"]
        
        # Exponential backoff
        delay = min(2 ** retry_count, 60)  # Max 60 seconds
        await asyncio.sleep(delay)
        
        logger.info(
            f"Retrying task {task.id} on agent {agent.id} "
            f"(attempt {retry_count + 1}/{self.max_retries})"
        )
        
        return {
            "strategy": "retry",
            "action": "retry_execution",
            "delay": delay,
            "agent_id": agent.id
        }
    
    async def _reassign_strategy(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        context: ExecutionContext,
        failure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reassign task to different agent"""
        # Find alternative agent
        available_agents = [
            a for a in context.agents
            if a.id != agent.id and a.status != "failed"
        ]
        
        if not available_agents:
            logger.warning(f"No alternative agents available for task {task.id}")
            return {
                "strategy": "reassign",
                "action": "failed",
                "reason": "no_available_agents"
            }
        
        # Select best alternative
        best_agent = min(available_agents, key=lambda a: a.load)
        
        logger.info(
            f"Reassigning task {task.id} from agent {agent.id} "
            f"to agent {best_agent.id}"
        )
        
        return {
            "strategy": "reassign",
            "action": "reassign_task",
            "new_agent_id": best_agent.id,
            "old_agent_id": agent.id
        }
    
    async def _checkpoint_strategy(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        context: ExecutionContext,
        failure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resume from last checkpoint"""
        # Check for saved checkpoint
        checkpoint = await self._load_checkpoint(task.id)
        
        if checkpoint:
            logger.info(
                f"Resuming task {task.id} from checkpoint "
                f"(progress: {checkpoint.get('progress', 0)}%)"
            )
            
            return {
                "strategy": "checkpoint",
                "action": "resume_from_checkpoint",
                "checkpoint": checkpoint,
                "agent_id": agent.id
            }
        else:
            # No checkpoint, retry from beginning
            return await self._retry_strategy(task, agent, context, failure_info)
    
    async def _replicate_strategy(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        context: ExecutionContext,
        failure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replicate task execution on multiple agents"""
        # Find available agents for replication
        num_replicas = min(3, len(context.agents))
        available_agents = [
            a for a in context.agents
            if a.status != "failed" and a.load < 0.8
        ][:num_replicas]
        
        if len(available_agents) < 2:
            # Not enough agents for replication
            return await self._retry_strategy(task, agent, context, failure_info)
        
        logger.info(
            f"Replicating task {task.id} on {len(available_agents)} agents "
            f"for fault tolerance"
        )
        
        return {
            "strategy": "replicate",
            "action": "replicate_execution",
            "replica_agents": [a.id for a in available_agents],
            "quorum": len(available_agents) // 2 + 1  # Majority quorum
        }
    
    async def create_checkpoint(
        self,
        task: SwarmTask,
        progress: float,
        state: Dict[str, Any]
    ):
        """Create execution checkpoint"""
        checkpoint = {
            "task_id": task.id,
            "progress": progress,
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store checkpoint (in production, use persistent storage)
        cache = RedisCache()
        await cache.set(
            f"checkpoint:{task.id}",
            checkpoint,
            timeout=3600,  # 1 hour
            strategy=CacheStrategy.STANDARD
        )
    
    async def _load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for task"""
        cache = RedisCache()
        checkpoint = cache.get(
            f"checkpoint:{task_id}",
            strategy=CacheStrategy.STANDARD
        )
        return checkpoint


class RuntimeOptimizer:
    """Runtime optimization for swarm execution"""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=100)
        self.performance_baseline = {}
        self.optimization_interval = 30  # seconds
        self.last_optimization = time.time()
        
    async def optimize_runtime(
        self,
        context: ExecutionContext,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize runtime execution parameters"""
        current_time = time.time()
        
        if current_time - self.last_optimization < self.optimization_interval:
            return {}  # Too soon for optimization
        
        self.last_optimization = current_time
        
        # Analyze current performance
        performance_analysis = await self._analyze_performance(metrics)
        
        # Generate optimizations
        optimizations = {}
        
        # Task batching optimization
        if performance_analysis["overhead_ratio"] > 0.3:
            optimizations["batch_size"] = min(
                metrics.get("batch_size", 1) * 2,
                10
            )
        
        # Parallelism optimization
        if performance_analysis["cpu_utilization"] < 0.6:
            optimizations["parallelism"] = min(
                metrics.get("parallelism", 1) + 1,
                len(context.agents)
            )
        elif performance_analysis["cpu_utilization"] > 0.9:
            optimizations["parallelism"] = max(
                metrics.get("parallelism", 1) - 1,
                1
            )
        
        # Communication optimization
        if performance_analysis["communication_latency"] > 100:  # ms
            optimizations["message_batching"] = True
            optimizations["compression"] = True
        
        # Cache optimization
        if performance_analysis["cache_hit_rate"] < 0.5:
            optimizations["cache_size"] = metrics.get("cache_size", 1000) * 1.5
            optimizations["cache_ttl"] = metrics.get("cache_ttl", 300) * 2
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": metrics,
            "analysis": performance_analysis,
            "optimizations": optimizations
        })
        
        return optimizations
    
    async def _analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        analysis = {
            "overhead_ratio": 0.0,
            "cpu_utilization": 0.0,
            "communication_latency": 0.0,
            "cache_hit_rate": 0.0,
            "efficiency": 0.0
        }
        
        # Calculate overhead ratio
        total_time = metrics.get("total_execution_time", 1)
        actual_work_time = metrics.get("actual_work_time", 0)
        analysis["overhead_ratio"] = 1 - (actual_work_time / total_time) if total_time > 0 else 0
        
        # CPU utilization
        analysis["cpu_utilization"] = metrics.get("avg_cpu_usage", 0.5)
        
        # Communication metrics
        analysis["communication_latency"] = metrics.get("avg_message_latency", 50)
        
        # Cache performance
        cache_hits = metrics.get("cache_hits", 0)
        cache_misses = metrics.get("cache_misses", 0)
        total_cache_ops = cache_hits + cache_misses
        analysis["cache_hit_rate"] = cache_hits / total_cache_ops if total_cache_ops > 0 else 0
        
        # Overall efficiency
        completed_tasks = metrics.get("completed_tasks", 0)
        failed_tasks = metrics.get("failed_tasks", 0)
        total_tasks = completed_tasks + failed_tasks
        analysis["efficiency"] = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        return analysis


class SwarmExecutionEngine:
    """Main execution engine for swarm tasks"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine_id = f"engine_{uuid4().hex[:8]}"
        
        # Core components
        self.scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer()
        self.fault_manager = FaultToleranceManager()
        self.runtime_optimizer = RuntimeOptimizer()
        
        # Execution state
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history = deque(maxlen=100)
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_tasks_executed": 0,
            "average_execution_time": 0.0
        }
        
        logger.info(f"Swarm Execution Engine initialized: {self.engine_id}")
    
    async def execute_swarm(
        self,
        context: ExecutionContext,
        monitoring_callback: Optional[Callable] = None
    ) -> SwarmResult:
        """Execute swarm with full orchestration"""
        execution_id = f"exec_{uuid4().hex[:8]}"
        start_time = time.time()
        
        self.active_executions[execution_id] = context
        self.metrics["total_executions"] += 1
        
        try:
            # Create execution plan
            plan = await self.scheduler.create_execution_plan(context)
            
            logger.info(
                f"Starting execution {execution_id} with "
                f"{len(context.tasks)} tasks and {len(context.agents)} agents"
            )
            
            # Initialize result
            result = SwarmResult(
                objective_id=context.objective.id if context.objective else "",
                status="running"
            )
            
            # Start monitoring
            monitor_task = asyncio.create_task(
                self._monitor_execution(
                    execution_id,
                    context,
                    plan,
                    monitoring_callback
                )
            )
            
            # Execute tasks
            execution_results = await self._execute_plan(
                execution_id,
                context,
                plan
            )
            
            # Cancel monitoring
            monitor_task.cancel()
            
            # Compile results
            result.status = self._determine_final_status(execution_results)
            result.results = execution_results["task_results"]
            result.completed_tasks = execution_results["completed"]
            result.failed_tasks = execution_results["failed"]
            result.execution_time = time.time() - start_time
            result.agent_utilization = self._calculate_utilization(context.agents)
            result.confidence = execution_results.get("confidence", 0.8)
            
            # Update metrics
            if result.status == "completed":
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1
                
            self.metrics["total_tasks_executed"] += len(result.completed_tasks)
            
            # Record execution
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": datetime.now(timezone.utc),
                "duration": result.execution_time,
                "status": result.status,
                "task_count": len(context.tasks),
                "agent_count": len(context.agents)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Execution {execution_id} failed: {e}")
            self.metrics["failed_executions"] += 1
            
            return SwarmResult(
                objective_id=context.objective.id if context.objective else "",
                status="failed",
                results={"error": str(e)},
                execution_time=time.time() - start_time
            )
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_plan(
        self,
        execution_id: str,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute the task plan"""
        results = {
            "task_results": {},
            "completed": [],
            "failed": [],
            "agent_performance": {}
        }
        
        # Create task queues for each agent
        agent_queues = {
            agent_id: asyncio.Queue()
            for agent_id in plan.agent_assignments.keys()
        }
        
        # Populate queues
        for agent_id, task_ids in plan.agent_assignments.items():
            for task_id in task_ids:
                task = next((t for t in context.tasks if t.id == task_id), None)
                if task:
                    await agent_queues[agent_id].put(task)
        
        # Start agent executors
        agent_tasks = []
        for agent in context.agents:
            if agent.id in agent_queues:
                executor_task = asyncio.create_task(
                    self._agent_executor(
                        agent,
                        agent_queues[agent.id],
                        context,
                        results
                    )
                )
                agent_tasks.append(executor_task)
        
        # Wait for all agents to complete
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Calculate confidence based on success rate
        total_tasks = len(results["completed"]) + len(results["failed"])
        if total_tasks > 0:
            results["confidence"] = len(results["completed"]) / total_tasks
        else:
            results["confidence"] = 0.0
        
        return results
    
    async def _agent_executor(
        self,
        agent: SwarmAgent,
        task_queue: asyncio.Queue,
        context: ExecutionContext,
        results: Dict[str, Any]
    ):
        """Execute tasks for a single agent"""
        agent_completed = 0
        agent_failed = 0
        
        while not task_queue.empty():
            try:
                # Get next task
                task = await asyncio.wait_for(task_queue.get(), timeout=1.0)
                
                # Update agent state
                agent.status = "busy"
                agent.current_task = task.id
                
                # Execute task
                logger.debug(f"Agent {agent.id} executing task {task.id}")
                
                task_result = await self._execute_single_task(
                    task,
                    agent,
                    context
                )
                
                # Handle result
                if task_result["status"] == "completed":
                    results["completed"].append(task.id)
                    results["task_results"][task.id] = task_result["result"]
                    agent_completed += 1
                    agent.completed_tasks.append(task.id)
                else:
                    # Handle failure
                    recovery = await self.fault_manager.handle_task_failure(
                        task,
                        agent,
                        Exception(task_result.get("error", "Unknown error")),
                        context
                    )
                    
                    if recovery["action"] == "retry_execution":
                        # Retry task
                        await task_queue.put(task)
                    elif recovery["action"] == "reassign_task":
                        # Task will be reassigned
                        results["failed"].append(task.id)
                        agent_failed += 1
                    else:
                        results["failed"].append(task.id)
                        agent_failed += 1
                
                # Update agent load
                agent.load = max(0, agent.load - 0.1)
                
            except asyncio.TimeoutError:
                # No more tasks
                break
            except Exception as e:
                logger.error(f"Agent {agent.id} executor error: {e}")
                agent_failed += 1
        
        # Update agent state
        agent.status = "idle"
        agent.current_task = None
        
        # Record agent performance
        results["agent_performance"][agent.id] = {
            "completed": agent_completed,
            "failed": agent_failed,
            "efficiency": agent_completed / (agent_completed + agent_failed) if (agent_completed + agent_failed) > 0 else 0
        }
    
    async def _execute_single_task(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute a single task"""
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        task.agent_id = agent.id
        
        try:
            # Simulate task execution (in production, would call actual task handler)
            execution_time = np.random.exponential(5)  # Average 5 seconds
            await asyncio.sleep(execution_time)
            
            # Simulate success/failure (90% success rate)
            if random.random() < 0.9:
                task.status = "completed"
                task.completed_at = datetime.now(timezone.utc)
                task.result = {
                    "output": f"Task {task.id} completed by {agent.id}",
                    "execution_time": execution_time,
                    "metadata": {"agent_type": agent.type}
                }
                
                return {
                    "status": "completed",
                    "result": task.result
                }
            else:
                raise Exception("Simulated task failure")
                
        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now(timezone.utc)
            
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _monitor_execution(
        self,
        execution_id: str,
        context: ExecutionContext,
        plan: ExecutionPlan,
        callback: Optional[Callable] = None
    ):
        """Monitor execution progress and optimize"""
        monitor_interval = 5  # seconds
        
        while execution_id in self.active_executions:
            try:
                # Collect metrics
                metrics = await self._collect_execution_metrics(context, plan)
                
                # Update agent metrics
                for agent in context.agents:
                    await self.load_balancer.update_agent_metrics(
                        agent,
                        {
                            "cpu_usage": random.random() * 0.8,  # Simulated
                            "memory_usage": random.random() * 0.7,  # Simulated
                            "active_tasks": 1 if agent.current_task else 0,
                            "queue_length": 0  # Would check actual queue
                        }
                    )
                
                # Check if rebalancing needed
                if await self.load_balancer.should_rebalance(context.agents):
                    active_tasks = {
                        t.id: t for t in context.tasks
                        if t.status in ["pending", "running"]
                    }
                    
                    new_assignments = await self.load_balancer.rebalance_tasks(
                        context.agents,
                        active_tasks,
                        plan
                    )
                    
                    # Apply rebalancing (would update queues in production)
                    logger.info(f"Rebalanced {len(new_assignments)} task assignments")
                
                # Runtime optimization
                optimizations = await self.runtime_optimizer.optimize_runtime(
                    context,
                    metrics
                )
                
                if optimizations:
                    logger.info(f"Applied runtime optimizations: {optimizations}")
                
                # Callback for external monitoring
                if callback:
                    await callback({
                        "execution_id": execution_id,
                        "metrics": metrics,
                        "optimizations": optimizations,
                        "timestamp": datetime.now(timezone.utc)
                    })
                
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error for {execution_id}: {e}")
                await asyncio.sleep(monitor_interval)
    
    async def _collect_execution_metrics(
        self,
        context: ExecutionContext,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Collect current execution metrics"""
        completed_tasks = sum(
            1 for t in context.tasks if t.status == "completed"
        )
        failed_tasks = sum(
            1 for t in context.tasks if t.status == "failed"
        )
        running_tasks = sum(
            1 for t in context.tasks if t.status == "running"
        )
        pending_tasks = sum(
            1 for t in context.tasks if t.status == "pending"
        )
        
        total_tasks = len(context.tasks)
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Agent utilization
        busy_agents = sum(1 for a in context.agents if a.status == "busy")
        avg_agent_load = np.mean([a.load for a in context.agents])
        
        return {
            "progress_percentage": progress,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "pending_tasks": pending_tasks,
            "busy_agents": busy_agents,
            "total_agents": len(context.agents),
            "average_agent_load": avg_agent_load,
            "parallelism_level": plan.parallelism_level,
            "estimated_remaining_time": self._estimate_remaining_time(
                pending_tasks + running_tasks,
                completed_tasks,
                context.start_time
            )
        }
    
    def _estimate_remaining_time(
        self,
        remaining_tasks: int,
        completed_tasks: int,
        start_time: datetime
    ) -> float:
        """Estimate remaining execution time"""
        if completed_tasks == 0:
            return remaining_tasks * 5.0  # Default 5 seconds per task
        
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        avg_task_time = elapsed / completed_tasks
        
        return remaining_tasks * avg_task_time
    
    def _determine_final_status(self, execution_results: Dict[str, Any]) -> str:
        """Determine final execution status"""
        completed = len(execution_results["completed"])
        failed = len(execution_results["failed"])
        
        if failed == 0:
            return "completed"
        elif completed == 0:
            return "failed"
        else:
            return "partial"
    
    def _calculate_utilization(self, agents: List[SwarmAgent]) -> Dict[str, float]:
        """Calculate agent utilization"""
        utilization = {}
        
        for agent in agents:
            completed = len(agent.completed_tasks)
            # Simple utilization based on completed tasks
            utilization[agent.id] = min(1.0, completed * 0.1)
        
        return utilization
    
    async def shutdown(self):
        """Gracefully shutdown execution engine"""
        logger.info(f"Shutting down execution engine {self.engine_id}")
        
        # Cancel active executions
        for execution_id in list(self.active_executions.keys()):
            logger.warning(f"Cancelling active execution: {execution_id}")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Save metrics
        logger.info(f"Final metrics: {self.metrics}")
