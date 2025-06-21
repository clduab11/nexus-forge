"""
Resource Optimizer Module
Implements advanced resource allocation, dynamic scaling, and intelligent load distribution
for ADK Hackathon optimization
"""

import asyncio
import logging
import psutil
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .advanced_load_balancer import EnhancedLoadBalancer
from .scalability.enterprise_scalability import AutoScaler

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources to optimize"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK_IO = "disk_io"
    GPU = "gpu"
    TASK_QUEUE = "task_queue"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    WORST_FIT = "worst_fit"
    ROUND_ROBIN = "round_robin"
    BIN_PACKING = "bin_packing"
    HUNGARIAN = "hungarian"  # Optimal assignment
    GENETIC = "genetic"  # GA-based optimization
    ML_BASED = "ml_based"  # Machine learning based


@dataclass
class ResourceMetrics:
    """Real-time resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    network_throughput_mbps: float
    disk_io_mbps: float
    gpu_utilization: Optional[float] = None
    task_queue_size: int = 0
    active_connections: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class ResourceAllocation:
    """Resource allocation for a task/agent"""
    allocation_id: str
    task_id: str
    resource_type: ResourceType
    amount: float
    server_id: str
    priority: int
    start_time: datetime
    duration_estimate: Optional[timedelta] = None
    actual_usage: Optional[float] = None
    cost: float = 0.0


@dataclass
class OptimizationResult:
    """Result of resource optimization"""
    timestamp: datetime
    strategy: AllocationStrategy
    allocations: List[ResourceAllocation]
    total_cost: float
    efficiency_score: float
    load_balance_score: float
    recommendations: List[Dict[str, Any]]
    predicted_savings: float


class ResourceOptimizer:
    """Advanced resource optimization engine"""
    
    def __init__(
        self,
        load_balancer: Optional[EnhancedLoadBalancer] = None,
        auto_scaler: Optional[AutoScaler] = None
    ):
        self.load_balancer = load_balancer or EnhancedLoadBalancer()
        self.auto_scaler = auto_scaler or AutoScaler()
        
        # Resource tracking
        self.resource_pools: Dict[str, Dict[ResourceType, float]] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history = deque(maxlen=10000)
        
        # Metrics collection
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Optimization parameters
        self.optimization_interval = 30  # seconds
        self.last_optimization = time.time()
        self.optimization_history = deque(maxlen=100)
        
        # Machine learning components
        self.prediction_models = {}
        self.anomaly_detectors = {}
        
        # Cost models
        self.resource_costs = {
            ResourceType.CPU: 0.05,  # per core-hour
            ResourceType.MEMORY: 0.01,  # per GB-hour
            ResourceType.NETWORK: 0.02,  # per GB transferred
            ResourceType.DISK_IO: 0.003,  # per GB
            ResourceType.GPU: 0.50,  # per GPU-hour
        }
        
        # Advanced features
        self.enable_predictive_scaling = True
        self.enable_cost_optimization = True
        self.enable_ml_allocation = True
    
    async def collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Network metrics (simplified)
        net_io = psutil.net_io_counters()
        network_throughput_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) if disk_io else 0
        
        # GPU metrics (if available)
        gpu_utilization = await self._get_gpu_utilization()
        
        # Application-specific metrics
        task_queue_size = len(self.allocations)
        active_connections = sum(
            s.current_connections for s in self.load_balancer.server_nodes.values()
        )
        
        # Calculate response time from load balancer
        response_times = []
        for server in self.load_balancer.server_nodes.values():
            if hasattr(server, 'avg_response_time'):
                response_times.append(server.avg_response_time)
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Calculate cost
        cost_per_hour = self._calculate_current_cost()
        
        return ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            network_throughput_mbps=network_throughput_mbps,
            disk_io_mbps=disk_io_mbps,
            gpu_utilization=gpu_utilization,
            task_queue_size=task_queue_size,
            active_connections=active_connections,
            response_time_ms=avg_response_time,
            error_rate=0.0,  # Would be calculated from actual errors
            cost_per_hour=cost_per_hour
        )
    
    async def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            # Check for NVIDIA GPUs
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                utilizations = [float(x) for x in result.stdout.strip().split('\n')]
                return statistics.mean(utilizations)
        except:
            pass
        return None
    
    def _calculate_current_cost(self) -> float:
        """Calculate current resource cost per hour"""
        total_cost = 0.0
        
        # CPU cost
        cpu_cores = psutil.cpu_count()
        cpu_utilization = psutil.cpu_percent() / 100
        total_cost += cpu_cores * cpu_utilization * self.resource_costs[ResourceType.CPU]
        
        # Memory cost
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        memory_utilization = psutil.virtual_memory().percent / 100
        total_cost += memory_gb * memory_utilization * self.resource_costs[ResourceType.MEMORY]
        
        # Add other resource costs...
        
        return total_cost
    
    async def optimize_resource_allocation(
        self,
        tasks: List[Dict[str, Any]],
        strategy: AllocationStrategy = AllocationStrategy.ML_BASED
    ) -> OptimizationResult:
        """Optimize resource allocation for tasks"""
        start_time = datetime.utcnow()
        
        # Collect current metrics
        current_metrics = await self.collect_system_metrics()
        
        # Store metrics history
        for server_id in self.load_balancer.server_nodes:
            self.metrics_history[server_id].append(current_metrics)
        
        # Prepare optimization data
        available_resources = await self._calculate_available_resources()
        task_requirements = await self._estimate_task_requirements(tasks)
        
        # Apply selected strategy
        allocations = []
        
        if strategy == AllocationStrategy.ML_BASED and self.enable_ml_allocation:
            allocations = await self._ml_based_allocation(
                tasks, task_requirements, available_resources
            )
        elif strategy == AllocationStrategy.HUNGARIAN:
            allocations = await self._hungarian_allocation(
                tasks, task_requirements, available_resources
            )
        elif strategy == AllocationStrategy.BIN_PACKING:
            allocations = await self._bin_packing_allocation(
                tasks, task_requirements, available_resources
            )
        elif strategy == AllocationStrategy.GENETIC:
            allocations = await self._genetic_allocation(
                tasks, task_requirements, available_resources
            )
        else:
            allocations = await self._basic_allocation(
                tasks, task_requirements, available_resources, strategy
            )
        
        # Calculate optimization metrics
        total_cost = sum(a.cost for a in allocations)
        efficiency_score = await self._calculate_efficiency_score(allocations)
        load_balance_score = await self._calculate_load_balance_score()
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            current_metrics, allocations, efficiency_score
        )
        
        # Predict savings
        predicted_savings = await self._predict_cost_savings(allocations)
        
        result = OptimizationResult(
            timestamp=start_time,
            strategy=strategy,
            allocations=allocations,
            total_cost=total_cost,
            efficiency_score=efficiency_score,
            load_balance_score=load_balance_score,
            recommendations=recommendations,
            predicted_savings=predicted_savings
        )
        
        # Store optimization result
        self.optimization_history.append(result)
        self.last_optimization = time.time()
        
        # Apply allocations
        await self._apply_allocations(allocations)
        
        return result
    
    async def _calculate_available_resources(self) -> Dict[str, Dict[ResourceType, float]]:
        """Calculate available resources per server"""
        available = {}
        
        for server_id, server in self.load_balancer.server_nodes.items():
            # Get server-specific metrics
            cpu_available = (100 - server.cpu_usage) if hasattr(server, 'cpu_usage') else 50.0
            memory_available = 8192  # MB, would be queried from actual server
            
            # Account for existing allocations
            existing_cpu = sum(
                a.amount for a in self.allocations.values()
                if a.server_id == server_id and a.resource_type == ResourceType.CPU
            )
            existing_memory = sum(
                a.amount for a in self.allocations.values()
                if a.server_id == server_id and a.resource_type == ResourceType.MEMORY
            )
            
            available[server_id] = {
                ResourceType.CPU: max(0, cpu_available - existing_cpu),
                ResourceType.MEMORY: max(0, memory_available - existing_memory),
                ResourceType.NETWORK: 1000,  # Mbps
                ResourceType.DISK_IO: 500,  # MB/s
                ResourceType.TASK_QUEUE: server.max_connections - server.current_connections,
            }
        
        return available
    
    async def _estimate_task_requirements(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[ResourceType, float]]:
        """Estimate resource requirements for tasks"""
        requirements = []
        
        for task in tasks:
            # Use historical data if available
            task_type = task.get('type', 'default')
            
            if task_type in self.performance_baselines:
                baseline = self.performance_baselines[task_type]
                req = {
                    ResourceType.CPU: baseline.get('cpu', 10.0),
                    ResourceType.MEMORY: baseline.get('memory', 512.0),
                    ResourceType.NETWORK: baseline.get('network', 10.0),
                    ResourceType.DISK_IO: baseline.get('disk_io', 5.0),
                }
            else:
                # Default estimates based on task properties
                complexity = task.get('complexity', 1.0)
                data_size = task.get('data_size_mb', 100)
                
                req = {
                    ResourceType.CPU: 10.0 * complexity,
                    ResourceType.MEMORY: 256.0 + data_size * 2,
                    ResourceType.NETWORK: data_size / 10,
                    ResourceType.DISK_IO: data_size / 20,
                }
            
            requirements.append(req)
        
        return requirements
    
    async def _ml_based_allocation(
        self,
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]]
    ) -> List[ResourceAllocation]:
        """Machine learning based resource allocation"""
        allocations = []
        
        # Prepare features for ML model
        features = []
        for i, (task, req) in enumerate(zip(tasks, requirements)):
            task_features = [
                req[ResourceType.CPU],
                req[ResourceType.MEMORY],
                req[ResourceType.NETWORK],
                task.get('priority', 1),
                task.get('deadline_minutes', 60),
                len(available),  # Number of available servers
            ]
            features.append(task_features)
        
        # Predict optimal allocations
        # In production, this would use a trained model
        predictions = await self._predict_allocations(features)
        
        # Convert predictions to allocations
        for i, (task, pred) in enumerate(zip(tasks, predictions)):
            server_id = pred['server_id']
            
            allocation = ResourceAllocation(
                allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{i}",
                task_id=task['id'],
                resource_type=ResourceType.CPU,  # Primary resource
                amount=requirements[i][ResourceType.CPU],
                server_id=server_id,
                priority=task.get('priority', 1),
                start_time=datetime.utcnow(),
                duration_estimate=timedelta(minutes=task.get('estimated_duration', 10)),
                cost=self._calculate_allocation_cost(requirements[i])
            )
            
            allocations.append(allocation)
        
        return allocations
    
    async def _predict_allocations(self, features: List[List[float]]) -> List[Dict[str, Any]]:
        """Predict optimal allocations using ML model"""
        # Simplified prediction logic
        predictions = []
        
        servers = list(self.load_balancer.server_nodes.keys())
        
        for feature_set in features:
            # Simple heuristic: allocate to least loaded server
            best_server = min(
                servers,
                key=lambda s: self.load_balancer.server_nodes[s].load_score
            )
            
            predictions.append({
                'server_id': best_server,
                'confidence': 0.85,
                'expected_performance': 0.9
            })
        
        return predictions
    
    async def _hungarian_allocation(
        self,
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]]
    ) -> List[ResourceAllocation]:
        """Hungarian algorithm for optimal task-server assignment"""
        servers = list(available.keys())
        n_tasks = len(tasks)
        n_servers = len(servers)
        
        # Create cost matrix
        cost_matrix = np.zeros((n_tasks, n_servers))
        
        for i, (task, req) in enumerate(zip(tasks, requirements)):
            for j, server_id in enumerate(servers):
                # Calculate assignment cost
                server_resources = available[server_id]
                
                # Check if server has enough resources
                if all(server_resources.get(r, 0) >= req.get(r, 0) for r in ResourceType):
                    # Cost based on resource utilization and server load
                    server = self.load_balancer.server_nodes[server_id]
                    
                    utilization_cost = sum(
                        req.get(r, 0) / max(server_resources.get(r, 1), 1)
                        for r in ResourceType
                    )
                    
                    load_cost = server.load_score
                    priority_factor = 1 / task.get('priority', 1)
                    
                    cost_matrix[i, j] = (utilization_cost + load_cost) * priority_factor
                else:
                    # Impossible assignment
                    cost_matrix[i, j] = np.inf
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create allocations
        allocations = []
        for task_idx, server_idx in zip(row_indices, col_indices):
            if cost_matrix[task_idx, server_idx] < np.inf:
                server_id = servers[server_idx]
                task = tasks[task_idx]
                
                allocation = ResourceAllocation(
                    allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{task_idx}",
                    task_id=task['id'],
                    resource_type=ResourceType.CPU,
                    amount=requirements[task_idx][ResourceType.CPU],
                    server_id=server_id,
                    priority=task.get('priority', 1),
                    start_time=datetime.utcnow(),
                    duration_estimate=timedelta(minutes=task.get('estimated_duration', 10)),
                    cost=self._calculate_allocation_cost(requirements[task_idx])
                )
                
                allocations.append(allocation)
        
        return allocations
    
    async def _bin_packing_allocation(
        self,
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]]
    ) -> List[ResourceAllocation]:
        """Bin packing algorithm for efficient resource utilization"""
        allocations = []
        
        # Sort tasks by resource requirements (largest first)
        sorted_tasks = sorted(
            zip(tasks, requirements),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )
        
        # Track remaining capacity per server
        remaining = {
            server_id: resources.copy()
            for server_id, resources in available.items()
        }
        
        for task, req in sorted_tasks:
            allocated = False
            
            # Try to fit task in servers (best fit)
            best_server = None
            best_fit_score = float('inf')
            
            for server_id, capacity in remaining.items():
                # Check if task fits
                if all(capacity.get(r, 0) >= req.get(r, 0) for r in ResourceType):
                    # Calculate fit score (smaller is better)
                    fit_score = sum(
                        capacity.get(r, 0) - req.get(r, 0)
                        for r in ResourceType
                    )
                    
                    if fit_score < best_fit_score:
                        best_fit_score = fit_score
                        best_server = server_id
            
            if best_server:
                # Allocate to best fitting server
                allocation = ResourceAllocation(
                    allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{task['id']}",
                    task_id=task['id'],
                    resource_type=ResourceType.CPU,
                    amount=req[ResourceType.CPU],
                    server_id=best_server,
                    priority=task.get('priority', 1),
                    start_time=datetime.utcnow(),
                    duration_estimate=timedelta(minutes=task.get('estimated_duration', 10)),
                    cost=self._calculate_allocation_cost(req)
                )
                
                allocations.append(allocation)
                
                # Update remaining capacity
                for r in ResourceType:
                    if r in remaining[best_server]:
                        remaining[best_server][r] -= req.get(r, 0)
        
        return allocations
    
    async def _genetic_allocation(
        self,
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]]
    ) -> List[ResourceAllocation]:
        """Genetic algorithm for resource allocation optimization"""
        # GA parameters
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        servers = list(available.keys())
        n_tasks = len(tasks)
        
        # Initialize population (random allocations)
        population = []
        for _ in range(population_size):
            individual = [np.random.choice(servers) for _ in range(n_tasks)]
            population.append(individual)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = await self._evaluate_allocation_fitness(
                    individual, tasks, requirements, available
                )
                fitness_scores.append(fitness)
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(
                    population_size, tournament_size, replace=False
                )
                winner_idx = max(
                    tournament_indices,
                    key=lambda i: fitness_scores[i]
                )
                new_population.append(population[winner_idx].copy())
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if np.random.random() < crossover_rate:
                    # Single-point crossover
                    crossover_point = np.random.randint(1, n_tasks)
                    new_population[i][crossover_point:], new_population[i+1][crossover_point:] = \
                        new_population[i+1][crossover_point:], new_population[i][crossover_point:]
            
            # Mutation
            for individual in new_population:
                for i in range(n_tasks):
                    if np.random.random() < mutation_rate:
                        individual[i] = np.random.choice(servers)
            
            population = new_population
        
        # Get best individual
        fitness_scores = []
        for individual in population:
            fitness = await self._evaluate_allocation_fitness(
                individual, tasks, requirements, available
            )
            fitness_scores.append(fitness)
        
        best_idx = np.argmax(fitness_scores)
        best_allocation = population[best_idx]
        
        # Convert to allocations
        allocations = []
        for i, (task, server_id) in enumerate(zip(tasks, best_allocation)):
            allocation = ResourceAllocation(
                allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{i}",
                task_id=task['id'],
                resource_type=ResourceType.CPU,
                amount=requirements[i][ResourceType.CPU],
                server_id=server_id,
                priority=task.get('priority', 1),
                start_time=datetime.utcnow(),
                duration_estimate=timedelta(minutes=task.get('estimated_duration', 10)),
                cost=self._calculate_allocation_cost(requirements[i])
            )
            allocations.append(allocation)
        
        return allocations
    
    async def _evaluate_allocation_fitness(
        self,
        allocation: List[str],
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]]
    ) -> float:
        """Evaluate fitness of an allocation"""
        fitness = 100.0
        
        # Check resource constraints
        server_loads = defaultdict(lambda: defaultdict(float))
        
        for task, req, server_id in zip(tasks, requirements, allocation):
            for resource_type, amount in req.items():
                server_loads[server_id][resource_type] += amount
        
        # Penalize overallocation
        for server_id, loads in server_loads.items():
            for resource_type, load in loads.items():
                available_amount = available[server_id].get(resource_type, 0)
                if load > available_amount:
                    fitness -= 50 * (load - available_amount) / available_amount
        
        # Reward load balancing
        load_values = [sum(loads.values()) for loads in server_loads.values()]
        if load_values:
            load_variance = np.var(load_values)
            fitness -= load_variance * 0.1
        
        # Consider priorities
        for task, server_id in zip(tasks, allocation):
            server = self.load_balancer.server_nodes.get(server_id)
            if server:
                # Higher priority tasks on better servers
                fitness += task.get('priority', 1) * server.health_score
        
        return max(0, fitness)
    
    async def _basic_allocation(
        self,
        tasks: List[Dict[str, Any]],
        requirements: List[Dict[ResourceType, float]],
        available: Dict[str, Dict[ResourceType, float]],
        strategy: AllocationStrategy
    ) -> List[ResourceAllocation]:
        """Basic allocation strategies"""
        allocations = []
        
        if strategy == AllocationStrategy.ROUND_ROBIN:
            servers = list(available.keys())
            for i, (task, req) in enumerate(zip(tasks, requirements)):
                server_id = servers[i % len(servers)]
                
                allocation = ResourceAllocation(
                    allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{i}",
                    task_id=task['id'],
                    resource_type=ResourceType.CPU,
                    amount=req[ResourceType.CPU],
                    server_id=server_id,
                    priority=task.get('priority', 1),
                    start_time=datetime.utcnow(),
                    cost=self._calculate_allocation_cost(req)
                )
                allocations.append(allocation)
        
        elif strategy == AllocationStrategy.FIRST_FIT:
            for task, req in zip(tasks, requirements):
                for server_id, capacity in available.items():
                    if all(capacity.get(r, 0) >= req.get(r, 0) for r in ResourceType):
                        allocation = ResourceAllocation(
                            allocation_id=f"alloc-{datetime.utcnow().timestamp()}-{task['id']}",
                            task_id=task['id'],
                            resource_type=ResourceType.CPU,
                            amount=req[ResourceType.CPU],
                            server_id=server_id,
                            priority=task.get('priority', 1),
                            start_time=datetime.utcnow(),
                            cost=self._calculate_allocation_cost(req)
                        )
                        allocations.append(allocation)
                        
                        # Update available capacity
                        for r in ResourceType:
                            if r in capacity:
                                capacity[r] -= req.get(r, 0)
                        break
        
        return allocations
    
    def _calculate_allocation_cost(self, requirements: Dict[ResourceType, float]) -> float:
        """Calculate cost of resource allocation"""
        total_cost = 0.0
        
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_costs:
                total_cost += amount * self.resource_costs[resource_type]
        
        return total_cost
    
    async def _calculate_efficiency_score(self, allocations: List[ResourceAllocation]) -> float:
        """Calculate resource utilization efficiency"""
        if not allocations:
            return 0.0
        
        # Calculate overall resource utilization
        total_allocated = defaultdict(float)
        total_available = defaultdict(float)
        
        for allocation in allocations:
            total_allocated[allocation.resource_type] += allocation.amount
        
        for server_resources in self.resource_pools.values():
            for resource_type, amount in server_resources.items():
                total_available[resource_type] += amount
        
        # Calculate efficiency as ratio of allocated to available
        efficiency_scores = []
        for resource_type in ResourceType:
            if resource_type in total_available and total_available[resource_type] > 0:
                efficiency = total_allocated[resource_type] / total_available[resource_type]
                efficiency_scores.append(min(efficiency, 1.0))
        
        return statistics.mean(efficiency_scores) * 100 if efficiency_scores else 0.0
    
    async def _calculate_load_balance_score(self) -> float:
        """Calculate load balance score across servers"""
        server_loads = []
        
        for server in self.load_balancer.server_nodes.values():
            server_loads.append(server.load_score)
        
        if not server_loads:
            return 0.0
        
        # Perfect balance = 100, high variance = lower score
        load_variance = statistics.variance(server_loads) if len(server_loads) > 1 else 0
        max_variance = 100.0  # Theoretical maximum variance
        
        balance_score = max(0, 100 - (load_variance / max_variance * 100))
        return balance_score
    
    async def _generate_recommendations(
        self,
        metrics: ResourceMetrics,
        allocations: List[ResourceAllocation],
        efficiency_score: float
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if metrics.cpu_percent > 80:
            recommendations.append({
                "type": "scale_up",
                "resource": "CPU",
                "reason": f"CPU utilization at {metrics.cpu_percent:.1f}%",
                "priority": "high",
                "estimated_impact": "20% performance improvement"
            })
        elif metrics.cpu_percent < 20:
            recommendations.append({
                "type": "scale_down",
                "resource": "CPU",
                "reason": f"CPU underutilized at {metrics.cpu_percent:.1f}%",
                "priority": "medium",
                "estimated_savings": f"${metrics.cost_per_hour * 0.3:.2f}/hour"
            })
        
        # Memory recommendations
        if metrics.memory_percent > 85:
            recommendations.append({
                "type": "optimize_memory",
                "resource": "Memory",
                "reason": f"Memory usage at {metrics.memory_percent:.1f}%",
                "priority": "high",
                "actions": ["Enable swap", "Increase memory allocation", "Optimize caching"]
            })
        
        # Load balancing recommendations
        load_balance_score = await self._calculate_load_balance_score()
        if load_balance_score < 70:
            recommendations.append({
                "type": "rebalance",
                "reason": f"Load imbalance detected (score: {load_balance_score:.1f})",
                "priority": "medium",
                "actions": ["Redistribute tasks", "Adjust server weights"]
            })
        
        # Efficiency recommendations
        if efficiency_score < 60:
            recommendations.append({
                "type": "improve_efficiency",
                "reason": f"Low resource efficiency ({efficiency_score:.1f}%)",
                "priority": "medium",
                "actions": ["Consolidate tasks", "Implement task batching", "Review allocation strategy"]
            })
        
        # Cost optimization
        if self.enable_cost_optimization and metrics.cost_per_hour > 100:
            recommendations.append({
                "type": "cost_optimization",
                "reason": f"High operational cost (${metrics.cost_per_hour:.2f}/hour)",
                "priority": "high",
                "actions": [
                    "Use spot instances",
                    "Implement auto-scaling policies",
                    "Schedule non-critical tasks during off-peak hours"
                ]
            })
        
        return recommendations
    
    async def _predict_cost_savings(self, allocations: List[ResourceAllocation]) -> float:
        """Predict potential cost savings from optimization"""
        current_cost = self._calculate_current_cost()
        optimized_cost = sum(a.cost for a in allocations)
        
        # Factor in efficiency improvements
        efficiency_factor = 0.85  # Assume 15% efficiency improvement
        predicted_cost = optimized_cost * efficiency_factor
        
        savings = max(0, current_cost - predicted_cost)
        return savings
    
    async def _apply_allocations(self, allocations: List[ResourceAllocation]):
        """Apply resource allocations to the system"""
        for allocation in allocations:
            # Store allocation
            self.allocations[allocation.allocation_id] = allocation
            self.allocation_history.append(allocation)
            
            # Update server load in load balancer
            if allocation.server_id in self.load_balancer.server_nodes:
                server = self.load_balancer.server_nodes[allocation.server_id]
                # Simulate resource usage update
                server.current_connections += 1
    
    async def dynamic_resource_adjustment(self):
        """Continuously adjust resources based on real-time metrics"""
        while True:
            try:
                # Collect metrics
                metrics = await self.collect_system_metrics()
                
                # Check for anomalies
                anomalies = await self._detect_anomalies(metrics)
                
                if anomalies:
                    logger.warning(f"Resource anomalies detected: {anomalies}")
                    
                    # Trigger immediate optimization
                    await self._handle_anomalies(anomalies)
                
                # Predictive scaling
                if self.enable_predictive_scaling:
                    predictions = await self._predict_resource_needs()
                    
                    if predictions['scale_action']:
                        await self._execute_predictive_scaling(predictions)
                
                # Regular optimization check
                if time.time() - self.last_optimization > self.optimization_interval:
                    # Get pending tasks
                    pending_tasks = await self._get_pending_tasks()
                    
                    if pending_tasks:
                        await self.optimize_resource_allocation(
                            pending_tasks,
                            AllocationStrategy.ML_BASED
                        )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in dynamic resource adjustment: {e}")
                await asyncio.sleep(10)
    
    async def _detect_anomalies(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Detect resource usage anomalies"""
        anomalies = []
        
        # CPU spike detection
        cpu_history = [m.cpu_percent for m in list(self.metrics_history.values())[-10:]]
        if cpu_history and metrics.cpu_percent > statistics.mean(cpu_history) + 2 * statistics.stdev(cpu_history):
            anomalies.append({
                "type": "cpu_spike",
                "severity": "high",
                "value": metrics.cpu_percent,
                "threshold": statistics.mean(cpu_history) + 2 * statistics.stdev(cpu_history)
            })
        
        # Memory leak detection
        memory_history = [m.memory_percent for m in list(self.metrics_history.values())[-20:]]
        if len(memory_history) > 10:
            # Check for continuous increase
            increasing_count = sum(
                1 for i in range(1, len(memory_history))
                if memory_history[i] > memory_history[i-1]
            )
            if increasing_count > len(memory_history) * 0.8:
                anomalies.append({
                    "type": "memory_leak",
                    "severity": "medium",
                    "trend": "increasing",
                    "rate": (memory_history[-1] - memory_history[0]) / len(memory_history)
                })
        
        # Response time degradation
        if metrics.response_time_ms > 1000:  # 1 second threshold
            anomalies.append({
                "type": "performance_degradation",
                "severity": "high",
                "response_time_ms": metrics.response_time_ms
            })
        
        return anomalies
    
    async def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies"""
        for anomaly in anomalies:
            if anomaly['type'] == 'cpu_spike' and anomaly['severity'] == 'high':
                # Distribute load to other servers
                await self._emergency_load_distribution()
            
            elif anomaly['type'] == 'memory_leak':
                # Schedule server restart or memory cleanup
                logger.warning("Memory leak detected, scheduling cleanup")
                await self._schedule_memory_cleanup()
            
            elif anomaly['type'] == 'performance_degradation':
                # Reduce load on affected servers
                await self._reduce_server_load()
    
    async def _predict_resource_needs(self) -> Dict[str, Any]:
        """Predict future resource needs"""
        # Simple trend-based prediction
        cpu_history = [m.cpu_percent for m in list(self.metrics_history.values())[-30:]]
        memory_history = [m.memory_percent for m in list(self.metrics_history.values())[-30:]]
        
        if len(cpu_history) < 10:
            return {"scale_action": None}
        
        # Calculate trends
        cpu_trend = np.polyfit(range(len(cpu_history)), cpu_history, 1)[0]
        memory_trend = np.polyfit(range(len(memory_history)), memory_history, 1)[0]
        
        # Predict values in 5 minutes
        predicted_cpu = cpu_history[-1] + cpu_trend * 60  # 60 data points = 5 minutes
        predicted_memory = memory_history[-1] + memory_trend * 60
        
        prediction = {
            "predicted_cpu": predicted_cpu,
            "predicted_memory": predicted_memory,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "scale_action": None
        }
        
        # Determine scaling action
        if predicted_cpu > 90 or predicted_memory > 90:
            prediction["scale_action"] = "scale_up"
            prediction["reason"] = f"Predicted high utilization: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%"
        elif predicted_cpu < 20 and predicted_memory < 30:
            prediction["scale_action"] = "scale_down"
            prediction["reason"] = f"Predicted low utilization: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%"
        
        return prediction
    
    async def _execute_predictive_scaling(self, predictions: Dict[str, Any]):
        """Execute predictive scaling actions"""
        if predictions['scale_action'] == 'scale_up':
            # Add resources proactively
            logger.info(f"Predictive scale-up: {predictions['reason']}")
            
            # Trigger auto-scaler
            metrics = {
                "cpu_usage": predictions['predicted_cpu'],
                "memory_usage": predictions['predicted_memory']
            }
            
            for service in self.auto_scaler.scaling_policies:
                await self.auto_scaler.evaluate_scaling(service, metrics)
        
        elif predictions['scale_action'] == 'scale_down':
            # Schedule resource reduction
            logger.info(f"Predictive scale-down: {predictions['reason']}")
            # Would implement gradual scale-down logic
    
    async def _emergency_load_distribution(self):
        """Emergency load distribution during anomalies"""
        # Find overloaded servers
        overloaded_servers = [
            server for server in self.load_balancer.server_nodes.values()
            if server.load_score > 0.8
        ]
        
        # Find underutilized servers
        underutilized_servers = [
            server for server in self.load_balancer.server_nodes.values()
            if server.load_score < 0.3
        ]
        
        if overloaded_servers and underutilized_servers:
            # Redistribute connections
            for overloaded in overloaded_servers:
                excess_connections = overloaded.current_connections - (overloaded.max_connections * 0.7)
                
                if excess_connections > 0:
                    # Distribute to underutilized servers
                    per_server = excess_connections // len(underutilized_servers)
                    
                    for underutilized in underutilized_servers:
                        # Simulate connection migration
                        transfer_count = min(
                            per_server,
                            underutilized.max_connections - underutilized.current_connections
                        )
                        
                        overloaded.current_connections -= transfer_count
                        underutilized.current_connections += transfer_count
                        
                        logger.info(
                            f"Migrated {transfer_count} connections from "
                            f"{overloaded.id} to {underutilized.id}"
                        )
    
    async def _schedule_memory_cleanup(self):
        """Schedule memory cleanup operations"""
        # In production, this would trigger actual cleanup
        logger.info("Scheduling memory cleanup operations")
        
        # Clear caches
        for server in self.load_balancer.server_nodes.values():
            if hasattr(server, 'clear_cache'):
                await server.clear_cache()
    
    async def _reduce_server_load(self):
        """Reduce load on servers experiencing performance issues"""
        # Temporarily reduce server weights
        for server in self.load_balancer.server_nodes.values():
            if server.avg_response_time > 1000:  # 1 second
                server.weight *= 0.5
                logger.info(f"Reduced weight for server {server.id} due to high response time")
    
    async def _get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of pending tasks to allocate"""
        # In production, this would query actual task queue
        # For now, return empty list to prevent continuous optimization
        return []
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        current_metrics = await self.collect_system_metrics()
        
        # Calculate various metrics
        efficiency_score = await self._calculate_efficiency_score(list(self.allocations.values()))
        load_balance_score = await self._calculate_load_balance_score()
        
        # Get recent optimization results
        recent_optimizations = list(self.optimization_history)[-10:]
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "active_connections": current_metrics.active_connections,
                "response_time_ms": current_metrics.response_time_ms,
                "cost_per_hour": current_metrics.cost_per_hour
            },
            "optimization_scores": {
                "efficiency": efficiency_score,
                "load_balance": load_balance_score,
                "overall": (efficiency_score + load_balance_score) / 2
            },
            "active_allocations": len(self.allocations),
            "recent_optimizations": [
                {
                    "timestamp": opt.timestamp.isoformat(),
                    "strategy": opt.strategy.value,
                    "allocations": len(opt.allocations),
                    "cost": opt.total_cost,
                    "savings": opt.predicted_savings
                }
                for opt in recent_optimizations
            ],
            "recommendations": [],
            "predicted_savings": 0.0
        }
        
        # Add recommendations
        if recent_optimizations:
            latest = recent_optimizations[-1]
            report["recommendations"] = latest.recommendations
            report["predicted_savings"] = latest.predicted_savings
        
        return report
