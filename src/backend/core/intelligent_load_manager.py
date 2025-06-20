"""
Intelligent Load Manager
Advanced load management with AI-driven optimization and predictive capabilities
"""

import asyncio
import heapq
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.stats import beta, gamma
import networkx as nx

from .advanced_load_balancer import EnhancedLoadBalancer, ServerNode
from .resource_optimizer import ResourceOptimizer, ResourceType, AllocationStrategy

logger = logging.getLogger(__name__)


class LoadPattern(Enum):
    """Types of load patterns"""
    STEADY = "steady"
    PERIODIC = "periodic"
    BURST = "burst"
    GRADUAL_INCREASE = "gradual_increase"
    GRADUAL_DECREASE = "gradual_decrease"
    RANDOM = "random"
    SEASONAL = "seasonal"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Task:
    """Task representation with resource requirements"""
    task_id: str
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    estimated_duration: timedelta
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    affinity_server: Optional[str] = None
    anti_affinity_servers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_server: Optional[str] = None
    actual_duration: Optional[timedelta] = None
    performance_score: float = 0.0


@dataclass
class LoadSnapshot:
    """Snapshot of system load at a point in time"""
    timestamp: datetime
    total_tasks: int
    active_tasks: int
    queued_tasks: int
    avg_response_time: float
    throughput: float
    error_rate: float
    resource_utilization: Dict[ResourceType, float]
    server_loads: Dict[str, float]
    pattern: LoadPattern
    anomaly_score: float = 0.0


class IntelligentLoadManager:
    """Advanced load management system with AI capabilities"""
    
    def __init__(
        self,
        load_balancer: Optional[EnhancedLoadBalancer] = None,
        resource_optimizer: Optional[ResourceOptimizer] = None
    ):
        self.load_balancer = load_balancer or EnhancedLoadBalancer()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer(self.load_balancer)
        
        # Task management
        self.task_queue: List[Task] = []  # Min heap by priority
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks = deque(maxlen=10000)
        self.task_dependencies = nx.DiGraph()
        
        # Load tracking
        self.load_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.pattern_detector = LoadPatternDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Intelligent features
        self.task_predictor = TaskDurationPredictor()
        self.load_forecaster = LoadForecaster()
        self.optimization_engine = OptimizationEngine()
        
        # Performance tracking
        self.sla_targets = {
            "response_time_ms": 100,
            "error_rate": 0.01,
            "availability": 0.999
        }
        self.performance_history = deque(maxlen=10000)
        
        # Configuration
        self.enable_predictive_scheduling = True
        self.enable_auto_optimization = True
        self.enable_smart_routing = True
        self.batch_size = 10
        self.scheduling_interval = 1.0  # seconds
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        # Add to dependency graph if has dependencies
        if task.dependencies:
            self.task_dependencies.add_node(task.task_id, task=task)
            for dep in task.dependencies:
                self.task_dependencies.add_edge(dep, task.task_id)
        
        # Check if task can be scheduled immediately
        if await self._can_schedule_task(task):
            await self._schedule_task(task)
        else:
            # Add to priority queue
            heapq.heappush(self.task_queue, (task.priority.value, task.created_at, task))
            logger.info(f"Task {task.task_id} queued with priority {task.priority.name}")
        
        return task.task_id
    
    async def _can_schedule_task(self, task: Task) -> bool:
        """Check if task can be scheduled immediately"""
        # Check dependencies
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
        
        # Check resource availability
        available_servers = await self._find_suitable_servers(task)
        return len(available_servers) > 0
    
    async def _find_suitable_servers(self, task: Task) -> List[ServerNode]:
        """Find servers suitable for task execution"""
        suitable_servers = []
        
        for server in self.load_balancer.server_nodes.values():
            if not server.is_healthy:
                continue
            
            # Check anti-affinity
            if server.id in task.anti_affinity_servers:
                continue
            
            # Check resource availability
            if await self._server_has_resources(server, task):
                suitable_servers.append(server)
        
        # Sort by preference
        if task.affinity_server and task.affinity_server in [s.id for s in suitable_servers]:
            # Move affinity server to front
            suitable_servers.sort(key=lambda s: s.id != task.affinity_server)
        else:
            # Sort by load
            suitable_servers.sort(key=lambda s: s.load_score)
        
        return suitable_servers
    
    async def _server_has_resources(self, server: ServerNode, task: Task) -> bool:
        """Check if server has required resources"""
        # Get current server resources
        server_resources = await self.resource_optimizer._calculate_available_resources()
        
        if server.id not in server_resources:
            return False
        
        available = server_resources[server.id]
        
        # Check each resource requirement
        for resource_type, required in task.resource_requirements.items():
            if available.get(resource_type, 0) < required:
                return False
        
        return True
    
    async def _schedule_task(self, task: Task):
        """Schedule task for execution"""
        # Find optimal server
        server = await self._select_optimal_server(task)
        
        if not server:
            logger.warning(f"No suitable server found for task {task.task_id}")
            heapq.heappush(self.task_queue, (task.priority.value, task.created_at, task))
            return
        
        # Assign task
        task.assigned_server = server.id
        task.started_at = datetime.utcnow()
        self.active_tasks[task.task_id] = task
        
        # Update server load
        server.current_connections += 1
        
        # Create resource allocation
        await self.resource_optimizer.optimize_resource_allocation(
            [{"id": task.task_id, "type": "scheduled", "priority": task.priority.value}],
            AllocationStrategy.ML_BASED
        )
        
        # Start task execution monitoring
        asyncio.create_task(self._monitor_task_execution(task))
        
        logger.info(f"Task {task.task_id} scheduled on server {server.id}")
    
    async def _select_optimal_server(self, task: Task) -> Optional[ServerNode]:
        """Select optimal server for task execution"""
        suitable_servers = await self._find_suitable_servers(task)
        
        if not suitable_servers:
            return None
        
        if not self.enable_smart_routing:
            return suitable_servers[0]
        
        # Smart routing based on multiple factors
        scores = {}
        
        for server in suitable_servers:
            score = 100.0
            
            # Factor 1: Current load
            score -= server.load_score * 30
            
            # Factor 2: Historical performance for similar tasks
            similar_perf = await self._get_server_performance_score(server.id, task)
            score += similar_perf * 20
            
            # Factor 3: Predicted task duration on this server
            predicted_duration = await self.task_predictor.predict_duration(task, server)
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.utcnow()).total_seconds()
                if predicted_duration.total_seconds() > time_to_deadline:
                    score -= 50  # Penalty for missing deadline
            
            # Factor 4: Resource efficiency
            resource_efficiency = await self._calculate_resource_efficiency(server, task)
            score += resource_efficiency * 15
            
            # Factor 5: Network locality (if applicable)
            if task.metadata.get('data_location'):
                locality_score = self._calculate_locality_score(server, task.metadata['data_location'])
                score += locality_score * 10
            
            scores[server.id] = max(0, score)
        
        # Select server with highest score
        best_server_id = max(scores, key=scores.get)
        return next(s for s in suitable_servers if s.id == best_server_id)
    
    async def _get_server_performance_score(self, server_id: str, task: Task) -> float:
        """Get historical performance score for server on similar tasks"""
        similar_tasks = [
            t for t in self.completed_tasks
            if t.assigned_server == server_id and
            t.metadata.get('type') == task.metadata.get('type')
        ]
        
        if not similar_tasks:
            return 50.0  # Neutral score
        
        # Calculate average performance score
        perf_scores = [t.performance_score for t in similar_tasks[-10:]]  # Last 10 similar tasks
        return statistics.mean(perf_scores)
    
    async def _calculate_resource_efficiency(self, server: ServerNode, task: Task) -> float:
        """Calculate resource utilization efficiency for task on server"""
        server_resources = await self.resource_optimizer._calculate_available_resources()
        available = server_resources.get(server.id, {})
        
        efficiency_scores = []
        
        for resource_type, required in task.resource_requirements.items():
            available_amount = available.get(resource_type, 0)
            if available_amount > 0:
                # Efficiency is better when resource usage is closer to available
                # without wasting resources
                utilization = required / available_amount
                efficiency = 1 - abs(0.7 - utilization)  # Optimal at 70% utilization
                efficiency_scores.append(efficiency)
        
        return statistics.mean(efficiency_scores) * 100 if efficiency_scores else 0
    
    def _calculate_locality_score(self, server: ServerNode, data_location: str) -> float:
        """Calculate data locality score"""
        # Simplified locality calculation
        if server.region == data_location:
            return 100.0
        elif server.region.split('-')[0] == data_location.split('-')[0]:  # Same continent
            return 50.0
        else:
            return 0.0
    
    async def _monitor_task_execution(self, task: Task):
        """Monitor task execution and handle completion"""
        try:
            # Simulate task execution
            predicted_duration = await self.task_predictor.predict_duration(task)
            await asyncio.sleep(predicted_duration.total_seconds())
            
            # Mark task as completed
            task.completed_at = datetime.utcnow()
            task.actual_duration = task.completed_at - task.started_at
            
            # Calculate performance score
            if task.deadline:
                if task.completed_at <= task.deadline:
                    task.performance_score = 100.0
                else:
                    overrun = (task.completed_at - task.deadline).total_seconds()
                    task.performance_score = max(0, 100 - overrun / 60)  # -1 point per minute late
            else:
                # Score based on prediction accuracy
                predicted_seconds = predicted_duration.total_seconds()
                actual_seconds = task.actual_duration.total_seconds()
                accuracy = 1 - abs(predicted_seconds - actual_seconds) / actual_seconds
                task.performance_score = accuracy * 100
            
            # Update tracking
            del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            # Update predictor with actual duration
            await self.task_predictor.update_prediction_model(task)
            
            # Free up server resources
            if task.assigned_server in self.load_balancer.server_nodes:
                server = self.load_balancer.server_nodes[task.assigned_server]
                server.current_connections = max(0, server.current_connections - 1)
            
            # Check for dependent tasks
            if task.task_id in self.task_dependencies:
                await self._schedule_dependent_tasks(task.task_id)
            
            logger.info(
                f"Task {task.task_id} completed in {task.actual_duration}, "
                f"performance score: {task.performance_score:.1f}"
            )
            
        except Exception as e:
            logger.error(f"Error monitoring task {task.task_id}: {e}")
            task.performance_score = 0.0
    
    async def _schedule_dependent_tasks(self, completed_task_id: str):
        """Schedule tasks that were waiting for this task"""
        if completed_task_id not in self.task_dependencies:
            return
        
        dependent_tasks = list(self.task_dependencies.successors(completed_task_id))
        
        for dep_task_id in dependent_tasks:
            # Check if all dependencies are satisfied
            dependencies = list(self.task_dependencies.predecessors(dep_task_id))
            all_completed = all(dep in [t.task_id for t in self.completed_tasks] for dep in dependencies)
            
            if all_completed:
                # Find the task in queue and schedule it
                for i, (priority, created_at, task) in enumerate(self.task_queue):
                    if task.task_id == dep_task_id:
                        self.task_queue.pop(i)
                        heapq.heapify(self.task_queue)
                        
                        await self._schedule_task(task)
                        break
    
    async def process_task_queue(self):
        """Continuously process queued tasks"""
        while True:
            try:
                if self.task_queue:
                    # Batch processing for efficiency
                    batch = []
                    batch_size = min(self.batch_size, len(self.task_queue))
                    
                    for _ in range(batch_size):
                        if self.task_queue:
                            _, _, task = heapq.heappop(self.task_queue)
                            if await self._can_schedule_task(task):
                                batch.append(task)
                            else:
                                # Put back in queue
                                heapq.heappush(self.task_queue, (task.priority.value, task.created_at, task))
                    
                    # Schedule batch
                    for task in batch:
                        await self._schedule_task(task)
                
                # Collect load snapshot
                await self._collect_load_snapshot()
                
                # Run optimizations if enabled
                if self.enable_auto_optimization:
                    await self._run_auto_optimization()
                
                await asyncio.sleep(self.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in task queue processing: {e}")
                await asyncio.sleep(5)
    
    async def _collect_load_snapshot(self):
        """Collect current load snapshot"""
        # Calculate metrics
        total_tasks = len(self.active_tasks) + len(self.task_queue)
        active_tasks = len(self.active_tasks)
        queued_tasks = len(self.task_queue)
        
        # Response times
        response_times = []
        for server in self.load_balancer.server_nodes.values():
            if hasattr(server, 'avg_response_time'):
                response_times.append(server.avg_response_time)
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Throughput (tasks completed per minute)
        recent_completed = [
            t for t in self.completed_tasks
            if t.completed_at and t.completed_at > datetime.utcnow() - timedelta(minutes=1)
        ]
        throughput = len(recent_completed)
        
        # Error rate (simplified)
        error_rate = sum(1 for t in recent_completed if t.performance_score < 50) / len(recent_completed) if recent_completed else 0
        
        # Resource utilization
        metrics = await self.resource_optimizer.collect_system_metrics()
        resource_utilization = {
            ResourceType.CPU: metrics.cpu_percent,
            ResourceType.MEMORY: metrics.memory_percent,
            ResourceType.NETWORK: min(metrics.network_throughput_mbps / 1000 * 100, 100),  # Assume 1Gbps max
        }
        
        # Server loads
        server_loads = {
            server.id: server.load_score
            for server in self.load_balancer.server_nodes.values()
        }
        
        # Detect pattern
        pattern = await self.pattern_detector.detect_pattern(self.load_history)
        
        # Check for anomalies
        anomaly_score = await self.anomaly_detector.calculate_anomaly_score(
            avg_response_time, throughput, error_rate
        )
        
        snapshot = LoadSnapshot(
            timestamp=datetime.utcnow(),
            total_tasks=total_tasks,
            active_tasks=active_tasks,
            queued_tasks=queued_tasks,
            avg_response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            resource_utilization=resource_utilization,
            server_loads=server_loads,
            pattern=pattern,
            anomaly_score=anomaly_score
        )
        
        self.load_history.append(snapshot)
    
    async def _run_auto_optimization(self):
        """Run automatic optimization based on current load"""
        if not self.load_history:
            return
        
        latest_snapshot = self.load_history[-1]
        
        # Check if optimization needed
        optimization_needed = False
        recommendations = []
        
        # High queue length
        if latest_snapshot.queued_tasks > 50:
            optimization_needed = True
            recommendations.append({
                "type": "scale_up",
                "reason": f"High queue length: {latest_snapshot.queued_tasks} tasks",
                "urgency": "high"
            })
        
        # Poor response time
        if latest_snapshot.avg_response_time > self.sla_targets['response_time_ms']:
            optimization_needed = True
            recommendations.append({
                "type": "optimize_routing",
                "reason": f"Response time {latest_snapshot.avg_response_time:.1f}ms exceeds SLA",
                "urgency": "medium"
            })
        
        # Load imbalance
        if latest_snapshot.server_loads:
            load_variance = statistics.variance(latest_snapshot.server_loads.values())
            if load_variance > 0.1:  # 10% variance threshold
                optimization_needed = True
                recommendations.append({
                    "type": "rebalance",
                    "reason": f"Load imbalance detected (variance: {load_variance:.3f})",
                    "urgency": "medium"
                })
        
        # Anomaly detected
        if latest_snapshot.anomaly_score > 0.7:
            optimization_needed = True
            recommendations.append({
                "type": "investigate_anomaly",
                "reason": f"Anomaly detected (score: {latest_snapshot.anomaly_score:.2f})",
                "urgency": "high"
            })
        
        if optimization_needed:
            await self.optimization_engine.optimize(
                latest_snapshot,
                recommendations,
                self.load_balancer,
                self.resource_optimizer
            )
    
    async def predict_future_load(self, horizon_minutes: int = 30) -> List[LoadSnapshot]:
        """Predict future load patterns"""
        if len(self.load_history) < 60:  # Need at least 1 hour of data
            return []
        
        return await self.load_forecaster.forecast(
            list(self.load_history),
            horizon_minutes
        )
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.load_history:
            return {"status": "no_data"}
        
        # Calculate metrics over last hour
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_snapshots = [s for s in self.load_history if s.timestamp > hour_ago]
        
        if not recent_snapshots:
            return {"status": "insufficient_data"}
        
        # Average metrics
        avg_response_time = statistics.mean(s.avg_response_time for s in recent_snapshots)
        avg_throughput = statistics.mean(s.throughput for s in recent_snapshots)
        avg_error_rate = statistics.mean(s.error_rate for s in recent_snapshots)
        avg_queue_length = statistics.mean(s.queued_tasks for s in recent_snapshots)
        
        # SLA compliance
        sla_compliance = {
            "response_time": sum(1 for s in recent_snapshots if s.avg_response_time <= self.sla_targets['response_time_ms']) / len(recent_snapshots),
            "error_rate": sum(1 for s in recent_snapshots if s.error_rate <= self.sla_targets['error_rate']) / len(recent_snapshots),
        }
        
        # Task completion statistics
        completed_tasks_hour = [
            t for t in self.completed_tasks
            if t.completed_at and t.completed_at > hour_ago
        ]
        
        task_stats = {
            "total_completed": len(completed_tasks_hour),
            "avg_duration": statistics.mean(
                t.actual_duration.total_seconds()
                for t in completed_tasks_hour
                if t.actual_duration
            ) if completed_tasks_hour else 0,
            "on_time_completion": sum(
                1 for t in completed_tasks_hour
                if t.deadline and t.completed_at <= t.deadline
            ) / len([t for t in completed_tasks_hour if t.deadline]) if any(t.deadline for t in completed_tasks_hour) else 1.0,
        }
        
        # Resource utilization
        avg_resource_util = {}
        for resource_type in ResourceType:
            if resource_type in recent_snapshots[0].resource_utilization:
                avg_resource_util[resource_type.value] = statistics.mean(
                    s.resource_utilization.get(resource_type, 0)
                    for s in recent_snapshots
                )
        
        # Pattern analysis
        patterns = defaultdict(int)
        for s in recent_snapshots:
            patterns[s.pattern.value] += 1
        
        dominant_pattern = max(patterns, key=patterns.get) if patterns else "unknown"
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": "last_hour",
            "metrics": {
                "avg_response_time_ms": avg_response_time,
                "avg_throughput_per_min": avg_throughput,
                "avg_error_rate": avg_error_rate,
                "avg_queue_length": avg_queue_length,
            },
            "sla_compliance": sla_compliance,
            "task_statistics": task_stats,
            "resource_utilization": avg_resource_util,
            "load_patterns": dict(patterns),
            "dominant_pattern": dominant_pattern,
            "optimization_recommendations": await self._generate_optimization_recommendations(recent_snapshots),
            "predicted_load": await self._get_load_predictions(),
        }
        
        return report
    
    async def _generate_optimization_recommendations(self, snapshots: List[LoadSnapshot]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on recent performance"""
        recommendations = []
        
        # Analyze trends
        response_times = [s.avg_response_time for s in snapshots]
        queue_lengths = [s.queued_tasks for s in snapshots]
        
        # Response time trend
        if len(response_times) > 10:
            trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            if trend > 1:  # Increasing by >1ms per snapshot
                recommendations.append({
                    "type": "performance_degradation",
                    "description": "Response times are increasing",
                    "action": "Consider scaling up or optimizing slow operations",
                    "priority": "high"
                })
        
        # Queue buildup
        if queue_lengths and statistics.mean(queue_lengths) > 20:
            recommendations.append({
                "type": "queue_backlog",
                "description": f"Average queue length is {statistics.mean(queue_lengths):.1f}",
                "action": "Increase processing capacity or optimize task scheduling",
                "priority": "medium"
            })
        
        # Resource bottlenecks
        for snapshot in snapshots[-5:]:  # Check recent snapshots
            for resource_type, utilization in snapshot.resource_utilization.items():
                if utilization > 80:
                    recommendations.append({
                        "type": "resource_bottleneck",
                        "description": f"{resource_type.value} utilization at {utilization:.1f}%",
                        "action": f"Optimize {resource_type.value} usage or add more capacity",
                        "priority": "high" if utilization > 90 else "medium"
                    })
                    break
        
        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = (rec['type'], rec['description'])
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    async def _get_load_predictions(self) -> Dict[str, Any]:
        """Get load predictions for planning"""
        predictions = await self.predict_future_load(30)
        
        if not predictions:
            return {"status": "insufficient_data"}
        
        # Summarize predictions
        predicted_peaks = max(p.total_tasks for p in predictions) if predictions else 0
        predicted_avg = statistics.mean(p.total_tasks for p in predictions) if predictions else 0
        
        return {
            "next_30_min": {
                "predicted_peak_tasks": predicted_peaks,
                "predicted_avg_tasks": predicted_avg,
                "confidence": 0.85  # Placeholder confidence score
            }
        }


class LoadPatternDetector:
    """Detects load patterns from historical data"""
    
    async def detect_pattern(self, history: deque) -> LoadPattern:
        """Detect the current load pattern"""
        if len(history) < 10:
            return LoadPattern.RANDOM
        
        # Extract task counts
        task_counts = [s.total_tasks for s in list(history)[-30:]]
        
        if not task_counts:
            return LoadPattern.RANDOM
        
        # Calculate statistics
        mean_load = statistics.mean(task_counts)
        std_load = statistics.stdev(task_counts) if len(task_counts) > 1 else 0
        
        # Detect patterns
        if std_load < mean_load * 0.1:  # Low variance
            return LoadPattern.STEADY
        
        # Check for trend
        if len(task_counts) > 5:
            trend = np.polyfit(range(len(task_counts)), task_counts, 1)[0]
            
            if trend > mean_load * 0.02:  # 2% increase per interval
                return LoadPattern.GRADUAL_INCREASE
            elif trend < -mean_load * 0.02:
                return LoadPattern.GRADUAL_DECREASE
        
        # Check for periodicity
        if len(task_counts) > 20:
            # Simple periodicity check using autocorrelation
            autocorr = np.correlate(task_counts, task_counts, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            if peaks and peaks[0] > 5:  # Period of at least 5 intervals
                return LoadPattern.PERIODIC
        
        # Check for bursts
        burst_threshold = mean_load + 2 * std_load
        burst_count = sum(1 for load in task_counts if load > burst_threshold)
        
        if burst_count > len(task_counts) * 0.1:  # More than 10% bursts
            return LoadPattern.BURST
        
        return LoadPattern.RANDOM


class AnomalyDetector:
    """Detects anomalies in system behavior"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.5  # Standard deviations
    
    async def calculate_anomaly_score(
        self,
        response_time: float,
        throughput: float,
        error_rate: float
    ) -> float:
        """Calculate anomaly score (0-1)"""
        if not self.baseline_metrics:
            # Initialize baselines
            self.baseline_metrics = {
                "response_time": {"mean": response_time, "std": response_time * 0.2},
                "throughput": {"mean": throughput, "std": throughput * 0.3},
                "error_rate": {"mean": error_rate, "std": error_rate * 0.5}
            }
            return 0.0
        
        scores = []
        
        # Response time anomaly
        rt_baseline = self.baseline_metrics["response_time"]
        rt_zscore = abs((response_time - rt_baseline["mean"]) / (rt_baseline["std"] + 1e-6))
        scores.append(min(rt_zscore / self.anomaly_threshold, 1.0))
        
        # Throughput anomaly
        tp_baseline = self.baseline_metrics["throughput"]
        tp_zscore = abs((throughput - tp_baseline["mean"]) / (tp_baseline["std"] + 1e-6))
        scores.append(min(tp_zscore / self.anomaly_threshold, 1.0))
        
        # Error rate anomaly
        er_baseline = self.baseline_metrics["error_rate"]
        er_zscore = abs((error_rate - er_baseline["mean"]) / (er_baseline["std"] + 1e-6))
        scores.append(min(er_zscore / self.anomaly_threshold, 1.0))
        
        # Update baselines with exponential moving average
        alpha = 0.1
        self.baseline_metrics["response_time"]["mean"] = (
            alpha * response_time + (1 - alpha) * rt_baseline["mean"]
        )
        self.baseline_metrics["throughput"]["mean"] = (
            alpha * throughput + (1 - alpha) * tp_baseline["mean"]
        )
        self.baseline_metrics["error_rate"]["mean"] = (
            alpha * error_rate + (1 - alpha) * er_baseline["mean"]
        )
        
        # Combined anomaly score
        return statistics.mean(scores)


class TaskDurationPredictor:
    """Predicts task execution duration"""
    
    def __init__(self):
        self.duration_history = defaultdict(list)
        self.server_performance = defaultdict(lambda: {"factor": 1.0})
    
    async def predict_duration(
        self,
        task: Task,
        server: Optional[ServerNode] = None
    ) -> timedelta:
        """Predict task duration"""
        task_type = task.metadata.get('type', 'default')
        
        # Get historical durations for this task type
        history = self.duration_history.get(task_type, [])
        
        if not history:
            # Use estimated duration
            base_duration = task.estimated_duration
        else:
            # Use historical average
            avg_seconds = statistics.mean(history[-20:])  # Last 20 executions
            base_duration = timedelta(seconds=avg_seconds)
        
        # Adjust for server performance
        if server:
            perf_factor = self.server_performance[server.id]["factor"]
            adjusted_seconds = base_duration.total_seconds() * perf_factor
            return timedelta(seconds=adjusted_seconds)
        
        return base_duration
    
    async def update_prediction_model(self, completed_task: Task):
        """Update prediction model with actual duration"""
        if not completed_task.actual_duration:
            return
        
        task_type = completed_task.metadata.get('type', 'default')
        actual_seconds = completed_task.actual_duration.total_seconds()
        
        # Update history
        self.duration_history[task_type].append(actual_seconds)
        
        # Limit history size
        if len(self.duration_history[task_type]) > 1000:
            self.duration_history[task_type] = self.duration_history[task_type][-1000:]
        
        # Update server performance factor
        if completed_task.assigned_server and completed_task.estimated_duration:
            expected_seconds = completed_task.estimated_duration.total_seconds()
            if expected_seconds > 0:
                perf_factor = actual_seconds / expected_seconds
                
                # Exponential moving average
                current_factor = self.server_performance[completed_task.assigned_server]["factor"]
                new_factor = 0.2 * perf_factor + 0.8 * current_factor
                self.server_performance[completed_task.assigned_server]["factor"] = new_factor


class LoadForecaster:
    """Forecasts future load patterns"""
    
    async def forecast(
        self,
        history: List[LoadSnapshot],
        horizon_minutes: int
    ) -> List[LoadSnapshot]:
        """Forecast future load snapshots"""
        if len(history) < 60:
            return []
        
        # Extract time series data
        task_counts = [s.total_tasks for s in history]
        
        # Simple moving average forecast
        window_size = min(30, len(task_counts) // 2)
        recent_avg = statistics.mean(task_counts[-window_size:])
        
        # Calculate trend
        if len(task_counts) > window_size:
            trend = np.polyfit(range(window_size), task_counts[-window_size:], 1)[0]
        else:
            trend = 0
        
        # Generate forecasts
        forecasts = []
        base_time = history[-1].timestamp
        
        for i in range(horizon_minutes):
            # Predict task count
            predicted_tasks = max(0, recent_avg + trend * i)
            
            # Add some randomness
            predicted_tasks += np.random.normal(0, recent_avg * 0.1)
            
            # Create forecast snapshot
            forecast = LoadSnapshot(
                timestamp=base_time + timedelta(minutes=i+1),
                total_tasks=int(predicted_tasks),
                active_tasks=int(predicted_tasks * 0.8),  # Assume 80% active
                queued_tasks=int(predicted_tasks * 0.2),
                avg_response_time=history[-1].avg_response_time,  # Assume constant
                throughput=predicted_tasks / 5,  # Rough estimate
                error_rate=history[-1].error_rate,
                resource_utilization=history[-1].resource_utilization,
                server_loads=history[-1].server_loads,
                pattern=LoadPattern.RANDOM,
                anomaly_score=0.0
            )
            
            forecasts.append(forecast)
        
        return forecasts


class OptimizationEngine:
    """Handles system optimization based on load conditions"""
    
    async def optimize(
        self,
        snapshot: LoadSnapshot,
        recommendations: List[Dict[str, Any]],
        load_balancer: EnhancedLoadBalancer,
        resource_optimizer: ResourceOptimizer
    ):
        """Execute optimization based on recommendations"""
        for rec in recommendations:
            try:
                if rec['type'] == 'scale_up':
                    await self._handle_scale_up(load_balancer, resource_optimizer)
                elif rec['type'] == 'optimize_routing':
                    await self._optimize_routing(load_balancer)
                elif rec['type'] == 'rebalance':
                    await self._rebalance_load(load_balancer)
                elif rec['type'] == 'investigate_anomaly':
                    await self._investigate_anomaly(snapshot)
                
                logger.info(f"Applied optimization: {rec['type']} - {rec['reason']}")
                
            except Exception as e:
                logger.error(f"Failed to apply optimization {rec['type']}: {e}")
    
    async def _handle_scale_up(
        self,
        load_balancer: EnhancedLoadBalancer,
        resource_optimizer: ResourceOptimizer
    ):
        """Handle scale-up recommendation"""
        # Trigger auto-scaler
        metrics = {
            "cpu_usage": 85.0,  # Simulate high load
            "queue_size": 100
        }
        
        for service in resource_optimizer.auto_scaler.scaling_policies:
            await resource_optimizer.auto_scaler.evaluate_scaling(service, metrics)
    
    async def _optimize_routing(self, load_balancer: EnhancedLoadBalancer):
        """Optimize request routing"""
        # Switch to more intelligent routing strategy
        if hasattr(load_balancer, 'custom_strategies'):
            # Enable ML-based routing
            load_balancer.strategy = "ml_optimized"
    
    async def _rebalance_load(self, load_balancer: EnhancedLoadBalancer):
        """Rebalance load across servers"""
        # Get server loads
        server_loads = {
            server.id: server.current_connections
            for server in load_balancer.server_nodes.values()
        }
        
        if not server_loads:
            return
        
        avg_load = statistics.mean(server_loads.values())
        
        # Adjust server weights
        for server_id, load in server_loads.items():
            server = load_balancer.server_nodes[server_id]
            
            if load > avg_load * 1.2:  # Overloaded
                server.weight *= 0.8
            elif load < avg_load * 0.8:  # Underutilized
                server.weight *= 1.2
    
    async def _investigate_anomaly(self, snapshot: LoadSnapshot):
        """Investigate detected anomaly"""
        logger.warning(
            f"Investigating anomaly at {snapshot.timestamp}: "
            f"score={snapshot.anomaly_score:.2f}, "
            f"response_time={snapshot.avg_response_time:.1f}ms, "
            f"error_rate={snapshot.error_rate:.3f}"
        )
        
        # In production, this would trigger detailed diagnostics