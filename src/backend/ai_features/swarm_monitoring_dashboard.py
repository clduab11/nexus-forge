"""
Swarm Monitoring Dashboard and Analytics
Real-time monitoring, visualization, and analytics for swarm intelligence system
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

from nexus_forge.core.cache import CacheStrategy, RedisCache
from nexus_forge.core.monitoring import get_logger
from nexus_forge.integrations.supabase.coordination_client import SupabaseCoordinationClient

from .swarm_intelligence import (
    SwarmAgent,
    SwarmMessage,
    SwarmObjective,
    SwarmTask,
    SwarmResult,
    CommunicationType,
    EmergenceBehavior,
    SwarmPattern,
)

logger = get_logger(__name__)


# Monitoring Data Classes
@dataclass
class SwarmMetrics:
    """Real-time swarm metrics"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    swarm_id: str = ""
    agent_count: int = 0
    active_agents: int = 0
    task_count: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0
    swarm_efficiency: float = 0.0
    emergence_score: float = 0.0
    communication_volume: int = 0
    coordination_overhead: float = 0.0
    pattern: SwarmPattern = SwarmPattern.ADAPTIVE


@dataclass
class AgentMetrics:
    """Individual agent performance metrics"""
    agent_id: str = ""
    agent_type: str = ""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_time: float = 0.0
    current_load: float = 0.0
    performance_score: float = 1.0
    communication_count: int = 0
    collaboration_score: float = 0.0
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)


@dataclass
class EmergenceMetrics:
    """Emergence pattern detection metrics"""
    pattern_type: EmergenceBehavior = EmergenceBehavior.CONSENSUS
    detection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strength: float = 0.0
    participating_agents: List[str] = field(default_factory=list)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationMetrics:
    """Communication pattern metrics"""
    message_count: int = 0
    broadcast_count: int = 0
    unicast_count: int = 0
    multicast_count: int = 0
    pheromone_count: int = 0
    average_latency: float = 0.0
    message_size_avg: float = 0.0
    network_utilization: float = 0.0
    hotspot_agents: List[str] = field(default_factory=list)


class SwarmMonitor:
    """Core monitoring system for swarm intelligence"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.emergence_history: deque = deque(maxlen=100)
        self.communication_metrics = CommunicationMetrics()
        
        # Time series data
        self.time_series_data = {
            "efficiency": deque(maxlen=buffer_size),
            "task_completion_rate": deque(maxlen=buffer_size),
            "emergence_score": deque(maxlen=buffer_size),
            "agent_utilization": deque(maxlen=buffer_size),
            "communication_volume": deque(maxlen=buffer_size),
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "efficiency_low": 0.5,
            "failure_rate_high": 0.3,
            "communication_overload": 1000,
            "agent_overload": 0.9,
            "emergence_anomaly": 0.1,
        }
        
        # Active alerts
        self.active_alerts: List[Dict[str, Any]] = []
        
    async def collect_metrics(
        self,
        swarm_id: str,
        agents: List[SwarmAgent],
        tasks: List[SwarmTask],
        messages: List[SwarmMessage],
        pattern: SwarmPattern
    ) -> SwarmMetrics:
        """Collect current swarm metrics"""
        metrics = SwarmMetrics(
            swarm_id=swarm_id,
            agent_count=len(agents),
            pattern=pattern
        )
        
        # Agent metrics
        metrics.active_agents = sum(1 for a in agents if a.status != "idle")
        
        # Task metrics
        metrics.task_count = len(tasks)
        metrics.completed_tasks = sum(1 for t in tasks if t.status == "completed")
        metrics.failed_tasks = sum(1 for t in tasks if t.status == "failed")
        
        # Calculate average task duration
        durations = []
        for task in tasks:
            if task.completed_at and task.started_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                durations.append(duration)
        
        if durations:
            metrics.average_task_duration = np.mean(durations)
        
        # Swarm efficiency
        if metrics.task_count > 0:
            metrics.swarm_efficiency = metrics.completed_tasks / metrics.task_count
        
        # Communication metrics
        metrics.communication_volume = len(messages)
        
        # Emergence score (simplified)
        metrics.emergence_score = await self._calculate_emergence_score(agents, messages)
        
        # Coordination overhead
        metrics.coordination_overhead = await self._calculate_coordination_overhead(
            agents, messages, tasks
        )
        
        # Store metrics
        self.metrics_buffer.append(metrics)
        
        # Update time series
        self._update_time_series(metrics)
        
        # Check for alerts
        await self._check_alerts(metrics)
        
        return metrics
    
    async def update_agent_metrics(self, agent: SwarmAgent):
        """Update metrics for individual agent"""
        if agent.id not in self.agent_metrics:
            self.agent_metrics[agent.id] = AgentMetrics(
                agent_id=agent.id,
                agent_type=agent.type
            )
        
        metrics = self.agent_metrics[agent.id]
        
        # Update basic metrics
        metrics.tasks_completed = len(agent.completed_tasks)
        metrics.current_load = agent.load
        metrics.performance_score = agent.performance_score
        metrics.position = agent.position.tolist() if hasattr(agent.position, 'tolist') else []
        metrics.velocity = agent.velocity.tolist() if hasattr(agent.velocity, 'tolist') else []
        metrics.neighbors = list(agent.neighbors)
        
        # Calculate collaboration score
        if metrics.neighbors:
            metrics.collaboration_score = len(metrics.neighbors) / 10.0  # Normalized
        
    async def detect_emergence(
        self,
        agents: List[SwarmAgent],
        messages: List[SwarmMessage],
        behavior: EmergenceBehavior,
        strength: float
    ) -> EmergenceMetrics:
        """Record detected emergence pattern"""
        emergence = EmergenceMetrics(
            pattern_type=behavior,
            strength=strength,
            participating_agents=[a.id for a in agents if a.status == "busy"]
        )
        
        # Store in history
        self.emergence_history.append(emergence)
        
        # Check for anomalies
        if strength < self.alert_thresholds["emergence_anomaly"]:
            await self._raise_alert(
                "emergence_anomaly",
                f"Weak emergence detected: {behavior.value} with strength {strength:.2f}"
            )
        
        return emergence
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization"""
        latest_metrics = self.metrics_buffer[-1] if self.metrics_buffer else None
        
        return {
            "overview": self._get_overview_data(latest_metrics),
            "agents": self._get_agent_data(),
            "tasks": self._get_task_data(),
            "emergence": self._get_emergence_data(),
            "communication": self._get_communication_data(),
            "time_series": self._get_time_series_data(),
            "alerts": self.active_alerts,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_overview_data(self, metrics: Optional[SwarmMetrics]) -> Dict[str, Any]:
        """Get overview metrics"""
        if not metrics:
            return {}
            
        return {
            "swarm_id": metrics.swarm_id,
            "pattern": metrics.pattern.value,
            "agent_count": metrics.agent_count,
            "active_agents": metrics.active_agents,
            "task_progress": {
                "total": metrics.task_count,
                "completed": metrics.completed_tasks,
                "failed": metrics.failed_tasks,
                "in_progress": metrics.task_count - metrics.completed_tasks - metrics.failed_tasks
            },
            "efficiency": metrics.swarm_efficiency,
            "emergence_score": metrics.emergence_score,
            "coordination_overhead": metrics.coordination_overhead
        }
    
    def _get_agent_data(self) -> List[Dict[str, Any]]:
        """Get agent performance data"""
        agent_data = []
        
        for agent_id, metrics in self.agent_metrics.items():
            agent_data.append({
                "id": agent_id,
                "type": metrics.agent_type,
                "performance": metrics.performance_score,
                "load": metrics.current_load,
                "tasks_completed": metrics.tasks_completed,
                "collaboration_score": metrics.collaboration_score,
                "position": metrics.position,
                "neighbors": len(metrics.neighbors)
            })
        
        return sorted(agent_data, key=lambda x: x["performance"], reverse=True)
    
    def _get_task_data(self) -> Dict[str, Any]:
        """Get task execution data"""
        if not self.metrics_buffer:
            return {}
            
        recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 samples
        
        return {
            "completion_rate": np.mean([m.swarm_efficiency for m in recent_metrics]),
            "average_duration": np.mean([m.average_task_duration for m in recent_metrics]),
            "failure_rate": np.mean([
                m.failed_tasks / m.task_count if m.task_count > 0 else 0
                for m in recent_metrics
            ]),
            "throughput": self._calculate_throughput(recent_metrics)
        }
    
    def _get_emergence_data(self) -> List[Dict[str, Any]]:
        """Get emergence pattern data"""
        emergence_data = []
        
        # Count patterns
        pattern_counts = defaultdict(int)
        pattern_strengths = defaultdict(list)
        
        for emergence in self.emergence_history:
            pattern_counts[emergence.pattern_type.value] += 1
            pattern_strengths[emergence.pattern_type.value].append(emergence.strength)
        
        for pattern, count in pattern_counts.items():
            emergence_data.append({
                "pattern": pattern,
                "count": count,
                "average_strength": np.mean(pattern_strengths[pattern]),
                "max_strength": max(pattern_strengths[pattern])
            })
        
        return emergence_data
    
    def _get_communication_data(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            "total_messages": self.communication_metrics.message_count,
            "message_types": {
                "broadcast": self.communication_metrics.broadcast_count,
                "unicast": self.communication_metrics.unicast_count,
                "multicast": self.communication_metrics.multicast_count,
                "pheromone": self.communication_metrics.pheromone_count
            },
            "average_latency": self.communication_metrics.average_latency,
            "network_utilization": self.communication_metrics.network_utilization,
            "hotspots": self.communication_metrics.hotspot_agents[:5]  # Top 5
        }
    
    def _get_time_series_data(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get time series data for charts"""
        time_series = {}
        
        for metric_name, data_points in self.time_series_data.items():
            if data_points:
                # Convert to (timestamp, value) pairs
                time_series[metric_name] = [
                    (i, value) for i, value in enumerate(data_points)
                ]
        
        return time_series
    
    def _update_time_series(self, metrics: SwarmMetrics):
        """Update time series data"""
        self.time_series_data["efficiency"].append(metrics.swarm_efficiency)
        
        completion_rate = (
            metrics.completed_tasks / metrics.task_count
            if metrics.task_count > 0 else 0
        )
        self.time_series_data["task_completion_rate"].append(completion_rate)
        
        self.time_series_data["emergence_score"].append(metrics.emergence_score)
        
        utilization = metrics.active_agents / metrics.agent_count if metrics.agent_count > 0 else 0
        self.time_series_data["agent_utilization"].append(utilization)
        
        self.time_series_data["communication_volume"].append(metrics.communication_volume)
    
    async def _calculate_emergence_score(
        self,
        agents: List[SwarmAgent],
        messages: List[SwarmMessage]
    ) -> float:
        """Calculate overall emergence score"""
        if not agents:
            return 0.0
            
        # Factors for emergence
        scores = []
        
        # Position clustering
        positions = np.array([a.position for a in agents if hasattr(a, 'position')])
        if len(positions) > 1:
            clustering_score = 1.0 - np.std(positions) / (np.mean(positions) + 1e-6)
            scores.append(clustering_score)
        
        # Velocity alignment (for flocking)
        velocities = np.array([a.velocity for a in agents if hasattr(a, 'velocity')])
        if len(velocities) > 1:
            alignment_score = 1.0 - np.std(velocities) / (np.mean(np.abs(velocities)) + 1e-6)
            scores.append(alignment_score)
        
        # Communication patterns
        if messages:
            broadcast_ratio = sum(1 for m in messages if m.type == CommunicationType.BROADCAST) / len(messages)
            scores.append(broadcast_ratio)
        
        return np.mean(scores) if scores else 0.0
    
    async def _calculate_coordination_overhead(
        self,
        agents: List[SwarmAgent],
        messages: List[SwarmMessage],
        tasks: List[SwarmTask]
    ) -> float:
        """Calculate coordination overhead"""
        if not tasks:
            return 0.0
            
        # Message to task ratio
        message_overhead = len(messages) / len(tasks) if tasks else 0
        
        # Idle agent ratio
        idle_ratio = sum(1 for a in agents if a.status == "idle") / len(agents) if agents else 0
        
        # Failed task ratio
        failed_ratio = sum(1 for t in tasks if t.status == "failed") / len(tasks)
        
        # Combined overhead (lower is better)
        overhead = (message_overhead * 0.3 + idle_ratio * 0.4 + failed_ratio * 0.3) / 10
        
        return min(overhead, 1.0)
    
    def _calculate_throughput(self, metrics_list: List[SwarmMetrics]) -> float:
        """Calculate task throughput"""
        if len(metrics_list) < 2:
            return 0.0
            
        # Tasks completed in time window
        first = metrics_list[0]
        last = metrics_list[-1]
        
        tasks_completed = last.completed_tasks - first.completed_tasks
        time_window = (last.timestamp - first.timestamp).total_seconds()
        
        return tasks_completed / time_window if time_window > 0 else 0.0
    
    async def _check_alerts(self, metrics: SwarmMetrics):
        """Check for alert conditions"""
        # Efficiency alert
        if metrics.swarm_efficiency < self.alert_thresholds["efficiency_low"]:
            await self._raise_alert(
                "low_efficiency",
                f"Swarm efficiency below threshold: {metrics.swarm_efficiency:.2f}"
            )
        
        # Failure rate alert
        failure_rate = metrics.failed_tasks / metrics.task_count if metrics.task_count > 0 else 0
        if failure_rate > self.alert_thresholds["failure_rate_high"]:
            await self._raise_alert(
                "high_failure_rate",
                f"Task failure rate above threshold: {failure_rate:.2f}"
            )
        
        # Communication overload
        if metrics.communication_volume > self.alert_thresholds["communication_overload"]:
            await self._raise_alert(
                "communication_overload",
                f"Communication volume excessive: {metrics.communication_volume} messages"
            )
    
    async def _raise_alert(self, alert_type: str, message: str):
        """Raise an alert"""
        alert = {
            "id": f"alert_{uuid4().hex[:8]}",
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self._determine_severity(alert_type)
        }
        
        self.active_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.active_alerts) > 20:
            self.active_alerts = self.active_alerts[-20:]
        
        logger.warning(f"Alert raised: {alert_type} - {message}")
    
    def _determine_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        high_severity_types = ["communication_overload", "emergence_anomaly"]
        medium_severity_types = ["high_failure_rate", "agent_overload"]
        
        if alert_type in high_severity_types:
            return "high"
        elif alert_type in medium_severity_types:
            return "medium"
        else:
            return "low"


class SwarmAnalytics:
    """Advanced analytics for swarm behavior and performance"""
    
    def __init__(self):
        self.analysis_history = deque(maxlen=1000)
        self.pattern_models = {}
        self.performance_baselines = {}
        
    async def analyze_swarm_performance(
        self,
        metrics_history: List[SwarmMetrics],
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Comprehensive swarm performance analysis"""
        if not metrics_history:
            return {}
            
        # Filter by time window
        cutoff_time = datetime.now(timezone.utc) - time_window
        recent_metrics = [
            m for m in metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        analysis = {
            "performance_trends": self._analyze_performance_trends(recent_metrics),
            "efficiency_analysis": self._analyze_efficiency(recent_metrics),
            "emergence_patterns": self._analyze_emergence_patterns(recent_metrics),
            "bottleneck_analysis": self._analyze_bottlenecks(recent_metrics),
            "optimization_opportunities": self._identify_optimizations(recent_metrics),
            "anomaly_detection": await self._detect_anomalies(recent_metrics)
        }
        
        # Store analysis
        self.analysis_history.append({
            "timestamp": datetime.now(timezone.utc),
            "window": time_window.total_seconds(),
            "analysis": analysis
        })
        
        return analysis
    
    def _analyze_performance_trends(
        self,
        metrics: List[SwarmMetrics]
    ) -> Dict[str, Any]:
        """Analyze performance trends"""
        if len(metrics) < 2:
            return {}
            
        # Extract time series
        timestamps = [m.timestamp.timestamp() for m in metrics]
        efficiencies = [m.swarm_efficiency for m in metrics]
        task_rates = [m.completed_tasks / m.task_count if m.task_count > 0 else 0 for m in metrics]
        
        # Calculate trends
        efficiency_trend = self._calculate_trend(timestamps, efficiencies)
        task_rate_trend = self._calculate_trend(timestamps, task_rates)
        
        return {
            "efficiency_trend": {
                "slope": efficiency_trend["slope"],
                "direction": "improving" if efficiency_trend["slope"] > 0 else "declining",
                "r_squared": efficiency_trend["r_squared"]
            },
            "task_completion_trend": {
                "slope": task_rate_trend["slope"],
                "direction": "improving" if task_rate_trend["slope"] > 0 else "declining",
                "r_squared": task_rate_trend["r_squared"]
            },
            "volatility": {
                "efficiency": np.std(efficiencies),
                "task_rate": np.std(task_rates)
            }
        }
    
    def _analyze_efficiency(self, metrics: List[SwarmMetrics]) -> Dict[str, Any]:
        """Detailed efficiency analysis"""
        efficiencies = [m.swarm_efficiency for m in metrics]
        agent_utilizations = [m.active_agents / m.agent_count if m.agent_count > 0 else 0 for m in metrics]
        
        # Statistical analysis
        return {
            "average_efficiency": np.mean(efficiencies),
            "efficiency_std": np.std(efficiencies),
            "min_efficiency": np.min(efficiencies),
            "max_efficiency": np.max(efficiencies),
            "efficiency_percentiles": {
                "p25": np.percentile(efficiencies, 25),
                "p50": np.percentile(efficiencies, 50),
                "p75": np.percentile(efficiencies, 75),
                "p95": np.percentile(efficiencies, 95)
            },
            "agent_utilization": {
                "average": np.mean(agent_utilizations),
                "peak": np.max(agent_utilizations)
            },
            "efficiency_stability": 1.0 - (np.std(efficiencies) / (np.mean(efficiencies) + 1e-6))
        }
    
    def _analyze_emergence_patterns(
        self,
        metrics: List[SwarmMetrics]
    ) -> Dict[str, Any]:
        """Analyze emergence pattern occurrences"""
        emergence_scores = [m.emergence_score for m in metrics]
        
        # Detect emergence events (scores above threshold)
        emergence_threshold = 0.7
        emergence_events = sum(1 for score in emergence_scores if score > emergence_threshold)
        
        return {
            "average_emergence_score": np.mean(emergence_scores),
            "emergence_frequency": emergence_events / len(metrics) if metrics else 0,
            "emergence_stability": np.std(emergence_scores),
            "strong_emergence_ratio": sum(1 for s in emergence_scores if s > 0.8) / len(emergence_scores)
        }
    
    def _analyze_bottlenecks(self, metrics: List[SwarmMetrics]) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Communication bottleneck
        comm_volumes = [m.communication_volume for m in metrics]
        if np.mean(comm_volumes) > 500:  # Threshold
            bottlenecks.append({
                "type": "communication",
                "severity": "high" if np.mean(comm_volumes) > 1000 else "medium",
                "description": f"High communication volume: {np.mean(comm_volumes):.0f} avg messages"
            })
        
        # Task completion bottleneck
        completion_rates = [
            m.completed_tasks / m.task_count if m.task_count > 0 else 0
            for m in metrics
        ]
        if np.mean(completion_rates) < 0.7:
            bottlenecks.append({
                "type": "task_completion",
                "severity": "high" if np.mean(completion_rates) < 0.5 else "medium",
                "description": f"Low task completion rate: {np.mean(completion_rates):.2f}"
            })
        
        # Coordination overhead
        overhead_values = [m.coordination_overhead for m in metrics]
        if np.mean(overhead_values) > 0.3:
            bottlenecks.append({
                "type": "coordination_overhead",
                "severity": "medium",
                "description": f"High coordination overhead: {np.mean(overhead_values):.2f}"
            })
        
        return {
            "bottlenecks": bottlenecks,
            "bottleneck_count": len(bottlenecks),
            "primary_bottleneck": bottlenecks[0]["type"] if bottlenecks else None
        }
    
    def _identify_optimizations(
        self,
        metrics: List[SwarmMetrics]
    ) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        optimizations = []
        
        # Pattern-based optimizations
        patterns = [m.pattern for m in metrics]
        pattern_counts = defaultdict(int)
        for p in patterns:
            pattern_counts[p] += 1
        
        # Check if pattern switching might help
        dominant_pattern = max(pattern_counts, key=pattern_counts.get)
        pattern_efficiency = defaultdict(list)
        
        for m in metrics:
            pattern_efficiency[m.pattern].append(m.swarm_efficiency)
        
        for pattern, efficiencies in pattern_efficiency.items():
            if pattern != dominant_pattern and np.mean(efficiencies) > np.mean(pattern_efficiency[dominant_pattern]):
                optimizations.append({
                    "type": "pattern_switch",
                    "recommendation": f"Switch from {dominant_pattern.value} to {pattern.value}",
                    "expected_improvement": np.mean(efficiencies) - np.mean(pattern_efficiency[dominant_pattern]),
                    "confidence": 0.7
                })
        
        # Agent count optimization
        agent_counts = [m.agent_count for m in metrics]
        efficiencies = [m.swarm_efficiency for m in metrics]
        
        if len(set(agent_counts)) > 1:  # Varying agent counts
            # Find optimal agent count
            agent_efficiency_map = defaultdict(list)
            for m in metrics:
                agent_efficiency_map[m.agent_count].append(m.swarm_efficiency)
            
            optimal_count = max(agent_efficiency_map, key=lambda k: np.mean(agent_efficiency_map[k]))
            current_avg_count = np.mean(agent_counts)
            
            if abs(optimal_count - current_avg_count) > 2:
                optimizations.append({
                    "type": "agent_count",
                    "recommendation": f"Adjust agent count to {optimal_count}",
                    "current": current_avg_count,
                    "optimal": optimal_count,
                    "confidence": 0.8
                })
        
        # Communication optimization
        comm_overhead = [m.coordination_overhead for m in metrics]
        if np.mean(comm_overhead) > 0.2:
            optimizations.append({
                "type": "communication",
                "recommendation": "Reduce communication frequency or implement message batching",
                "current_overhead": np.mean(comm_overhead),
                "target_overhead": 0.1,
                "confidence": 0.9
            })
        
        return optimizations
    
    async def _detect_anomalies(
        self,
        metrics: List[SwarmMetrics]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in swarm behavior"""
        anomalies = []
        
        # Efficiency anomalies
        efficiencies = [m.swarm_efficiency for m in metrics]
        if len(efficiencies) > 10:
            # Use z-score for anomaly detection
            z_scores = stats.zscore(efficiencies)
            
            for i, (z_score, metric) in enumerate(zip(z_scores, metrics)):
                if abs(z_score) > 2:  # 2 standard deviations
                    anomalies.append({
                        "type": "efficiency_anomaly",
                        "timestamp": metric.timestamp.isoformat(),
                        "value": metric.swarm_efficiency,
                        "z_score": z_score,
                        "description": f"Unusual efficiency: {metric.swarm_efficiency:.2f}"
                    })
        
        # Emergence anomalies
        emergence_scores = [m.emergence_score for m in metrics]
        if len(emergence_scores) > 10:
            # Sudden drops in emergence
            for i in range(1, len(emergence_scores)):
                if emergence_scores[i] < emergence_scores[i-1] * 0.5:  # 50% drop
                    anomalies.append({
                        "type": "emergence_collapse",
                        "timestamp": metrics[i].timestamp.isoformat(),
                        "previous_score": emergence_scores[i-1],
                        "current_score": emergence_scores[i],
                        "description": "Sudden drop in emergence score"
                    })
        
        # Communication spikes
        comm_volumes = [m.communication_volume for m in metrics]
        if len(comm_volumes) > 5:
            avg_comm = np.mean(comm_volumes)
            for i, (volume, metric) in enumerate(zip(comm_volumes, metrics)):
                if volume > avg_comm * 3:  # 3x average
                    anomalies.append({
                        "type": "communication_spike",
                        "timestamp": metric.timestamp.isoformat(),
                        "volume": volume,
                        "average": avg_comm,
                        "description": f"Communication spike: {volume} messages"
                    })
        
        return anomalies
    
    def _calculate_trend(
        self,
        x: List[float],
        y: List[float]
    ) -> Dict[str, float]:
        """Calculate linear trend"""
        if len(x) < 2:
            return {"slope": 0.0, "r_squared": 0.0}
            
        # Linear regression
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Normalize x to avoid numerical issues
        x_normalized = (x_array - x_array.min()) / (x_array.max() - x_array.min() + 1e-6)
        
        # Calculate slope and r-squared
        slope, intercept, r_value, _, _ = stats.linregress(x_normalized, y_array)
        
        return {
            "slope": slope,
            "r_squared": r_value ** 2
        }
    
    async def generate_report(
        self,
        swarm_id: str,
        metrics_history: List[SwarmMetrics],
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Generate comprehensive swarm performance report"""
        analysis = await self.analyze_swarm_performance(metrics_history, time_window)
        
        report = {
            "report_id": f"report_{uuid4().hex[:8]}",
            "swarm_id": swarm_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "time_window": {
                "start": (datetime.now(timezone.utc) - time_window).isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
                "duration_hours": time_window.total_seconds() / 3600
            },
            "executive_summary": self._generate_executive_summary(analysis),
            "detailed_analysis": analysis,
            "recommendations": self._generate_recommendations(analysis),
            "visualizations": self._prepare_visualization_data(metrics_history)
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        perf_trends = analysis.get("performance_trends", {})
        efficiency = analysis.get("efficiency_analysis", {})
        
        return {
            "overall_health": self._determine_health_score(analysis),
            "key_metrics": {
                "average_efficiency": efficiency.get("average_efficiency", 0),
                "efficiency_trend": perf_trends.get("efficiency_trend", {}).get("direction", "stable"),
                "bottleneck_count": analysis.get("bottleneck_analysis", {}).get("bottleneck_count", 0),
                "anomaly_count": len(analysis.get("anomaly_detection", []))
            },
            "summary": self._generate_summary_text(analysis)
        }
    
    def _determine_health_score(self, analysis: Dict[str, Any]) -> str:
        """Determine overall swarm health"""
        score = 100
        
        # Deduct for poor efficiency
        avg_efficiency = analysis.get("efficiency_analysis", {}).get("average_efficiency", 0)
        if avg_efficiency < 0.5:
            score -= 30
        elif avg_efficiency < 0.7:
            score -= 15
        
        # Deduct for bottlenecks
        bottleneck_count = analysis.get("bottleneck_analysis", {}).get("bottleneck_count", 0)
        score -= bottleneck_count * 10
        
        # Deduct for anomalies
        anomaly_count = len(analysis.get("anomaly_detection", []))
        score -= anomaly_count * 5
        
        # Deduct for negative trends
        efficiency_trend = analysis.get("performance_trends", {}).get("efficiency_trend", {})
        if efficiency_trend.get("direction") == "declining":
            score -= 10
        
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _generate_summary_text(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        efficiency = analysis.get("efficiency_analysis", {}).get("average_efficiency", 0)
        trend = analysis.get("performance_trends", {}).get("efficiency_trend", {}).get("direction", "stable")
        bottlenecks = analysis.get("bottleneck_analysis", {}).get("bottlenecks", [])
        
        summary = f"The swarm is operating at {efficiency:.1%} average efficiency with a {trend} trend. "
        
        if bottlenecks:
            summary += f"There are {len(bottlenecks)} identified bottlenecks, primarily in {bottlenecks[0]['type']}. "
        
        optimizations = analysis.get("optimization_opportunities", [])
        if optimizations:
            summary += f"We've identified {len(optimizations)} optimization opportunities that could improve performance."
        
        return summary
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on optimization opportunities
        for opt in analysis.get("optimization_opportunities", []):
            recommendations.append({
                "priority": "high" if opt.get("confidence", 0) > 0.8 else "medium",
                "type": opt["type"],
                "action": opt["recommendation"],
                "expected_impact": opt.get("expected_improvement", "moderate"),
                "implementation_effort": "low"  # Would be calculated based on type
            })
        
        # Based on bottlenecks
        for bottleneck in analysis.get("bottleneck_analysis", {}).get("bottlenecks", []):
            if bottleneck["type"] == "communication":
                recommendations.append({
                    "priority": bottleneck["severity"],
                    "type": "architecture",
                    "action": "Implement message batching and compression",
                    "expected_impact": "30% reduction in communication overhead",
                    "implementation_effort": "medium"
                })
            elif bottleneck["type"] == "task_completion":
                recommendations.append({
                    "priority": bottleneck["severity"],
                    "type": "algorithm",
                    "action": "Review task allocation algorithm and agent capabilities",
                    "expected_impact": "20% improvement in completion rate",
                    "implementation_effort": "high"
                })
        
        return sorted(recommendations, key=lambda x: x["priority"] == "high", reverse=True)
    
    def _prepare_visualization_data(
        self,
        metrics_history: List[SwarmMetrics]
    ) -> Dict[str, Any]:
        """Prepare data for visualization"""
        return {
            "efficiency_timeline": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.swarm_efficiency
                }
                for m in metrics_history
            ],
            "agent_utilization_timeline": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.active_agents / m.agent_count if m.agent_count > 0 else 0
                }
                for m in metrics_history
            ],
            "task_completion_timeline": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "completed": m.completed_tasks,
                    "failed": m.failed_tasks,
                    "total": m.task_count
                }
                for m in metrics_history
            ],
            "emergence_score_timeline": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.emergence_score
                }
                for m in metrics_history
            ]
        }


class SwarmDashboard:
    """Web dashboard interface for swarm monitoring"""
    
    def __init__(self, monitor: SwarmMonitor, analytics: SwarmAnalytics):
        self.monitor = monitor
        self.analytics = analytics
        self.update_interval = 5  # seconds
        self.dashboard_config = {
            "theme": "dark",
            "refresh_rate": 5000,  # milliseconds
            "max_data_points": 100,
            "chart_types": {
                "efficiency": "line",
                "agents": "bar",
                "emergence": "heatmap",
                "communication": "network"
            }
        }
        
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return {
            "config": self.dashboard_config,
            "layout": self._get_dashboard_layout(),
            "widgets": self._get_widget_definitions()
        }
    
    def _get_dashboard_layout(self) -> List[Dict[str, Any]]:
        """Define dashboard layout"""
        return [
            {
                "row": 1,
                "widgets": [
                    {"id": "overview", "col": 1, "width": 4, "height": 2},
                    {"id": "efficiency_chart", "col": 5, "width": 4, "height": 2},
                    {"id": "alerts", "col": 9, "width": 4, "height": 2}
                ]
            },
            {
                "row": 2,
                "widgets": [
                    {"id": "agent_performance", "col": 1, "width": 6, "height": 3},
                    {"id": "task_progress", "col": 7, "width": 6, "height": 3}
                ]
            },
            {
                "row": 3,
                "widgets": [
                    {"id": "emergence_patterns", "col": 1, "width": 4, "height": 2},
                    {"id": "communication_flow", "col": 5, "width": 4, "height": 2},
                    {"id": "optimization_suggestions", "col": 9, "width": 4, "height": 2}
                ]
            }
        ]
    
    def _get_widget_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Define dashboard widgets"""
        return {
            "overview": {
                "type": "stats",
                "title": "Swarm Overview",
                "metrics": ["agent_count", "task_progress", "efficiency", "emergence_score"]
            },
            "efficiency_chart": {
                "type": "line_chart",
                "title": "Efficiency Over Time",
                "data_source": "time_series.efficiency",
                "y_axis": "Efficiency",
                "x_axis": "Time"
            },
            "alerts": {
                "type": "alert_list",
                "title": "Active Alerts",
                "max_items": 5,
                "severity_colors": {
                    "high": "#ff4444",
                    "medium": "#ffaa00",
                    "low": "#44ff44"
                }
            },
            "agent_performance": {
                "type": "table",
                "title": "Agent Performance",
                "columns": ["id", "type", "performance", "load", "tasks_completed"],
                "sortable": True,
                "filterable": True
            },
            "task_progress": {
                "type": "progress_chart",
                "title": "Task Execution Progress",
                "categories": ["completed", "in_progress", "failed", "pending"]
            },
            "emergence_patterns": {
                "type": "pattern_viz",
                "title": "Emergence Patterns",
                "visualization": "radial"
            },
            "communication_flow": {
                "type": "network_graph",
                "title": "Agent Communication Network",
                "node_size": "message_count",
                "edge_weight": "message_frequency"
            },
            "optimization_suggestions": {
                "type": "recommendation_list",
                "title": "Optimization Opportunities",
                "priority_sort": True
            }
        }
    
    async def stream_updates(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream real-time dashboard updates"""
        while True:
            try:
                # Get latest data
                dashboard_data = self.monitor.get_dashboard_data()
                
                # Add timestamp
                dashboard_data["update_timestamp"] = datetime.now(timezone.utc).isoformat()
                
                yield dashboard_data
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                yield {"error": str(e)}
                await asyncio.sleep(self.update_interval)