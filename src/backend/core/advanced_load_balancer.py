"""
Advanced Load Balancer Enhancement Module
Implements sophisticated load balancing algorithms with dynamic scaling
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .load_balancer import (
    ServerNode, LoadBalancingStrategy, AdvancedLoadBalancer as BaseLoadBalancer
)

logger = logging.getLogger(__name__)


class PredictiveModel(Enum):
    """Predictive model types for load forecasting"""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class ServerProfile:
    """Detailed server performance profile"""
    server_id: str
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    response_time_distribution: List[float] = field(default_factory=list)
    failure_rate: float = 0.0
    recovery_time: float = 0.0
    peak_capacity: int = 0
    optimal_load_range: Tuple[float, float] = (0.3, 0.7)
    specializations: Set[str] = field(default_factory=set)
    cost_per_request: float = 0.001
    energy_efficiency: float = 1.0
    last_optimization: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoadPattern:
    """Load pattern analysis"""
    pattern_type: str  # "periodic", "burst", "gradual", "random"
    period: Optional[timedelta] = None
    amplitude: float = 0.0
    trend: float = 0.0  # Positive for increasing, negative for decreasing
    seasonality: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics"""
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cost_efficiency: float = 0.0
    carbon_footprint: float = 0.0
    sla_compliance: float = 1.0


class EnhancedLoadBalancer(BaseLoadBalancer):
    """Enhanced load balancer with advanced algorithms"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        super().__init__(strategy)
        
        # Enhanced components
        self.server_profiles: Dict[str, ServerProfile] = {}
        self.load_patterns: Dict[str, LoadPattern] = {}
        self.performance_metrics = PerformanceMetrics()
        
        # Predictive components
        self.prediction_models: Dict[str, Any] = {}
        self.historical_load_data = deque(maxlen=10000)
        self.anomaly_threshold = 3.0  # Standard deviations
        
        # Advanced strategies
        self.custom_strategies = {
            "consistent_hash": self._consistent_hash_selection,
            "power_of_two": self._power_of_two_selection,
            "join_shortest_queue": self._join_shortest_queue_selection,
            "least_outstanding_requests": self._least_outstanding_requests_selection,
            "adaptive_weighted": self._adaptive_weighted_selection,
            "cost_optimized": self._cost_optimized_selection,
            "ml_optimized": self._ml_optimized_selection,
        }
        
        # Dynamic scaling parameters
        self.scaling_policies = {
            "aggressive": {"up": 0.7, "down": 0.2, "cooldown": 60},
            "moderate": {"up": 0.8, "down": 0.3, "cooldown": 120},
            "conservative": {"up": 0.9, "down": 0.4, "cooldown": 300},
        }
        self.current_scaling_policy = "moderate"
        self.scaling_history = deque(maxlen=100)
        
        # Circuit breaker pattern
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_threshold = 0.5  # 50% failure rate
        self.circuit_breaker_timeout = 30  # seconds
        
        # Request routing optimization
        self.routing_cache = {}
        self.routing_cache_ttl = 10  # seconds
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> server_id
        
        # Performance optimization
        self.optimization_interval = 60  # seconds
        self.last_optimization = time.time()
        
    def initialize_server_profile(self, server: ServerNode):
        """Initialize detailed server profile"""
        if server.id not in self.server_profiles:
            self.server_profiles[server.id] = ServerProfile(
                server_id=server.id,
                peak_capacity=server.max_connections,
                specializations=set(server.region.split("-"))  # Extract specializations from region
            )
    
    async def select_server(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Enhanced server selection with custom strategies"""
        # Check for sticky session
        if request_context and "session_id" in request_context:
            session_id = request_context["session_id"]
            if session_id in self.sticky_sessions:
                server_id = self.sticky_sessions[session_id]
                if server_id in self.server_nodes:
                    server = self.server_nodes[server_id]
                    if server.is_healthy:
                        return server
        
        # Check custom strategies
        strategy_name = request_context.get("strategy") if request_context else None
        if strategy_name in self.custom_strategies:
            return await self.custom_strategies[strategy_name](request_context)
        
        # Fall back to base implementation
        return await super().select_server(request_context)
    
    async def _consistent_hash_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Consistent hashing for better cache utilization"""
        if not request_context or "key" not in request_context:
            return self._round_robin_selection(list(self.server_nodes.values()))
        
        key = request_context["key"]
        hash_value = hash(key)
        
        # Create sorted list of servers by hash
        servers = [(hash(s.id), s) for s in self.server_nodes.values() if s.is_healthy]
        servers.sort(key=lambda x: x[0])
        
        if not servers:
            return None
        
        # Find the server with hash >= key hash
        for server_hash, server in servers:
            if server_hash >= hash_value:
                return server
        
        # Wrap around to first server
        return servers[0][1]
    
    async def _power_of_two_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Power of two choices for better load distribution"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        if len(healthy_servers) == 1:
            return healthy_servers[0]
        
        # Randomly select two servers
        import random
        candidates = random.sample(healthy_servers, min(2, len(healthy_servers)))
        
        # Choose the one with lower load
        return min(candidates, key=lambda s: s.load_score)
    
    async def _join_shortest_queue_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Join shortest queue algorithm"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Find server with shortest queue (least connections)
        return min(healthy_servers, key=lambda s: s.current_connections)
    
    async def _least_outstanding_requests_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Select server with least outstanding requests"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Calculate outstanding requests (connections * avg response time)
        def outstanding_score(server):
            return server.current_connections * (server.avg_response_time / 1000)
        
        return min(healthy_servers, key=outstanding_score)
    
    async def _adaptive_weighted_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Adaptive weighted selection based on real-time performance"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Calculate adaptive weights
        weights = {}
        for server in healthy_servers:
            profile = self.server_profiles.get(server.id)
            if not profile:
                self.initialize_server_profile(server)
                profile = self.server_profiles[server.id]
            
            # Multi-factor weight calculation
            base_weight = server.weight
            health_factor = server.health_score
            performance_factor = 1 / (server.avg_response_time + 1)
            capacity_factor = 1 - (server.current_connections / server.max_connections)
            
            # Check if server is in optimal load range
            load_ratio = server.current_connections / server.max_connections
            optimal_factor = 1.0
            if profile.optimal_load_range:
                if load_ratio < profile.optimal_load_range[0]:
                    optimal_factor = 1.2  # Prefer underutilized
                elif load_ratio > profile.optimal_load_range[1]:
                    optimal_factor = 0.8  # Avoid overutilized
            
            weights[server.id] = (
                base_weight * health_factor * performance_factor * 
                capacity_factor * optimal_factor
            )
        
        # Weighted random selection
        total_weight = sum(weights.values())
        if total_weight == 0:
            return healthy_servers[0]
        
        import random
        rand_value = random.uniform(0, total_weight)
        cumulative = 0
        
        for server in healthy_servers:
            cumulative += weights[server.id]
            if rand_value <= cumulative:
                return server
        
        return healthy_servers[-1]
    
    async def _cost_optimized_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Select server optimizing for cost efficiency"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Calculate cost score for each server
        cost_scores = {}
        
        for server in healthy_servers:
            profile = self.server_profiles.get(server.id)
            if not profile:
                self.initialize_server_profile(server)
                profile = self.server_profiles[server.id]
            
            # Cost factors
            base_cost = profile.cost_per_request
            load_penalty = server.capacity_utilization * 0.5  # Higher load = higher cost
            energy_factor = 1 / profile.energy_efficiency
            
            cost_scores[server.id] = base_cost * (1 + load_penalty) * energy_factor
        
        # Select server with lowest cost
        best_server_id = min(cost_scores.keys(), key=lambda k: cost_scores[k])
        return self.server_nodes[best_server_id]
    
    async def _ml_optimized_selection(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """Machine learning optimized selection"""
        healthy_servers = [s for s in self.server_nodes.values() if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Extract features for ML model
        features = []
        for server in healthy_servers:
            server_features = [
                server.current_connections / server.max_connections,
                server.health_score,
                server.avg_response_time / 1000,
                server.capacity_utilization,
                len(self.server_profiles.get(server.id, ServerProfile(server.id)).performance_history)
            ]
            features.append(server_features)
        
        # Simple ML model (in production, use trained model)
        # For now, use a weighted scoring based on historical performance
        scores = []
        for i, server in enumerate(healthy_servers):
            profile = self.server_profiles.get(server.id)
            if profile and profile.performance_history:
                # Calculate historical success rate
                recent_history = list(profile.performance_history)[-100:]
                success_rate = sum(1 for h in recent_history if h.get("success", False)) / len(recent_history)
            else:
                success_rate = 0.9  # Default
            
            # Combine with current state
            score = (
                success_rate * 0.5 +
                (1 - features[i][0]) * 0.3 +  # Available capacity
                features[i][1] * 0.2  # Health score
            )
            scores.append(score)
        
        # Select server with highest score
        best_idx = scores.index(max(scores))
        return healthy_servers[best_idx]
    
    async def analyze_load_patterns(self) -> Dict[str, LoadPattern]:
        """Analyze historical load to identify patterns"""
        if len(self.historical_load_data) < 100:
            return {}
        
        # Convert to numpy array for analysis
        load_data = np.array([d["load"] for d in self.historical_load_data])
        timestamps = np.array([d["timestamp"] for d in self.historical_load_data])
        
        patterns = {}
        
        # Detect periodicity using FFT
        fft_result = np.fft.fft(load_data)
        frequencies = np.fft.fftfreq(len(load_data))
        
        # Find dominant frequency
        dominant_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
        dominant_freq = frequencies[dominant_idx]
        
        if dominant_freq > 0:
            period_seconds = 1 / dominant_freq
            patterns["periodic"] = LoadPattern(
                pattern_type="periodic",
                period=timedelta(seconds=period_seconds),
                amplitude=np.std(load_data),
                confidence=np.abs(fft_result[dominant_idx]) / len(load_data)
            )
        
        # Detect trend
        x = np.arange(len(load_data))
        trend_coef = np.polyfit(x, load_data, 1)[0]
        
        patterns["trend"] = LoadPattern(
            pattern_type="gradual",
            trend=trend_coef,
            confidence=abs(trend_coef) / np.mean(load_data)
        )
        
        # Detect bursts
        rolling_mean = np.convolve(load_data, np.ones(10)/10, mode='valid')
        rolling_std = np.array([np.std(load_data[i:i+10]) for i in range(len(load_data)-9)])
        burst_threshold = rolling_mean + 2 * rolling_std
        
        burst_count = sum(load_data[10:] > burst_threshold)
        if burst_count > len(load_data) * 0.05:  # More than 5% bursts
            patterns["burst"] = LoadPattern(
                pattern_type="burst",
                amplitude=np.max(load_data) - np.mean(load_data),
                confidence=burst_count / len(load_data)
            )
        
        self.load_patterns = patterns
        return patterns
    
    async def predict_future_load(
        self, 
        horizon_minutes: int = 30,
        model: PredictiveModel = PredictiveModel.ENSEMBLE
    ) -> List[Tuple[datetime, float]]:
        """Predict future load using various models"""
        if len(self.historical_load_data) < 100:
            # Not enough data, return simple forecast
            current_load = self._get_current_total_load()
            predictions = []
            for i in range(horizon_minutes):
                future_time = datetime.utcnow() + timedelta(minutes=i)
                predictions.append((future_time, current_load))
            return predictions
        
        # Prepare data
        load_data = np.array([d["load"] for d in self.historical_load_data])
        
        predictions = []
        
        if model == PredictiveModel.LINEAR_REGRESSION:
            # Simple linear regression
            x = np.arange(len(load_data))
            coeffs = np.polyfit(x, load_data, 1)
            
            for i in range(horizon_minutes):
                future_x = len(load_data) + i
                future_load = coeffs[0] * future_x + coeffs[1]
                future_time = datetime.utcnow() + timedelta(minutes=i)
                predictions.append((future_time, max(0, future_load)))
        
        elif model == PredictiveModel.EXPONENTIAL_SMOOTHING:
            # Exponential smoothing
            alpha = 0.3
            smoothed = [load_data[0]]
            
            for i in range(1, len(load_data)):
                smoothed.append(alpha * load_data[i] + (1 - alpha) * smoothed[-1])
            
            last_value = smoothed[-1]
            trend = smoothed[-1] - smoothed[-2] if len(smoothed) > 1 else 0
            
            for i in range(horizon_minutes):
                future_load = last_value + trend * i
                future_time = datetime.utcnow() + timedelta(minutes=i)
                predictions.append((future_time, max(0, future_load)))
        
        elif model == PredictiveModel.ENSEMBLE:
            # Ensemble of multiple models
            linear_pred = await self.predict_future_load(horizon_minutes, PredictiveModel.LINEAR_REGRESSION)
            exp_pred = await self.predict_future_load(horizon_minutes, PredictiveModel.EXPONENTIAL_SMOOTHING)
            
            # Average predictions
            for i in range(horizon_minutes):
                avg_load = (linear_pred[i][1] + exp_pred[i][1]) / 2
                
                # Add pattern-based adjustments
                if "periodic" in self.load_patterns:
                    pattern = self.load_patterns["periodic"]
                    if pattern.period:
                        # Add periodic component
                        phase = (i * 60) % pattern.period.total_seconds()
                        periodic_component = pattern.amplitude * math.sin(2 * math.pi * phase / pattern.period.total_seconds())
                        avg_load += periodic_component * pattern.confidence
                
                predictions.append((linear_pred[i][0], max(0, avg_load)))
        
        return predictions
    
    async def optimize_server_allocation(self) -> Dict[str, Any]:
        """Optimize server allocation based on predictions and patterns"""
        # Get current state
        current_servers = len(self.server_nodes)
        total_capacity = sum(s.max_connections for s in self.server_nodes.values())
        current_load = self._get_current_total_load()
        
        # Predict future load
        predictions = await self.predict_future_load(30)  # 30 minute horizon
        max_predicted_load = max(pred[1] for pred in predictions)
        
        # Analyze patterns
        patterns = await self.analyze_load_patterns()
        
        # Optimization decisions
        recommendations = {
            "current_servers": current_servers,
            "current_capacity": total_capacity,
            "current_load": current_load,
            "predicted_max_load": max_predicted_load,
            "recommendations": []
        }
        
        # Check if scaling needed
        policy = self.scaling_policies[self.current_scaling_policy]
        load_ratio = max_predicted_load / total_capacity if total_capacity > 0 else 0
        
        if load_ratio > policy["up"]:
            # Scale up recommendation
            additional_servers = math.ceil((max_predicted_load - total_capacity * 0.7) / 1000)
            recommendations["recommendations"].append({
                "action": "scale_up",
                "servers": additional_servers,
                "reason": f"Predicted load ratio {load_ratio:.2f} exceeds threshold {policy['up']}"
            })
        
        elif load_ratio < policy["down"] and current_servers > self.min_nodes:
            # Scale down recommendation
            servers_to_remove = min(
                current_servers - self.min_nodes,
                math.floor((total_capacity * policy["down"] - max_predicted_load) / 1000)
            )
            if servers_to_remove > 0:
                recommendations["recommendations"].append({
                    "action": "scale_down",
                    "servers": servers_to_remove,
                    "reason": f"Predicted load ratio {load_ratio:.2f} below threshold {policy['down']}"
                })
        
        # Server rebalancing recommendations
        if "burst" in patterns and patterns["burst"].confidence > 0.1:
            recommendations["recommendations"].append({
                "action": "enable_burst_handling",
                "reason": "Burst pattern detected with high confidence"
            })
        
        # Geographic optimization
        if hasattr(self, 'request_geography'):
            geo_recommendations = self._optimize_geographic_distribution()
            recommendations["recommendations"].extend(geo_recommendations)
        
        return recommendations
    
    def _get_current_total_load(self) -> float:
        """Calculate current total load across all servers"""
        total_connections = sum(s.current_connections for s in self.server_nodes.values())
        total_capacity = sum(s.max_connections for s in self.server_nodes.values())
        return (total_connections / total_capacity * 100) if total_capacity > 0 else 0
    
    def _optimize_geographic_distribution(self) -> List[Dict[str, Any]]:
        """Optimize geographic distribution of servers"""
        recommendations = []
        
        # Analyze request distribution by region
        region_loads = defaultdict(float)
        for server in self.server_nodes.values():
            region_loads[server.region] += server.current_connections
        
        # Find imbalanced regions
        avg_load = sum(region_loads.values()) / len(region_loads) if region_loads else 0
        
        for region, load in region_loads.items():
            if load > avg_load * 1.5:
                recommendations.append({
                    "action": "add_server_to_region",
                    "region": region,
                    "reason": f"Region {region} has {load/avg_load:.1f}x average load"
                })
            elif load < avg_load * 0.5 and len(region_loads) > 1:
                recommendations.append({
                    "action": "consider_removing_server_from_region",
                    "region": region,
                    "reason": f"Region {region} has only {load/avg_load:.1f}x average load"
                })
        
        return recommendations
    
    async def apply_circuit_breaker(self, server_id: str, error: Exception):
        """Apply circuit breaker pattern to failing servers"""
        if server_id not in self.circuit_breakers:
            self.circuit_breakers[server_id] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed",  # closed, open, half-open
                "last_state_change": datetime.utcnow()
            }
        
        breaker = self.circuit_breakers[server_id]
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Check if circuit should open
        if breaker["state"] == "closed":
            failure_window = datetime.utcnow() - timedelta(seconds=60)
            recent_failures = breaker["failures"]  # Simplified, should track time window
            
            if recent_failures >= 5:  # 5 failures in 60 seconds
                breaker["state"] = "open"
                breaker["last_state_change"] = datetime.utcnow()
                logger.warning(f"Circuit breaker opened for server {server_id}")
                
                # Mark server as unhealthy
                if server_id in self.server_nodes:
                    self.server_nodes[server_id].health_score = 0.1
        
        # Check if circuit should attempt half-open
        elif breaker["state"] == "open":
            time_since_open = (datetime.utcnow() - breaker["last_state_change"]).seconds
            if time_since_open >= self.circuit_breaker_timeout:
                breaker["state"] = "half-open"
                breaker["failures"] = 0
                logger.info(f"Circuit breaker half-open for server {server_id}")
    
    async def record_request_success(self, server_id: str, response_time: float):
        """Record successful request for circuit breaker and profiling"""
        # Update circuit breaker
        if server_id in self.circuit_breakers:
            breaker = self.circuit_breakers[server_id]
            if breaker["state"] == "half-open":
                # Success in half-open state, close the circuit
                breaker["state"] = "closed"
                breaker["failures"] = 0
                logger.info(f"Circuit breaker closed for server {server_id}")
        
        # Update server profile
        if server_id not in self.server_profiles:
            self.initialize_server_profile(self.server_nodes[server_id])
        
        profile = self.server_profiles[server_id]
        profile.performance_history.append({
            "timestamp": datetime.utcnow(),
            "response_time": response_time,
            "success": True
        })
        
        # Update response time distribution
        profile.response_time_distribution.append(response_time)
        if len(profile.response_time_distribution) > 1000:
            profile.response_time_distribution = profile.response_time_distribution[-1000:]
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced load balancing statistics"""
        base_stats = self.get_load_balancing_stats()
        
        # Add enhanced metrics
        enhanced_stats = {
            **base_stats,
            "enhanced_metrics": {
                "latency_p50": self.performance_metrics.latency_p50,
                "latency_p95": self.performance_metrics.latency_p95,
                "latency_p99": self.performance_metrics.latency_p99,
                "throughput": self.performance_metrics.throughput,
                "cost_efficiency": self.performance_metrics.cost_efficiency,
                "sla_compliance": self.performance_metrics.sla_compliance,
            },
            "load_patterns": {
                pattern_type: {
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "trend": pattern.trend
                }
                for pattern_type, pattern in self.load_patterns.items()
            },
            "circuit_breakers": {
                server_id: breaker["state"]
                for server_id, breaker in self.circuit_breakers.items()
            },
            "scaling_policy": self.current_scaling_policy,
            "optimization_recommendations": []
        }
        
        # Add recent optimization recommendations
        if hasattr(self, '_last_optimization_result'):
            enhanced_stats["optimization_recommendations"] = self._last_optimization_result.get("recommendations", [])
        
        return enhanced_stats
