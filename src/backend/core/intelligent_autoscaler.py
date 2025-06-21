"""
Intelligent Auto-Scaler
Advanced auto-scaling with predictive capabilities and multi-dimensional optimization
"""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .scalability.enterprise_scalability import AutoScaler, ScalingPolicy, ScalingEvent
from .resource_optimizer import ResourceOptimizer
from .intelligent_load_manager import IntelligentLoadManager
import statistics

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Advanced scaling strategies"""
    REACTIVE = "reactive"  # Traditional threshold-based
    PREDICTIVE = "predictive"  # ML-based prediction
    PROACTIVE = "proactive"  # Pattern-based proactive
    HYBRID = "hybrid"  # Combination of strategies
    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Maximize performance
    BALANCED = "balanced"  # Balance cost and performance


class ScalingDecision(Enum):
    """Scaling decisions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Add more instances
    SCALE_IN = "scale_in"  # Remove instances
    VERTICAL_SCALE = "vertical_scale"  # Change instance type
    NO_ACTION = "no_action"
    DEFER = "defer"  # Wait for more data


@dataclass
class ScalingTarget:
    """Target configuration for scaling"""
    service: str
    min_instances: int
    max_instances: int
    target_utilization: float
    cost_per_instance: float
    instance_types: List[Dict[str, Any]]
    scaling_step: int = 1
    cooldown_seconds: int = 300
    warm_up_seconds: int = 120


@dataclass
class ScalingPlan:
    """Detailed scaling plan"""
    plan_id: str
    timestamp: datetime
    strategy: ScalingStrategy
    decision: ScalingDecision
    target: ScalingTarget
    current_state: Dict[str, Any]
    desired_state: Dict[str, Any]
    reason: str
    confidence: float
    estimated_cost_impact: float
    estimated_performance_impact: float
    execution_steps: List[Dict[str, Any]]
    rollback_plan: Optional[Dict[str, Any]] = None


@dataclass
class PredictionModel:
    """ML prediction model for scaling"""
    model_type: str
    model: Any
    feature_names: List[str]
    last_trained: datetime
    accuracy_score: float
    prediction_horizon: int  # minutes


class IntelligentAutoScaler:
    """Advanced auto-scaling system with ML and optimization"""
    
    def __init__(
        self,
        base_scaler: Optional[AutoScaler] = None,
        resource_optimizer: Optional[ResourceOptimizer] = None,
        load_manager: Optional[IntelligentLoadManager] = None
    ):
        self.base_scaler = base_scaler or AutoScaler()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer()
        self.load_manager = load_manager
        
        # Scaling configuration
        self.scaling_targets: Dict[str, ScalingTarget] = {}
        self.active_plans: Dict[str, ScalingPlan] = {}
        self.scaling_history = deque(maxlen=1000)
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.model_update_interval = 3600  # 1 hour
        self.last_model_update = time.time()
        
        # Metrics collection
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours
        self.anomaly_buffer = deque(maxlen=100)
        
        # Optimization parameters
        self.cost_weight = 0.4
        self.performance_weight = 0.6
        self.stability_bonus = 0.1
        
        # Advanced features
        self.enable_predictive_scaling = True
        self.enable_cost_optimization = True
        self.enable_multi_region = True
        self.enable_spot_instances = True
        
        # Pattern detection
        self.pattern_library = self._initialize_pattern_library()
        self.detected_patterns: Dict[str, List[str]] = defaultdict(list)
    
    def configure_scaling_target(self, target: ScalingTarget):
        """Configure scaling target for a service"""
        self.scaling_targets[target.service] = target
        
        # Initialize in base scaler
        self.base_scaler.configure_auto_scaling(
            service=target.service,
            policy=ScalingPolicy.PREDICTIVE,
            min_instances=target.min_instances,
            max_instances=target.max_instances,
            scale_up_threshold=target.target_utilization + 20,
            scale_down_threshold=target.target_utilization - 20,
            cooldown_seconds=target.cooldown_seconds
        )
        
        logger.info(f"Configured scaling target for {target.service}")
    
    async def evaluate_scaling_needs(
        self,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID
    ) -> List[ScalingPlan]:
        """Evaluate scaling needs and create plans"""
        plans = []
        
        for service, target in self.scaling_targets.items():
            # Collect current metrics
            metrics = await self._collect_service_metrics(service)
            
            if not metrics:
                continue
            
            # Store metrics
            self.metrics_buffer[service].append(metrics)
            
            # Evaluate based on strategy
            if strategy == ScalingStrategy.REACTIVE:
                plan = await self._reactive_evaluation(service, target, metrics)
            elif strategy == ScalingStrategy.PREDICTIVE:
                plan = await self._predictive_evaluation(service, target, metrics)
            elif strategy == ScalingStrategy.PROACTIVE:
                plan = await self._proactive_evaluation(service, target, metrics)
            elif strategy == ScalingStrategy.COST_OPTIMIZED:
                plan = await self._cost_optimized_evaluation(service, target, metrics)
            elif strategy == ScalingStrategy.PERFORMANCE_OPTIMIZED:
                plan = await self._performance_optimized_evaluation(service, target, metrics)
            elif strategy == ScalingStrategy.HYBRID:
                plan = await self._hybrid_evaluation(service, target, metrics)
            else:  # BALANCED
                plan = await self._balanced_evaluation(service, target, metrics)
            
            if plan and plan.decision != ScalingDecision.NO_ACTION:
                plans.append(plan)
                self.active_plans[plan.plan_id] = plan
        
        return plans
    
    async def _collect_service_metrics(self, service: str) -> Dict[str, Any]:
        """Collect comprehensive metrics for a service"""
        # Get resource metrics
        resource_metrics = await self.resource_optimizer.collect_system_metrics()
        
        # Get load metrics if available
        load_metrics = None
        if self.load_manager and self.load_manager.load_history:
            latest_snapshot = self.load_manager.load_history[-1]
            load_metrics = {
                "total_tasks": latest_snapshot.total_tasks,
                "queued_tasks": latest_snapshot.queued_tasks,
                "avg_response_time": latest_snapshot.avg_response_time,
                "throughput": latest_snapshot.throughput,
                "error_rate": latest_snapshot.error_rate
            }
        
        # Get service-specific metrics
        current_instances = self.base_scaler.current_instances.get(service, 1)
        
        return {
            "timestamp": datetime.utcnow(),
            "service": service,
            "current_instances": current_instances,
            "cpu_utilization": resource_metrics.cpu_percent,
            "memory_utilization": resource_metrics.memory_percent,
            "network_throughput": resource_metrics.network_throughput_mbps,
            "active_connections": resource_metrics.active_connections,
            "response_time_ms": resource_metrics.response_time_ms,
            "error_rate": resource_metrics.error_rate,
            "cost_per_hour": resource_metrics.cost_per_hour,
            "load_metrics": load_metrics
        }
    
    async def _reactive_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Traditional threshold-based scaling"""
        current_instances = metrics["current_instances"]
        cpu_util = metrics["cpu_utilization"]
        memory_util = metrics["memory_utilization"]
        
        # Determine scaling decision
        decision = ScalingDecision.NO_ACTION
        reason = ""
        
        # Combined utilization
        utilization = max(cpu_util, memory_util)
        
        if utilization > target.target_utilization + 20:  # Scale up threshold
            if current_instances < target.max_instances:
                decision = ScalingDecision.SCALE_OUT
                reason = f"High utilization: {utilization:.1f}%"
        elif utilization < target.target_utilization - 20:  # Scale down threshold
            if current_instances > target.min_instances:
                decision = ScalingDecision.SCALE_IN
                reason = f"Low utilization: {utilization:.1f}%"
        
        if decision == ScalingDecision.NO_ACTION:
            return None
        
        # Calculate desired state
        if decision == ScalingDecision.SCALE_OUT:
            desired_instances = min(
                current_instances + target.scaling_step,
                target.max_instances
            )
        else:
            desired_instances = max(
                current_instances - target.scaling_step,
                target.min_instances
            )
        
        plan = ScalingPlan(
            plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            strategy=ScalingStrategy.REACTIVE,
            decision=decision,
            target=target,
            current_state={"instances": current_instances, "utilization": utilization},
            desired_state={"instances": desired_instances},
            reason=reason,
            confidence=0.9,
            estimated_cost_impact=self._calculate_cost_impact(
                current_instances, desired_instances, target
            ),
            estimated_performance_impact=self._calculate_performance_impact(
                utilization, desired_instances, current_instances
            ),
            execution_steps=self._create_execution_steps(
                service, current_instances, desired_instances, decision
            )
        )
        
        return plan
    
    async def _predictive_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """ML-based predictive scaling"""
        if not self.enable_predictive_scaling:
            return await self._reactive_evaluation(service, target, metrics)
        
        # Update prediction models if needed
        if time.time() - self.last_model_update > self.model_update_interval:
            await self._update_prediction_models()
        
        # Get or create prediction model
        if service not in self.prediction_models:
            await self._train_prediction_model(service)
        
        model = self.prediction_models.get(service)
        if not model:
            return await self._reactive_evaluation(service, target, metrics)
        
        # Prepare features
        features = self._extract_features(service, metrics)
        
        # Predict future load
        try:
            prediction = model.model.predict([features])[0]
            
            # Predict utilization in N minutes
            predicted_utilization = prediction * 100  # Convert to percentage
            current_instances = metrics["current_instances"]
            
            # Calculate required instances
            required_instances = math.ceil(
                (predicted_utilization / target.target_utilization) * current_instances
            )
            
            # Apply bounds
            required_instances = max(
                target.min_instances,
                min(required_instances, target.max_instances)
            )
            
            # Determine decision
            if required_instances > current_instances:
                decision = ScalingDecision.SCALE_OUT
                reason = f"Predicted high utilization: {predicted_utilization:.1f}% in {model.prediction_horizon} minutes"
            elif required_instances < current_instances:
                decision = ScalingDecision.SCALE_IN
                reason = f"Predicted low utilization: {predicted_utilization:.1f}% in {model.prediction_horizon} minutes"
            else:
                return None
            
            plan = ScalingPlan(
                plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                strategy=ScalingStrategy.PREDICTIVE,
                decision=decision,
                target=target,
                current_state={
                    "instances": current_instances,
                    "utilization": max(metrics["cpu_utilization"], metrics["memory_utilization"])
                },
                desired_state={
                    "instances": required_instances,
                    "predicted_utilization": predicted_utilization
                },
                reason=reason,
                confidence=model.accuracy_score,
                estimated_cost_impact=self._calculate_cost_impact(
                    current_instances, required_instances, target
                ),
                estimated_performance_impact=self._calculate_performance_impact(
                    predicted_utilization, required_instances, current_instances
                ),
                execution_steps=self._create_execution_steps(
                    service, current_instances, required_instances, decision
                )
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Prediction failed for {service}: {e}")
            return await self._reactive_evaluation(service, target, metrics)
    
    async def _proactive_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Pattern-based proactive scaling"""
        # Detect patterns in historical data
        patterns = await self._detect_patterns(service)
        
        if not patterns:
            return await self._reactive_evaluation(service, target, metrics)
        
        current_instances = metrics["current_instances"]
        decision = ScalingDecision.NO_ACTION
        reason = ""
        confidence = 0.0
        
        # Check for known patterns
        for pattern in patterns:
            if pattern["type"] == "daily_peak":
                # Check if approaching peak time
                current_hour = datetime.utcnow().hour
                peak_hour = pattern["peak_hour"]
                
                if abs(current_hour - peak_hour) <= 1:  # Within 1 hour of peak
                    if current_instances < pattern["recommended_instances"]:
                        decision = ScalingDecision.SCALE_OUT
                        reason = f"Approaching daily peak at {peak_hour}:00"
                        confidence = pattern["confidence"]
                        required_instances = pattern["recommended_instances"]
                        break
                
            elif pattern["type"] == "weekly_pattern":
                # Check day of week patterns
                current_day = datetime.utcnow().weekday()
                if current_day in pattern["high_load_days"]:
                    if current_instances < pattern["recommended_instances"]:
                        decision = ScalingDecision.SCALE_OUT
                        reason = "High load day pattern detected"
                        confidence = pattern["confidence"]
                        required_instances = pattern["recommended_instances"]
                        break
        
        if decision == ScalingDecision.NO_ACTION:
            return None
        
        plan = ScalingPlan(
            plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            strategy=ScalingStrategy.PROACTIVE,
            decision=decision,
            target=target,
            current_state={"instances": current_instances},
            desired_state={"instances": required_instances},
            reason=reason,
            confidence=confidence,
            estimated_cost_impact=self._calculate_cost_impact(
                current_instances, required_instances, target
            ),
            estimated_performance_impact=0.2,  # Proactive = better performance
            execution_steps=self._create_execution_steps(
                service, current_instances, required_instances, decision
            )
        )
        
        return plan
    
    async def _cost_optimized_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Cost-optimized scaling decisions"""
        current_instances = metrics["current_instances"]
        utilization = max(metrics["cpu_utilization"], metrics["memory_utilization"])
        
        # Calculate current cost
        current_cost = current_instances * target.cost_per_instance
        
        # Find most cost-effective configuration
        best_config = None
        min_cost = current_cost
        
        for instances in range(target.min_instances, target.max_instances + 1):
            # Estimate utilization with this instance count
            estimated_util = (utilization * current_instances) / instances
            
            # Check if configuration is viable
            if estimated_util > 90:  # Too high utilization
                continue
            
            # Calculate cost including potential spot instances
            if self.enable_spot_instances and instances > target.min_instances:
                # Use spot for non-critical capacity
                on_demand = target.min_instances
                spot = instances - on_demand
                cost = (on_demand * target.cost_per_instance + 
                       spot * target.cost_per_instance * 0.3)  # 70% discount
            else:
                cost = instances * target.cost_per_instance
            
            # Add penalty for very low utilization (waste)
            if estimated_util < 30:
                cost *= 1.2
            
            if cost < min_cost:
                min_cost = cost
                best_config = instances
        
        if best_config and best_config != current_instances:
            if best_config > current_instances:
                decision = ScalingDecision.SCALE_OUT
            else:
                decision = ScalingDecision.SCALE_IN
            
            reason = f"Cost optimization: ${current_cost:.2f}/hr -> ${min_cost:.2f}/hr"
            
            plan = ScalingPlan(
                plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                strategy=ScalingStrategy.COST_OPTIMIZED,
                decision=decision,
                target=target,
                current_state={
                    "instances": current_instances,
                    "cost_per_hour": current_cost
                },
                desired_state={
                    "instances": best_config,
                    "cost_per_hour": min_cost
                },
                reason=reason,
                confidence=0.85,
                estimated_cost_impact=min_cost - current_cost,
                estimated_performance_impact=self._calculate_performance_impact(
                    utilization, best_config, current_instances
                ),
                execution_steps=self._create_execution_steps(
                    service, current_instances, best_config, decision
                )
            )
            
            return plan
        
        return None
    
    async def _performance_optimized_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Performance-optimized scaling decisions"""
        current_instances = metrics["current_instances"]
        response_time = metrics["response_time_ms"]
        error_rate = metrics["error_rate"]
        
        # Performance thresholds
        target_response_time = 100  # ms
        target_error_rate = 0.001
        
        # Check if performance is degraded
        if response_time > target_response_time or error_rate > target_error_rate:
            # Scale aggressively
            if current_instances < target.max_instances:
                # Calculate required instances for target performance
                if response_time > target_response_time:
                    scale_factor = response_time / target_response_time
                else:
                    scale_factor = error_rate / target_error_rate
                
                required_instances = min(
                    math.ceil(current_instances * scale_factor),
                    target.max_instances
                )
                
                if required_instances > current_instances:
                    plan = ScalingPlan(
                        plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
                        timestamp=datetime.utcnow(),
                        strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED,
                        decision=ScalingDecision.SCALE_OUT,
                        target=target,
                        current_state={
                            "instances": current_instances,
                            "response_time_ms": response_time,
                            "error_rate": error_rate
                        },
                        desired_state={"instances": required_instances},
                        reason=f"Performance degradation: {response_time:.0f}ms response time",
                        confidence=0.95,
                        estimated_cost_impact=self._calculate_cost_impact(
                            current_instances, required_instances, target
                        ),
                        estimated_performance_impact=0.5,  # Significant improvement expected
                        execution_steps=self._create_execution_steps(
                            service, current_instances, required_instances, ScalingDecision.SCALE_OUT
                        )
                    )
                    
                    return plan
        
        # Check if we can scale down without impacting performance
        elif response_time < target_response_time * 0.5 and error_rate < target_error_rate * 0.5:
            if current_instances > target.min_instances:
                # Conservative scale down
                test_instances = current_instances - 1
                
                # Estimate new response time
                estimated_response_time = response_time * (current_instances / test_instances)
                
                if estimated_response_time < target_response_time * 0.8:
                    plan = ScalingPlan(
                        plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
                        timestamp=datetime.utcnow(),
                        strategy=ScalingStrategy.PERFORMANCE_OPTIMIZED,
                        decision=ScalingDecision.SCALE_IN,
                        target=target,
                        current_state={
                            "instances": current_instances,
                            "response_time_ms": response_time
                        },
                        desired_state={"instances": test_instances},
                        reason="Excess capacity with good performance",
                        confidence=0.8,
                        estimated_cost_impact=self._calculate_cost_impact(
                            current_instances, test_instances, target
                        ),
                        estimated_performance_impact=-0.1,  # Slight degradation
                        execution_steps=self._create_execution_steps(
                            service, current_instances, test_instances, ScalingDecision.SCALE_IN
                        )
                    )
                    
                    return plan
        
        return None
    
    async def _hybrid_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Hybrid strategy combining multiple approaches"""
        # Collect recommendations from different strategies
        strategies = [
            self._reactive_evaluation,
            self._predictive_evaluation,
            self._proactive_evaluation,
            self._cost_optimized_evaluation,
            self._performance_optimized_evaluation
        ]
        
        recommendations = []
        for strategy_func in strategies:
            try:
                plan = await strategy_func(service, target, metrics)
                if plan:
                    recommendations.append(plan)
            except Exception as e:
                logger.error(f"Strategy evaluation failed: {e}")
        
        if not recommendations:
            return None
        
        # Score and rank recommendations
        scored_plans = []
        for plan in recommendations:
            score = self._score_scaling_plan(plan, metrics)
            scored_plans.append((score, plan))
        
        # Select best plan
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan = scored_plans[0]
        
        # Update plan with hybrid strategy
        best_plan.strategy = ScalingStrategy.HYBRID
        best_plan.confidence = best_score
        
        return best_plan
    
    async def _balanced_evaluation(
        self,
        service: str,
        target: ScalingTarget,
        metrics: Dict[str, Any]
    ) -> Optional[ScalingPlan]:
        """Balanced strategy considering both cost and performance"""
        current_instances = metrics["current_instances"]
        utilization = max(metrics["cpu_utilization"], metrics["memory_utilization"])
        response_time = metrics["response_time_ms"]
        current_cost = current_instances * target.cost_per_instance
        
        # Define acceptable ranges
        util_range = (target.target_utilization - 10, target.target_utilization + 10)
        response_range = (0, 150)  # ms
        
        # Check if current state is balanced
        if (util_range[0] <= utilization <= util_range[1] and 
            response_time <= response_range[1]):
            return None  # Already balanced
        
        # Find optimal balance
        best_config = current_instances
        best_score = 0
        
        for instances in range(target.min_instances, target.max_instances + 1):
            # Estimate metrics with this configuration
            est_util = (utilization * current_instances) / instances
            est_response = response_time * (instances / current_instances) ** -0.7  # Empirical
            est_cost = instances * target.cost_per_instance
            
            # Calculate balance score
            util_score = 1 - abs(est_util - target.target_utilization) / 100
            perf_score = max(0, 1 - est_response / 200)  # Normalize to 0-1
            cost_score = 1 - est_cost / (target.max_instances * target.cost_per_instance)
            
            # Weighted balance
            balance_score = (
                util_score * 0.3 +
                perf_score * 0.4 +
                cost_score * 0.3
            )
            
            if balance_score > best_score:
                best_score = balance_score
                best_config = instances
        
        if best_config != current_instances:
            if best_config > current_instances:
                decision = ScalingDecision.SCALE_OUT
            else:
                decision = ScalingDecision.SCALE_IN
            
            plan = ScalingPlan(
                plan_id=f"plan-{service}-{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                strategy=ScalingStrategy.BALANCED,
                decision=decision,
                target=target,
                current_state={
                    "instances": current_instances,
                    "balance_score": self._calculate_balance_score(metrics)
                },
                desired_state={
                    "instances": best_config,
                    "balance_score": best_score
                },
                reason="Optimizing for balanced cost and performance",
                confidence=best_score,
                estimated_cost_impact=self._calculate_cost_impact(
                    current_instances, best_config, target
                ),
                estimated_performance_impact=self._calculate_performance_impact(
                    utilization, best_config, current_instances
                ),
                execution_steps=self._create_execution_steps(
                    service, current_instances, best_config, decision
                )
            )
            
            return plan
        
        return None
    
    def _score_scaling_plan(self, plan: ScalingPlan, metrics: Dict[str, Any]) -> float:
        """Score a scaling plan based on multiple factors"""
        score = 0.0
        
        # Factor 1: Strategy appropriateness
        if plan.strategy == ScalingStrategy.PREDICTIVE and self.enable_predictive_scaling:
            score += 0.2
        elif plan.strategy == ScalingStrategy.COST_OPTIMIZED and self.enable_cost_optimization:
            score += 0.15
        
        # Factor 2: Confidence
        score += plan.confidence * 0.3
        
        # Factor 3: Cost impact
        if plan.estimated_cost_impact < 0:  # Saves money
            score += 0.2
        elif plan.estimated_cost_impact > 0:  # Costs more
            score -= min(0.1, plan.estimated_cost_impact / 100)
        
        # Factor 4: Performance impact
        score += plan.estimated_performance_impact * 0.2
        
        # Factor 5: Stability (avoid frequent changes)
        recent_changes = sum(
            1 for event in self.scaling_history
            if event.timestamp > datetime.utcnow() - timedelta(hours=1)
        )
        if recent_changes < 2:
            score += self.stability_bonus
        
        return min(1.0, max(0.0, score))
    
    def _calculate_cost_impact(
        self,
        current_instances: int,
        desired_instances: int,
        target: ScalingTarget
    ) -> float:
        """Calculate cost impact of scaling"""
        current_cost = current_instances * target.cost_per_instance
        desired_cost = desired_instances * target.cost_per_instance
        
        return desired_cost - current_cost
    
    def _calculate_performance_impact(
        self,
        current_utilization: float,
        desired_instances: int,
        current_instances: int
    ) -> float:
        """Estimate performance impact of scaling"""
        if desired_instances > current_instances:
            # Scaling up improves performance
            improvement = (desired_instances - current_instances) / current_instances
            return min(0.5, improvement * 0.3)
        elif desired_instances < current_instances:
            # Scaling down may degrade performance
            new_utilization = (current_utilization * current_instances) / desired_instances
            if new_utilization > 80:
                return -0.3  # Significant degradation
            elif new_utilization > 60:
                return -0.1  # Minor degradation
            else:
                return 0  # No impact
        
        return 0
    
    def _calculate_balance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate current balance score"""
        util = max(metrics["cpu_utilization"], metrics["memory_utilization"])
        response_time = metrics["response_time_ms"]
        cost = metrics["cost_per_hour"]
        
        # Normalize metrics
        util_score = 1 - abs(util - 60) / 100  # Target 60% utilization
        perf_score = max(0, 1 - response_time / 200)
        cost_score = 1 - cost / 1000  # Assume $1000/hr is max
        
        return (util_score + perf_score + cost_score) / 3
    
    def _create_execution_steps(
        self,
        service: str,
        current: int,
        desired: int,
        decision: ScalingDecision
    ) -> List[Dict[str, Any]]:
        """Create detailed execution steps for scaling"""
        steps = []
        
        if decision in [ScalingDecision.SCALE_OUT, ScalingDecision.SCALE_UP]:
            instances_to_add = desired - current
            
            steps.append({
                "step": 1,
                "action": "provision_instances",
                "details": {
                    "count": instances_to_add,
                    "type": "standard",  # Could be spot for cost optimization
                    "timeout_seconds": 300
                }
            })
            
            steps.append({
                "step": 2,
                "action": "configure_instances",
                "details": {
                    "service": service,
                    "configuration": "standard"
                }
            })
            
            steps.append({
                "step": 3,
                "action": "update_load_balancer",
                "details": {
                    "add_instances": instances_to_add
                }
            })
            
            steps.append({
                "step": 4,
                "action": "verify_health",
                "details": {
                    "health_check_url": "/health",
                    "expected_instances": desired
                }
            })
            
        elif decision in [ScalingDecision.SCALE_IN, ScalingDecision.SCALE_DOWN]:
            instances_to_remove = current - desired
            
            steps.append({
                "step": 1,
                "action": "drain_connections",
                "details": {
                    "instances": instances_to_remove,
                    "timeout_seconds": 120
                }
            })
            
            steps.append({
                "step": 2,
                "action": "remove_from_load_balancer",
                "details": {
                    "instances": instances_to_remove
                }
            })
            
            steps.append({
                "step": 3,
                "action": "terminate_instances",
                "details": {
                    "count": instances_to_remove,
                    "grace_period_seconds": 30
                }
            })
            
            steps.append({
                "step": 4,
                "action": "verify_capacity",
                "details": {
                    "expected_instances": desired,
                    "monitor_duration_seconds": 60
                }
            })
        
        return steps
    
    async def execute_scaling_plan(self, plan: ScalingPlan) -> bool:
        """Execute a scaling plan"""
        logger.info(f"Executing scaling plan {plan.plan_id}: {plan.reason}")
        
        success = True
        start_time = time.time()
        
        try:
            # Execute each step
            for step in plan.execution_steps:
                logger.info(f"Executing step {step['step']}: {step['action']}")
                
                # Simulate execution (in production, would call actual APIs)
                await asyncio.sleep(2)
                
                # For now, update the base scaler
                if step['action'] == 'verify_health':
                    self.base_scaler.current_instances[plan.target.service] = plan.desired_state['instances']
            
            # Record scaling event
            event = ScalingEvent(
                event_id=plan.plan_id,
                timestamp=plan.timestamp,
                scaling_policy=ScalingPolicy.CUSTOM,
                trigger_metric=plan.strategy.value,
                trigger_value=0,  # Would be actual metric value
                threshold=0,
                action=plan.decision.value,
                instances_before=plan.current_state['instances'],
                instances_after=plan.desired_state['instances'],
                region="global",
                success=True,
                duration_seconds=time.time() - start_time
            )
            
            self.scaling_history.append(event)
            self.base_scaler.scaling_history.append(event)
            
            logger.info(f"Scaling plan {plan.plan_id} executed successfully")
            
        except Exception as e:
            logger.error(f"Failed to execute scaling plan {plan.plan_id}: {e}")
            success = False
            
            # Execute rollback if available
            if plan.rollback_plan:
                await self._execute_rollback(plan)
        
        finally:
            # Remove from active plans
            if plan.plan_id in self.active_plans:
                del self.active_plans[plan.plan_id]
        
        return success
    
    async def _execute_rollback(self, plan: ScalingPlan):
        """Execute rollback for failed scaling plan"""
        logger.warning(f"Executing rollback for plan {plan.plan_id}")
        # Would implement actual rollback logic
        pass
    
    async def _detect_patterns(self, service: str) -> List[Dict[str, Any]]:
        """Detect patterns in service load"""
        if service not in self.metrics_buffer or len(self.metrics_buffer[service]) < 168:  # 1 week
            return []
        
        patterns = []
        metrics_list = list(self.metrics_buffer[service])
        
        # Daily pattern detection
        hourly_loads = defaultdict(list)
        for metric in metrics_list:
            hour = metric['timestamp'].hour
            utilization = max(metric['cpu_utilization'], metric['memory_utilization'])
            hourly_loads[hour].append(utilization)
        
        # Find peak hours
        avg_hourly = {hour: statistics.mean(loads) for hour, loads in hourly_loads.items()}
        peak_hour = max(avg_hourly.items(), key=lambda x: x[1])[0]
        peak_load = avg_hourly[peak_hour]
        
        if peak_load > 70:  # Significant peak
            patterns.append({
                "type": "daily_peak",
                "peak_hour": peak_hour,
                "peak_load": peak_load,
                "recommended_instances": math.ceil(
                    self.scaling_targets[service].max_instances * (peak_load / 100)
                ),
                "confidence": 0.8
            })
        
        # Weekly pattern detection
        daily_loads = defaultdict(list)
        for metric in metrics_list:
            day = metric['timestamp'].weekday()
            utilization = max(metric['cpu_utilization'], metric['memory_utilization'])
            daily_loads[day].append(utilization)
        
        avg_daily = {day: statistics.mean(loads) for day, loads in daily_loads.items()}
        high_load_days = [day for day, load in avg_daily.items() if load > 60]
        
        if high_load_days:
            patterns.append({
                "type": "weekly_pattern",
                "high_load_days": high_load_days,
                "recommended_instances": self.scaling_targets[service].max_instances,
                "confidence": 0.7
            })
        
        self.detected_patterns[service] = [p["type"] for p in patterns]
        
        return patterns
    
    def _extract_features(self, service: str, metrics: Dict[str, Any]) -> List[float]:
        """Extract features for ML prediction"""
        features = [
            metrics['cpu_utilization'],
            metrics['memory_utilization'],
            metrics['network_throughput'],
            metrics['active_connections'],
            metrics['response_time_ms'],
            metrics['error_rate'],
            datetime.utcnow().hour,  # Hour of day
            datetime.utcnow().weekday(),  # Day of week
            metrics['current_instances']
        ]
        
        # Add historical features if available
        if service in self.metrics_buffer and len(self.metrics_buffer[service]) > 10:
            recent_metrics = list(self.metrics_buffer[service])[-10:]
            
            # Add rolling averages
            features.append(
                statistics.mean(m['cpu_utilization'] for m in recent_metrics)
            )
            features.append(
                statistics.mean(m['memory_utilization'] for m in recent_metrics)
            )
            
            # Add trend
            cpu_values = [m['cpu_utilization'] for m in recent_metrics]
            if len(cpu_values) > 1:
                trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
                features.append(trend)
            else:
                features.append(0)
        else:
            features.extend([50, 50, 0])  # Default values
        
        return features
    
    async def _train_prediction_model(self, service: str):
        """Train prediction model for a service"""
        if service not in self.metrics_buffer or len(self.metrics_buffer[service]) < 100:
            return
        
        logger.info(f"Training prediction model for {service}")
        
        # Prepare training data
        metrics_list = list(self.metrics_buffer[service])
        X = []
        y = []
        
        # Create features and labels
        for i in range(len(metrics_list) - 10):
            features = self._extract_features(service, metrics_list[i])
            # Label is max utilization in next 10 minutes
            future_utils = [
                max(m['cpu_utilization'], m['memory_utilization'])
                for m in metrics_list[i+1:i+11]
            ]
            label = max(future_utils) / 100  # Normalize to 0-1
            
            X.append(features)
            y.append(label)
        
        # Train model
        try:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_test, y_test)
            
            # Store model
            self.prediction_models[service] = PredictionModel(
                model_type="random_forest",
                model=model,
                feature_names=[
                    "cpu_util", "mem_util", "network", "connections",
                    "response_time", "error_rate", "hour", "day",
                    "instances", "cpu_avg", "mem_avg", "trend"
                ],
                last_trained=datetime.utcnow(),
                accuracy_score=score,
                prediction_horizon=10
            )
            
            logger.info(f"Model trained for {service} with accuracy: {score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {service}: {e}")
    
    async def _update_prediction_models(self):
        """Update all prediction models"""
        for service in self.scaling_targets:
            await self._train_prediction_model(service)
        
        self.last_model_update = time.time()
    
    def _initialize_pattern_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize known patterns library"""
        return {
            "business_hours": {
                "description": "Higher load during business hours",
                "detection": lambda h: 9 <= h <= 17,
                "scale_factor": 1.5
            },
            "weekend": {
                "description": "Lower load on weekends",
                "detection": lambda d: d in [5, 6],
                "scale_factor": 0.7
            },
            "month_end": {
                "description": "Higher load at month end",
                "detection": lambda day: day >= 28,
                "scale_factor": 1.3
            },
            "black_friday": {
                "description": "Extreme load on Black Friday",
                "detection": lambda date: date.month == 11 and date.day == 24,
                "scale_factor": 3.0
            }
        }
    
    async def get_scaling_insights(self) -> Dict[str, Any]:
        """Get comprehensive scaling insights"""
        insights = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "overall_metrics": {
                "total_instances": sum(
                    self.base_scaler.current_instances.values()
                ),
                "total_cost_per_hour": 0,
                "scaling_events_last_hour": 0,
                "active_plans": len(self.active_plans)
            },
            "patterns_detected": dict(self.detected_patterns),
            "model_performance": {},
            "recommendations": []
        }
        
        # Service-specific insights
        for service, target in self.scaling_targets.items():
            current = self.base_scaler.current_instances.get(service, 0)
            cost = current * target.cost_per_instance
            
            insights["services"][service] = {
                "current_instances": current,
                "min_instances": target.min_instances,
                "max_instances": target.max_instances,
                "cost_per_hour": cost,
                "utilization": "N/A",
                "scaling_events_24h": sum(
                    1 for event in self.scaling_history
                    if event.timestamp > datetime.utcnow() - timedelta(hours=24)
                    and service in event.event_id
                )
            }
            
            insights["overall_metrics"]["total_cost_per_hour"] += cost
        
        # Model performance
        for service, model in self.prediction_models.items():
            insights["model_performance"][service] = {
                "accuracy": model.accuracy_score,
                "last_trained": model.last_trained.isoformat(),
                "prediction_horizon_minutes": model.prediction_horizon
            }
        
        # Recent scaling events
        recent_events = [
            event for event in self.scaling_history
            if event.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        insights["overall_metrics"]["scaling_events_last_hour"] = len(recent_events)
        
        # Generate recommendations
        if insights["overall_metrics"]["scaling_events_last_hour"] > 5:
            insights["recommendations"].append({
                "type": "high_scaling_frequency",
                "description": "Consider adjusting scaling thresholds",
                "priority": "medium"
            })
        
        if insights["overall_metrics"]["total_cost_per_hour"] > 500:
            insights["recommendations"].append({
                "type": "cost_optimization",
                "description": "Enable spot instances for non-critical workloads",
                "priority": "high"
            })
        
        return insights
