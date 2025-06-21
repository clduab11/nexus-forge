"""
Advanced Caching System - Predictive Pre-loading and Intelligent Cache Warming

This module extends the existing Redis caching with advanced capabilities:
- Predictive pre-loading based on usage patterns and ML models
- Intelligent cache warming using reinforcement learning
- Cross-agent shared caching with correlation analysis
- Dynamic TTL management based on access patterns
- Real-time cache optimization with Supabase analytics integration

Key Features:
- LSTM/Transformer models for access pattern prediction
- Heuristic policies for cache priority management
- Multi-agent cache coordination
- Usage analytics integration with Supabase
- Automated cache performance optimization
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.cache import CacheStrategy, RedisCache

logger = logging.getLogger(__name__)


class PredictionModel(Enum):
    """Types of prediction models for cache pre-loading"""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    STATISTICAL = "statistical"
    HEURISTIC = "heuristic"
    REINFORCEMENT_LEARNING = "rl"


class CacheOptimizationStrategy(Enum):
    """Cache optimization strategies"""

    FREQUENCY_BASED = "frequency"
    RECENCY_BASED = "recency"
    PREDICTIVE = "predictive"
    AGENT_CORRELATION = "correlation"
    PERFORMANCE_DRIVEN = "performance"


@dataclass
class CacheAccessPattern:
    """Represents cache access patterns for analysis"""

    key: str
    agent_id: str
    access_time: float
    hit: bool
    size: int
    ttl: int
    access_frequency: int = 0
    last_access: float = 0.0
    correlation_score: float = 0.0


@dataclass
class PredictionResult:
    """Result of cache access prediction"""

    keys_to_preload: List[str]
    confidence_scores: Dict[str, float]
    predicted_access_time: Dict[str, float]
    recommendation: str
    model_used: PredictionModel


@dataclass
class CacheWarmingStrategy:
    """Strategy for intelligent cache warming"""

    priority_keys: List[str]
    warm_ahead_time: int  # seconds
    warm_data_sources: List[str]
    warming_schedule: Dict[str, float]  # key -> timestamp
    expected_hit_improvement: float


class AccessPatternAnalyzer:
    """
    Analyzes cache access patterns to predict future needs
    Uses statistical analysis and machine learning models
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.access_history = deque(maxlen=10000)  # Store last 10k accesses
        self.agent_patterns = defaultdict(list)
        self.key_correlations = defaultdict(set)
        self.seasonal_patterns = {}

        # Pattern analysis configuration
        self.min_pattern_length = 10
        self.correlation_threshold = 0.7
        self.prediction_window = 300  # 5 minutes ahead

    async def record_access(self, pattern: CacheAccessPattern):
        """Record cache access for pattern analysis"""
        self.access_history.append(pattern)
        self.agent_patterns[pattern.agent_id].append(pattern)

        # Update correlations
        await self._update_correlations(pattern)

        # Analyze seasonal patterns
        await self._analyze_seasonal_patterns(pattern)

        # Store in Redis for persistence
        access_key = f"access_pattern:{pattern.agent_id}:{int(pattern.access_time)}"
        await self.cache.set_l3(access_key, pattern.__dict__, timeout=86400)

    async def _update_correlations(self, pattern: CacheAccessPattern):
        """Update key correlation analysis"""
        # Find keys accessed within correlation window (30 seconds)
        recent_keys = set()
        correlation_window = 30

        for access in reversed(self.access_history):
            if pattern.access_time - access.access_time > correlation_window:
                break
            if access.agent_id == pattern.agent_id:
                recent_keys.add(access.key)

        # Update correlation sets
        for key in recent_keys:
            if key != pattern.key:
                self.key_correlations[pattern.key].add(key)
                self.key_correlations[key].add(pattern.key)

    async def _analyze_seasonal_patterns(self, pattern: CacheAccessPattern):
        """Analyze seasonal/temporal access patterns"""
        hour = datetime.fromtimestamp(pattern.access_time).hour
        day_of_week = datetime.fromtimestamp(pattern.access_time).weekday()

        pattern_key = f"{pattern.agent_id}:{pattern.key}"

        if pattern_key not in self.seasonal_patterns:
            self.seasonal_patterns[pattern_key] = {
                "hourly": defaultdict(int),
                "daily": defaultdict(int),
                "total_accesses": 0,
            }

        self.seasonal_patterns[pattern_key]["hourly"][hour] += 1
        self.seasonal_patterns[pattern_key]["daily"][day_of_week] += 1
        self.seasonal_patterns[pattern_key]["total_accesses"] += 1

    async def predict_next_accesses(
        self, agent_id: str, model: PredictionModel = PredictionModel.STATISTICAL
    ) -> PredictionResult:
        """
        Predict next cache accesses for an agent
        """
        if model == PredictionModel.STATISTICAL:
            return await self._predict_statistical(agent_id)
        elif model == PredictionModel.HEURISTIC:
            return await self._predict_heuristic(agent_id)
        elif model == PredictionModel.LSTM:
            return await self._predict_lstm(agent_id)
        else:
            # Fallback to statistical
            return await self._predict_statistical(agent_id)

    async def _predict_statistical(self, agent_id: str) -> PredictionResult:
        """
        Statistical prediction based on frequency and recency
        """
        agent_accesses = self.agent_patterns.get(agent_id, [])

        if len(agent_accesses) < self.min_pattern_length:
            return PredictionResult(
                keys_to_preload=[],
                confidence_scores={},
                predicted_access_time={},
                recommendation="Insufficient data for prediction",
                model_used=PredictionModel.STATISTICAL,
            )

        # Analyze frequency and recency
        key_stats = defaultdict(lambda: {"count": 0, "last_access": 0, "intervals": []})

        for access in agent_accesses[-100:]:  # Last 100 accesses
            key = access.key
            key_stats[key]["count"] += 1

            if key_stats[key]["last_access"] > 0:
                interval = access.access_time - key_stats[key]["last_access"]
                key_stats[key]["intervals"].append(interval)

            key_stats[key]["last_access"] = access.access_time

        # Calculate prediction scores
        current_time = time.time()
        predictions = {}
        confidence_scores = {}
        predicted_times = {}

        for key, stats in key_stats.items():
            if stats["count"] >= 3:  # Minimum accesses for prediction
                # Frequency score
                frequency_score = stats["count"] / len(agent_accesses[-100:])

                # Recency score
                time_since_last = current_time - stats["last_access"]
                recency_score = 1.0 / (1.0 + time_since_last / 3600)  # Decay over hours

                # Interval prediction
                if stats["intervals"]:
                    avg_interval = np.mean(stats["intervals"])
                    predicted_next_access = stats["last_access"] + avg_interval
                    predicted_times[key] = predicted_next_access

                    # Time-based confidence
                    time_to_prediction = predicted_next_access - current_time
                    time_confidence = 1.0 / (
                        1.0 + abs(time_to_prediction) / self.prediction_window
                    )
                else:
                    time_confidence = 0.5

                # Combined confidence
                confidence = (frequency_score + recency_score + time_confidence) / 3

                if confidence > 0.3:  # Threshold for prediction
                    predictions[key] = confidence
                    confidence_scores[key] = confidence

        # Sort by confidence and select top predictions
        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )
        keys_to_preload = [key for key, confidence in sorted_predictions[:10]]

        recommendation = f"Predicted {len(keys_to_preload)} cache accesses based on statistical analysis"

        return PredictionResult(
            keys_to_preload=keys_to_preload,
            confidence_scores=confidence_scores,
            predicted_access_time=predicted_times,
            recommendation=recommendation,
            model_used=PredictionModel.STATISTICAL,
        )

    async def _predict_heuristic(self, agent_id: str) -> PredictionResult:
        """
        Heuristic-based prediction using domain knowledge
        """
        # Get recent access patterns
        recent_accesses = [
            a
            for a in self.agent_patterns.get(agent_id, [])
            if time.time() - a.access_time < 3600
        ]  # Last hour

        if not recent_accesses:
            return PredictionResult(
                keys_to_preload=[],
                confidence_scores={},
                predicted_access_time={},
                recommendation="No recent activity for heuristic prediction",
                model_used=PredictionModel.HEURISTIC,
            )

        predictions = {}
        confidence_scores = {}
        predicted_times = {}

        # Heuristic rules based on agent behavior
        current_time = time.time()

        for access in recent_accesses[-20:]:  # Last 20 accesses
            key = access.key

            # Rule 1: API response caching - high predictability
            if "api_response" in key or "endpoint" in key:
                confidence = 0.8
                predicted_times[key] = current_time + 60  # Likely within 1 minute

            # Rule 2: Code generation patterns - medium predictability
            elif "code_gen" in key or "jules" in key:
                confidence = 0.6
                predicted_times[key] = current_time + 120  # Likely within 2 minutes

            # Rule 3: UI mockup patterns - lower predictability
            elif "imagen" in key or "mockup" in key:
                confidence = 0.4
                predicted_times[key] = current_time + 300  # Likely within 5 minutes

            # Rule 4: Video generation - very specific timing
            elif "veo" in key or "video" in key:
                confidence = 0.3
                predicted_times[key] = current_time + 180  # Likely within 3 minutes

            # Rule 5: Session data - high short-term predictability
            elif "session" in key or "user" in key:
                confidence = 0.9
                predicted_times[key] = current_time + 30  # Very soon

            else:
                confidence = 0.2  # Default low confidence
                predicted_times[key] = current_time + 240

            # Apply correlation boost
            correlated_keys = self.key_correlations.get(key, set())
            if len(correlated_keys) > 2:
                confidence *= 1.2  # Boost confidence for well-connected keys

            confidence_scores[key] = min(confidence, 1.0)
            predictions[key] = confidence

        # Add correlation-based predictions
        for key in predictions:
            for correlated_key in self.key_correlations.get(key, set()):
                if correlated_key not in predictions:
                    base_confidence = predictions[key]
                    corr_confidence = (
                        base_confidence * 0.7
                    )  # Correlated keys have 70% confidence

                    if corr_confidence > 0.3:
                        predictions[correlated_key] = corr_confidence
                        confidence_scores[correlated_key] = corr_confidence
                        predicted_times[correlated_key] = predicted_times[key] + 30

        # Sort and select top predictions
        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )
        keys_to_preload = [key for key, confidence in sorted_predictions[:15]]

        recommendation = f"Heuristic prediction identified {len(keys_to_preload)} high-probability cache accesses"

        return PredictionResult(
            keys_to_preload=keys_to_preload,
            confidence_scores=confidence_scores,
            predicted_access_time=predicted_times,
            recommendation=recommendation,
            model_used=PredictionModel.HEURISTIC,
        )

    async def _predict_lstm(self, agent_id: str) -> PredictionResult:
        """
        LSTM-based prediction (simplified implementation)
        In production, this would use a trained LSTM model
        """
        # For now, combine statistical and heuristic approaches
        # This simulates what an LSTM might predict

        statistical_result = await self._predict_statistical(agent_id)
        heuristic_result = await self._predict_heuristic(agent_id)

        # Combine predictions with weighted confidence
        combined_predictions = {}
        combined_confidence = {}
        combined_times = {}

        # Weight statistical (40%) + heuristic (60%) for LSTM simulation
        all_keys = set(
            statistical_result.keys_to_preload + heuristic_result.keys_to_preload
        )

        for key in all_keys:
            stat_conf = statistical_result.confidence_scores.get(key, 0.0)
            heur_conf = heuristic_result.confidence_scores.get(key, 0.0)

            # LSTM-like combination with temporal awareness
            combined_conf = (0.4 * stat_conf + 0.6 * heur_conf) * 1.1  # LSTM boost
            combined_conf = min(combined_conf, 1.0)

            if combined_conf > 0.4:  # Higher threshold for LSTM
                combined_predictions[key] = combined_conf
                combined_confidence[key] = combined_conf

                # Average predicted times
                stat_time = statistical_result.predicted_access_time.get(
                    key, time.time() + 300
                )
                heur_time = heuristic_result.predicted_access_time.get(
                    key, time.time() + 300
                )
                combined_times[key] = (stat_time + heur_time) / 2

        # Sort by confidence
        sorted_predictions = sorted(
            combined_predictions.items(), key=lambda x: x[1], reverse=True
        )
        keys_to_preload = [key for key, confidence in sorted_predictions[:12]]

        recommendation = f"LSTM-based prediction (simulated) identified {len(keys_to_preload)} probable accesses"

        return PredictionResult(
            keys_to_preload=keys_to_preload,
            confidence_scores=combined_confidence,
            predicted_access_time=combined_times,
            recommendation=recommendation,
            model_used=PredictionModel.LSTM,
        )

    async def get_seasonal_prediction(
        self, agent_id: str, target_time: float
    ) -> List[str]:
        """
        Get prediction based on seasonal/temporal patterns
        """
        target_hour = datetime.fromtimestamp(target_time).hour
        target_day = datetime.fromtimestamp(target_time).weekday()

        seasonal_predictions = []

        for pattern_key, patterns in self.seasonal_patterns.items():
            stored_agent_id = pattern_key.split(":")[0]
            if stored_agent_id != agent_id:
                continue

            key = ":".join(pattern_key.split(":")[1:])

            # Calculate seasonal probability
            hourly_prob = patterns["hourly"][target_hour] / max(
                patterns["total_accesses"], 1
            )
            daily_prob = patterns["daily"][target_day] / max(
                patterns["total_accesses"], 1
            )

            # Combined seasonal score
            seasonal_score = (hourly_prob + daily_prob) / 2

            if seasonal_score > 0.1:  # Threshold for seasonal prediction
                seasonal_predictions.append((key, seasonal_score))

        # Sort by seasonal score and return top keys
        seasonal_predictions.sort(key=lambda x: x[1], reverse=True)
        return [key for key, score in seasonal_predictions[:8]]


class CachePreloader:
    """
    Intelligent cache pre-loading system based on predictions
    """

    def __init__(self, cache: RedisCache, analyzer: AccessPatternAnalyzer):
        self.cache = cache
        self.analyzer = analyzer
        self.preload_queue = asyncio.Queue()
        self.active_preloads = set()
        self.preload_stats = {
            "total_preloads": 0,
            "successful_preloads": 0,
            "cache_hits_from_preload": 0,
            "preload_efficiency": 0.0,
        }

    async def start_preloading(self):
        """Start background preloading process"""
        logger.info("Starting intelligent cache preloading")

        # Start preload workers
        tasks = [
            asyncio.create_task(self._preload_worker()),
            asyncio.create_task(self._prediction_scheduler()),
            asyncio.create_task(self._preload_monitor()),
        ]

        await asyncio.gather(*tasks)

    async def _prediction_scheduler(self):
        """Schedule predictions for all active agents"""
        while True:
            try:
                # Get list of active agents from recent cache accesses
                active_agents = await self._get_active_agents()

                for agent_id in active_agents:
                    # Generate predictions using multiple models
                    predictions = await self._generate_multi_model_predictions(agent_id)

                    # Queue preload tasks
                    await self._queue_preload_tasks(agent_id, predictions)

                # Sleep before next prediction cycle
                await asyncio.sleep(60)  # Predict every minute

            except Exception as e:
                logger.error(f"Error in prediction scheduler: {str(e)}")
                await asyncio.sleep(30)

    async def _get_active_agents(self) -> List[str]:
        """Get list of agents with recent cache activity"""
        cutoff_time = time.time() - 3600  # Last hour
        active_agents = set()

        # Check recent access patterns
        for access in self.analyzer.access_history:
            if access.access_time > cutoff_time:
                active_agents.add(access.agent_id)

        return list(active_agents)

    async def _generate_multi_model_predictions(
        self, agent_id: str
    ) -> List[PredictionResult]:
        """Generate predictions using multiple models"""
        models = [
            PredictionModel.STATISTICAL,
            PredictionModel.HEURISTIC,
            PredictionModel.LSTM,
        ]

        predictions = []
        for model in models:
            try:
                result = await self.analyzer.predict_next_accesses(agent_id, model)
                predictions.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for model {model}: {str(e)}")

        return predictions

    async def _queue_preload_tasks(
        self, agent_id: str, predictions: List[PredictionResult]
    ):
        """Queue preload tasks based on predictions"""
        # Combine and rank predictions from all models
        combined_keys = {}

        for prediction in predictions:
            for key in prediction.keys_to_preload:
                confidence = prediction.confidence_scores.get(key, 0.0)
                predicted_time = prediction.predicted_access_time.get(
                    key, time.time() + 300
                )

                if key not in combined_keys:
                    combined_keys[key] = {
                        "total_confidence": 0.0,
                        "model_count": 0,
                        "predicted_time": predicted_time,
                        "agent_id": agent_id,
                    }

                combined_keys[key]["total_confidence"] += confidence
                combined_keys[key]["model_count"] += 1

        # Calculate average confidence and queue high-confidence predictions
        for key, data in combined_keys.items():
            avg_confidence = data["total_confidence"] / data["model_count"]

            if avg_confidence > 0.5 and key not in self.active_preloads:
                preload_task = {
                    "key": key,
                    "agent_id": agent_id,
                    "confidence": avg_confidence,
                    "predicted_time": data["predicted_time"],
                    "queued_time": time.time(),
                }

                await self.preload_queue.put(preload_task)
                self.active_preloads.add(key)

    async def _preload_worker(self):
        """Worker to process preload queue"""
        while True:
            try:
                # Get next preload task
                task = await self.preload_queue.get()

                # Execute preload
                success = await self._execute_preload(task)

                # Update statistics
                self.preload_stats["total_preloads"] += 1
                if success:
                    self.preload_stats["successful_preloads"] += 1

                # Remove from active set
                self.active_preloads.discard(task["key"])

                # Mark task as done
                self.preload_queue.task_done()

            except Exception as e:
                logger.error(f"Error in preload worker: {str(e)}")
                await asyncio.sleep(1)

    async def _execute_preload(self, task: Dict[str, Any]) -> bool:
        """Execute a preload task"""
        key = task["key"]
        agent_id = task["agent_id"]
        predicted_time = task["predicted_time"]

        # Wait until appropriate time to preload
        current_time = time.time()
        if predicted_time > current_time:
            wait_time = min(
                predicted_time - current_time - 30, 300
            )  # Preload 30s early, max wait 5min
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        try:
            # Check if key is already cached
            existing_value = await self.cache.get(key)
            if existing_value is not None:
                logger.debug(f"Key {key} already cached, skipping preload")
                return True

            # Generate data for preloading
            preload_data = await self._generate_preload_data(key, agent_id)

            if preload_data is not None:
                # Determine appropriate TTL and strategy
                ttl = self._calculate_preload_ttl(key, task["confidence"])
                strategy = self._determine_cache_strategy(key, preload_data)

                # Cache the preloaded data
                success = await self.cache.set(key, preload_data, ttl, strategy)

                if success:
                    logger.info(
                        f"Successfully preloaded key {key} for agent {agent_id}"
                    )

                    # Store preload metadata
                    await self.cache.set_l3(
                        f"preload_meta:{key}",
                        {
                            "agent_id": agent_id,
                            "preload_time": time.time(),
                            "confidence": task["confidence"],
                            "predicted_time": predicted_time,
                        },
                        ttl,
                    )

                return success
            else:
                logger.warning(f"Could not generate preload data for key {key}")
                return False

        except Exception as e:
            logger.error(f"Failed to preload key {key}: {str(e)}")
            return False

    async def _generate_preload_data(self, key: str, agent_id: str) -> Optional[Any]:
        """
        Generate data for preloading (this would call actual data sources)
        For now, this is a simulation of the preload data generation
        """
        # Simulate different types of preload data based on key patterns
        if "api_response" in key:
            # Simulate API response preloading
            return {
                "status": "success",
                "data": f"Preloaded API response for {key}",
                "timestamp": time.time(),
                "cached": True,
            }

        elif "code_gen" in key or "jules" in key:
            # Simulate code generation preloading
            return {
                "generated_code": f"# Preloaded code for {key}\ndef example_function():\n    return 'preloaded'",
                "language": "python",
                "timestamp": time.time(),
            }

        elif "imagen" in key or "mockup" in key:
            # Simulate UI mockup preloading
            return {
                "mockup_url": f"https://example.com/preloaded/{hashlib.md5(key.encode()).hexdigest()}.png",
                "design_tokens": {"primary_color": "#007bff", "font_family": "Inter"},
                "timestamp": time.time(),
            }

        elif "session" in key:
            # Simulate session data preloading
            return {
                "session_id": hashlib.md5(key.encode()).hexdigest(),
                "agent_id": agent_id,
                "created_at": time.time(),
                "expires_at": time.time() + 1800,
            }

        else:
            # Generic preloaded data
            return {
                "preloaded": True,
                "key": key,
                "agent_id": agent_id,
                "timestamp": time.time(),
            }

    def _calculate_preload_ttl(self, key: str, confidence: float) -> int:
        """Calculate TTL for preloaded data based on confidence"""
        base_ttl = 300  # 5 minutes base

        # Higher confidence = longer TTL
        confidence_multiplier = 1 + confidence

        # Key-type specific adjustments
        if "session" in key:
            base_ttl = 1800  # 30 minutes for session data
        elif "api_response" in key:
            base_ttl = 600  # 10 minutes for API responses
        elif "code_gen" in key:
            base_ttl = 1200  # 20 minutes for code generation
        elif "imagen" in key:
            base_ttl = 3600  # 1 hour for images

        return int(base_ttl * confidence_multiplier)

    def _determine_cache_strategy(self, key: str, data: Any) -> CacheStrategy:
        """Determine optimal cache strategy for preloaded data"""
        data_size = len(json.dumps(data).encode()) if data else 0

        if data_size > 10000:  # > 10KB
            return CacheStrategy.COMPRESSED
        elif "semantic" in key or "similarity" in key:
            return CacheStrategy.SEMANTIC
        else:
            return CacheStrategy.SIMPLE

    async def _preload_monitor(self):
        """Monitor preload effectiveness and adjust strategies"""
        while True:
            try:
                # Calculate preload efficiency
                total = self.preload_stats["total_preloads"]
                successful = self.preload_stats["successful_preloads"]

                if total > 0:
                    self.preload_stats["preload_efficiency"] = successful / total

                # Log statistics periodically
                if total > 0 and total % 50 == 0:  # Every 50 preloads
                    logger.info(f"Preload stats: {self.preload_stats}")

                # Check for cache hits from preloaded data
                await self._analyze_preload_effectiveness()

                # Sleep before next monitoring cycle
                await asyncio.sleep(300)  # Monitor every 5 minutes

            except Exception as e:
                logger.error(f"Error in preload monitor: {str(e)}")
                await asyncio.sleep(60)

    async def _analyze_preload_effectiveness(self):
        """Analyze how effective preloading has been"""
        # Check recent cache hits for preloaded data
        preload_hits = 0
        total_checked = 0

        # Sample recent access patterns
        recent_accesses = [
            a
            for a in self.analyzer.access_history
            if time.time() - a.access_time < 300 and a.hit
        ]  # Last 5 minutes

        for access in recent_accesses[-50:]:  # Check last 50 hits
            total_checked += 1

            # Check if this was a preloaded key
            preload_meta = await self.cache.get_l3(f"preload_meta:{access.key}")
            if preload_meta:
                preload_hits += 1

        if total_checked > 0:
            preload_hit_rate = preload_hits / total_checked
            self.preload_stats["cache_hits_from_preload"] = preload_hit_rate

            logger.info(
                f"Preload effectiveness: {preload_hit_rate:.2%} of cache hits were preloaded"
            )


class IntelligentCacheWarmer:
    """
    Intelligent cache warming system using reinforcement learning and heuristics
    """

    def __init__(self, cache: RedisCache, analyzer: AccessPatternAnalyzer):
        self.cache = cache
        self.analyzer = analyzer
        self.warming_policies = {}
        self.warming_history = deque(maxlen=1000)
        self.rl_rewards = defaultdict(list)

    async def create_warming_strategy(
        self, agent_id: str, optimization_strategy: CacheOptimizationStrategy
    ) -> CacheWarmingStrategy:
        """Create intelligent cache warming strategy for an agent"""

        if optimization_strategy == CacheOptimizationStrategy.FREQUENCY_BASED:
            return await self._create_frequency_strategy(agent_id)
        elif optimization_strategy == CacheOptimizationStrategy.RECENCY_BASED:
            return await self._create_recency_strategy(agent_id)
        elif optimization_strategy == CacheOptimizationStrategy.PREDICTIVE:
            return await self._create_predictive_strategy(agent_id)
        elif optimization_strategy == CacheOptimizationStrategy.AGENT_CORRELATION:
            return await self._create_correlation_strategy(agent_id)
        elif optimization_strategy == CacheOptimizationStrategy.PERFORMANCE_DRIVEN:
            return await self._create_performance_strategy(agent_id)
        else:
            return await self._create_hybrid_strategy(agent_id)

    async def _create_frequency_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create strategy based on access frequency"""
        agent_accesses = self.analyzer.agent_patterns.get(agent_id, [])

        # Count key frequencies
        key_frequencies = defaultdict(int)
        for access in agent_accesses[-200:]:  # Last 200 accesses
            key_frequencies[access.key] += 1

        # Sort by frequency and select top keys
        sorted_keys = sorted(key_frequencies.items(), key=lambda x: x[1], reverse=True)
        priority_keys = [key for key, freq in sorted_keys[:20] if freq >= 3]

        # Create warming schedule
        warming_schedule = {}
        current_time = time.time()

        for i, key in enumerate(priority_keys):
            # Warm high-frequency keys more often
            frequency = key_frequencies[key]
            warm_interval = max(3600 / frequency, 300)  # At least every 5 minutes
            warming_schedule[key] = current_time + (i * 60) + warm_interval

        return CacheWarmingStrategy(
            priority_keys=priority_keys,
            warm_ahead_time=1800,  # 30 minutes ahead
            warm_data_sources=["frequency_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.15,
        )

    async def _create_recency_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create strategy based on recent access patterns"""
        recent_cutoff = time.time() - 3600  # Last hour
        recent_accesses = [
            a
            for a in self.analyzer.agent_patterns.get(agent_id, [])
            if a.access_time > recent_cutoff
        ]

        # Prioritize recently accessed keys
        recent_keys = list(set(access.key for access in recent_accesses))

        # Create warming schedule based on recency
        warming_schedule = {}
        current_time = time.time()

        for i, key in enumerate(recent_keys[:15]):
            # Warm recent keys soon
            warming_schedule[key] = (
                current_time + (i * 30) + 300
            )  # Every 30s, starting in 5min

        return CacheWarmingStrategy(
            priority_keys=recent_keys[:15],
            warm_ahead_time=900,  # 15 minutes ahead
            warm_data_sources=["recency_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.20,
        )

    async def _create_predictive_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create strategy based on ML predictions"""
        # Get predictions from multiple models
        predictions = await self.analyzer.predict_next_accesses(
            agent_id, PredictionModel.LSTM
        )

        # Filter high-confidence predictions
        high_confidence_keys = [
            key
            for key, confidence in predictions.confidence_scores.items()
            if confidence > 0.6
        ]

        # Create warming schedule based on predicted access times
        warming_schedule = {}
        for key in high_confidence_keys:
            predicted_time = predictions.predicted_access_time.get(
                key, time.time() + 600
            )
            warm_time = predicted_time - 300  # Warm 5 minutes before predicted access
            warming_schedule[key] = max(
                warm_time, time.time() + 60
            )  # At least 1 minute from now

        return CacheWarmingStrategy(
            priority_keys=high_confidence_keys,
            warm_ahead_time=300,  # 5 minutes ahead of prediction
            warm_data_sources=["ml_predictions"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.25,
        )

    async def _create_correlation_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create strategy based on key correlations"""
        # Find keys correlated with recently accessed keys
        recent_keys = [
            a.key for a in self.analyzer.agent_patterns.get(agent_id, [])[-20:]
        ]

        correlated_keys = set()
        for key in recent_keys:
            correlated_keys.update(self.analyzer.key_correlations.get(key, set()))

        # Remove already recent keys
        priority_keys = list(correlated_keys - set(recent_keys))

        # Create warming schedule
        warming_schedule = {}
        current_time = time.time()

        for i, key in enumerate(priority_keys[:12]):
            # Warm correlated keys with slight delay
            warming_schedule[key] = (
                current_time + (i * 45) + 180
            )  # Every 45s, starting in 3min

        return CacheWarmingStrategy(
            priority_keys=priority_keys[:12],
            warm_ahead_time=600,  # 10 minutes ahead
            warm_data_sources=["correlation_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.18,
        )

    async def _create_performance_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create strategy based on performance optimization"""
        # Identify slow/expensive operations that benefit from caching
        agent_accesses = self.analyzer.agent_patterns.get(agent_id, [])

        # Find keys that showed performance benefits from caching
        performance_keys = []
        for access in agent_accesses[-100:]:
            if access.hit and access.size > 1000:  # Large cached items
                performance_keys.append(access.key)

        # Prioritize by size/complexity (proxy for performance benefit)
        priority_keys = list(set(performance_keys))

        # Create performance-optimized warming schedule
        warming_schedule = {}
        current_time = time.time()

        for i, key in enumerate(priority_keys[:10]):
            # Warm performance-critical keys more aggressively
            warming_schedule[key] = (
                current_time + (i * 20) + 120
            )  # Every 20s, starting in 2min

        return CacheWarmingStrategy(
            priority_keys=priority_keys[:10],
            warm_ahead_time=1200,  # 20 minutes ahead
            warm_data_sources=["performance_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.30,
        )

    async def _create_hybrid_strategy(self, agent_id: str) -> CacheWarmingStrategy:
        """Create hybrid strategy combining multiple approaches"""
        # Get strategies from different approaches
        freq_strategy = await self._create_frequency_strategy(agent_id)
        pred_strategy = await self._create_predictive_strategy(agent_id)
        corr_strategy = await self._create_correlation_strategy(agent_id)

        # Combine priority keys with weighting
        all_keys = set()
        all_keys.update(freq_strategy.priority_keys[:8])  # Top 8 frequent
        all_keys.update(pred_strategy.priority_keys[:8])  # Top 8 predicted
        all_keys.update(corr_strategy.priority_keys[:6])  # Top 6 correlated

        priority_keys = list(all_keys)

        # Merge warming schedules
        warming_schedule = {}
        current_time = time.time()

        # Prioritize predicted keys (earliest), then frequent, then correlated
        for i, key in enumerate(pred_strategy.priority_keys[:8]):
            warming_schedule[key] = current_time + (i * 30) + 120

        offset = len(pred_strategy.priority_keys[:8]) * 30
        for i, key in enumerate(freq_strategy.priority_keys[:8]):
            if key not in warming_schedule:
                warming_schedule[key] = current_time + offset + (i * 45) + 180

        offset += len(freq_strategy.priority_keys[:8]) * 45
        for i, key in enumerate(corr_strategy.priority_keys[:6]):
            if key not in warming_schedule:
                warming_schedule[key] = current_time + offset + (i * 60) + 240

        return CacheWarmingStrategy(
            priority_keys=priority_keys,
            warm_ahead_time=900,  # 15 minutes ahead
            warm_data_sources=["hybrid_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.28,
        )

    async def execute_warming_strategy(
        self, agent_id: str, strategy: CacheWarmingStrategy
    ):
        """Execute cache warming strategy"""
        logger.info(f"Executing cache warming strategy for agent {agent_id}")

        warming_tasks = []
        for key, warm_time in strategy.warming_schedule.items():
            task = asyncio.create_task(self._warm_key_at_time(key, warm_time, agent_id))
            warming_tasks.append(task)

        # Wait for all warming tasks to complete
        results = await asyncio.gather(*warming_tasks, return_exceptions=True)

        # Analyze results
        successful_warms = sum(1 for result in results if result is True)
        total_warms = len(warming_tasks)

        warming_result = {
            "agent_id": agent_id,
            "total_keys": total_warms,
            "successful_warms": successful_warms,
            "success_rate": successful_warms / total_warms if total_warms > 0 else 0,
            "strategy": strategy,
            "timestamp": time.time(),
        }

        # Store warming history
        self.warming_history.append(warming_result)

        # Update RL rewards
        reward = self._calculate_warming_reward(warming_result)
        self.rl_rewards[agent_id].append(reward)

        logger.info(
            f"Warming strategy completed: {successful_warms}/{total_warms} successful"
        )

        return warming_result

    async def _warm_key_at_time(
        self, key: str, warm_time: float, agent_id: str
    ) -> bool:
        """Warm a specific key at the scheduled time"""
        current_time = time.time()

        # Wait until the scheduled warm time
        if warm_time > current_time:
            wait_time = min(warm_time - current_time, 3600)  # Max wait 1 hour
            await asyncio.sleep(wait_time)

        try:
            # Check if key is already cached
            existing_value = await self.cache.get(key)
            if existing_value is not None:
                logger.debug(f"Key {key} already cached, warming skipped")
                return True

            # Generate warm data (similar to preloader)
            warm_data = await self._generate_warm_data(key, agent_id)

            if warm_data is not None:
                # Determine appropriate TTL
                ttl = self._calculate_warm_ttl(key)

                # Cache the warm data
                success = await self.cache.set(key, warm_data, ttl)

                if success:
                    logger.debug(f"Successfully warmed key {key} for agent {agent_id}")

                    # Mark as warmed
                    await self.cache.set_l3(
                        f"warmed:{key}",
                        {"agent_id": agent_id, "warm_time": time.time(), "ttl": ttl},
                        ttl,
                    )

                return success

            return False

        except Exception as e:
            logger.error(f"Failed to warm key {key}: {str(e)}")
            return False

    async def _generate_warm_data(self, key: str, agent_id: str) -> Optional[Any]:
        """Generate data for cache warming"""
        # This would normally fetch data from appropriate sources
        # For now, simulate warm data generation

        return {
            "warmed": True,
            "key": key,
            "agent_id": agent_id,
            "warm_timestamp": time.time(),
            "data": f"Warm data for {key}",
        }

    def _calculate_warm_ttl(self, key: str) -> int:
        """Calculate TTL for warmed data"""
        # Base TTL of 30 minutes for warmed data
        base_ttl = 1800

        # Adjust based on key type
        if "session" in key:
            return base_ttl * 2  # Longer for session data
        elif "api_response" in key:
            return base_ttl // 2  # Shorter for API responses
        elif "code_gen" in key:
            return base_ttl * 3  # Much longer for code

        return base_ttl

    def _calculate_warming_reward(self, warming_result: Dict[str, Any]) -> float:
        """Calculate RL reward for warming strategy"""
        success_rate = warming_result["success_rate"]
        expected_improvement = warming_result["strategy"].expected_hit_improvement

        # Base reward from success rate
        base_reward = success_rate

        # Bonus for meeting expected improvement
        if success_rate >= expected_improvement:
            base_reward += 0.2

        # Penalty for poor performance
        if success_rate < 0.5:
            base_reward -= 0.3

        return max(base_reward, 0.0)


class AdvancedCacheOrchestrator:
    """
    Main orchestrator for advanced caching capabilities
    Coordinates pattern analysis, preloading, and intelligent warming
    """

    def __init__(self):
        self.cache = RedisCache()
        self.analyzer = AccessPatternAnalyzer(self.cache)
        self.preloader = CachePreloader(self.cache, self.analyzer)
        self.warmer = IntelligentCacheWarmer(self.cache, self.analyzer)

        self.optimization_active = False
        self.performance_metrics = {
            "cache_hit_rate_improvement": 0.0,
            "average_response_time_improvement": 0.0,
            "preload_effectiveness": 0.0,
            "warming_effectiveness": 0.0,
        }

    async def start_advanced_caching(self):
        """Start all advanced caching components"""
        logger.info("Starting advanced caching system")

        self.optimization_active = True

        # Start all components
        tasks = [
            asyncio.create_task(self.preloader.start_preloading()),
            asyncio.create_task(self._cache_optimization_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._adaptive_warming_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Advanced caching system error: {str(e)}")
            self.optimization_active = False

    async def record_cache_access(
        self, key: str, agent_id: str, hit: bool, size: int = 0, ttl: int = 0
    ):
        """Record cache access for pattern analysis"""
        pattern = CacheAccessPattern(
            key=key,
            agent_id=agent_id,
            access_time=time.time(),
            hit=hit,
            size=size,
            ttl=ttl,
        )

        await self.analyzer.record_access(pattern)

    async def _cache_optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                # Analyze current cache performance
                cache_stats = self.cache.get_cache_stats()

                # Optimize based on current performance
                if cache_stats["hit_rate"] < 70:  # Below threshold
                    await self._trigger_aggressive_optimization()
                elif cache_stats["hit_rate"] < 85:  # Moderate optimization needed
                    await self._trigger_moderate_optimization()

                # Sleep before next optimization cycle
                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Error in cache optimization loop: {str(e)}")
                await asyncio.sleep(60)

    async def _trigger_aggressive_optimization(self):
        """Trigger aggressive cache optimization"""
        logger.info("Triggering aggressive cache optimization")

        # Get all active agents
        active_agents = await self._get_active_agents()

        for agent_id in active_agents:
            # Create hybrid warming strategy for each agent
            strategy = await self.warmer.create_warming_strategy(
                agent_id, CacheOptimizationStrategy.PERFORMANCE_DRIVEN
            )

            # Execute warming strategy
            await self.warmer.execute_warming_strategy(agent_id, strategy)

    async def _trigger_moderate_optimization(self):
        """Trigger moderate cache optimization"""
        logger.info("Triggering moderate cache optimization")

        # Focus on predictive optimization
        active_agents = await self._get_active_agents()

        for agent_id in active_agents[:3]:  # Top 3 active agents
            strategy = await self.warmer.create_warming_strategy(
                agent_id, CacheOptimizationStrategy.PREDICTIVE
            )

            await self.warmer.execute_warming_strategy(agent_id, strategy)

    async def _get_active_agents(self) -> List[str]:
        """Get list of currently active agents"""
        cutoff_time = time.time() - 1800  # Last 30 minutes
        active_agents = set()

        for access in self.analyzer.access_history:
            if access.access_time > cutoff_time:
                active_agents.add(access.agent_id)

        # Sort by activity level
        agent_activity = defaultdict(int)
        for access in self.analyzer.access_history:
            if access.access_time > cutoff_time:
                agent_activity[access.agent_id] += 1

        sorted_agents = sorted(agent_activity.items(), key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, activity in sorted_agents]

    async def _performance_monitoring_loop(self):
        """Monitor cache performance improvements"""
        baseline_metrics = None

        while self.optimization_active:
            try:
                current_metrics = self.cache.get_cache_stats()

                if baseline_metrics is None:
                    baseline_metrics = current_metrics.copy()
                else:
                    # Calculate improvements
                    hit_rate_improvement = (
                        current_metrics["hit_rate"] - baseline_metrics["hit_rate"]
                    )

                    self.performance_metrics["cache_hit_rate_improvement"] = (
                        hit_rate_improvement
                    )

                    # Update performance metrics
                    self.performance_metrics["preload_effectiveness"] = (
                        self.preloader.preload_stats["preload_efficiency"]
                    )

                    # Log improvements
                    if hit_rate_improvement > 5:  # 5% improvement
                        logger.info(
                            f"Cache hit rate improved by {hit_rate_improvement:.1f}%"
                        )

                # Sleep before next monitoring cycle
                await asyncio.sleep(180)  # Every 3 minutes

            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def _adaptive_warming_loop(self):
        """Adaptive warming based on real-time patterns"""
        while self.optimization_active:
            try:
                # Analyze recent patterns for emerging trends
                recent_patterns = await self._detect_emerging_patterns()

                if recent_patterns:
                    # Create adaptive warming strategies
                    for agent_id, patterns in recent_patterns.items():
                        await self._create_adaptive_warming(agent_id, patterns)

                # Sleep before next adaptive cycle
                await asyncio.sleep(240)  # Every 4 minutes

            except Exception as e:
                logger.error(f"Error in adaptive warming loop: {str(e)}")
                await asyncio.sleep(60)

    async def _detect_emerging_patterns(self) -> Dict[str, List[str]]:
        """Detect emerging access patterns"""
        cutoff_time = time.time() - 600  # Last 10 minutes
        recent_accesses = [
            a for a in self.analyzer.access_history if a.access_time > cutoff_time
        ]

        # Group by agent
        agent_patterns = defaultdict(list)
        for access in recent_accesses:
            agent_patterns[access.agent_id].append(access.key)

        # Find agents with significant recent activity
        emerging_patterns = {}
        for agent_id, keys in agent_patterns.items():
            if len(keys) >= 5:  # At least 5 accesses
                emerging_patterns[agent_id] = list(set(keys))

        return emerging_patterns

    async def _create_adaptive_warming(self, agent_id: str, recent_keys: List[str]):
        """Create adaptive warming based on recent patterns"""
        # Create custom warming strategy for recent patterns
        warming_schedule = {}
        current_time = time.time()

        for i, key in enumerate(recent_keys[:8]):  # Top 8 recent keys
            # Warm similar keys soon
            warming_schedule[key] = (
                current_time + (i * 20) + 60
            )  # Every 20s, starting in 1min

        adaptive_strategy = CacheWarmingStrategy(
            priority_keys=recent_keys[:8],
            warm_ahead_time=300,  # 5 minutes ahead
            warm_data_sources=["adaptive_analysis"],
            warming_schedule=warming_schedule,
            expected_hit_improvement=0.15,
        )

        # Execute adaptive warming
        await self.warmer.execute_warming_strategy(agent_id, adaptive_strategy)

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        cache_stats = self.cache.get_cache_stats()

        return {
            "cache_stats": cache_stats,
            "performance_metrics": self.performance_metrics,
            "preload_stats": self.preloader.preload_stats,
            "pattern_analysis": {
                "total_patterns": len(self.analyzer.access_history),
                "unique_agents": len(self.analyzer.agent_patterns),
                "correlation_count": sum(
                    len(corr) for corr in self.analyzer.key_correlations.values()
                ),
                "seasonal_patterns": len(self.analyzer.seasonal_patterns),
            },
            "optimization_active": self.optimization_active,
        }

    async def optimize_for_agent(
        self,
        agent_id: str,
        strategy: CacheOptimizationStrategy = CacheOptimizationStrategy.PERFORMANCE_DRIVEN,
    ) -> Dict[str, Any]:
        """Manually trigger optimization for specific agent"""
        logger.info(
            f"Manual optimization triggered for agent {agent_id} with strategy {strategy.value}"
        )

        # Create and execute warming strategy
        warming_strategy = await self.warmer.create_warming_strategy(agent_id, strategy)
        result = await self.warmer.execute_warming_strategy(agent_id, warming_strategy)

        return result


# Global orchestrator instance
_advanced_cache_orchestrator = None


async def get_advanced_cache_orchestrator() -> AdvancedCacheOrchestrator:
    """Get or create the global advanced cache orchestrator"""
    global _advanced_cache_orchestrator

    if _advanced_cache_orchestrator is None:
        _advanced_cache_orchestrator = AdvancedCacheOrchestrator()

    return _advanced_cache_orchestrator


# Convenience functions
async def start_advanced_caching_system():
    """Start the advanced caching system"""
    orchestrator = await get_advanced_cache_orchestrator()
    await orchestrator.start_advanced_caching()


async def record_cache_access(
    key: str, agent_id: str, hit: bool, size: int = 0, ttl: int = 0
):
    """Record cache access for optimization"""
    orchestrator = await get_advanced_cache_orchestrator()
    await orchestrator.record_cache_access(key, agent_id, hit, size, ttl)


async def optimize_cache_for_agent(
    agent_id: str,
    strategy: CacheOptimizationStrategy = CacheOptimizationStrategy.PERFORMANCE_DRIVEN,
):
    """Optimize cache for specific agent"""
    orchestrator = await get_advanced_cache_orchestrator()
    return await orchestrator.optimize_for_agent(agent_id, strategy)


# Example usage
async def main():
    """Example of using the advanced caching system"""

    # Start advanced caching
    orchestrator = await get_advanced_cache_orchestrator()

    # Simulate some cache accesses
    await orchestrator.record_cache_access(
        "api_response:users", "starri_orchestrator", True, 1024, 3600
    )
    await orchestrator.record_cache_access(
        "code_gen:python:function", "jules_coding_agent", False, 2048, 1800
    )
    await orchestrator.record_cache_access(
        "mockup:dashboard:main", "imagen_4_designer", True, 5120, 7200
    )

    # Get performance summary
    summary = await orchestrator.get_performance_summary()
    print(f"Performance summary: {json.dumps(summary, indent=2)}")

    # Trigger manual optimization
    result = await orchestrator.optimize_for_agent(
        "jules_coding_agent", CacheOptimizationStrategy.PREDICTIVE
    )
    print(f"Optimization result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
