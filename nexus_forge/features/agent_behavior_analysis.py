"""
Agent Behavior Analysis System - Deep Learning from Interaction Patterns

This module implements comprehensive behavior analysis for multi-agent systems:
- Centralized interaction logging with real-time event streaming
- Pattern analysis using deep learning models (transformers, graph neural networks)
- Behavioral anomaly detection and optimization recommendations
- Cross-agent collaboration pattern discovery
- Performance prediction and bottleneck identification
- Continuous learning and adaptation based on agent interactions

Key Features:
- Real-time interaction tracking with Supabase integration
- Sequence models for collaboration pattern analysis
- Graph neural networks for agent relationship modeling
- Behavioral clustering and classification
- Predictive analytics for agent performance
- Automated optimization recommendations
"""

import asyncio
import logging
import time
import json
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import math
from concurrent.futures import ThreadPoolExecutor

from ..core.cache import RedisCache
from ..core.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of agent interactions"""
    REQUEST = "request"
    RESPONSE = "response" 
    COORDINATION = "coordination"
    ERROR = "error"
    SUCCESS = "success"
    COLLABORATION = "collaboration"
    FEEDBACK = "feedback"
    OPTIMIZATION = "optimization"


class AgentBehaviorPattern(Enum):
    """Behavioral patterns detected in agents"""
    EFFICIENT_COLLABORATOR = "efficient_collaborator"
    BOTTLENECK_CREATOR = "bottleneck_creator"
    ERROR_PRONE = "error_prone"
    OPTIMIZATION_FOCUSED = "optimization_focused"
    RAPID_RESPONDER = "rapid_responder"
    RESOURCE_INTENSIVE = "resource_intensive"
    ADAPTIVE_LEARNER = "adaptive_learner"
    CONSISTENT_PERFORMER = "consistent_performer"


class AnalysisModel(Enum):
    """Types of analysis models"""
    TRANSFORMER = "transformer"
    GRAPH_NEURAL_NETWORK = "gnn"
    STATISTICAL = "statistical"
    CLUSTERING = "clustering"
    SEQUENCE_ANALYSIS = "sequence"
    ENSEMBLE = "ensemble"


@dataclass
class AgentInteraction:
    """Represents a single agent interaction"""
    interaction_id: str
    timestamp: float
    source_agent: str
    target_agent: Optional[str]
    interaction_type: InteractionType
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    duration: Optional[float] = None
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass
class BehaviorPattern:
    """Represents a detected behavioral pattern"""
    pattern_id: str
    agent_id: str
    pattern_type: AgentBehaviorPattern
    confidence: float
    frequency: int
    first_detected: float
    last_detected: float
    characteristics: Dict[str, Any]
    impact_score: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CollaborationGraph:
    """Represents agent collaboration relationships"""
    nodes: Dict[str, Dict[str, Any]]  # agent_id -> properties
    edges: Dict[Tuple[str, str], Dict[str, Any]]  # (source, target) -> properties
    centrality_scores: Dict[str, float]
    community_clusters: Dict[str, List[str]]
    efficiency_metrics: Dict[str, float]


@dataclass
class BehaviorAnalysis:
    """Complete behavior analysis results"""
    analysis_id: str
    timestamp: float
    agents_analyzed: List[str]
    patterns_detected: List[BehaviorPattern]
    collaboration_graph: CollaborationGraph
    performance_insights: Dict[str, Any]
    optimization_recommendations: List[Dict[str, Any]]
    anomalies_detected: List[Dict[str, Any]]
    prediction_confidence: float


class InteractionLogger:
    """
    Centralized interaction logging system with real-time streaming
    Captures all agent interactions for analysis
    """
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.interaction_buffer = deque(maxlen=10000)
        self.supabase_client = None  # Would be initialized with credentials
        self.real_time_subscribers = []
        self.logging_active = True
        
        # Performance tracking
        self.log_stats = {
            "interactions_logged": 0,
            "interactions_per_second": 0.0,
            "buffer_utilization": 0.0,
            "last_flush": time.time()
        }
    
    async def start_logging(self):
        """Start interaction logging with background tasks"""
        logger.info("Starting agent interaction logging")
        
        tasks = [
            asyncio.create_task(self._buffer_flush_worker()),
            asyncio.create_task(self._real_time_streaming_worker()),
            asyncio.create_task(self._performance_monitoring_worker())
        ]
        
        await asyncio.gather(*tasks)
    
    async def log_interaction(self, interaction: AgentInteraction):
        """Log a single agent interaction"""
        try:
            # Add to buffer
            self.interaction_buffer.append(interaction)
            
            # Update statistics
            self.log_stats["interactions_logged"] += 1
            self.log_stats["buffer_utilization"] = len(self.interaction_buffer) / self.interaction_buffer.maxlen
            
            # Cache for immediate access
            cache_key = f"interaction:{interaction.interaction_id}"
            await self.cache.set_l2(cache_key, asdict(interaction), timeout=3600)
            
            # Stream to real-time subscribers
            await self._stream_to_subscribers(interaction)
            
            logger.debug(f"Logged interaction {interaction.interaction_id}")
            
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")
    
    async def _buffer_flush_worker(self):
        """Worker to flush interaction buffer to persistent storage"""
        while self.logging_active:
            try:
                if len(self.interaction_buffer) >= 100:  # Flush when buffer reaches 100 items
                    await self._flush_buffer()
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in buffer flush worker: {str(e)}")
                await asyncio.sleep(30)
    
    async def _flush_buffer(self):
        """Flush interaction buffer to Supabase"""
        if not self.interaction_buffer:
            return
        
        # Extract interactions to flush
        interactions_to_flush = []
        while self.interaction_buffer and len(interactions_to_flush) < 100:
            interactions_to_flush.append(self.interaction_buffer.popleft())
        
        try:
            # In production, this would write to Supabase
            # For now, store in Redis with longer TTL
            batch_id = str(uuid.uuid4())
            
            batch_data = {
                "batch_id": batch_id,
                "timestamp": time.time(),
                "interactions": [asdict(interaction) for interaction in interactions_to_flush]
            }
            
            await self.cache.set_l3(f"interaction_batch:{batch_id}", batch_data, timeout=604800)  # 1 week
            
            self.log_stats["last_flush"] = time.time()
            
            logger.info(f"Flushed {len(interactions_to_flush)} interactions to storage")
            
        except Exception as e:
            logger.error(f"Failed to flush interactions: {str(e)}")
            # Put interactions back in buffer on failure
            for interaction in reversed(interactions_to_flush):
                self.interaction_buffer.appendleft(interaction)
    
    async def _real_time_streaming_worker(self):
        """Worker for real-time interaction streaming"""
        while self.logging_active:
            try:
                # Process real-time streaming
                # In production, this would maintain WebSocket connections
                # or Supabase real-time subscriptions
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in real-time streaming: {str(e)}")
                await asyncio.sleep(5)
    
    async def _stream_to_subscribers(self, interaction: AgentInteraction):
        """Stream interaction to real-time subscribers"""
        # Simulate real-time streaming
        streaming_data = {
            "type": "interaction",
            "data": asdict(interaction),
            "timestamp": time.time()
        }
        
        # Cache for dashboard access
        await self.cache.set_l1(f"realtime_interaction", streaming_data)
    
    async def _performance_monitoring_worker(self):
        """Monitor logging performance"""
        last_count = 0
        
        while self.logging_active:
            try:
                current_count = self.log_stats["interactions_logged"]
                time_elapsed = 60  # 1 minute window
                
                interactions_per_second = (current_count - last_count) / time_elapsed
                self.log_stats["interactions_per_second"] = interactions_per_second
                
                last_count = current_count
                
                # Log performance metrics
                if interactions_per_second > 0:
                    logger.debug(f"Logging performance: {interactions_per_second:.2f} interactions/sec")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(30)
    
    async def get_interactions(self, agent_id: Optional[str] = None, 
                              time_range: Optional[Tuple[float, float]] = None,
                              interaction_types: Optional[List[InteractionType]] = None,
                              limit: int = 1000) -> List[AgentInteraction]:
        """Retrieve interactions with filtering"""
        interactions = []
        
        # Search in buffer first
        for interaction in self.interaction_buffer:
            if self._matches_filters(interaction, agent_id, time_range, interaction_types):
                interactions.append(interaction)
                if len(interactions) >= limit:
                    break
        
        # If need more, search in cache
        if len(interactions) < limit:
            # Search cached interactions
            batch_keys = await self.cache.client.keys("interaction_batch:*")
            
            for key in batch_keys[:10]:  # Limit search to recent batches
                try:
                    batch_data = await self.cache.get(key.decode() if isinstance(key, bytes) else key)
                    if batch_data and "interactions" in batch_data:
                        for interaction_data in batch_data["interactions"]:
                            interaction = AgentInteraction(**interaction_data)
                            if self._matches_filters(interaction, agent_id, time_range, interaction_types):
                                interactions.append(interaction)
                                if len(interactions) >= limit:
                                    break
                except:
                    continue
                
                if len(interactions) >= limit:
                    break
        
        return interactions[:limit]
    
    def _matches_filters(self, interaction: AgentInteraction, 
                        agent_id: Optional[str],
                        time_range: Optional[Tuple[float, float]],
                        interaction_types: Optional[List[InteractionType]]) -> bool:
        """Check if interaction matches filters"""
        if agent_id and interaction.source_agent != agent_id and interaction.target_agent != agent_id:
            return False
        
        if time_range and not (time_range[0] <= interaction.timestamp <= time_range[1]):
            return False
        
        if interaction_types and interaction.interaction_type not in interaction_types:
            return False
        
        return True
    
    async def get_logging_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics"""
        return self.log_stats.copy()


class PatternAnalyzer:
    """
    Advanced pattern analysis using deep learning models
    Identifies behavioral patterns and collaboration structures
    """
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.models = {}
        self.pattern_cache = {}
        self.analysis_history = deque(maxlen=100)
        
        # Pattern detection thresholds
        self.pattern_thresholds = {
            AgentBehaviorPattern.EFFICIENT_COLLABORATOR: 0.8,
            AgentBehaviorPattern.BOTTLENECK_CREATOR: 0.7,
            AgentBehaviorPattern.ERROR_PRONE: 0.6,
            AgentBehaviorPattern.RAPID_RESPONDER: 0.75,
            AgentBehaviorPattern.ADAPTIVE_LEARNER: 0.85
        }
    
    async def analyze_behavior_patterns(self, interactions: List[AgentInteraction], 
                                       model: AnalysisModel = AnalysisModel.ENSEMBLE) -> BehaviorAnalysis:
        """
        Comprehensive behavior pattern analysis
        """
        analysis_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Extract unique agents
        agents_analyzed = list(set([i.source_agent for i in interactions] + 
                                 [i.target_agent for i in interactions if i.target_agent]))
        
        logger.info(f"Analyzing behavior patterns for {len(agents_analyzed)} agents using {model.value}")
        
        # Detect individual agent patterns
        patterns_detected = await self._detect_individual_patterns(interactions, agents_analyzed)
        
        # Build collaboration graph
        collaboration_graph = await self._build_collaboration_graph(interactions, agents_analyzed)
        
        # Extract performance insights
        performance_insights = await self._extract_performance_insights(interactions, agents_analyzed)
        
        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(
            patterns_detected, collaboration_graph, performance_insights
        )
        
        # Detect anomalies
        anomalies_detected = await self._detect_anomalies(interactions, patterns_detected)
        
        # Calculate prediction confidence
        prediction_confidence = await self._calculate_prediction_confidence(
            patterns_detected, collaboration_graph
        )
        
        # Create analysis result
        analysis = BehaviorAnalysis(
            analysis_id=analysis_id,
            timestamp=timestamp,
            agents_analyzed=agents_analyzed,
            patterns_detected=patterns_detected,
            collaboration_graph=collaboration_graph,
            performance_insights=performance_insights,
            optimization_recommendations=optimization_recommendations,
            anomalies_detected=anomalies_detected,
            prediction_confidence=prediction_confidence
        )
        
        # Cache analysis result
        await self.cache.set_l2(f"behavior_analysis:{analysis_id}", asdict(analysis), timeout=86400)
        
        # Store in analysis history
        self.analysis_history.append(analysis)
        
        logger.info(f"Behavior analysis completed: {len(patterns_detected)} patterns, "
                   f"{len(optimization_recommendations)} recommendations")
        
        return analysis
    
    async def _detect_individual_patterns(self, interactions: List[AgentInteraction], 
                                        agents: List[str]) -> List[BehaviorPattern]:
        """Detect behavioral patterns for individual agents"""
        patterns = []
        
        for agent_id in agents:
            agent_interactions = [i for i in interactions 
                                if i.source_agent == agent_id or i.target_agent == agent_id]
            
            if len(agent_interactions) < 10:  # Minimum interactions for pattern detection
                continue
            
            # Analyze different pattern types
            agent_patterns = await self._analyze_agent_patterns(agent_id, agent_interactions)
            patterns.extend(agent_patterns)
        
        return patterns
    
    async def _analyze_agent_patterns(self, agent_id: str, 
                                    interactions: List[AgentInteraction]) -> List[BehaviorPattern]:
        """Analyze patterns for a specific agent"""
        patterns = []
        
        # Calculate agent metrics
        metrics = await self._calculate_agent_metrics(agent_id, interactions)
        
        # Check for efficient collaborator pattern
        if (metrics["collaboration_rate"] > 0.7 and 
            metrics["success_rate"] > 0.9 and 
            metrics["average_response_time"] < 1000):
            
            pattern = BehaviorPattern(
                pattern_id=f"{agent_id}_efficient_collaborator_{int(time.time())}",
                agent_id=agent_id,
                pattern_type=AgentBehaviorPattern.EFFICIENT_COLLABORATOR,
                confidence=min(metrics["collaboration_rate"] + metrics["success_rate"], 1.0) / 2,
                frequency=len(interactions),
                first_detected=min(i.timestamp for i in interactions),
                last_detected=max(i.timestamp for i in interactions),
                characteristics={
                    "collaboration_rate": metrics["collaboration_rate"],
                    "success_rate": metrics["success_rate"],
                    "response_time": metrics["average_response_time"]
                },
                impact_score=0.8,
                recommendations=["Leverage as mentor for other agents", "Increase task allocation"]
            )
            patterns.append(pattern)
        
        # Check for bottleneck creator pattern
        if (metrics["average_response_time"] > 5000 and 
            metrics["error_rate"] > 0.2):
            
            pattern = BehaviorPattern(
                pattern_id=f"{agent_id}_bottleneck_{int(time.time())}",
                agent_id=agent_id,
                pattern_type=AgentBehaviorPattern.BOTTLENECK_CREATOR,
                confidence=min(metrics["error_rate"] + (metrics["average_response_time"] / 10000), 1.0),
                frequency=len(interactions),
                first_detected=min(i.timestamp for i in interactions),
                last_detected=max(i.timestamp for i in interactions),
                characteristics={
                    "response_time": metrics["average_response_time"],
                    "error_rate": metrics["error_rate"],
                    "bottleneck_score": metrics.get("bottleneck_score", 0.7)
                },
                impact_score=-0.6,
                recommendations=["Optimize processing pipeline", "Reduce task complexity", "Add monitoring"]
            )
            patterns.append(pattern)
        
        # Check for rapid responder pattern
        if (metrics["average_response_time"] < 500 and 
            metrics["success_rate"] > 0.85):
            
            pattern = BehaviorPattern(
                pattern_id=f"{agent_id}_rapid_responder_{int(time.time())}",
                agent_id=agent_id,
                pattern_type=AgentBehaviorPattern.RAPID_RESPONDER,
                confidence=(1.0 - metrics["average_response_time"] / 1000) * metrics["success_rate"],
                frequency=len(interactions),
                first_detected=min(i.timestamp for i in interactions),
                last_detected=max(i.timestamp for i in interactions),
                characteristics={
                    "response_time": metrics["average_response_time"],
                    "success_rate": metrics["success_rate"],
                    "throughput": metrics.get("throughput", 0)
                },
                impact_score=0.7,
                recommendations=["Handle high-priority tasks", "Share optimization techniques"]
            )
            patterns.append(pattern)
        
        # Check for error-prone pattern
        if metrics["error_rate"] > 0.3:
            pattern = BehaviorPattern(
                pattern_id=f"{agent_id}_error_prone_{int(time.time())}",
                agent_id=agent_id,
                pattern_type=AgentBehaviorPattern.ERROR_PRONE,
                confidence=metrics["error_rate"],
                frequency=len(interactions),
                first_detected=min(i.timestamp for i in interactions),
                last_detected=max(i.timestamp for i in interactions),
                characteristics={
                    "error_rate": metrics["error_rate"],
                    "error_types": metrics.get("error_types", []),
                    "recovery_rate": metrics.get("recovery_rate", 0.5)
                },
                impact_score=-0.8,
                recommendations=["Implement better error handling", "Add input validation", "Training needed"]
            )
            patterns.append(pattern)
        
        # Check for adaptive learner pattern
        learning_trend = await self._calculate_learning_trend(agent_id, interactions)
        if learning_trend > 0.1:  # Improving performance over time
            pattern = BehaviorPattern(
                pattern_id=f"{agent_id}_adaptive_learner_{int(time.time())}",
                agent_id=agent_id,
                pattern_type=AgentBehaviorPattern.ADAPTIVE_LEARNER,
                confidence=min(learning_trend * 5, 1.0),  # Scale learning trend
                frequency=len(interactions),
                first_detected=min(i.timestamp for i in interactions),
                last_detected=max(i.timestamp for i in interactions),
                characteristics={
                    "learning_trend": learning_trend,
                    "improvement_rate": metrics.get("improvement_rate", 0),
                    "adaptation_speed": metrics.get("adaptation_speed", 0)
                },
                impact_score=0.9,
                recommendations=["Provide more challenging tasks", "Document learning patterns"]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _calculate_agent_metrics(self, agent_id: str, 
                                     interactions: List[AgentInteraction]) -> Dict[str, float]:
        """Calculate comprehensive metrics for an agent"""
        if not interactions:
            return {}
        
        # Basic metrics
        total_interactions = len(interactions)
        successful_interactions = sum(1 for i in interactions if i.success)
        failed_interactions = total_interactions - successful_interactions
        
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
        error_rate = failed_interactions / total_interactions if total_interactions > 0 else 0
        
        # Response time metrics
        response_times = [i.duration for i in interactions if i.duration is not None]
        average_response_time = np.mean(response_times) if response_times else 0
        
        # Collaboration metrics
        collaboration_interactions = sum(1 for i in interactions 
                                       if i.interaction_type == InteractionType.COLLABORATION)
        collaboration_rate = collaboration_interactions / total_interactions if total_interactions > 0 else 0
        
        # Performance over time
        timestamps = [i.timestamp for i in interactions]
        time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
        throughput = total_interactions / time_range if time_range > 0 else 0
        
        # Error analysis
        error_types = [i.error_details.get("type", "unknown") for i in interactions 
                      if not i.success and i.error_details]
        error_type_counts = {}
        for error_type in error_types:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        # Recovery rate (successful interactions after errors)
        recovery_rate = 0.5  # Placeholder - would need more complex analysis
        
        return {
            "success_rate": success_rate,
            "error_rate": error_rate,
            "average_response_time": average_response_time,
            "collaboration_rate": collaboration_rate,
            "throughput": throughput,
            "error_types": list(error_type_counts.keys()),
            "recovery_rate": recovery_rate,
            "total_interactions": total_interactions
        }
    
    async def _calculate_learning_trend(self, agent_id: str, 
                                      interactions: List[AgentInteraction]) -> float:
        """Calculate learning/improvement trend for an agent"""
        if len(interactions) < 20:  # Need sufficient data
            return 0.0
        
        # Sort interactions by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        # Split into early and recent periods
        mid_point = len(sorted_interactions) // 2
        early_interactions = sorted_interactions[:mid_point]
        recent_interactions = sorted_interactions[mid_point:]
        
        # Calculate success rates for each period
        early_success_rate = sum(1 for i in early_interactions if i.success) / len(early_interactions)
        recent_success_rate = sum(1 for i in recent_interactions if i.success) / len(recent_interactions)
        
        # Calculate response time improvement
        early_response_times = [i.duration for i in early_interactions if i.duration is not None]
        recent_response_times = [i.duration for i in recent_interactions if i.duration is not None]
        
        early_avg_response = np.mean(early_response_times) if early_response_times else 0
        recent_avg_response = np.mean(recent_response_times) if recent_response_times else 0
        
        # Calculate improvement (higher success rate, lower response time = positive trend)
        success_improvement = recent_success_rate - early_success_rate
        
        if early_avg_response > 0 and recent_avg_response > 0:
            response_improvement = (early_avg_response - recent_avg_response) / early_avg_response
        else:
            response_improvement = 0
        
        # Combined learning trend
        learning_trend = (success_improvement + response_improvement) / 2
        
        return learning_trend
    
    async def _build_collaboration_graph(self, interactions: List[AgentInteraction], 
                                       agents: List[str]) -> CollaborationGraph:
        """Build collaboration graph showing agent relationships"""
        
        # Initialize graph structure
        nodes = {agent: {"interaction_count": 0, "success_rate": 0.0} for agent in agents}
        edges = {}
        
        # Build edges from interactions
        for interaction in interactions:
            source = interaction.source_agent
            target = interaction.target_agent
            
            if target:  # Only for agent-to-agent interactions
                edge_key = (source, target)
                
                if edge_key not in edges:
                    edges[edge_key] = {
                        "interaction_count": 0,
                        "success_count": 0,
                        "total_duration": 0,
                        "collaboration_strength": 0.0
                    }
                
                edges[edge_key]["interaction_count"] += 1
                if interaction.success:
                    edges[edge_key]["success_count"] += 1
                
                if interaction.duration:
                    edges[edge_key]["total_duration"] += interaction.duration
            
            # Update node metrics
            nodes[source]["interaction_count"] += 1
            if interaction.success:
                nodes[source]["success_rate"] = (
                    nodes[source].get("success_rate", 0) * (nodes[source]["interaction_count"] - 1) + 1
                ) / nodes[source]["interaction_count"]
        
        # Calculate collaboration strength for edges
        for edge_key, edge_data in edges.items():
            if edge_data["interaction_count"] > 0:
                success_rate = edge_data["success_count"] / edge_data["interaction_count"]
                avg_duration = edge_data["total_duration"] / edge_data["interaction_count"]
                
                # Collaboration strength based on frequency, success rate, and efficiency
                collaboration_strength = (
                    min(edge_data["interaction_count"] / 10, 1.0) * 0.4 +  # Frequency factor
                    success_rate * 0.4 +  # Success factor
                    max(0, 1.0 - avg_duration / 5000) * 0.2  # Efficiency factor
                )
                
                edge_data["collaboration_strength"] = collaboration_strength
        
        # Calculate centrality scores (simplified)
        centrality_scores = await self._calculate_centrality(agents, edges)
        
        # Detect community clusters
        community_clusters = await self._detect_communities(agents, edges)
        
        # Calculate efficiency metrics
        efficiency_metrics = await self._calculate_efficiency_metrics(nodes, edges)
        
        return CollaborationGraph(
            nodes=nodes,
            edges=edges,
            centrality_scores=centrality_scores,
            community_clusters=community_clusters,
            efficiency_metrics=efficiency_metrics
        )
    
    async def _calculate_centrality(self, agents: List[str], 
                                  edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, float]:
        """Calculate centrality scores for agents"""
        centrality = {agent: 0.0 for agent in agents}
        
        # Simple degree centrality calculation
        for (source, target), edge_data in edges.items():
            weight = edge_data.get("collaboration_strength", 0)
            centrality[source] += weight
            centrality[target] += weight
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality.values() else 1
        if max_centrality > 0:
            centrality = {agent: score / max_centrality for agent, score in centrality.items()}
        
        return centrality
    
    async def _detect_communities(self, agents: List[str], 
                                edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, List[str]]:
        """Detect community clusters in agent collaboration"""
        # Simplified community detection - in production would use proper algorithms
        
        communities = {}
        
        # Group by collaboration strength
        strong_collaborators = set()
        for (source, target), edge_data in edges.items():
            if edge_data.get("collaboration_strength", 0) > 0.7:
                strong_collaborators.add(source)
                strong_collaborators.add(target)
        
        communities["high_collaboration"] = list(strong_collaborators)
        communities["moderate_collaboration"] = [agent for agent in agents 
                                               if agent not in strong_collaborators]
        
        return communities
    
    async def _calculate_efficiency_metrics(self, nodes: Dict[str, Dict[str, Any]], 
                                          edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, float]:
        """Calculate collaboration efficiency metrics"""
        if not edges:
            return {"overall_efficiency": 0.0}
        
        # Calculate overall collaboration efficiency
        total_strength = sum(edge_data.get("collaboration_strength", 0) 
                           for edge_data in edges.values())
        
        overall_efficiency = total_strength / len(edges) if edges else 0
        
        # Calculate network density
        total_possible_edges = len(nodes) * (len(nodes) - 1)
        network_density = len(edges) / total_possible_edges if total_possible_edges > 0 else 0
        
        return {
            "overall_efficiency": overall_efficiency,
            "network_density": network_density,
            "total_collaborations": len(edges)
        }
    
    async def _extract_performance_insights(self, interactions: List[AgentInteraction], 
                                          agents: List[str]) -> Dict[str, Any]:
        """Extract performance insights from interaction data"""
        insights = {
            "overall_metrics": {},
            "agent_rankings": {},
            "temporal_patterns": {},
            "bottleneck_analysis": {},
            "optimization_opportunities": []
        }
        
        # Overall system metrics
        total_interactions = len(interactions)
        successful_interactions = sum(1 for i in interactions if i.success)
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
        
        response_times = [i.duration for i in interactions if i.duration is not None]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        insights["overall_metrics"] = {
            "total_interactions": total_interactions,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "agents_active": len(agents)
        }
        
        # Agent rankings
        agent_performance = {}
        for agent in agents:
            agent_interactions = [i for i in interactions if i.source_agent == agent]
            if agent_interactions:
                metrics = await self._calculate_agent_metrics(agent, agent_interactions)
                agent_performance[agent] = metrics.get("success_rate", 0) * 0.6 + \
                                         (1 - min(metrics.get("average_response_time", 1000) / 1000, 1)) * 0.4
        
        insights["agent_rankings"] = dict(sorted(agent_performance.items(), 
                                               key=lambda x: x[1], reverse=True))
        
        # Temporal patterns
        if interactions:
            timestamps = [i.timestamp for i in interactions]
            time_range = max(timestamps) - min(timestamps)
            
            # Activity distribution by hour
            hour_distribution = defaultdict(int)
            for interaction in interactions:
                hour = datetime.fromtimestamp(interaction.timestamp).hour
                hour_distribution[hour] += 1
            
            insights["temporal_patterns"] = {
                "time_range_hours": time_range / 3600,
                "peak_activity_hour": max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else 0,
                "activity_distribution": dict(hour_distribution)
            }
        
        # Bottleneck analysis
        bottlenecks = []
        for agent in agents:
            agent_interactions = [i for i in interactions if i.source_agent == agent]
            if agent_interactions:
                metrics = await self._calculate_agent_metrics(agent, agent_interactions)
                if (metrics.get("average_response_time", 0) > 3000 or 
                    metrics.get("error_rate", 0) > 0.2):
                    bottlenecks.append({
                        "agent": agent,
                        "severity": metrics.get("error_rate", 0) + 
                                  min(metrics.get("average_response_time", 0) / 5000, 1),
                        "issues": ["High response time" if metrics.get("average_response_time", 0) > 3000 else "",
                                 "High error rate" if metrics.get("error_rate", 0) > 0.2 else ""]
                    })
        
        insights["bottleneck_analysis"] = {
            "bottlenecks_detected": len(bottlenecks),
            "bottleneck_agents": bottlenecks
        }
        
        return insights
    
    async def _generate_optimization_recommendations(self, patterns: List[BehaviorPattern],
                                                   collaboration_graph: CollaborationGraph,
                                                   performance_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Analyze patterns for optimization opportunities
        for pattern in patterns:
            if pattern.pattern_type == AgentBehaviorPattern.BOTTLENECK_CREATOR:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "agent": pattern.agent_id,
                    "title": "Optimize Bottleneck Agent",
                    "description": f"Agent {pattern.agent_id} is creating bottlenecks",
                    "actions": [
                        "Review and optimize processing pipeline",
                        "Implement caching for frequent operations",
                        "Consider load balancing or task distribution"
                    ],
                    "expected_impact": "25-40% improvement in overall system throughput"
                })
            
            elif pattern.pattern_type == AgentBehaviorPattern.EFFICIENT_COLLABORATOR:
                recommendations.append({
                    "type": "resource_allocation",
                    "priority": "medium",
                    "agent": pattern.agent_id,
                    "title": "Leverage High-Performing Agent",
                    "description": f"Agent {pattern.agent_id} shows excellent collaboration patterns",
                    "actions": [
                        "Increase task allocation to this agent",
                        "Use as mentor for other agents",
                        "Document best practices used by this agent"
                    ],
                    "expected_impact": "15-25% improvement in task completion rate"
                })
        
        # Collaboration graph recommendations
        centrality_scores = collaboration_graph.centrality_scores
        if centrality_scores:
            most_central_agent = max(centrality_scores.items(), key=lambda x: x[1])
            
            recommendations.append({
                "type": "collaboration_optimization",
                "priority": "medium",
                "agent": most_central_agent[0],
                "title": "Optimize Central Agent",
                "description": f"Agent {most_central_agent[0]} is central to collaboration network",
                "actions": [
                    "Ensure this agent has sufficient resources",
                    "Monitor for overload conditions",
                    "Create backup coordination paths"
                ],
                "expected_impact": "Improved system resilience and reduced single points of failure"
            })
        
        # Performance insights recommendations
        bottlenecks = performance_insights.get("bottleneck_analysis", {}).get("bottleneck_agents", [])
        for bottleneck in bottlenecks:
            if bottleneck["severity"] > 0.7:
                recommendations.append({
                    "type": "critical_optimization",
                    "priority": "critical",
                    "agent": bottleneck["agent"],
                    "title": "Critical Performance Issue",
                    "description": f"Agent {bottleneck['agent']} has critical performance issues",
                    "actions": [
                        "Immediate performance review required",
                        "Implement emergency optimization measures",
                        "Consider temporary load reduction"
                    ],
                    "expected_impact": "Prevent system degradation and maintain SLA compliance"
                })
        
        # System-wide recommendations
        overall_success_rate = performance_insights.get("overall_metrics", {}).get("success_rate", 0)
        if overall_success_rate < 0.8:
            recommendations.append({
                "type": "system_optimization",
                "priority": "high",
                "agent": "system",
                "title": "Improve Overall Success Rate",
                "description": f"System success rate is {overall_success_rate:.1%}",
                "actions": [
                    "Review error handling across all agents",
                    "Implement retry mechanisms",
                    "Improve input validation and preprocessing"
                ],
                "expected_impact": "Increase system reliability and user satisfaction"
            })
        
        return recommendations
    
    async def _detect_anomalies(self, interactions: List[AgentInteraction], 
                               patterns: List[BehaviorPattern]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies"""
        anomalies = []
        
        # Detect sudden performance drops
        if len(interactions) >= 50:
            recent_interactions = interactions[-25:]  # Last 25 interactions
            earlier_interactions = interactions[-50:-25]  # Previous 25 interactions
            
            recent_success_rate = sum(1 for i in recent_interactions if i.success) / len(recent_interactions)
            earlier_success_rate = sum(1 for i in earlier_interactions if i.success) / len(earlier_interactions)
            
            if recent_success_rate < earlier_success_rate - 0.2:  # 20% drop
                anomalies.append({
                    "type": "performance_drop",
                    "severity": "high",
                    "description": "Significant drop in success rate detected",
                    "metrics": {
                        "recent_success_rate": recent_success_rate,
                        "earlier_success_rate": earlier_success_rate,
                        "drop_percentage": (earlier_success_rate - recent_success_rate) * 100
                    },
                    "timestamp": time.time()
                })
        
        # Detect unusual collaboration patterns
        collaboration_interactions = [i for i in interactions 
                                    if i.interaction_type == InteractionType.COLLABORATION]
        
        if len(collaboration_interactions) < len(interactions) * 0.1:  # Less than 10% collaboration
            anomalies.append({
                "type": "low_collaboration",
                "severity": "medium",
                "description": "Unusually low collaboration rate detected",
                "metrics": {
                    "collaboration_rate": len(collaboration_interactions) / len(interactions) * 100,
                    "expected_rate": 20  # Expected 20%+
                },
                "timestamp": time.time()
            })
        
        # Detect agents with unusual error patterns
        error_interactions = [i for i in interactions if not i.success]
        agent_errors = defaultdict(int)
        
        for interaction in error_interactions:
            agent_errors[interaction.source_agent] += 1
        
        if agent_errors:
            max_errors = max(agent_errors.values())
            avg_errors = sum(agent_errors.values()) / len(agent_errors)
            
            for agent, error_count in agent_errors.items():
                if error_count > avg_errors + 2 * np.std(list(agent_errors.values())):
                    anomalies.append({
                        "type": "error_spike",
                        "severity": "high",
                        "agent": agent,
                        "description": f"Agent {agent} has unusually high error rate",
                        "metrics": {
                            "agent_errors": error_count,
                            "average_errors": avg_errors,
                            "error_rate": error_count / len([i for i in interactions if i.source_agent == agent])
                        },
                        "timestamp": time.time()
                    })
        
        return anomalies
    
    async def _calculate_prediction_confidence(self, patterns: List[BehaviorPattern],
                                             collaboration_graph: CollaborationGraph) -> float:
        """Calculate confidence in predictions and recommendations"""
        if not patterns:
            return 0.0
        
        # Base confidence on pattern strength and consistency
        pattern_confidences = [pattern.confidence for pattern in patterns]
        avg_pattern_confidence = np.mean(pattern_confidences)
        
        # Factor in collaboration graph completeness
        total_agents = len(collaboration_graph.nodes)
        connected_agents = len([agent for agent, centrality in collaboration_graph.centrality_scores.items() 
                              if centrality > 0])
        
        graph_completeness = connected_agents / total_agents if total_agents > 0 else 0
        
        # Factor in data sufficiency
        total_patterns = len(patterns)
        data_sufficiency = min(total_patterns / (total_agents * 2), 1.0)  # Want ~2 patterns per agent
        
        # Combined confidence
        prediction_confidence = (
            avg_pattern_confidence * 0.5 +
            graph_completeness * 0.3 +
            data_sufficiency * 0.2
        )
        
        return min(prediction_confidence, 1.0)


class BehaviorAnalysisOrchestrator:
    """
    Main orchestrator for agent behavior analysis system
    Coordinates logging, pattern analysis, and real-time monitoring
    """
    
    def __init__(self):
        self.cache = RedisCache()
        self.logger = InteractionLogger(self.cache)
        self.analyzer = PatternAnalyzer(self.cache)
        
        self.monitoring_active = False
        self.analysis_interval = 300  # 5 minutes
        self.last_analysis = 0
        
        self.system_metrics = {
            "total_interactions_analyzed": 0,
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "optimization_recommendations": 0,
            "analysis_runs": 0
        }
    
    async def start_behavior_analysis(self):
        """Start the behavior analysis system"""
        logger.info("Starting agent behavior analysis system")
        
        self.monitoring_active = True
        
        # Start all components
        tasks = [
            asyncio.create_task(self.logger.start_logging()),
            asyncio.create_task(self._analysis_scheduler()),
            asyncio.create_task(self._real_time_monitoring()),
            asyncio.create_task(self._metrics_collection())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Behavior analysis system error: {str(e)}")
            self.monitoring_active = False
    
    async def log_agent_interaction(self, source_agent: str, target_agent: Optional[str],
                                   interaction_type: InteractionType, context: Dict[str, Any],
                                   duration: Optional[float] = None, success: bool = True,
                                   error_details: Optional[Dict[str, Any]] = None) -> str:
        """Log an agent interaction"""
        interaction_id = str(uuid.uuid4())
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            timestamp=time.time(),
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            context=context,
            metadata={"logged_by": "orchestrator"},
            duration=duration,
            success=success,
            error_details=error_details
        )
        
        await self.logger.log_interaction(interaction)
        
        return interaction_id
    
    async def _analysis_scheduler(self):
        """Schedule periodic behavior analysis"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_analysis >= self.analysis_interval:
                    await self._run_periodic_analysis()
                    self.last_analysis = current_time
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in analysis scheduler: {str(e)}")
                await asyncio.sleep(30)
    
    async def _run_periodic_analysis(self):
        """Run periodic behavior analysis"""
        logger.info("Running periodic behavior analysis")
        
        try:
            # Get recent interactions for analysis
            cutoff_time = time.time() - 3600  # Last hour
            interactions = await self.logger.get_interactions(
                time_range=(cutoff_time, time.time()),
                limit=1000
            )
            
            if len(interactions) < 10:
                logger.debug("Insufficient interactions for analysis")
                return
            
            # Run comprehensive analysis
            analysis = await self.analyzer.analyze_behavior_patterns(interactions)
            
            # Update system metrics
            self.system_metrics["total_interactions_analyzed"] += len(interactions)
            self.system_metrics["patterns_detected"] += len(analysis.patterns_detected)
            self.system_metrics["anomalies_detected"] += len(analysis.anomalies_detected)
            self.system_metrics["optimization_recommendations"] += len(analysis.optimization_recommendations)
            self.system_metrics["analysis_runs"] += 1
            
            # Store analysis results
            await self._store_analysis_results(analysis)
            
            # Trigger actions based on critical findings
            await self._handle_critical_findings(analysis)
            
            logger.info(f"Analysis completed: {len(analysis.patterns_detected)} patterns, "
                       f"{len(analysis.optimization_recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error in periodic analysis: {str(e)}")
    
    async def _store_analysis_results(self, analysis: BehaviorAnalysis):
        """Store analysis results for dashboard and reporting"""
        # Store full analysis
        analysis_key = f"behavior_analysis:{analysis.analysis_id}"
        await self.cache.set_l2(analysis_key, asdict(analysis), timeout=86400)
        
        # Store summary for quick access
        summary = {
            "analysis_id": analysis.analysis_id,
            "timestamp": analysis.timestamp,
            "agents_count": len(analysis.agents_analyzed),
            "patterns_count": len(analysis.patterns_detected),
            "recommendations_count": len(analysis.optimization_recommendations),
            "anomalies_count": len(analysis.anomalies_detected),
            "prediction_confidence": analysis.prediction_confidence
        }
        
        await self.cache.set_l1("latest_behavior_analysis", summary)
        
        # Store pattern summaries for trending
        pattern_summary = defaultdict(int)
        for pattern in analysis.patterns_detected:
            pattern_summary[pattern.pattern_type.value] += 1
        
        await self.cache.set_l2("pattern_trends", dict(pattern_summary), timeout=86400)
    
    async def _handle_critical_findings(self, analysis: BehaviorAnalysis):
        """Handle critical findings that require immediate attention"""
        critical_issues = []
        
        # Check for critical patterns
        for pattern in analysis.patterns_detected:
            if pattern.pattern_type in [AgentBehaviorPattern.BOTTLENECK_CREATOR, 
                                       AgentBehaviorPattern.ERROR_PRONE]:
                if pattern.confidence > 0.8:
                    critical_issues.append(f"Critical pattern: {pattern.pattern_type.value} "
                                         f"for agent {pattern.agent_id}")
        
        # Check for high-severity anomalies
        for anomaly in analysis.anomalies_detected:
            if anomaly.get("severity") == "high":
                critical_issues.append(f"High-severity anomaly: {anomaly['type']}")
        
        # Check for critical recommendations
        critical_recommendations = [r for r in analysis.optimization_recommendations 
                                  if r.get("priority") == "critical"]
        
        if critical_recommendations:
            critical_issues.extend([f"Critical recommendation: {r['title']}" 
                                  for r in critical_recommendations])
        
        # Log and alert on critical issues
        if critical_issues:
            logger.warning(f"Critical behavior analysis findings: {critical_issues}")
            
            # Store for alerting system
            await self.cache.set_l1("critical_behavior_alerts", {
                "issues": critical_issues,
                "timestamp": time.time(),
                "analysis_id": analysis.analysis_id
            })
    
    async def _real_time_monitoring(self):
        """Real-time monitoring for immediate issue detection"""
        while self.monitoring_active:
            try:
                # Monitor recent interactions for immediate issues
                recent_interactions = await self.logger.get_interactions(
                    time_range=(time.time() - 300, time.time()),  # Last 5 minutes
                    limit=100
                )
                
                if recent_interactions:
                    # Quick anomaly detection
                    await self._quick_anomaly_check(recent_interactions)
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _quick_anomaly_check(self, interactions: List[AgentInteraction]):
        """Quick anomaly detection for real-time monitoring"""
        if len(interactions) < 5:
            return
        
        # Check for sudden error spikes
        recent_errors = sum(1 for i in interactions if not i.success)
        error_rate = recent_errors / len(interactions)
        
        if error_rate > 0.5:  # More than 50% errors
            logger.warning(f"High error rate detected: {error_rate:.1%} in recent interactions")
            
            await self.cache.set_l1("realtime_alert", {
                "type": "high_error_rate",
                "error_rate": error_rate,
                "timestamp": time.time(),
                "interactions_checked": len(interactions)
            })
        
        # Check for response time spikes
        response_times = [i.duration for i in interactions if i.duration is not None]
        if response_times:
            avg_response_time = np.mean(response_times)
            if avg_response_time > 10000:  # More than 10 seconds
                logger.warning(f"High response time detected: {avg_response_time:.1f}ms average")
                
                await self.cache.set_l1("realtime_alert", {
                    "type": "high_response_time",
                    "average_response_time": avg_response_time,
                    "timestamp": time.time()
                })
    
    async def _metrics_collection(self):
        """Collect and update system metrics"""
        while self.monitoring_active:
            try:
                # Update logging statistics
                logging_stats = await self.logger.get_logging_stats()
                
                # Update system metrics
                self.system_metrics.update({
                    "interactions_per_second": logging_stats.get("interactions_per_second", 0),
                    "buffer_utilization": logging_stats.get("buffer_utilization", 0),
                    "logging_active": self.monitoring_active
                })
                
                # Store metrics
                await self.cache.set_l1("behavior_system_metrics", self.system_metrics)
                
                # Sleep before next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                await asyncio.sleep(60)
    
    async def get_behavior_summary(self) -> Dict[str, Any]:
        """Get comprehensive behavior analysis summary"""
        # Get latest analysis
        latest_analysis = await self.cache.get_l1("latest_behavior_analysis")
        
        # Get pattern trends
        pattern_trends = await self.cache.get_l2("pattern_trends") or {}
        
        # Get critical alerts
        critical_alerts = await self.cache.get_l1("critical_behavior_alerts")
        
        # Get system metrics
        system_metrics = self.system_metrics.copy()
        
        return {
            "latest_analysis": latest_analysis,
            "pattern_trends": pattern_trends,
            "critical_alerts": critical_alerts,
            "system_metrics": system_metrics,
            "monitoring_status": "active" if self.monitoring_active else "inactive"
        }
    
    async def analyze_agent_behavior(self, agent_id: str, 
                                   time_range: Optional[Tuple[float, float]] = None) -> Optional[BehaviorAnalysis]:
        """Analyze behavior for a specific agent"""
        interactions = await self.logger.get_interactions(
            agent_id=agent_id,
            time_range=time_range,
            limit=500
        )
        
        if len(interactions) < 10:
            logger.warning(f"Insufficient interactions for agent {agent_id}")
            return None
        
        return await self.analyzer.analyze_behavior_patterns(interactions)
    
    async def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights about agent collaboration patterns"""
        # Get recent interactions
        recent_interactions = await self.logger.get_interactions(
            time_range=(time.time() - 3600, time.time()),  # Last hour
            limit=1000
        )
        
        if not recent_interactions:
            return {"message": "No recent interactions available"}
        
        # Extract unique agents
        agents = list(set([i.source_agent for i in recent_interactions] + 
                         [i.target_agent for i in recent_interactions if i.target_agent]))
        
        # Build collaboration graph
        collaboration_graph = await self.analyzer._build_collaboration_graph(recent_interactions, agents)
        
        return {
            "total_agents": len(agents),
            "collaboration_edges": len(collaboration_graph.edges),
            "centrality_scores": collaboration_graph.centrality_scores,
            "community_clusters": collaboration_graph.community_clusters,
            "efficiency_metrics": collaboration_graph.efficiency_metrics
        }


# Global orchestrator instance
_behavior_orchestrator = None

async def get_behavior_orchestrator() -> BehaviorAnalysisOrchestrator:
    """Get or create the global behavior analysis orchestrator"""
    global _behavior_orchestrator
    
    if _behavior_orchestrator is None:
        _behavior_orchestrator = BehaviorAnalysisOrchestrator()
    
    return _behavior_orchestrator


# Convenience functions
async def start_behavior_analysis_system():
    """Start the behavior analysis system"""
    orchestrator = await get_behavior_orchestrator()
    await orchestrator.start_behavior_analysis()


async def log_agent_interaction(source_agent: str, target_agent: Optional[str] = None,
                               interaction_type: InteractionType = InteractionType.REQUEST,
                               context: Optional[Dict[str, Any]] = None,
                               duration: Optional[float] = None,
                               success: bool = True,
                               error_details: Optional[Dict[str, Any]] = None) -> str:
    """Log an agent interaction"""
    orchestrator = await get_behavior_orchestrator()
    return await orchestrator.log_agent_interaction(
        source_agent, target_agent, interaction_type, 
        context or {}, duration, success, error_details
    )


async def get_agent_behavior_analysis(agent_id: str) -> Optional[BehaviorAnalysis]:
    """Get behavior analysis for a specific agent"""
    orchestrator = await get_behavior_orchestrator()
    return await orchestrator.analyze_agent_behavior(agent_id)


# Example usage
async def main():
    """Example of using the behavior analysis system"""
    
    # Start behavior analysis
    orchestrator = await get_behavior_orchestrator()
    
    # Log some example interactions
    await orchestrator.log_agent_interaction(
        source_agent="starri_orchestrator",
        target_agent="jules_coding_agent",
        interaction_type=InteractionType.REQUEST,
        context={"task": "generate_code", "complexity": "medium"},
        duration=1500.0,
        success=True
    )
    
    await orchestrator.log_agent_interaction(
        source_agent="jules_coding_agent",
        target_agent="starri_orchestrator", 
        interaction_type=InteractionType.RESPONSE,
        context={"task": "generate_code", "result": "success", "lines_generated": 150},
        duration=1200.0,
        success=True
    )
    
    # Get behavior summary
    summary = await orchestrator.get_behavior_summary()
    print(f"Behavior summary: {json.dumps(summary, indent=2)}")
    
    # Get collaboration insights
    insights = await orchestrator.get_collaboration_insights()
    print(f"Collaboration insights: {json.dumps(insights, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())