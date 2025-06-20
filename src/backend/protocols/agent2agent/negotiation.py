"""
Capability Negotiation Engine for Agent2Agent Protocol

Handles capability matching, task negotiation, and contract management
between agents.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .core import Agent2AgentMessage, MessageType, Agent2AgentProtocol
from .discovery import AgentInfo

logger = logging.getLogger(__name__)


class NegotiationStatus(Enum):
    """Status of negotiation session"""
    INITIATED = "initiated"
    NEGOTIATING = "negotiating"
    CONTRACTED = "contracted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Task:
    """Task representation for negotiation"""
    id: str
    name: str
    description: str
    required_capabilities: List[str]
    optional_capabilities: List[str]
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    priority: int = 0
    max_duration: Optional[int] = None  # seconds
    success_criteria: Dict[str, Any] = None
    reward: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(**data)


@dataclass
class ContractTerms:
    """Terms of contract between agents"""
    response_time: int  # milliseconds
    success_rate: float  # 0.0 to 1.0
    availability: float  # 0.0 to 1.0
    cost: Optional[float] = None
    resource_limits: Dict[str, Any] = None
    quality_metrics: Dict[str, float] = None
    penalties: Dict[str, Any] = None
    

@dataclass
class TaskContract:
    """Contract for task execution between agents"""
    contract_id: str
    task_id: str
    requester_id: str
    provider_id: str
    terms: ContractTerms
    created_at: float
    expires_at: Optional[float] = None
    status: str = "active"
    performance_history: List[Dict[str, Any]] = None
    
    def is_valid(self) -> bool:
        """Check if contract is still valid"""
        if self.status != "active":
            return False
            
        if self.expires_at and time.time() > self.expires_at:
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "task_id": self.task_id,
            "requester_id": self.requester_id,
            "provider_id": self.provider_id,
            "terms": asdict(self.terms),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "performance_history": self.performance_history or []
        }


@dataclass
class NegotiationSession:
    """Session for negotiating task execution"""
    session_id: str
    task: Task
    requester_id: str
    provider_id: str
    status: NegotiationStatus
    created_at: float
    proposals: List[Dict[str, Any]]
    counter_proposals: List[Dict[str, Any]]
    final_terms: Optional[ContractTerms] = None
    rejection_reason: Optional[str] = None
    

class SemanticCapabilityMatcher:
    """Semantic matching of capabilities using embeddings"""
    
    def __init__(self):
        self.capability_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim = 128
        
    def _generate_embedding(self, capability: str) -> np.ndarray:
        """Generate embedding for capability"""
        # In production, use a proper embedding model
        # For now, use a simple hash-based approach
        np.random.seed(hash(capability) % 2**32)
        return np.random.randn(self.embedding_dim)
        
    def get_embedding(self, capability: str) -> np.ndarray:
        """Get or generate embedding for capability"""
        if capability not in self.capability_embeddings:
            self.capability_embeddings[capability] = self._generate_embedding(capability)
        return self.capability_embeddings[capability]
        
    def match_capabilities(
        self,
        required: List[str],
        available: List[str],
        threshold: float = 0.8
    ) -> float:
        """Calculate semantic match score between capability sets"""
        if not required:
            return 1.0
            
        if not available:
            return 0.0
            
        # Get embeddings
        req_embeddings = np.array([self.get_embedding(cap) for cap in required])
        avail_embeddings = np.array([self.get_embedding(cap) for cap in available])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(req_embeddings, avail_embeddings)
        
        # For each required capability, find best match
        best_matches = np.max(similarity_matrix, axis=1)
        
        # Calculate overall match score
        matches_above_threshold = np.sum(best_matches >= threshold)
        match_score = matches_above_threshold / len(required)
        
        return match_score
        
    def find_missing_capabilities(
        self,
        required: List[str],
        available: List[str],
        threshold: float = 0.8
    ) -> List[str]:
        """Find required capabilities not covered by available ones"""
        if not required:
            return []
            
        if not available:
            return required
            
        # Get embeddings
        req_embeddings = np.array([self.get_embedding(cap) for cap in required])
        avail_embeddings = np.array([self.get_embedding(cap) for cap in available])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(req_embeddings, avail_embeddings)
        
        # Find unmatched capabilities
        missing = []
        for i, cap in enumerate(required):
            best_match = np.max(similarity_matrix[i])
            if best_match < threshold:
                missing.append(cap)
                
        return missing


class PerformancePredictor:
    """Predict agent performance for tasks"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def record_performance(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        duration: float,
        resource_usage: Dict[str, float]
    ):
        """Record agent performance for a task"""
        self.performance_history[agent_id].append({
            "task_type": task_type,
            "success": success,
            "duration": duration,
            "resource_usage": resource_usage,
            "timestamp": time.time()
        })
        
    def predict_performance(
        self,
        agent_id: str,
        task: Task
    ) -> Dict[str, float]:
        """Predict agent performance for a task"""
        history = self.performance_history.get(agent_id, [])
        
        if not history:
            # No history, return default predictions
            return {
                "success_probability": 0.8,
                "expected_duration": task.max_duration or 300,
                "confidence": 0.1
            }
            
        # Filter relevant history
        relevant_history = [
            h for h in history
            if self._is_similar_task(h["task_type"], task)
        ]
        
        if not relevant_history:
            relevant_history = history[-10:]  # Use last 10 tasks
            
        # Calculate metrics
        successes = sum(1 for h in relevant_history if h["success"])
        success_rate = successes / len(relevant_history)
        
        durations = [h["duration"] for h in relevant_history if h["success"]]
        avg_duration = np.mean(durations) if durations else (task.max_duration or 300)
        
        # Confidence based on amount of relevant data
        confidence = min(len(relevant_history) / 10, 1.0)
        
        return {
            "success_probability": success_rate,
            "expected_duration": avg_duration,
            "confidence": confidence
        }
        
    def _is_similar_task(self, task_type: str, task: Task) -> bool:
        """Check if task type is similar to given task"""
        # Simple string matching for now
        return (task_type.lower() in task.name.lower() or
                task.name.lower() in task_type.lower())


class CapabilityNegotiationEngine:
    """Engine for capability matching and task negotiation"""
    
    def __init__(self, protocol: Agent2AgentProtocol):
        self.protocol = protocol
        self.capability_matcher = SemanticCapabilityMatcher()
        self.performance_predictor = PerformancePredictor()
        self.negotiation_sessions: Dict[str, NegotiationSession] = {}
        self.contracts: Dict[str, TaskContract] = {}
        
        # Configuration
        self.negotiation_timeout = 60  # seconds
        self.max_negotiation_rounds = 5
        
        # Setup protocol handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup message handlers for negotiation"""
        self.protocol.message_handler.register_handler(
            MessageType.TASK_PROPOSE,
            self._handle_task_proposal
        )
        self.protocol.message_handler.register_handler(
            MessageType.TASK_ACCEPT,
            self._handle_task_accept
        )
        self.protocol.message_handler.register_handler(
            MessageType.TASK_REJECT,
            self._handle_task_reject
        )
        
    async def negotiate_task(
        self,
        task: Task,
        candidate_agents: List[AgentInfo]
    ) -> Optional[TaskContract]:
        """Negotiate task execution with candidate agents"""
        
        # Rank agents by capability match
        ranked_agents = await self._rank_agents_for_task(task, candidate_agents)
        
        if not ranked_agents:
            logger.warning(f"No suitable agents found for task {task.id}")
            return None
            
        # Try negotiation with top candidates
        for agent, score in ranked_agents[:3]:  # Try top 3
            contract = await self._negotiate_with_agent(task, agent)
            if contract:
                return contract
                
        return None
        
    async def _rank_agents_for_task(
        self,
        task: Task,
        agents: List[AgentInfo]
    ) -> List[Tuple[AgentInfo, float]]:
        """Rank agents by suitability for task"""
        agent_scores = []
        
        for agent in agents:
            # Calculate capability match
            capability_score = self.capability_matcher.match_capabilities(
                task.required_capabilities,
                agent.capabilities
            )
            
            # Skip if doesn't meet minimum requirements
            if capability_score < 0.7:
                continue
                
            # Predict performance
            performance = self.performance_predictor.predict_performance(
                agent.id,
                task
            )
            
            # Calculate composite score
            score = (
                capability_score * 0.4 +
                performance["success_probability"] * 0.4 +
                performance["confidence"] * 0.2
            )
            
            # Apply performance metrics if available
            if agent.performance_metrics:
                if "response_time" in agent.performance_metrics:
                    # Prefer faster agents
                    response_score = 1.0 / (1.0 + agent.performance_metrics["response_time"] / 1000)
                    score = score * 0.8 + response_score * 0.2
                    
            agent_scores.append((agent, score))
            
        # Sort by score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        return agent_scores
        
    async def _negotiate_with_agent(
        self,
        task: Task,
        agent: AgentInfo
    ) -> Optional[TaskContract]:
        """Negotiate contract with specific agent"""
        
        # Create negotiation session
        session = NegotiationSession(
            session_id=str(uuid.uuid4()),
            task=task,
            requester_id=self.protocol.agent_id,
            provider_id=agent.id,
            status=NegotiationStatus.INITIATED,
            created_at=time.time(),
            proposals=[],
            counter_proposals=[]
        )
        
        self.negotiation_sessions[session.session_id] = session
        
        # Create initial proposal
        initial_proposal = self._create_initial_proposal(task, agent)
        session.proposals.append(initial_proposal)
        
        # Send proposal
        proposal_message = Agent2AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_PROPOSE,
            sender=self.protocol.agent_id,
            recipient=agent.id,
            payload={
                "session_id": session.session_id,
                "task": task.to_dict(),
                "proposal": initial_proposal
            },
            timestamp=time.time(),
            requires_ack=True,
            priority=1
        )
        
        # Wait for response
        try:
            response = await asyncio.wait_for(
                self._wait_for_negotiation_response(session.session_id),
                timeout=self.negotiation_timeout
            )
            
            if session.status == NegotiationStatus.CONTRACTED:
                # Create contract
                contract = TaskContract(
                    contract_id=session.session_id,
                    task_id=task.id,
                    requester_id=self.protocol.agent_id,
                    provider_id=agent.id,
                    terms=session.final_terms,
                    created_at=time.time(),
                    expires_at=time.time() + (task.max_duration or 3600)
                )
                
                self.contracts[contract.contract_id] = contract
                return contract
                
        except asyncio.TimeoutError:
            logger.warning(f"Negotiation timeout with agent {agent.id}")
            session.status = NegotiationStatus.EXPIRED
            
        return None
        
    def _create_initial_proposal(self, task: Task, agent: AgentInfo) -> Dict[str, Any]:
        """Create initial proposal for task"""
        # Predict performance
        performance = self.performance_predictor.predict_performance(agent.id, task)
        
        # Create terms based on task requirements and agent capabilities
        proposed_terms = {
            "response_time": min(
                task.requirements.get("max_response_time", 1000),
                agent.performance_metrics.get("avg_response_time", 500)
            ),
            "success_rate": max(
                task.requirements.get("min_success_rate", 0.9),
                performance["success_probability"]
            ),
            "availability": agent.performance_metrics.get("availability", 0.99),
            "resource_limits": task.constraints.get("resources", {}),
            "quality_metrics": task.success_criteria or {}
        }
        
        # Add cost if specified
        if task.reward:
            proposed_terms["cost"] = task.reward
            
        return proposed_terms
        
    async def _wait_for_negotiation_response(self, session_id: str) -> Dict[str, Any]:
        """Wait for negotiation response"""
        session = self.negotiation_sessions.get(session_id)
        if not session:
            raise ValueError(f"Unknown negotiation session: {session_id}")
            
        # Wait for status change
        while session.status in [NegotiationStatus.INITIATED, NegotiationStatus.NEGOTIATING]:
            await asyncio.sleep(0.1)
            
        return {
            "status": session.status,
            "final_terms": session.final_terms,
            "rejection_reason": session.rejection_reason
        }
        
    async def _handle_task_proposal(self, message: Agent2AgentMessage) -> Optional[Agent2AgentMessage]:
        """Handle incoming task proposal"""
        session_id = message.payload.get("session_id")
        task_data = message.payload.get("task", {})
        proposal = message.payload.get("proposal", {})
        
        # Evaluate proposal
        task = Task.from_dict(task_data)
        decision = await self._evaluate_proposal(task, proposal)
        
        if decision["accept"]:
            # Accept proposal
            terms = ContractTerms(**proposal)
            
            return Agent2AgentMessage(
                id=str(uuid.uuid4()),
                correlation_id=message.id,
                type=MessageType.TASK_ACCEPT,
                sender=self.protocol.agent_id,
                recipient=message.sender,
                payload={
                    "session_id": session_id,
                    "terms": asdict(terms),
                    "message": "Proposal accepted"
                },
                timestamp=time.time()
            )
        else:
            # Reject or counter-propose
            return Agent2AgentMessage(
                id=str(uuid.uuid4()),
                correlation_id=message.id,
                type=MessageType.TASK_REJECT,
                sender=self.protocol.agent_id,
                recipient=message.sender,
                payload={
                    "session_id": session_id,
                    "reason": decision["reason"],
                    "counter_proposal": decision.get("counter_proposal")
                },
                timestamp=time.time()
            )
            
    async def _evaluate_proposal(
        self,
        task: Task,
        proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate task proposal"""
        # Check if we have required capabilities
        # This is simplified - real implementation would check actual agent capabilities
        
        # For now, accept if success rate is reasonable
        if proposal.get("success_rate", 0) >= 0.8:
            return {"accept": True}
        else:
            return {
                "accept": False,
                "reason": "Success rate too low",
                "counter_proposal": {
                    **proposal,
                    "success_rate": 0.85
                }
            }
            
    async def _handle_task_accept(self, message: Agent2AgentMessage) -> None:
        """Handle task acceptance"""
        session_id = message.payload.get("session_id")
        terms_data = message.payload.get("terms", {})
        
        session = self.negotiation_sessions.get(session_id)
        if session:
            session.status = NegotiationStatus.CONTRACTED
            session.final_terms = ContractTerms(**terms_data)
            
    async def _handle_task_reject(self, message: Agent2AgentMessage) -> None:
        """Handle task rejection"""
        session_id = message.payload.get("session_id")
        reason = message.payload.get("reason")
        counter_proposal = message.payload.get("counter_proposal")
        
        session = self.negotiation_sessions.get(session_id)
        if session:
            if counter_proposal and len(session.proposals) < self.max_negotiation_rounds:
                # Continue negotiation
                session.status = NegotiationStatus.NEGOTIATING
                session.counter_proposals.append(counter_proposal)
                # Would send new proposal here
            else:
                # End negotiation
                session.status = NegotiationStatus.REJECTED
                session.rejection_reason = reason
                
    async def monitor_contract_performance(self, contract_id: str) -> Dict[str, Any]:
        """Monitor performance of active contract"""
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Unknown contract: {contract_id}")
            
        # In real implementation, would track actual performance metrics
        # For now, return mock data
        return {
            "contract_id": contract_id,
            "status": "active" if contract.is_valid() else "expired",
            "performance": {
                "tasks_completed": 10,
                "success_rate": 0.95,
                "avg_response_time": 450,
                "sla_compliance": 0.98
            }
        }