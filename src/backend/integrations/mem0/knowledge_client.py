"""
Mem0 Knowledge Graph Integration for Nexus Forge

Provides intelligent knowledge management for agents, patterns, and relationships
using Mem0's graph-based memory system.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from nexus_forge.core.cache import CacheStrategy, RedisCache
from nexus_forge.core.exceptions import IntegrationError
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class Mem0KnowledgeClient:
    """
    Enterprise-grade Mem0 integration for knowledge graph operations

    Features:
    - Entity management (agents, patterns, workflows)
    - Relationship tracking
    - Pattern recognition and retrieval
    - Semantic search capabilities
    - Knowledge persistence
    """

    def __init__(
        self,
        api_key: str,
        orchestrator_id: str,
        base_url: str = "https://api.mem0.ai/v1",
        cache: Optional[RedisCache] = None,
    ):
        """Initialize Mem0 knowledge client"""
        self.api_key = api_key
        self.orchestrator_id = orchestrator_id
        self.base_url = base_url
        self.cache = cache or RedisCache()

        # Entity type mappings
        self.entity_types = {
            "agent": "AI_AGENT",
            "pattern": "KNOWLEDGE_PATTERN",
            "workflow": "WORKFLOW",
            "task": "TASK",
            "capability": "CAPABILITY",
            "orchestrator": "ORCHESTRATOR",
        }

        # Relationship type mappings
        self.relationship_types = {
            "executes": "EXECUTES",
            "requires": "REQUIRES",
            "produces": "PRODUCES",
            "coordinates": "COORDINATES",
            "implements": "IMPLEMENTS",
            "learns_from": "LEARNS_FROM",
            "optimizes": "OPTIMIZES",
        }

        # In-memory graph cache for fast lookups
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self.relationship_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Pattern recognition storage
        self.pattern_library: Dict[str, List[Dict[str, Any]]] = {
            "task_decomposition": [],
            "agent_coordination": [],
            "error_recovery": [],
            "optimization": [],
            "workflow_execution": [],
        }

        logger.info(
            f"Initialized Mem0 knowledge client for orchestrator: {orchestrator_id}"
        )

    async def initialize_orchestrator_knowledge(self):
        """Initialize knowledge graph with orchestrator entity and base patterns"""
        try:
            # Create orchestrator entity
            orchestrator_entity = {
                "name": f"Starri-{self.orchestrator_id}",
                "type": self.entity_types["orchestrator"],
                "properties": {
                    "model": "gemini-2.5-flash-thinking",
                    "capabilities": [
                        "deep_thinking",
                        "task_decomposition",
                        "agent_coordination",
                        "pattern_learning",
                    ],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            await self.create_entity(orchestrator_entity)

            # Initialize base patterns
            base_patterns = [
                {
                    "name": "sequential_task_execution",
                    "type": "workflow_pattern",
                    "description": "Execute tasks one after another",
                    "confidence": 0.9,
                },
                {
                    "name": "parallel_task_execution",
                    "type": "workflow_pattern",
                    "description": "Execute independent tasks simultaneously",
                    "confidence": 0.85,
                },
                {
                    "name": "error_retry_pattern",
                    "type": "recovery_pattern",
                    "description": "Retry failed tasks with exponential backoff",
                    "confidence": 0.95,
                },
            ]

            for pattern in base_patterns:
                await self.add_pattern_entity(pattern)

            logger.info("Initialized orchestrator knowledge base")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator knowledge: {e}")
            raise IntegrationError(f"Mem0 initialization failed: {e}")

    async def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """Create a new entity in the knowledge graph"""
        try:
            entity_id = entity_data.get("id", f"entity_{uuid4().hex[:8]}")

            # Store in cache
            self.entity_cache[entity_id] = {
                **entity_data,
                "id": entity_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store in Redis for persistence
            cache_key = f"mem0:entity:{entity_id}"
            self.cache.set(
                cache_key,
                self.entity_cache[entity_id],
                timeout=86400,  # 24 hours
                strategy=CacheStrategy.COMPRESSED,
            )

            logger.debug(
                f"Created entity: {entity_id} of type {entity_data.get('type')}"
            )
            return entity_id

        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            raise IntegrationError(f"Entity creation failed: {e}")

    async def create_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a relationship between two entities"""
        try:
            relationship_id = f"rel_{uuid4().hex[:8]}"

            relationship = {
                "id": relationship_id,
                "from": from_entity,
                "to": to_entity,
                "type": self.relationship_types.get(
                    relationship_type, relationship_type
                ),
                "properties": properties or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Update relationship cache
            if from_entity not in self.relationship_cache:
                self.relationship_cache[from_entity] = []
            self.relationship_cache[from_entity].append(relationship)

            # Store in Redis
            cache_key = f"mem0:relationship:{relationship_id}"
            self.cache.set(
                cache_key, relationship, timeout=86400, strategy=CacheStrategy.SIMPLE
            )

            logger.debug(
                f"Created relationship: {from_entity} -{relationship_type}-> {to_entity}"
            )
            return relationship_id

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise IntegrationError(f"Relationship creation failed: {e}")

    async def add_agent_entity(
        self, agent_id: str, agent_type: str, capabilities: List[str]
    ) -> str:
        """Add an agent entity to the knowledge graph"""
        entity_data = {
            "id": agent_id,
            "name": agent_id,
            "type": self.entity_types["agent"],
            "properties": {
                "agent_type": agent_type,
                "capabilities": capabilities,
                "status": "registered",
                "performance_metrics": {
                    "tasks_completed": 0,
                    "success_rate": 0.0,
                    "average_execution_time": 0.0,
                },
            },
        }

        entity_id = await self.create_entity(entity_data)

        # Create relationship: orchestrator coordinates agent
        await self.create_relationship(self.orchestrator_id, entity_id, "coordinates")

        # Create capability entities and relationships
        for capability in capabilities:
            cap_entity = {
                "id": f"cap_{capability}",
                "name": capability,
                "type": self.entity_types["capability"],
            }
            cap_id = await self.create_entity(cap_entity)

            # Agent implements capability
            await self.create_relationship(entity_id, cap_id, "implements")

        return entity_id

    async def add_thinking_pattern(
        self, pattern_type: str, pattern_content: Dict[str, Any], confidence: float
    ) -> str:
        """Add a thinking pattern to the knowledge graph"""
        pattern_entity = {
            "name": f"pattern_{pattern_type}_{uuid4().hex[:8]}",
            "type": self.entity_types["pattern"],
            "properties": {
                "pattern_type": pattern_type,
                "content": pattern_content,
                "confidence": confidence,
                "usage_count": 0,
                "success_rate": 0.0,
            },
        }

        pattern_id = await self.create_entity(pattern_entity)

        # Store in pattern library
        if pattern_type in self.pattern_library:
            self.pattern_library[pattern_type].append(
                {
                    "id": pattern_id,
                    "pattern": pattern_entity,
                    "embedding": self._generate_pattern_embedding(pattern_content),
                }
            )

        # Orchestrator learns from pattern
        await self.create_relationship(
            self.orchestrator_id,
            pattern_id,
            "learns_from",
            properties={"confidence": confidence},
        )

        return pattern_id

    async def add_pattern_entity(self, pattern: Dict[str, Any]) -> str:
        """Add a pattern entity to the knowledge graph"""
        pattern_entity = {
            "name": pattern["name"],
            "type": self.entity_types["pattern"],
            "properties": {
                "pattern_type": pattern.get("type", "general"),
                "description": pattern.get("description", ""),
                "confidence": pattern.get("confidence", 0.5),
                "usage_count": 0,
            },
        }

        return await self.create_entity(pattern_entity)

    async def add_execution_pattern(
        self, workflow_type: str, pattern_content: Dict[str, Any], success_rate: float
    ) -> str:
        """Add an execution pattern from workflow results"""
        pattern_id = await self.add_thinking_pattern(
            f"execution_{workflow_type}", pattern_content, success_rate
        )

        # Update pattern with execution-specific metrics
        if pattern_id in self.entity_cache:
            self.entity_cache[pattern_id]["properties"].update(
                {
                    "workflow_type": workflow_type,
                    "success_rate": success_rate,
                    "last_execution": datetime.now(timezone.utc).isoformat(),
                }
            )

        return pattern_id

    async def search_patterns(
        self,
        query: str,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for patterns in the knowledge graph"""
        try:
            results = []

            # Generate query embedding
            query_embedding = self._generate_pattern_embedding({"query": query})

            # Search in pattern library
            patterns_to_search = (
                self.pattern_library.get(pattern_type, [])
                if pattern_type
                else [p for patterns in self.pattern_library.values() for p in patterns]
            )

            # Calculate similarity scores
            scored_patterns = []
            for pattern_data in patterns_to_search:
                pattern = pattern_data["pattern"]
                if pattern["properties"]["confidence"] >= min_confidence:
                    similarity = self._calculate_similarity(
                        query_embedding, pattern_data["embedding"]
                    )
                    scored_patterns.append((similarity, pattern))

            # Sort by similarity and return top results
            scored_patterns.sort(key=lambda x: x[0], reverse=True)
            results = [pattern for _, pattern in scored_patterns[:limit]]

            # Cache search results
            cache_key = f"mem0:search:{hashlib.md5(query.encode()).hexdigest()}"
            self.cache.set(
                cache_key,
                results,
                timeout=300,  # 5 minutes
                strategy=CacheStrategy.SIMPLE,
            )

            return results

        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []

    async def get_agent_relationships(
        self, agent_id: str, relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships for an agent"""
        relationships = self.relationship_cache.get(agent_id, [])

        if relationship_type:
            relationships = [
                rel
                for rel in relationships
                if rel["type"]
                == self.relationship_types.get(relationship_type, relationship_type)
            ]

        return relationships

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        # Check cache first
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]

        # Check Redis
        cache_key = f"mem0:entity:{entity_id}"
        entity = self.cache.get(cache_key, CacheStrategy.COMPRESSED)

        if entity:
            self.entity_cache[entity_id] = entity
            return entity

        return None

    async def update_entity_metrics(self, entity_id: str, metrics: Dict[str, Any]):
        """Update metrics for an entity"""
        entity = await self.get_entity(entity_id)
        if entity:
            if "properties" not in entity:
                entity["properties"] = {}

            entity["properties"].update(metrics)
            entity["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Update caches
            self.entity_cache[entity_id] = entity
            cache_key = f"mem0:entity:{entity_id}"
            self.cache.set(
                cache_key, entity, timeout=86400, strategy=CacheStrategy.COMPRESSED
            )

    async def find_similar_workflows(
        self, workflow_description: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar workflows based on description"""
        # Search for workflow patterns
        similar_patterns = await self.search_patterns(
            workflow_description, pattern_type="workflow_execution", limit=limit
        )

        workflows = []
        for pattern in similar_patterns:
            # Extract workflow information from pattern
            if "workflow_id" in pattern.get("properties", {}):
                workflow_entity = await self.get_entity(
                    pattern["properties"]["workflow_id"]
                )
                if workflow_entity:
                    workflows.append(workflow_entity)

        return workflows

    async def get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        agent_entity = await self.get_entity(agent_id)

        if agent_entity:
            return agent_entity.get("properties", {}).get(
                "performance_metrics",
                {
                    "tasks_completed": 0,
                    "success_rate": 0.0,
                    "average_execution_time": 0.0,
                },
            )

        return {}

    async def save_orchestrator_state(
        self,
        orchestrator_id: str,
        metrics: Dict[str, Any],
        thinking_chains: Dict[str, List[Dict[str, Any]]],
    ):
        """Save orchestrator state and patterns"""
        try:
            # Update orchestrator entity
            await self.update_entity_metrics(
                orchestrator_id,
                {
                    "metrics": metrics,
                    "last_checkpoint": datetime.now(timezone.utc).isoformat(),
                    "thinking_chains_count": len(thinking_chains),
                },
            )

            # Save thinking chains as patterns
            for chain_id, chain in thinking_chains.items():
                if chain:  # Only save non-empty chains
                    await self.add_thinking_pattern(
                        "thinking_chain",
                        {
                            "chain_id": chain_id,
                            "steps": len(chain),
                            "final_confidence": chain[-1].get("confidence", 0.5),
                        },
                        chain[-1].get("confidence", 0.5),
                    )

            logger.info(
                f"Saved orchestrator state with {len(thinking_chains)} thinking chains"
            )

        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")

    def _generate_pattern_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate a simple embedding for pattern matching"""
        # In production, use a proper embedding model
        # For now, create a simple hash-based embedding
        content_str = json.dumps(content, sort_keys=True)
        hash_value = hashlib.sha256(content_str.encode()).hexdigest()

        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_value), 8):
            chunk = hash_value[i : i + 8]
            embedding.append(int(chunk, 16) / (16**8))

        return embedding

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        if len(embedding1) != len(embedding2):
            return 0.0

        # Simple dot product similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 * magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the entire knowledge graph"""
        return {
            "entities": list(self.entity_cache.values()),
            "relationships": [
                rel for rels in self.relationship_cache.values() for rel in rels
            ],
            "patterns": self.pattern_library,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def import_knowledge_graph(self, graph_data: Dict[str, Any]):
        """Import a knowledge graph"""
        try:
            # Import entities
            for entity in graph_data.get("entities", []):
                self.entity_cache[entity["id"]] = entity

            # Import relationships
            for relationship in graph_data.get("relationships", []):
                from_entity = relationship["from"]
                if from_entity not in self.relationship_cache:
                    self.relationship_cache[from_entity] = []
                self.relationship_cache[from_entity].append(relationship)

            # Import patterns
            for pattern_type, patterns in graph_data.get("patterns", {}).items():
                self.pattern_library[pattern_type] = patterns

            logger.info(
                f"Imported knowledge graph with {len(self.entity_cache)} entities"
            )

        except Exception as e:
            logger.error(f"Failed to import knowledge graph: {e}")
            raise IntegrationError(f"Knowledge graph import failed: {e}")
