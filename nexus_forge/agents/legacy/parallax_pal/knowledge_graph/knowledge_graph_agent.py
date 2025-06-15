"""
Knowledge Graph Agent for ParallaxMind

This agent is responsible for creating visual knowledge graphs from research results,
extracting entities and relationships, and providing a structured representation of knowledge.
"""

import json
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action

# Import schemas
from schemas.agent_messages import (
    KnowledgeGraphRequest, KnowledgeGraphResponse, KnowledgeGraph,
    KnowledgeGraphNode, KnowledgeGraphEdge
)

# Import config
from adk_config import get_config

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeGraphAgent(Agent):
    """
    Knowledge Graph Agent for ParallaxMind
    
    This agent is responsible for extracting entities and relationships from research results
    and creating interactive knowledge graphs for visualization.
    """
    
    def __init__(self):
        """Initialize knowledge graph agent with configuration."""
        super().__init__()
        self.config = get_config()
        self.logger = logging.getLogger("knowledge_graph_agent")
        self.node_types = ["concept", "entity", "event", "process", "property"]
        self.edge_types = ["hierarchical", "causal", "temporal", "relational", "property"]
    
    def initialize(self, context: AgentContext) -> None:
        """Initialize the agent with context."""
        self.context = context
        self.logger.info("Knowledge Graph agent initialized")
    
    @action
    def generate_knowledge_graph(self, request: Dict) -> Dict:
        """
        Generate a knowledge graph from research results.
        
        Args:
            request: The knowledge graph request
            
        Returns:
            Dict containing the knowledge graph
        """
        try:
            # Parse request
            if isinstance(request, str):
                try:
                    request = json.loads(request)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse request as JSON")
                    return self._create_empty_graph_response(str(uuid.uuid4()))
            
            request_id = request.get("request_id", str(uuid.uuid4()))
            query = request.get("query", "")
            research_response = request.get("research_response", {})
            max_nodes = request.get("max_nodes", 30)
            
            self.logger.info(f"Generating knowledge graph for query: {query}")
            
            # Extract information from research response
            summary = research_response.get("summary", "")
            sources = research_response.get("sources", [])
            focus_areas = research_response.get("focus_areas", [])
            
            # Check if we have enough information to generate a graph
            if not summary and not sources and not focus_areas:
                self.logger.warning("Insufficient information to generate knowledge graph")
                return self._create_empty_graph_response(request_id)
            
            # Extract entities and relationships
            entities, relationships = self._extract_entities_relationships(query, summary, sources, focus_areas)
            
            # Create knowledge graph
            graph = self._create_knowledge_graph(query, entities, relationships, max_nodes)
            
            # Generate a text summary of the graph
            graph_summary = self._generate_graph_summary(graph, query)
            
            # Create the response
            response = {
                "request_id": request_id,
                "graph": graph,
                "summary": graph_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating knowledge graph: {str(e)}")
            return self._create_empty_graph_response(request.get("request_id", str(uuid.uuid4())))
    
    def _extract_entities_relationships(self, query: str, summary: str, sources: List[Dict], focus_areas: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from research results.
        
        Args:
            query: The original research query
            summary: The research summary
            sources: The research sources
            focus_areas: The research focus areas
            
        Returns:
            Tuple containing lists of entities and relationships
        """
        try:
            # Create a prompt for extracting entities and relationships
            content_for_extraction = f"Research Query: {query}\n\n"
            
            # Add summary
            if summary:
                content_for_extraction += f"Research Summary:\n{summary[:2000]}\n\n"
            
            # Add focus areas
            if focus_areas:
                content_for_extraction += "Research Focus Areas:\n"
                for i, area in enumerate(focus_areas[:5], 1):
                    area_content = area.get("area", "") if isinstance(area, dict) else area
                    content_for_extraction += f"{i}. {area_content}\n"
                content_for_extraction += "\n"
            
            # Add source snippets
            if sources:
                content_for_extraction += "Source Information:\n"
                for i, source in enumerate(sources[:5], 1):
                    title = source.get("title", "Untitled")
                    snippet = source.get("snippet", "")
                    content_for_extraction += f"Source {i}: {title}\n{snippet[:200]}...\n\n"
            
            # Create extraction prompt
            extraction_prompt = f"""
            Based on the following research information, identify key entities and relationships to create a knowledge graph.
            
            RESEARCH INFORMATION:
            {content_for_extraction}
            
            TASK:
            1. Extract at least 15-25 important concepts, entities, and terms from the research.
            2. For each entity, provide:
               - A clear label
               - A brief description
               - A classification type (concept, entity, event, process, property)
               - A confidence score (0-1) indicating how central this entity is to the topic
            
            3. Identify at least 20-30 meaningful relationships between these entities.
            4. For each relationship, provide:
               - The source entity
               - The target entity
               - A relationship label describing the connection
               - A relationship type (hierarchical, causal, temporal, relational, property)
               - A weight/strength score (0-1)
            
            Format your response as JSON with two sections: "entities" and "relationships".
            
            Ensure that:
            - The main query topic is included as an entity
            - Entities and relationships form a connected graph
            - The entities selected represent the most important concepts in the research
            - Relationships accurately reflect the information in the sources
            """
            
            # Use vertex_ai_generate tool to extract entities and relationships
            extraction_task = Task(
                tool_name="vertex_ai_generate",
                task_input={
                    "model": self.config.primary_model,
                    "prompt": extraction_prompt
                }
            )
            
            # Execute extraction task
            extraction_result = self.context.execute_task(extraction_task)
            
            if not extraction_result or not isinstance(extraction_result, str):
                return self._create_fallback_entities_relationships(query)
            
            # Try to parse the result as JSON
            try:
                # Extract JSON from text if needed
                json_match = re.search(r'```json\s*(.*?)\s*```', extraction_result, re.DOTALL)
                if json_match:
                    extraction_json = json.loads(json_match.group(1))
                else:
                    # Try to parse the whole text as JSON
                    extraction_json = json.loads(extraction_result)
                
                entities = extraction_json.get("entities", [])
                relationships = extraction_json.get("relationships", [])
                
                # Validate entities and relationships
                if not entities or not relationships:
                    return self._create_fallback_entities_relationships(query)
                
                return entities, relationships
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error parsing extraction result as JSON: {str(e)}")
                return self._create_fallback_entities_relationships(query)
            
        except Exception as e:
            self.logger.error(f"Error extracting entities and relationships: {str(e)}")
            return self._create_fallback_entities_relationships(query)
    
    def _create_fallback_entities_relationships(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Create fallback entities and relationships when extraction fails.
        
        Args:
            query: The original research query
            
        Returns:
            Tuple containing lists of fallback entities and relationships
        """
        self.logger.info(f"Creating fallback entities and relationships for query: {query}")
        
        # Extract keywords from query
        words = query.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in {
            "what", "when", "where", "which", "who", "whose", "whom", "why", "how",
            "does", "could", "would", "should", "about", "with", "from", "have", "this"
        }]
        
        # Create main topic entity
        main_topic = {
            "id": "topic_1",
            "label": query,
            "description": f"Main research topic: {query}",
            "type": "concept",
            "confidence": 1.0
        }
        
        # Create entities from keywords
        entities = [main_topic]
        
        for i, keyword in enumerate(keywords[:10], 1):
            entities.append({
                "id": f"entity_{i}",
                "label": keyword.capitalize(),
                "description": f"Entity extracted from query: {keyword}",
                "type": self.node_types[i % len(self.node_types)],
                "confidence": 0.8 - (i * 0.05)
            })
        
        # Add some generic entities for a more complete graph
        generic_entities = [
            {
                "id": "entity_history",
                "label": f"History of {query}",
                "description": f"Historical development and context of {query}",
                "type": "concept",
                "confidence": 0.7
            },
            {
                "id": "entity_applications",
                "label": f"Applications",
                "description": f"Practical applications and uses related to {query}",
                "type": "process",
                "confidence": 0.75
            },
            {
                "id": "entity_challenges",
                "label": f"Challenges",
                "description": f"Challenges and limitations related to {query}",
                "type": "concept",
                "confidence": 0.65
            },
            {
                "id": "entity_future",
                "label": f"Future developments",
                "description": f"Potential future developments related to {query}",
                "type": "concept",
                "confidence": 0.6
            }
        ]
        
        entities.extend(generic_entities)
        
        # Create relationships
        relationships = []
        
        # Connect main topic to all entities
        for i, entity in enumerate(entities[1:], 1):
            relationships.append({
                "source": "topic_1",
                "target": entity["id"],
                "label": "relates to",
                "type": self.edge_types[i % len(self.edge_types)],
                "weight": 0.9 - (i * 0.05)
            })
        
        # Add some additional relationships between entities
        if len(entities) > 5:
            relationships.extend([
                {
                    "source": "entity_history",
                    "target": entities[1]["id"],
                    "label": "influences",
                    "type": "causal",
                    "weight": 0.7
                },
                {
                    "source": entities[1]["id"],
                    "target": "entity_applications",
                    "label": "enables",
                    "type": "causal",
                    "weight": 0.75
                },
                {
                    "source": "entity_challenges",
                    "target": "entity_future",
                    "label": "shapes",
                    "type": "causal",
                    "weight": 0.8
                }
            ])
        
        return entities, relationships
    
    def _create_knowledge_graph(self, query: str, entities: List[Dict], relationships: List[Dict], max_nodes: int) -> Dict:
        """
        Create a knowledge graph from entities and relationships.
        
        Args:
            query: The original research query
            entities: The extracted entities
            relationships: The extracted relationships
            max_nodes: The maximum number of nodes to include
            
        Returns:
            Dict containing the knowledge graph
        """
        try:
            # Find the main topic node
            main_topic_id = None
            
            # Look for an entity that matches the query
            for entity in entities:
                if entity.get("label", "").lower() in query.lower() or query.lower() in entity.get("label", "").lower():
                    main_topic_id = entity.get("id")
                    entity["confidence"] = 1.0  # Ensure main topic has highest confidence
                    break
            
            # If no main topic found, use the highest confidence entity
            if not main_topic_id and entities:
                main_topic_id = max(entities, key=lambda e: e.get("confidence", 0)).get("id")
            
            # If still no main topic, create one
            if not main_topic_id:
                main_topic_id = "topic_main"
                entities.append({
                    "id": main_topic_id,
                    "label": query,
                    "description": f"Main research topic: {query}",
                    "type": "concept",
                    "confidence": 1.0
                })
            
            # Sort entities by confidence and limit to max_nodes
            sorted_entities = sorted(entities, key=lambda e: e.get("confidence", 0), reverse=True)
            nodes = sorted_entities[:max_nodes]
            
            # Get the IDs of included nodes
            node_ids = {node.get("id") for node in nodes}
            
            # Filter relationships to only include connections between included nodes
            edges = [
                rel for rel in relationships 
                if rel.get("source") in node_ids and rel.get("target") in node_ids
            ]
            
            # Convert to KnowledgeGraphNode and KnowledgeGraphEdge objects
            kg_nodes = []
            for node in nodes:
                kg_nodes.append(KnowledgeGraphNode(
                    id=node.get("id", str(uuid.uuid4())),
                    label=node.get("label", "Unnamed Node"),
                    type=node.get("type", "concept"),
                    description=node.get("description"),
                    confidence=node.get("confidence", 0.5),
                    sources=[]  # No source tracking in this version
                ).dict())
            
            kg_edges = []
            for edge in edges:
                kg_edges.append(KnowledgeGraphEdge(
                    source=edge.get("source"),
                    target=edge.get("target"),
                    label=edge.get("label", "related to"),
                    type=edge.get("type", "relational"),
                    weight=edge.get("weight", 1.0),
                    confidence=edge.get("confidence", 0.8),
                    sources=[]  # No source tracking in this version
                ).dict())
            
            # Create the knowledge graph
            knowledge_graph = {
                "nodes": kg_nodes,
                "edges": kg_edges,
                "main_topic": main_topic_id,
                "query": query
            }
            
            return knowledge_graph
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge graph: {str(e)}")
            return {
                "nodes": [],
                "edges": [],
                "main_topic": "",
                "query": query
            }
    
    def _generate_graph_summary(self, graph: Dict, query: str) -> str:
        """
        Generate a text summary of the knowledge graph.
        
        Args:
            graph: The knowledge graph
            query: The original research query
            
        Returns:
            Text summary of the graph
        """
        try:
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            main_topic = graph.get("main_topic", "")
            
            if not nodes or not edges or not main_topic:
                return f"No knowledge graph could be generated for the query: {query}"
            
            # Find the main topic node
            main_node = next((n for n in nodes if n.get("id") == main_topic), None)
            main_topic_label = main_node.get("label", query) if main_node else query
            
            # Count node types
            node_type_counts = {}
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            
            # Count edge types
            edge_type_counts = {}
            for edge in edges:
                edge_type = edge.get("type", "unknown")
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            
            # Generate summary
            summary = f"Knowledge Graph for: {main_topic_label}\n\n"
            summary += f"This graph contains {len(nodes)} nodes and {len(edges)} relationships "
            summary += f"centered around the main topic of \"{main_topic_label}\".\n\n"
            
            # Node types summary
            summary += "Node Types:\n"
            for node_type, count in node_type_counts.items():
                summary += f"- {node_type.capitalize()}: {count} nodes\n"
            summary += "\n"
            
            # Edge types summary
            summary += "Relationship Types:\n"
            for edge_type, count in edge_type_counts.items():
                summary += f"- {edge_type.capitalize()}: {count} relationships\n"
            summary += "\n"
            
            # Key connections
            summary += "Key Connections:\n"
            
            # Find and list connections from main topic to other nodes
            main_topic_edges = [e for e in edges if e.get("source") == main_topic or e.get("target") == main_topic]
            main_topic_edges = sorted(main_topic_edges, key=lambda e: e.get("weight", 0), reverse=True)
            
            for edge in main_topic_edges[:5]:  # List top 5 connections
                source_id = edge.get("source")
                target_id = edge.get("target")
                
                # Find the connected node (the one that's not the main topic)
                connected_id = target_id if source_id == main_topic else source_id
                connected_node = next((n for n in nodes if n.get("id") == connected_id), None)
                
                if connected_node:
                    connected_label = connected_node.get("label", "Unknown")
                    relation_label = edge.get("label", "is related to")
                    
                    if source_id == main_topic:
                        summary += f"- {main_topic_label} {relation_label} {connected_label}\n"
                    else:
                        summary += f"- {connected_label} {relation_label} {main_topic_label}\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating graph summary: {str(e)}")
            return f"A knowledge graph was generated for the query: {query}"
    
    def _create_empty_graph_response(self, request_id: str) -> Dict:
        """
        Create an empty graph response when graph generation fails.
        
        Args:
            request_id: The request ID
            
        Returns:
            Dict containing an empty knowledge graph response
        """
        return {
            "request_id": request_id,
            "graph": {
                "nodes": [],
                "edges": [],
                "main_topic": "",
                "query": ""
            },
            "summary": "Unable to generate knowledge graph due to insufficient information.",
            "timestamp": datetime.now().isoformat()
        }