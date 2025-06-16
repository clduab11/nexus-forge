"""
Knowledge Graph Agent for ParallaxMind ADK Implementation

This agent creates, manages, and visualizes knowledge graphs to represent
the relationships between research topics, concepts, and sources.
"""

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class KnowledgeGraphAgent:
    """
    Specialized agent for creating and managing knowledge graphs.

    This agent is responsible for:
    - Extracting entities and relationships from research content
    - Building knowledge graphs from research results
    - Identifying concept connections and hierarchies
    - Generating visual representations
    - Supporting exploration and navigation
    """

    def __init__(self):
        """Initialize the Knowledge Graph Agent."""
        self.entity_cache = {}
        self.relationship_cache = {}
        self.graph_cache = {}

        # Entity extraction patterns
        self.entity_patterns = {
            "person": [
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Basic name pattern
                r"\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",  # Titles
            ],
            "organization": [
                r"\b[A-Z][A-Z]+\b",  # Acronyms
                r"\b(?:University|Institute|Corporation|Company|Inc\.|Ltd\.|LLC)\b",
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|Institute|Corporation|Company)\b",
            ],
            "location": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b",  # City, State/Country
                r"\b(?:United States|USA|UK|Canada|Australia|Germany|France|Japan|China|India)\b",
            ],
            "concept": [
                r"\b[a-z]+(?:\s+[a-z]+)*\b(?=\s+(?:theory|model|framework|approach|method|system|process))",
                r"\b(?:artificial intelligence|machine learning|deep learning|neural networks|blockchain|quantum computing)\b",
            ],
            "technology": [
                r"\b(?:AI|ML|IoT|API|GPU|CPU|SaaS|PaaS|IaaS|VR|AR|5G|6G)\b",
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:technology|platform|system|framework|library)\b",
            ],
        }

        # Relationship indicators
        self.relationship_indicators = {
            "causes": ["causes", "leads to", "results in", "triggers", "induces"],
            "enables": ["enables", "allows", "facilitates", "supports", "helps"],
            "requires": [
                "requires",
                "needs",
                "depends on",
                "relies on",
                "necessitates",
            ],
            "contains": ["contains", "includes", "comprises", "consists of", "has"],
            "improves": ["improves", "enhances", "increases", "boosts", "optimizes"],
            "reduces": ["reduces", "decreases", "minimizes", "lowers", "diminishes"],
            "relates_to": [
                "relates to",
                "connected to",
                "associated with",
                "linked to",
                "tied to",
            ],
            "competes_with": [
                "competes with",
                "rivals",
                "opposes",
                "conflicts with",
                "versus",
            ],
        }

        logger.info("Knowledge Graph Agent initialized")

    async def build_knowledge_graph(
        self, research_data: Dict, context: str = ""
    ) -> Dict:
        """
        Build a knowledge graph from research data.

        Args:
            research_data: Research results containing sources and analysis
            context: Research context for better entity extraction

        Returns:
            Dictionary containing the knowledge graph structure
        """
        try:
            logger.info("Building knowledge graph from research data")

            # Extract entities from all sources
            entities = await self._extract_entities(research_data, context)

            # Extract relationships
            relationships = await self._extract_relationships(research_data, entities)

            # Build graph structure
            graph_structure = await self._build_graph_structure(entities, relationships)

            # Calculate graph metrics
            metrics = await self._calculate_graph_metrics(graph_structure)

            # Generate visual layout
            layout = await self._generate_visual_layout(graph_structure)

            # Create clusters and hierarchies
            clusters = await self._identify_clusters(graph_structure, entities)

            # Generate navigation paths
            navigation_paths = await self._generate_navigation_paths(
                graph_structure, context
            )

            knowledge_graph = {
                "entities": entities,
                "relationships": relationships,
                "graph_structure": graph_structure,
                "visual_layout": layout,
                "clusters": clusters,
                "navigation_paths": navigation_paths,
                "metrics": metrics,
                "context": context,
                "created_at": datetime.now().isoformat(),
                "graph_id": hashlib.md5(
                    f"{context}{datetime.now()}".encode()
                ).hexdigest()[:12],
            }

            # Cache the graph
            graph_id = knowledge_graph["graph_id"]
            self.graph_cache[graph_id] = knowledge_graph

            logger.info(
                f"Built knowledge graph with {len(entities)} entities and {len(relationships)} relationships"
            )
            return knowledge_graph

        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {
                "entities": {},
                "relationships": [],
                "error": str(e),
                "graph_structure": {"nodes": [], "edges": []},
                "visual_layout": {"nodes": [], "edges": []},
            }

    async def enhance_knowledge_graph(
        self, graph_id: str, additional_data: Dict
    ) -> Dict:
        """
        Enhance an existing knowledge graph with additional data.

        Args:
            graph_id: ID of the existing knowledge graph
            additional_data: Additional research data to incorporate

        Returns:
            Enhanced knowledge graph
        """
        try:
            if graph_id not in self.graph_cache:
                return {"error": "Knowledge graph not found"}

            existing_graph = self.graph_cache[graph_id]

            # Extract new entities and relationships
            new_entities = await self._extract_entities(
                additional_data, existing_graph["context"]
            )
            new_relationships = await self._extract_relationships(
                additional_data, new_entities
            )

            # Merge with existing graph
            merged_entities = await self._merge_entities(
                existing_graph["entities"], new_entities
            )
            merged_relationships = await self._merge_relationships(
                existing_graph["relationships"], new_relationships
            )

            # Rebuild graph structure
            enhanced_structure = await self._build_graph_structure(
                merged_entities, merged_relationships
            )

            # Update visual layout
            enhanced_layout = await self._generate_visual_layout(enhanced_structure)

            # Update clusters
            enhanced_clusters = await self._identify_clusters(
                enhanced_structure, merged_entities
            )

            # Update metrics
            enhanced_metrics = await self._calculate_graph_metrics(enhanced_structure)

            enhanced_graph = {
                **existing_graph,
                "entities": merged_entities,
                "relationships": merged_relationships,
                "graph_structure": enhanced_structure,
                "visual_layout": enhanced_layout,
                "clusters": enhanced_clusters,
                "metrics": enhanced_metrics,
                "last_enhanced": datetime.now().isoformat(),
            }

            # Update cache
            self.graph_cache[graph_id] = enhanced_graph

            logger.info(f"Enhanced knowledge graph {graph_id}")
            return enhanced_graph

        except Exception as e:
            logger.error(f"Error enhancing knowledge graph: {e}")
            return {"error": str(e)}

    async def query_knowledge_graph(
        self, graph_id: str, query: str, query_type: str = "concept"
    ) -> Dict:
        """
        Query a knowledge graph for specific information.

        Args:
            graph_id: ID of the knowledge graph
            query: Query string
            query_type: Type of query (concept, entity, relationship, path)

        Returns:
            Query results
        """
        try:
            if graph_id not in self.graph_cache:
                return {"error": "Knowledge graph not found"}

            graph = self.graph_cache[graph_id]
            results = {"query": query, "query_type": query_type, "results": []}

            if query_type == "concept":
                results["results"] = await self._search_concepts(graph, query)
            elif query_type == "entity":
                results["results"] = await self._search_entities(graph, query)
            elif query_type == "relationship":
                results["results"] = await self._search_relationships(graph, query)
            elif query_type == "path":
                results["results"] = await self._find_paths(graph, query)
            else:
                # General search across all types
                results["results"] = await self._general_search(graph, query)

            return results

        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return {"error": str(e), "query": query}

    async def _extract_entities(self, research_data: Dict, context: str) -> Dict:
        """Extract entities from research data."""
        try:
            entities = defaultdict(list)
            text_content = ""

            # Collect all text content
            if "sources" in research_data:
                for source in research_data["sources"]:
                    text_content += (
                        f" {source.get('title', '')} {source.get('content', '')}"
                    )

            if "analysis" in research_data:
                text_content += f" {research_data['analysis']}"

            text_content += f" {context}"

            # Extract entities by type
            for entity_type, patterns in self.entity_patterns.items():
                type_entities = set()

                for pattern in patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = " ".join(match)

                        # Clean and validate entity
                        entity = match.strip()
                        if len(entity) > 2 and len(entity) < 100:  # Basic validation
                            type_entities.add(entity)

                # Convert to list with metadata
                for entity in type_entities:
                    entity_id = hashlib.md5(entity.encode()).hexdigest()[:8]
                    entities[entity_type].append(
                        {
                            "id": entity_id,
                            "name": entity,
                            "type": entity_type,
                            "frequency": text_content.lower().count(entity.lower()),
                            "importance": self._calculate_entity_importance(
                                entity, text_content
                            ),
                            "context_snippets": self._extract_context_snippets(
                                entity, text_content
                            ),
                        }
                    )

            # Sort entities by importance
            for entity_type in entities:
                entities[entity_type].sort(key=lambda x: x["importance"], reverse=True)
                entities[entity_type] = entities[entity_type][
                    :20
                ]  # Keep top 20 per type

            return dict(entities)

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}

    async def _extract_relationships(
        self, research_data: Dict, entities: Dict
    ) -> List[Dict]:
        """Extract relationships between entities."""
        try:
            relationships = []
            text_content = ""

            # Collect all text content
            if "sources" in research_data:
                for source in research_data["sources"]:
                    text_content += (
                        f" {source.get('title', '')} {source.get('content', '')}"
                    )

            if "analysis" in research_data:
                text_content += f" {research_data['analysis']}"

            # Get all entities as flat list
            all_entities = []
            for entity_type, entity_list in entities.items():
                all_entities.extend(entity_list)

            # Find relationships between entities
            for i, entity1 in enumerate(all_entities):
                for j, entity2 in enumerate(all_entities[i + 1 :], i + 1):
                    relationship = await self._find_relationship(
                        entity1, entity2, text_content
                    )
                    if relationship:
                        relationships.append(relationship)

            # Sort by strength and limit
            relationships.sort(key=lambda x: x.get("strength", 0), reverse=True)
            return relationships[:100]  # Keep top 100 relationships

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    async def _find_relationship(
        self, entity1: Dict, entity2: Dict, text_content: str
    ) -> Optional[Dict]:
        """Find relationship between two entities."""
        try:
            name1 = entity1["name"].lower()
            name2 = entity2["name"].lower()

            # Look for sentences containing both entities
            sentences = re.split(r"[.!?]+", text_content)
            relationship_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if name1 in sentence_lower and name2 in sentence_lower:
                    relationship_sentences.append(sentence.strip())

            if not relationship_sentences:
                return None

            # Determine relationship type
            relationship_type = "relates_to"  # Default
            relationship_strength = 0.1

            for rel_type, indicators in self.relationship_indicators.items():
                for indicator in indicators:
                    for sentence in relationship_sentences:
                        if indicator in sentence.lower():
                            relationship_type = rel_type
                            relationship_strength = 0.8
                            break
                    if relationship_strength > 0.5:
                        break
                if relationship_strength > 0.5:
                    break

            # Calculate relationship strength based on co-occurrence
            co_occurrence_count = len(relationship_sentences)
            if co_occurrence_count >= 3:
                relationship_strength = max(relationship_strength, 0.9)
            elif co_occurrence_count >= 2:
                relationship_strength = max(relationship_strength, 0.6)
            else:
                relationship_strength = max(relationship_strength, 0.3)

            return {
                "id": hashlib.md5(
                    f"{entity1['id']}-{entity2['id']}".encode()
                ).hexdigest()[:8],
                "source_entity": entity1["id"],
                "target_entity": entity2["id"],
                "relationship_type": relationship_type,
                "strength": relationship_strength,
                "evidence_sentences": relationship_sentences[
                    :3
                ],  # Keep top 3 as evidence
                "co_occurrence_count": co_occurrence_count,
            }

        except Exception as e:
            logger.error(f"Error finding relationship: {e}")
            return None

    async def _build_graph_structure(
        self, entities: Dict, relationships: List[Dict]
    ) -> Dict:
        """Build the graph structure for visualization."""
        try:
            nodes = []
            edges = []

            # Create entity ID to entity mapping
            entity_map = {}
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_map[entity["id"]] = entity

            # Create nodes
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    nodes.append(
                        {
                            "id": entity["id"],
                            "label": entity["name"],
                            "type": entity_type,
                            "size": min(max(entity["importance"] * 20, 10), 50),
                            "color": self._get_entity_color(entity_type),
                            "importance": entity["importance"],
                            "frequency": entity["frequency"],
                        }
                    )

            # Create edges
            for relationship in relationships:
                source_id = relationship["source_entity"]
                target_id = relationship["target_entity"]

                if source_id in entity_map and target_id in entity_map:
                    edges.append(
                        {
                            "id": relationship["id"],
                            "source": source_id,
                            "target": target_id,
                            "type": relationship["relationship_type"],
                            "strength": relationship["strength"],
                            "width": max(relationship["strength"] * 5, 1),
                            "color": self._get_relationship_color(
                                relationship["relationship_type"]
                            ),
                            "evidence_count": relationship.get(
                                "co_occurrence_count", 1
                            ),
                        }
                    )

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.error(f"Error building graph structure: {e}")
            return {"nodes": [], "edges": []}

    async def _generate_visual_layout(self, graph_structure: Dict) -> Dict:
        """Generate visual layout coordinates for the graph."""
        try:
            nodes = graph_structure["nodes"]
            edges = graph_structure["edges"]

            if not nodes:
                return {"nodes": [], "edges": []}

            # Simple force-directed layout simulation
            layout_nodes = []
            node_positions = {}

            # Initialize random positions
            import random

            random.seed(42)  # For consistent layouts

            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / len(nodes)
                radius = 100 + (node.get("importance", 0.5) * 50)

                x = radius * math.cos(angle)
                y = radius * math.sin(angle)

                layout_node = {**node, "x": x, "y": y, "fixed": False}

                layout_nodes.append(layout_node)
                node_positions[node["id"]] = {"x": x, "y": y}

            # Apply simple force simulation (simplified)
            for iteration in range(50):
                # Repulsion between nodes
                for i, node1 in enumerate(layout_nodes):
                    for j, node2 in enumerate(layout_nodes[i + 1 :], i + 1):
                        dx = node2["x"] - node1["x"]
                        dy = node2["y"] - node1["y"]
                        distance = math.sqrt(dx * dx + dy * dy)

                        if distance > 0:
                            force = 1000 / (distance * distance)
                            fx = force * dx / distance
                            fy = force * dy / distance

                            node1["x"] -= fx
                            node1["y"] -= fy
                            node2["x"] += fx
                            node2["y"] += fy

                # Attraction for connected nodes
                for edge in edges:
                    source_node = next(
                        (n for n in layout_nodes if n["id"] == edge["source"]), None
                    )
                    target_node = next(
                        (n for n in layout_nodes if n["id"] == edge["target"]), None
                    )

                    if source_node and target_node:
                        dx = target_node["x"] - source_node["x"]
                        dy = target_node["y"] - source_node["y"]
                        distance = math.sqrt(dx * dx + dy * dy)

                        if distance > 0:
                            force = edge["strength"] * 0.1
                            fx = force * dx / distance
                            fy = force * dy / distance

                            source_node["x"] += fx
                            source_node["y"] += fy
                            target_node["x"] -= fx
                            target_node["y"] -= fy

            # Prepare edges with coordinates
            layout_edges = []
            for edge in edges:
                source_pos = next(
                    (n for n in layout_nodes if n["id"] == edge["source"]), None
                )
                target_pos = next(
                    (n for n in layout_nodes if n["id"] == edge["target"]), None
                )

                if source_pos and target_pos:
                    layout_edges.append(
                        {
                            **edge,
                            "x1": source_pos["x"],
                            "y1": source_pos["y"],
                            "x2": target_pos["x"],
                            "y2": target_pos["y"],
                        }
                    )

            return {"nodes": layout_nodes, "edges": layout_edges}

        except Exception as e:
            logger.error(f"Error generating visual layout: {e}")
            return {"nodes": [], "edges": []}

    async def _identify_clusters(
        self, graph_structure: Dict, entities: Dict
    ) -> List[Dict]:
        """Identify clusters of related entities."""
        try:
            clusters = []
            nodes = graph_structure["nodes"]
            edges = graph_structure["edges"]

            # Group by entity type first
            type_clusters = defaultdict(list)
            for node in nodes:
                type_clusters[node["type"]].append(node)

            cluster_id = 0
            for entity_type, type_nodes in type_clusters.items():
                if len(type_nodes) >= 2:  # Only create clusters with 2+ nodes
                    clusters.append(
                        {
                            "id": f"cluster_{cluster_id}",
                            "name": f"{entity_type.title()} Concepts",
                            "type": "entity_type",
                            "nodes": [node["id"] for node in type_nodes],
                            "size": len(type_nodes),
                            "center_x": sum(node.get("x", 0) for node in type_nodes)
                            / len(type_nodes),
                            "center_y": sum(node.get("y", 0) for node in type_nodes)
                            / len(type_nodes),
                            "color": self._get_entity_color(entity_type),
                        }
                    )
                    cluster_id += 1

            # Identify highly connected clusters
            node_connections = defaultdict(int)
            for edge in edges:
                node_connections[edge["source"]] += 1
                node_connections[edge["target"]] += 1

            # Find hub nodes (highly connected)
            hub_threshold = len(edges) * 0.1  # Top 10% of connections
            hub_nodes = [
                node_id
                for node_id, connections in node_connections.items()
                if connections >= hub_threshold
            ]

            if hub_nodes:
                clusters.append(
                    {
                        "id": f"cluster_{cluster_id}",
                        "name": "Central Concepts",
                        "type": "hub_cluster",
                        "nodes": hub_nodes,
                        "size": len(hub_nodes),
                        "center_x": 0,
                        "center_y": 0,
                        "color": "#ff6b6b",
                    }
                )

            return clusters

        except Exception as e:
            logger.error(f"Error identifying clusters: {e}")
            return []

    async def _generate_navigation_paths(
        self, graph_structure: Dict, context: str
    ) -> List[Dict]:
        """Generate suggested navigation paths through the knowledge graph."""
        try:
            nodes = graph_structure["nodes"]
            edges = graph_structure["edges"]

            if not nodes or not edges:
                return []

            paths = []

            # Find paths from most important nodes
            important_nodes = sorted(
                nodes, key=lambda x: x.get("importance", 0), reverse=True
            )[:5]

            for start_node in important_nodes:
                # Find connected nodes
                connected_nodes = []
                for edge in edges:
                    if edge["source"] == start_node["id"]:
                        target_node = next(
                            (n for n in nodes if n["id"] == edge["target"]), None
                        )
                        if target_node:
                            connected_nodes.append((target_node, edge["strength"]))
                    elif edge["target"] == start_node["id"]:
                        source_node = next(
                            (n for n in nodes if n["id"] == edge["source"]), None
                        )
                        if source_node:
                            connected_nodes.append((source_node, edge["strength"]))

                if connected_nodes:
                    # Sort by connection strength
                    connected_nodes.sort(key=lambda x: x[1], reverse=True)

                    # Create a path with top 3 connected nodes
                    path_nodes = [start_node["id"]]
                    for connected_node, strength in connected_nodes[:3]:
                        path_nodes.append(connected_node["id"])

                    paths.append(
                        {
                            "id": f"path_{len(paths)}",
                            "name": f"Exploring {start_node['label']}",
                            "description": f"Key concepts related to {start_node['label']}",
                            "nodes": path_nodes,
                            "length": len(path_nodes),
                            "total_strength": sum(
                                strength for _, strength in connected_nodes[:3]
                            ),
                        }
                    )

            return paths[:10]  # Return top 10 paths

        except Exception as e:
            logger.error(f"Error generating navigation paths: {e}")
            return []

    async def _calculate_graph_metrics(self, graph_structure: Dict) -> Dict:
        """Calculate metrics for the knowledge graph."""
        try:
            nodes = graph_structure["nodes"]
            edges = graph_structure["edges"]

            if not nodes:
                return {}

            # Basic metrics
            node_count = len(nodes)
            edge_count = len(edges)

            # Calculate degree distribution
            degree_distribution = defaultdict(int)
            node_degrees = defaultdict(int)

            for edge in edges:
                node_degrees[edge["source"]] += 1
                node_degrees[edge["target"]] += 1

            for degree in node_degrees.values():
                degree_distribution[degree] += 1

            # Calculate density
            max_possible_edges = node_count * (node_count - 1) / 2
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0

            # Calculate average degree
            avg_degree = (
                sum(node_degrees.values()) / node_count if node_count > 0 else 0
            )

            # Find most connected nodes
            most_connected = sorted(
                node_degrees.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": round(density, 3),
                "average_degree": round(avg_degree, 2),
                "max_degree": max(node_degrees.values()) if node_degrees else 0,
                "min_degree": min(node_degrees.values()) if node_degrees else 0,
                "most_connected_nodes": most_connected,
                "degree_distribution": dict(degree_distribution),
            }

        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return {}

    def _calculate_entity_importance(self, entity: str, text_content: str) -> float:
        """Calculate the importance of an entity based on frequency and context."""
        try:
            frequency = text_content.lower().count(entity.lower())

            # Base importance on frequency
            importance = min(frequency / 10.0, 1.0)

            # Boost importance for entities in titles or headings
            title_patterns = [r"^.*?{}.*?$".format(re.escape(entity))]
            for pattern in title_patterns:
                if re.search(pattern, text_content, re.MULTILINE | re.IGNORECASE):
                    importance += 0.3

            # Boost for entities with capitals (proper nouns)
            if entity[0].isupper():
                importance += 0.2

            return min(importance, 1.0)

        except Exception:
            return 0.1

    def _extract_context_snippets(
        self, entity: str, text_content: str, max_snippets: int = 3
    ) -> List[str]:
        """Extract context snippets around entity mentions."""
        try:
            snippets = []
            sentences = re.split(r"[.!?]+", text_content)

            for sentence in sentences:
                if entity.lower() in sentence.lower() and len(snippets) < max_snippets:
                    snippets.append(sentence.strip())

            return snippets

        except Exception:
            return []

    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type."""
        color_map = {
            "person": "#3498db",
            "organization": "#e74c3c",
            "location": "#2ecc71",
            "concept": "#f39c12",
            "technology": "#9b59b6",
        }
        return color_map.get(entity_type, "#95a5a6")

    def _get_relationship_color(self, relationship_type: str) -> str:
        """Get color for relationship type."""
        color_map = {
            "causes": "#e74c3c",
            "enables": "#2ecc71",
            "requires": "#f39c12",
            "improves": "#3498db",
            "reduces": "#e67e22",
            "relates_to": "#95a5a6",
            "competes_with": "#c0392b",
        }
        return color_map.get(relationship_type, "#bdc3c7")

    async def _merge_entities(
        self, existing_entities: Dict, new_entities: Dict
    ) -> Dict:
        """Merge new entities with existing ones."""
        try:
            merged = dict(existing_entities)

            for entity_type, entity_list in new_entities.items():
                if entity_type not in merged:
                    merged[entity_type] = []

                # Add new entities, avoiding duplicates
                existing_names = {e["name"].lower() for e in merged[entity_type]}

                for entity in entity_list:
                    if entity["name"].lower() not in existing_names:
                        merged[entity_type].append(entity)
                        existing_names.add(entity["name"].lower())

            return merged

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            return existing_entities

    async def _merge_relationships(
        self, existing_relationships: List[Dict], new_relationships: List[Dict]
    ) -> List[Dict]:
        """Merge new relationships with existing ones."""
        try:
            merged = list(existing_relationships)

            # Create set of existing relationship pairs
            existing_pairs = {
                (r["source_entity"], r["target_entity"]) for r in existing_relationships
            }

            for relationship in new_relationships:
                pair = (relationship["source_entity"], relationship["target_entity"])
                reverse_pair = (
                    relationship["target_entity"],
                    relationship["source_entity"],
                )

                if pair not in existing_pairs and reverse_pair not in existing_pairs:
                    merged.append(relationship)
                    existing_pairs.add(pair)

            return merged

        except Exception as e:
            logger.error(f"Error merging relationships: {e}")
            return existing_relationships

    async def _search_concepts(self, graph: Dict, query: str) -> List[Dict]:
        """Search for concepts in the knowledge graph."""
        try:
            results = []
            query_lower = query.lower()

            for entity_type, entities in graph["entities"].items():
                for entity in entities:
                    if query_lower in entity["name"].lower():
                        results.append(
                            {
                                "type": "entity",
                                "entity": entity,
                                "relevance": self._calculate_relevance(
                                    query, entity["name"]
                                ),
                            }
                        )

            # Sort by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:10]

        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []

    async def _search_entities(self, graph: Dict, query: str) -> List[Dict]:
        """Search for specific entities."""
        return await self._search_concepts(graph, query)

    async def _search_relationships(self, graph: Dict, query: str) -> List[Dict]:
        """Search for relationships."""
        try:
            results = []
            query_lower = query.lower()

            for relationship in graph["relationships"]:
                if query_lower in relationship["relationship_type"]:
                    results.append(
                        {
                            "type": "relationship",
                            "relationship": relationship,
                            "relevance": self._calculate_relevance(
                                query, relationship["relationship_type"]
                            ),
                        }
                    )

            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:10]

        except Exception as e:
            logger.error(f"Error searching relationships: {e}")
            return []

    async def _find_paths(self, graph: Dict, query: str) -> List[Dict]:
        """Find paths in the knowledge graph."""
        try:
            # For simplicity, return navigation paths that contain the query term
            results = []
            query_lower = query.lower()

            for path in graph.get("navigation_paths", []):
                if (
                    query_lower in path["name"].lower()
                    or query_lower in path["description"].lower()
                ):
                    results.append(
                        {
                            "type": "path",
                            "path": path,
                            "relevance": self._calculate_relevance(query, path["name"]),
                        }
                    )

            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:5]

        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []

    async def _general_search(self, graph: Dict, query: str) -> List[Dict]:
        """General search across all graph elements."""
        try:
            results = []

            # Search concepts
            concept_results = await self._search_concepts(graph, query)
            results.extend(concept_results)

            # Search relationships
            relationship_results = await self._search_relationships(graph, query)
            results.extend(relationship_results)

            # Search paths
            path_results = await self._find_paths(graph, query)
            results.extend(path_results)

            # Sort all results by relevance
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:20]

        except Exception as e:
            logger.error(f"Error in general search: {e}")
            return []

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text."""
        try:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())

            if not query_words or not text_words:
                return 0.0

            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0
